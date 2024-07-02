from fastprogress.fastprogress import progress_bar
from functools import partial
import numpy as np
import jax
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow_probability.substrates.jax.distributions import InverseGamma as IG
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

from dynamax.ssm import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.inference import lgssm_posterior_sample_identity, lgssm_filter_identity, \
    lgssm_smoother_identity
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, \
    ParamsLGSSMEmissions, ParamsTVLGSSM
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update, mvn_posterior_update, \
    ig_posterior_update
from dynamax.utils.utils import pytree_stack, psd_solve, symmetrize

class SuffStatsLGSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statistics for LGSSM parameter estimation."""
    pass


class LinearGaussianSSM(SSM):
    r"""
    Linear Gaussian State Space Model.

    The model is defined as follows

    $$p(z_1) = \mathcal{N}(z_1 \mid m, S)$$
    $$p(z_t \mid z_{t-1}, u_t) = \mathcal{N}(z_t \mid F_t z_{t-1} + B_t u_t + b_t, Q_t)$$
    $$p(y_t \mid z_t) = \mathcal{N}(y_t \mid H_t z_t + D_t u_t + d_t, R_t)$$

    where

    * $z_t$ is a latent state of size `state_dim`,
    * $y_t$ is an emission of size `emission_dim`
    * $u_t$ is an input of size `input_dim` (defaults to 0)
    * $F$ = dynamics (transition) matrix
    * $B$ = optional input-to-state weight matrix
    * $b$ = optional input-to-state bias vector
    * $Q$ = covariance matrix of dynamics (system) noise
    * $H$ = emission (observation) matrix
    * $D$ = optional input-to-emission weight matrix
    * $d$ = optional input-to-emission bias vector
    * $R$ = covariance function for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state

    The parameters of the model are stored in a :class:`ParamsLGSSM`.
    You can create the parameters manually, or by calling :meth:`initialize`.

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term $b$. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term $d$. Defaults to True.

    """

    def __init__(
            self,
            state_dim: int,
            emission_dim: int,
            input_dim: int = 0,
            has_dynamics_bias: bool = True,
            has_emissions_bias: bool = True
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initialize(
            self,
            key: PRNGKey = jr.PRNGKey(0),
            initial_mean: Optional[Float[Array, "state_dim"]] = None,
            initial_covariance=None,
            dynamics_weights=None,
            dynamics_bias=None,
            dynamics_input_weights=None,
            dynamics_covariance=None,
            emission_weights=None,
            emission_bias=None,
            emission_input_weights=None,
            emission_covariance=None
    ) -> Tuple[ParamsLGSSM, ParamsLGSSM]:
        r"""Initialize model parameters that are set to None, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            initial_mean: parameter $m$. Defaults to None.
            initial_covariance: parameter $S$. Defaults to None.
            dynamics_weights: parameter $F$. Defaults to None.
            dynamics_bias: parameter $b$. Defaults to None.
            dynamics_input_weights: parameter $B$. Defaults to None.
            dynamics_covariance: parameter $Q$. Defaults to None.
            emission_weights: parameter $H$. Defaults to None.
            emission_bias: parameter $d$. Defaults to None.
            emission_input_weights: parameter $D$. Defaults to None.
            emission_covariance: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsLGSSM, ParamsLGSSM]: parameters and their properties.
        """

        # Arbitrary default values, for demo purposes.
        _initial_mean = jnp.zeros(self.state_dim)
        _initial_covariance = jnp.eye(self.state_dim)
        _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)
        _emission_weights = jr.normal(key, (self.emission_dim, self.state_dim))
        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # Create nested dictionary of params
        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=default(initial_mean, _initial_mean),
                cov=default(initial_covariance, _initial_covariance)),
            dynamics=ParamsLGSSMDynamics(
                weights=default(dynamics_weights, _dynamics_weights),
                bias=default(dynamics_bias, _dynamics_bias),
                input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                cov=default(dynamics_covariance, _dynamics_covariance),
                ar_dependency=None),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
                ar_dependency=None)
        )

        # The keys of param_props must match those of params!
        props = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            dynamics=ParamsLGSSMDynamics(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                ar_dependency=ParameterProperties()),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                ar_dependency=ParameterProperties())
        )
        return params, props

    def initial_distribution(
            self,
            params: ParamsLGSSM,
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean, params.initial.cov)

    def transition_distribution(
            self,
            params: ParamsLGSSM,
            state: Float[Array, "state_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights @ state + params.dynamics.input_weights @ inputs
        if self.has_dynamics_bias:
            mean += params.dynamics.bias
        return MVN(mean, params.dynamics.cov)

    def emission_distribution(
            self,
            params: ParamsLGSSM,
            state: Float[Array, "state_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias
        return MVN(mean, params.emissions.cov)

    def marginal_log_prob(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> Scalar:
        filtered_posterior = lgssm_filter(params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def filter(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMFiltered:
        return lgssm_filter(params, emissions, inputs)

    def smoother(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMSmoothed:
        return lgssm_smoother(params, emissions, inputs)

    def posterior_sample(
            self,
            key: PRNGKey,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> Float[Array, "ntime state_dim"]:
        return lgssm_posterior_sample(key, params, emissions, inputs)

    def posterior_predictive(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        posterior = lgssm_smoother(params, emissions, inputs)
        H = params.emissions.weights
        b = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + b if self.has_emissions_bias else posterior.smoothed_means @ H.T
        smoothed_emissions_cov = H @ posterior.smoothed_covariances @ H.T + R
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    # Expectation-maximization (EM) code
    def e_step(
            self,
            params: ParamsLGSSM,
            emissions: Union[Float[Array, "num_timesteps emission_dim"],
            Float[Array, "num_batches num_timesteps emission_dim"]],
            inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
            Float[Array, "num_batches num_timesteps input_dim"]]] = None,
    ) -> Tuple[SuffStatsLGSSM, Scalar]:
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(params, emissions, inputs)

        # shorthand
        Ex = posterior.smoothed_means
        Exp = posterior.smoothed_means[:-1]
        Exn = posterior.smoothed_means[1:]
        Vx = posterior.smoothed_covariances
        Vxp = posterior.smoothed_covariances[:-1]
        Vxn = posterior.smoothed_covariances[1:]
        Expxn = posterior.smoothed_cross_covariances

        # Append bias to the inputs
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        init_stats = (Ex0, Ex0x0T, 1)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                              num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[:self.state_dim, :self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
        if not self.has_emissions_bias:
            emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik

    def initialize_m_step_state(
            self,
            params: ParamsLGSSM,
            props: ParamsLGSSM
    ) -> Any:
        return None

    def m_step(
            self,
            params: ParamsLGSSM,
            props: ParamsLGSSM,
            batch_stats: SuffStatsLGSSM,
            m_step_state: Any
    ) -> Tuple[ParamsLGSSM, Any]:

        def fit_linear_regression(ExxT, ExyT, EyyT, N):
            # Solve a linear regression given sufficient statistics
            W = psd_solve(ExxT, ExyT).T
            Sigma = (EyyT - W @ ExyT - ExyT.T @ W.T + W @ ExxT @ W.T) / N
            return W, Sigma

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], None)

        HD, R = fit_linear_regression(*emission_stats)
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], None)

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q,
                                         ar_dependency=None),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                           ar_dependency=None)
        )
        return params, m_step_state


class LinearGaussianConjugateSSM(LinearGaussianSSM):
    r"""
    Linear Gaussian State Space Model with conjugate priors for the model parameters.

    The parameters are the same as LG-SSM. The priors are as follows:

    * p(m, S) = NIW(loc, mean_concentration, df, scale) # normal inverse wishart
    * p([F, B, b], Q) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
    * p([H, D, d], R) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term b. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term d. Defaults to True.

    """

    def __init__(self,
                 state_dim,
                 emission_dim,
                 input_dim=0,
                 has_dynamics_bias=True,
                 has_emissions_bias=True,
                 **kw_priors):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim,
                         has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias)

        # Initialize prior distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_prior = default_prior(
            'initial_prior',
            NIW(loc=jnp.zeros(self.state_dim),
                mean_concentration=1.,
                df=self.state_dim + 0.1,
                scale=jnp.eye(self.state_dim)))

        self.dynamics_prior = default_prior(
            'dynamics_prior',
            MNIW(loc=jnp.zeros((self.state_dim, self.state_dim + self.input_dim + self.has_dynamics_bias)),
                 col_precision=jnp.eye(self.state_dim + self.input_dim + self.has_dynamics_bias),
                 df=self.state_dim + 0.1,
                 scale=jnp.eye(self.state_dim)))

        self.emission_prior = default_prior(
            'emission_prior',
            MNIW(loc=jnp.zeros((self.emission_dim, self.state_dim + self.input_dim + self.has_emissions_bias)),
                 col_precision=jnp.eye(self.state_dim + self.input_dim + self.has_emissions_bias),
                 df=self.emission_dim + 0.1,
                 scale=jnp.eye(self.emission_dim)))

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return dict(inputs=(self.input_dim,)) if self.input_dim > 0 else dict()

    def log_prior(
            self,
            params: ParamsLGSSM
    ) -> Scalar:
        lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean))

        # dynamics
        dynamics_bias = params.dynamics.bias if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
        dynamics_matrix = jnp.column_stack((params.dynamics.weights,
                                            params.dynamics.input_weights,
                                            dynamics_bias))
        lp += self.dynamics_prior.log_prob((params.dynamics.cov, dynamics_matrix))

        emission_bias = params.emissions.bias if self.has_emissions_bias else jnp.zeros((self.emission_dim, 0))
        emission_matrix = jnp.column_stack((params.emissions.weights,
                                            params.emissions.input_weights,
                                            emission_bias))
        lp += self.emission_prior.log_prob((params.emissions.cov, emission_matrix))
        return lp

    def initialize_m_step_state(
            self,
            params: ParamsLGSSM,
            props: ParamsLGSSM
    ) -> Any:
        return None

    def m_step(
            self,
            params: ParamsLGSSM,
            props: ParamsLGSSM,
            batch_stats: SuffStatsLGSSM,
            m_step_state: Any):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MAP estimation jointly
        initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
        S, m = initial_posterior.mode()

        dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
        Q, FB = dynamics_posterior.mode()
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

        emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
        R, HD = emission_posterior.mode()
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q,
                                         ar_dependency=None),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                           ar_dependency=None)
        )
        return params, m_step_state

    def fit_blocked_gibbs(
            self,
            key: PRNGKey,
            initial_params: ParamsLGSSM,
            sample_size: int,
            emissions: Float[Array, "nbatch ntime emission_dim"],
            inputs: Optional[Float[Array, "nbatch ntime input_dim"]] = None
    ) -> ParamsLGSSM:
        r"""Estimate parameter posterior using block-Gibbs sampler.

        Args:
            key: random number key.
            initial_params: starting parameters.
            sample_size: how many samples to draw.
            emissions: set of observation sequences.
            inputs: optional set of input sequences.

        Returns:
            parameter object, where each field has `sample_size` copies as leading batch dimension.
        """
        num_timesteps = len(emissions)

        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        def sufficient_stats_from_sample(states):
            """Convert samples of states to sufficient statistics."""
            inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x, xp, xn = states, states[:-1], states[1:]
            u, up = inputs_joint, inputs_joint[:-1]
            y = emissions

            init_stats = (x[0], jnp.outer(x[0], x[0]), 1)

            # Quantities for the dynamics distribution
            # Let zp[t] = [x[t], u[t]] for t = 0...T-2
            sum_zpzpT = jnp.block([[xp.T @ xp, xp.T @ up], [up.T @ xp, up.T @ up]])
            sum_zpxnT = jnp.block([[xp.T @ xn], [up.T @ xn]])
            sum_xnxnT = xn.T @ xn
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, num_timesteps - 1)
            if not self.has_dynamics_bias:
                dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                  num_timesteps - 1)

            # Quantities for the emissions
            # Let z[t] = [x[t], u[t]] for t = 0...T-1
            sum_zzT = jnp.block([[x.T @ x, x.T @ u], [u.T @ x, u.T @ u]])
            sum_zyT = jnp.block([[x.T @ y], [u.T @ y]])
            sum_yyT = y.T @ y
            emission_stats = (sum_zzT, sum_zyT, sum_yyT, num_timesteps)
            if not self.has_emissions_bias:
                emission_stats = (sum_zzT[:-1, :-1], sum_zyT[:-1, :], sum_yyT, num_timesteps)

            return init_stats, dynamics_stats, emission_stats

        def lgssm_params_sample(rng, stats):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats
            rngs = iter(jr.split(rng, 3))

            # Sample the initial params
            initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
            S, m = initial_posterior.sample(seed=next(rngs))

            # Sample the dynamics params
            dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
            Q, FB = dynamics_posterior.sample(seed=next(rngs))
            F = FB[:, :self.state_dim]
            B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
                else (FB[:, self.state_dim:], None)

            # Sample the emission params
            emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
            R, HD = emission_posterior.sample(seed=next(rngs))
            H = HD[:, :self.state_dim]
            D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
                else (HD[:, self.state_dim:], None)

            params = ParamsLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q,
                                             ar_dependency=None),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               ar_dependency=None)
            )
            return params

        @jit
        def one_sample(_params, rng):
            rngs = jr.split(rng, 2)
            # Sample latent states
            # print(_params.dynamics.bias)
            states = lgssm_posterior_sample(rngs[0], _params, emissions, inputs)
            # Sample parameters
            _stats = sufficient_stats_from_sample(states)
            return lgssm_params_sample(rngs[1], _stats)

        sample_of_params = []
        keys = iter(jr.split(key, sample_size))
        current_params = initial_params
        for _ in progress_bar(range(sample_size)):
            sample_of_params.append(current_params)
            current_params = one_sample(current_params, next(keys))

        # print(sample_of_params[0])
        # print(sample_of_params[-1])
        return pytree_stack(sample_of_params)


class TimeVaryingLinearGaussianConjugateSSM(LinearGaussianSSM):
    r"""
    Time Varying Linear Gaussian State Space Model with conjugate priors for the model parameters.

    The model is defined as follows

    $$p(z_1) = \mathcal{N}(z_1 \mid m, S)$$
    $$p(A_1) = \mathcal{N}(z_1 \mid m, S)$$
    $$p(C_1) = \mathcal{N}(z_1 \mid m, S)$$
    $$p(z_t \mid z_{t-1}, u_t) = \mathcal{N}(z_t \mid F_t z_{t-1} + B_t u_t + b_t, Q_t)$$
    $$p(y_t \mid z_t) = \mathcal{N}(y_t \mid H_t z_t + D_t u_t + d_t, R_t)$$

    where

    * $z_t$ is a latent state of size `state_dim`,
    * $y_t$ is an emission of size `emission_dim`
    * $u_t$ is an input of size `input_dim` (defaults to 0)
    * $F_t$ = dynamics (transition) matrix
    * $B_t$ = optional input-to-state weight matrix
    * $b$ = optional input-to-state bias vector
    * $Q$ = covariance matrix of dynamics (system) noise
    * $H_t$ = emission (observation) matrix
    * $D_t$ = optional input-to-emission weight matrix
    * $d$ = optional input-to-emission bias vector
    * $R$ = covariance function for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state

    The parameters of the model are stored in a :class:`ParamsLGSSM`.
    You can create the parameters manually, or by calling :meth:`initialize`.

    The priors are as follows:

    * p(m, S) = NIW(loc, mean_concentration, df, scale) # normal inverse wishart
    * p([F, B, b], Q) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
    * p([H, D, d], R) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term b. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term d. Defaults to True.

    """

    def __init__(
            self,
            state_dim: int,
            emission_dim: int,
            input_dim: int = 0,
            num_trials: int = 0,  # number of trials
            sequence_length: int = 0,  # number of timesteps per trial
            time_varying_emissions: bool = True,
            has_dynamics_bias: bool = False,
            has_emissions_bias: bool = False,
            fix_initial: bool = False,
            fix_dynamics: bool = False,
            fix_emissions: bool = False,
            init_emissions_with_standard_normal: bool = True,
            normalize_emissions: bool = False,
            orthogonal_emissions_weights: bool = False,
            emission_weights_scale: float = 1.0,
            standardize_states: bool = False,
            standardize_per_latent_dim: bool = True,
            per_column_ar_dependency: bool=False,
            update_emissions_param_ar_dependency_variance: bool = False,
            update_initial_covariance: bool = False,
            update_dynamics_covariance: bool = False,  # learn diagonal covariance matrix
            update_emissions_covariance: bool = False,  # learn diagonal covariance matrix
            update_init_emissions_mean: bool = False,
            update_init_emissions_covariance: bool = False,
            EPS: float = 1e-6,
            **kw_priors
    ):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim,
                         has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias)

        self.num_trials = num_trials
        self.time_varying_emissions = time_varying_emissions
        self.sequence_length = sequence_length

        self.fix_initial = fix_initial
        self.fix_dynamics = fix_dynamics
        self.fix_emissions = fix_emissions

        self.init_emissions_with_standard_normal = init_emissions_with_standard_normal
        self.normalize_emissions = normalize_emissions
        self.orthogonal_emissions_weights = orthogonal_emissions_weights
        self.emission_weights_scale = emission_weights_scale

        self.per_column_ar_dependency = per_column_ar_dependency
        self.update_emissions_param_ar_dependency_variance = update_emissions_param_ar_dependency_variance
        self.update_dynamics_covariance = update_dynamics_covariance
        self.update_emissions_covariance = update_emissions_covariance
        self.update_initial_covariance = update_initial_covariance
        self.update_init_emissions_mean = update_init_emissions_mean
        self.update_init_emissions_covariance = update_init_emissions_covariance

        self.standardize_states = standardize_states
        self.standardize_per_latent_dim = standardize_per_latent_dim

        self.EPS = EPS

        # Initialize prior distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_mean_prior = default_prior(
            'initial_mean_prior',
            MVN(loc=jnp.zeros(self.state_dim),
                covariance_matrix=jnp.eye(self.state_dim)))

        self.initial_covariance_prior = default_prior(
            'initial_covariance_prior',
            IG(concentration=1.0, scale=1.0)
        )

        if update_emissions_param_ar_dependency_variance:
            self.emissions_ar_dependency_prior = default_prior(
                'emissions_ar_dependency_prior',
                IG(concentration=1.0, scale=1.0)
            )

        if update_dynamics_covariance:
            self.dynamics_covariance_prior = default_prior(
                'dynamics_covariance_prior',
                IG(concentration=1.0, scale=1.0)
            )

        if update_emissions_covariance:
            self.emissions_covariance_prior = default_prior(
                'emissions_covariance_prior',
                IG(concentration=1.0, scale=1.0)
            )

        # prior on dynamics parameters
        self.dynamics_prior = default_prior(
            'dynamics_prior',
            MVN(loc=jnp.zeros(self.state_dim * (self.state_dim + self.has_dynamics_bias)),
                covariance_matrix=jnp.eye(self.state_dim * (self.state_dim + self.has_dynamics_bias)))
        )

        if time_varying_emissions:
            # prior on initial emissions
            self.emission_prior = default_prior(
                'emission_prior',
                MVN(loc=jnp.zeros(self.emission_dim * (self.state_dim + self.has_emissions_bias)),
                    covariance_matrix=1e2 * jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)))
            )
            self.initial_emissions_covariance_prior = default_prior(
                'init_emissions_cov_prior',
                IG(concentration=1.0, scale=0.01)
            )
        else:
            # prior on emissions parameters
            self.emission_prior = default_prior(
                'emission_prior',
                MVN(loc=jnp.zeros(self.emission_dim * (self.state_dim + self.has_emissions_bias)),
                    covariance_matrix=1e2 * jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)))
            )

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return dict(inputs=(self.input_dim,)) if self.input_dim > 0 else dict()

    def initialize(
            self,
            key: PRNGKey = jr.PRNGKey(0),
            initial_mean: Optional[Float[Array, "state_dim"]] = None,
            initial_covariance=None,
            dynamics_weights=None,
            dynamics_bias=None,
            dynamics_input_weights=None,
            dynamics_covariance=None,
            stabilize_dynamics=True,
            emission_weights=None,
            emission_bias=None,
            emission_input_weights=None,
            emission_covariance=None,
            emissions_param_ar_dependency_variance=None,
            update_emissions_ar_var=False,
            initial_emissions_mean=None,
            initial_emissions_cov=None
    ) -> Tuple[ParamsTVLGSSM, ParamsTVLGSSM]:
        r"""Initialize model parameters that are set to None, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            initial_mean: parameter $m$. Defaults to None.
            initial_covariance: parameter $S$. Defaults to None.
            dynamics_weights: parameter $F$. Defaults to None.
            dynamics_bias: parameter $b$. Defaults to None.
            dynamics_input_weights: parameter $B$. Defaults to None.
            dynamics_covariance: parameter $Q$. Defaults to None.
            emission_weights: parameter $H$. Defaults to None.
            emission_bias: parameter $d$. Defaults to None.
            emission_input_weights: parameter $D$. Defaults to None.
            emission_covariance: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsLGSSM, ParamsLGSSM]: parameters and their properties.
        """

        # Arbitrary default values, for demo purposes.
        _initial_mean = jnp.zeros((self.num_trials, self.state_dim))
        _initial_covariance = jnp.tile(jnp.eye(self.state_dim)[None], (self.num_trials, 1, 1))

        if self.time_varying_emissions:
            _initial_emissions_mean = jnp.zeros(self.emission_dim * (self.state_dim + self.has_emissions_bias))
            # _initial_emissions_cov = jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias))
            if emissions_param_ar_dependency_variance is None:
                if self.per_column_ar_dependency:
                    _emissions_param_ar_dependency_variance = jnp.ones(self.state_dim + self.has_emissions_bias) * 1e-2
                    _initial_emissions_cov = jnp.ones(self.state_dim + self.has_emissions_bias) * 1e-2
                else:
                    _emissions_param_ar_dependency_variance = 1e-2
                    _initial_emissions_cov = 1e-2
            else:
                _emissions_param_ar_dependency_variance = emissions_param_ar_dependency_variance
                _initial_emissions_cov = emissions_param_ar_dependency_variance
        else:
            _initial_emissions_cov, _initial_emissions_mean = None, None
            _emissions_param_ar_dependency_variance = 0.0

        key1, key = jr.split(key, 2)
        _dynamics_weights = jr.normal(key1, shape=(self.state_dim, self.state_dim))
        if stabilize_dynamics:
            eigdecomp_result = np.linalg.eig(_dynamics_weights)
            eigenvalues = eigdecomp_result.eigenvalues
            eigenvectors = eigdecomp_result.eigenvectors
            dynamics = eigenvectors @ np.diag(eigenvalues / np.abs(eigenvalues)) @ np.linalg.inv(eigenvectors)
            _dynamics_weights = jnp.array(np.real(dynamics))

        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)

        if self.time_varying_emissions:
            if self.per_column_ar_dependency:
                _emissions_param_ar_dependency_cov = jnp.diag(jnp.tile(_emissions_param_ar_dependency_variance, self.emission_dim))
            else:
                _emissions_param_ar_dependency_cov = jnp.eye(
                    self.emission_dim * self.state_dim) * _emissions_param_ar_dependency_variance

            keys = jr.split(key, self.num_trials)
            key = keys[-1]
            def _get_emission_weights(prev_weights, current_key):
                current_weights_dist = MVN(loc=prev_weights,
                                           covariance_matrix=_emissions_param_ar_dependency_cov)
                current_weights = current_weights_dist.sample(seed=current_key)
                current_weights = current_weights.reshape(self.emission_dim, (self.state_dim + self.has_emissions_bias))
                if self.orthogonal_emissions_weights:
                    current_weights = jnp.linalg.svd(current_weights, full_matrices=False).U
                elif self.normalize_emissions:
                    current_weights = current_weights / jnp.linalg.norm(current_weights, ord=2, axis=0, keepdims=True)
                current_weights = current_weights.reshape(-1)
                return current_weights, current_weights

            key1, key = jr.split(key, 2)
            if self.init_emissions_with_standard_normal:
                #initial_emission_weights = jr.normal(key1, shape=(self.emission_dim * (self.state_dim + self.has_emissions_bias),))
                initial_emission_weights = jr.orthogonal(key1, self.emission_dim)
                initial_emission_weights = initial_emission_weights[:, :self.state_dim + self.has_emissions_bias].reshape(-1)
            else:
                initial_emission_weights = MVN(loc=jnp.zeros(self.emission_dim * (self.state_dim + self.has_emissions_bias), ),
                                               covariance_matrix=self.num_trials * _emissions_param_ar_dependency_cov).sample(seed=key1)

            initial_emission_weights = initial_emission_weights.reshape(self.emission_dim, (self.state_dim + self.has_emissions_bias))
            if self.orthogonal_emissions_weights:
                initial_emission_weights = jnp.linalg.svd(initial_emission_weights, full_matrices=False).U
            elif self.normalize_emissions:
                initial_emission_weights = initial_emission_weights / jnp.linalg.norm(initial_emission_weights, ord=2,
                                                                                      axis=0, keepdims=True)
            initial_emission_weights = initial_emission_weights.reshape(-1)
            _, _emission_weights = jax.lax.scan(_get_emission_weights, initial_emission_weights, keys[:-1])
            _emission_weights = jnp.concatenate([initial_emission_weights[None], _emission_weights])
            _emission_weights = _emission_weights.reshape(_emission_weights.shape[0],
                                                          self.emission_dim,
                                                          (self.state_dim + self.has_emissions_bias))
            _emission_weights = self.emission_weights_scale * _emission_weights

            if update_emissions_ar_var:
                if self.per_column_ar_dependency:
                    emissions_ar_diff = jnp.diff(_emission_weights, axis=0)
                    _emissions_param_ar_dependency_variance = vmap(jnp.var)(emissions_ar_diff.reshape(-1, (self.state_dim + self.has_emissions_bias)).T)
                else:
                    concatenated_emissions_weights = _emission_weights.reshape(self.num_trials, -1)
                    emissions_ar_diff = jnp.diff(concatenated_emissions_weights, axis=0)
                    _emissions_param_ar_dependency_variance = jnp.var(emissions_ar_diff.reshape(-1))

            if self.has_emissions_bias:
                _emission_weights, _emission_bias = _emission_weights[:, :, :-1], _emission_weights[:, :, -1]
            else:
                _emission_bias = None
        else:
            key1, key = jr.split(key, 2)
            _emission_weights = jr.normal(key1, shape=(self.emission_dim, self.state_dim))
            if self.orthogonal_emissions_weights:
                _emission_weights = jnp.linalg.svd(_emission_weights, full_matrices=False).U
                _emission_weights = self.emission_weights_scale * _emission_weights
            elif self.normalize_emissions:
                _emission_weights = _emission_weights / jnp.linalg.norm(_emission_weights, ord=2, axis=-2)[None]
                _emission_weights = self.emission_weights_scale * _emission_weights
            _emission_bias = jnp.zeros(self.emission_dim) if self.has_emissions_bias else None

        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # Create nested dictionary of params
        params = ParamsTVLGSSM(
            initial=ParamsLGSSMInitial(
                mean=default(initial_mean, _initial_mean),
                cov=default(initial_covariance, _initial_covariance)),
            dynamics=ParamsLGSSMDynamics(
                weights=default(dynamics_weights, _dynamics_weights),
                bias=default(dynamics_bias, _dynamics_bias),
                input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
                ar_dependency=_emissions_param_ar_dependency_variance),
            initial_emissions=ParamsLGSSMInitial(
                mean=default(initial_emissions_mean, _initial_emissions_mean),
                cov=default(initial_emissions_cov, _initial_emissions_cov))
        )

        # The keys of param_props must match those of params!
        props = ParamsTVLGSSM(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            dynamics=ParamsLGSSMDynamics(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                ar_dependency=ParameterProperties()),
            initial_emissions=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()))
        )
        return params, props

    # All SSMs support sampling
    def sample(
            self,
            params: ParameterSet,
            key: PRNGKey,
            num_timesteps: int,
            inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None,
    ) -> Tuple[Float[Array, "num_timesteps state_dim"],
    Float[Array, "num_timesteps emission_dim"]]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$ and (optionally) inputs $u_{1:T}$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            inputs: inputs $u_{1:T}$

        Returns:
            latent states and emissions

        """

        def _dynamics_outer_step(carry, outer_args):
            key, t = outer_args

            def _dynamics_step(prev_state, args):
                key, inpt, idx = args
                state = self.transition_distribution(params, prev_state, inpt).sample(seed=key)
                return state, state

            # Sample the initial state
            key1, key = jr.split(key, 2)
            initial_input = tree_map(lambda x: x[0], inputs)
            initial_state = self.initial_distribution(t, params, initial_input).sample(seed=key1)

            # Sample the remaining emissions and states
            next_keys = jr.split(key, self.sequence_length - 1)
            next_inputs = tree_map(lambda x: x[1:], inputs)
            next_indices = jnp.arange(1, self.sequence_length)
            _, next_states = lax.scan(_dynamics_step, initial_state, (next_keys, next_inputs, next_indices))

            # Concatenate the initial state and emission with the following ones
            expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
            states = tree_map(expand_and_cat, initial_state, next_states)

            return None, states

        keys = jr.split(key, self.num_trials + 1)
        _, states = lax.scan(_dynamics_outer_step, None, (keys[:-1], jnp.arange(self.num_trials)))

        if self.standardize_states:
            if self.standardize_per_latent_dim:
                axis = (0, 1)
            else:
                axis = (0, 1, 2)
            states_mean = jnp.mean(states, axis=axis, keepdims=True)
            states_std = jnp.std(states, axis=axis, keepdims=True)
            states = (states - states_mean) / states_std

        def _emissions_outer_step(carry, outer_args):
            key, state, t = outer_args

            def _emissions_step(prev_state, args):
                key, x, inpt, idx = args
                emission_distribution = self.emission_distribution(t, params, x, inpt)
                emission = emission_distribution.sample(seed=key)
                signal = emission_distribution.mean()
                return None, (emission, signal)

            # Sample the remaining emissions and states
            next_keys = jr.split(key, self.sequence_length)
            next_states = tree_map(lambda x: x, state)
            next_inputs = tree_map(lambda x: x, inputs)
            next_indices = jnp.arange(self.sequence_length)
            _, (next_emissions, next_signals) = lax.scan(_emissions_step, None, (next_keys, next_states,
                                                                                 next_inputs, next_indices))
            return None, (next_emissions, next_signals)

        keys = jr.split(keys[-1], self.num_trials)
        _, (emissions, signals) = lax.scan(_emissions_outer_step, None, (keys, states, jnp.arange(self.num_trials)))

        return states, emissions, signals

    def initial_distribution(
            self,
            timestep: int,
            params: ParamsLGSSM,
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean[timestep], params.initial.cov[timestep])

    def transition_distribution(
            self,
            params: ParamsTVLGSSM,
            state: Float[Array, "state_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights @ state + params.dynamics.input_weights @ inputs
        if self.has_dynamics_bias:
            mean += params.dynamics.bias
        return MVN(mean, params.dynamics.cov)

    def emission_distribution(
            self,
            timestep: int,
            params: ParamsTVLGSSM,
            state: Float[Array, "state_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        if self.time_varying_emissions:
            mean = params.emissions.weights[timestep] @ state + params.emissions.input_weights @ inputs
            if self.has_emissions_bias:
                mean += params.emissions.bias[timestep]
        else:
            mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
            if self.has_emissions_bias:
                mean += params.emissions.bias
        return MVN(mean, params.emissions.cov)

    def marginal_log_prob(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            add_prior_and_posterior: bool = True,
            states_samples: jnp.array = None,
    ) -> Scalar:

        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        trials = jnp.arange(num_trials, dtype=int)

        def _get_marginal_ll(emission, input, mask, trial_r):
            return lgssm_filter(params, emission, input, mask, trial_r).marginal_loglik

        _get_marginal_ll_vmap = vmap(_get_marginal_ll, in_axes=(0, 0, 0, 0))
        marginal_lls = _get_marginal_ll_vmap(emissions, inputs, masks, trials)
        marginal_ll = marginal_lls.sum()

        if add_prior_and_posterior:
            if self.per_column_ar_dependency:
                emissions_param_ar_dependency_cov = jnp.diag(jnp.tile(params.emissions.ar_dependency, self.emission_dim))
                emissions_initial_emissions_cov = jnp.diag(jnp.tile(params.initial_emissions.cov, self.emission_dim))
            else:
                emissions_param_ar_dependency_cov = jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)) * params.emissions.ar_dependency
                emissions_initial_emissions_cov = jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)) * params.initial_emissions.cov
            # add prior
            def _compute_trial_emissions_lp(prev_lp, current_t):
                current_param = params.emissions.weights[current_t]
                next_param = params.emissions.weights[current_t + 1]

                if self.has_emissions_bias:
                    current_param = jnp.hstack([current_param, params.emissions.bias[current_t][:, None]])
                    next_param = jnp.hstack([next_param, params.emissions.bias[current_t+1][:, None]])

                current_lp = prev_lp + MVN(loc=jnp.ravel(current_param),
                                           covariance_matrix=emissions_param_ar_dependency_cov).log_prob(jnp.ravel(next_param))
                return current_lp, None

            marginal_ll, _ = jax.lax.scan(_compute_trial_emissions_lp, marginal_ll, jnp.arange(num_trials - 1))
            marginal_ll += MVN(params.initial_emissions.mean, emissions_initial_emissions_cov).log_prob(
                jnp.ravel(params.emissions.weights[0]))

            # subtract posterior
            Rinv = jnp.linalg.inv(params.emissions.cov)
            y = emissions
            N = y.shape[-1]


            # UPDATE FROM HERE
            def _iterate_over_states_samples(prev_lp, current_sample):
                states = states_samples[current_sample]
                x, xp, xn = states, states[:, :-1], states[:, 1:]
                D = x.shape[-1]
                emissions_stats_1 = jnp.einsum('bt,bti,jk,btl->bjikl', masks, x, Rinv, x).reshape(num_trials, N * D,
                                                                                                  N * D)
                emissions_covs = jnp.linalg.inv(emissions_stats_1)
                emissions_stats_2 = jnp.einsum('bt,bti,ik,btl->bkl', masks, y, Rinv, x).reshape(num_trials, -1)
                emissions_y = jnp.einsum('bij,bj->bi', emissions_covs, emissions_stats_2)

                _emissions_params = ParamsLGSSM(
                    initial=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                               cov=params.initial_emissions.cov),
                    dynamics=ParamsLGSSMDynamics(weights=jnp.eye(N * D),
                                                 bias=None,
                                                 input_weights=None,
                                                 cov=emissions_param_ar_dependency_cov),
                    emissions=ParamsLGSSMEmissions(
                        weights=jnp.eye(N * D),
                        bias=None,
                        input_weights=None,
                        cov=emissions_covs,
                        ar_dependency=None)
                )

                _emissions_smoother = lgssm_smoother_identity(_emissions_params,
                                                              emissions_y,
                                                              jnp.zeros((num_trials, 0)),
                                                              jnp.ones(num_trials, dtype=bool))

                _emissions_smoothed_means = _emissions_smoother.smoothed_means
                _emissions_smoothed_covs = _emissions_smoother.smoothed_covariances

                def _compute_posterior_lp(_prev_lp, current_trial):
                    current_param = params.emissions.weights[current_trial]
                    _current_lp = _prev_lp + MVN(loc=_emissions_smoothed_means[current_trial],
                                                 covariance_matrix=_emissions_smoothed_covs[current_trial]).log_prob(
                        jnp.ravel(current_param))
                    return _current_lp, None

                current_lp, _ = jax.lax.scan(_compute_posterior_lp, 0.0, jnp.arange(num_trials))

                return None, current_lp

            _, current_lps = jax.lax.scan(_iterate_over_states_samples, None, jnp.arange(len(states_samples)))

            marginal_ll -= logsumexp(current_lps)
            marginal_ll += jnp.log(len(states_samples))

        return marginal_ll

    def marginal_log_prob_v2(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            states,
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            add_prior_and_posterior: bool = True,
            emissions_weights_samples: jnp.array = None,
    ) -> Scalar:

        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)

        x, xp, xn = states, states[:, :-1], states[:, 1:]
        y = emissions
        N, D = y.shape[-1], x.shape[-1]
        num_trials = y.shape[0]
        Rinv = jnp.linalg.inv(params.emissions.cov)
        emissions_stats_1 = jnp.einsum('bt,bti,jk,btl->bjikl', masks, x, Rinv, x).reshape(num_trials, N * D, N * D)
        emissions_covs = jnp.linalg.inv(emissions_stats_1)
        emissions_stats_2 = jnp.einsum('bt,bti,ik,btl->bkl', masks, y, Rinv, x).reshape(num_trials, -1)
        emissions_y = jnp.einsum('bij,bj->bi', emissions_covs, emissions_stats_2)

        if self.per_column_ar_dependency:
            emissions_param_ar_dependency_cov = jnp.diag(jnp.tile(params.emissions.ar_dependency, self.emission_dim))
        else:
            emissions_param_ar_dependency_cov = jnp.eye(self.emission_dim * self.state_dim) * params.emissions.ar_dependency

        _emissions_params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                       cov=params.initial_emissions.cov),
            dynamics=ParamsLGSSMDynamics(weights=jnp.eye(N * D),
                                         bias=None,
                                         input_weights=None,
                                         cov=emissions_param_ar_dependency_cov,
                                         ar_dependency=None),
            emissions=ParamsLGSSMEmissions(
                weights=jnp.eye(N * D),
                bias=None,
                input_weights=None,
                cov=emissions_covs,
                ar_dependency=None)
        )

        _emissions_filter = lgssm_filter_identity(_emissions_params,
                                                  emissions_y,
                                                  jnp.zeros((num_trials, 0)),
                                                  jnp.ones(num_trials, dtype=bool))

        marginal_ll = _emissions_filter.marginal_loglik
        if add_prior_and_posterior:
            # add prior
            marginal_ll += MVN(params.initial.mean, params.initial.cov).log_prob(states[:, 0]).sum()

            def _compute_dynamics_lp(prev_lp, current_t):
                current_state_mean = jnp.einsum('ij,rj->ri', params.dynamics.weights, states[:, current_t])
                new_lp = MVN(current_state_mean, params.dynamics.cov).log_prob(states[:, current_t + 1])
                masked_new_lp = jnp.nansum(masks[:, current_t + 1] * new_lp)
                current_lp = prev_lp + masked_new_lp
                return current_lp, None

            marginal_ll, _ = jax.lax.scan(_compute_dynamics_lp, marginal_ll, jnp.arange(self.sequence_length - 1))

            # subtract posterior
            def _iterate_over_emissions_samples(prev_lp, current_sample):
                emissions_weights = emissions_weights_samples[current_sample]

                _states_params = ParamsTVLGSSM(
                    initial=ParamsLGSSMInitial(mean=params.initial.mean, cov=params.initial.cov),
                    dynamics=ParamsLGSSMDynamics(weights=params.dynamics.weights, bias=params.dynamics.bias,
                                                 input_weights=params.dynamics.input_weights, cov=params.dynamics.cov),
                    emissions=ParamsLGSSMEmissions(weights=emissions_weights, bias=params.emissions.bias,
                                                   input_weights=params.emissions.input_weights,
                                                   cov=params.emissions.cov,
                                                   ar_dependency=params.emissions.ar_dependency),
                    initial_dynamics=ParamsLGSSMInitial(mean=params.initial_dynamics.mean,
                                                        cov=params.initial_dynamics.cov),
                    initial_emissions=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                                         cov=params.initial_emissions.cov),
                )

                trials = jnp.arange(num_trials, dtype=int)

                def _get_smoothers(emission, input, mask, trial_r):
                    smoother = lgssm_smoother(_states_params, emission, input, mask, trial_r)
                    return smoother.smoothed_means, smoother.smoothed_covariances

                _get_smoothers_vmap = vmap(_get_smoothers, in_axes=(0, 0, 0, 0))
                smoothed_means, smoothed_covs = _get_smoothers_vmap(emissions, inputs, masks, trials)

                states_flattened = states.reshape(-1, self.state_dim)
                smoothed_means = smoothed_means.reshape(-1, self.state_dim)
                smoothed_covs = smoothed_covs.reshape(-1, self.state_dim, self.state_dim)
                masks_flattened = masks.reshape(-1)

                def _compute_posterior_lp(_prev_lp, current_t):
                    current_param = states_flattened[current_t]
                    _current_lp = _prev_lp + masks_flattened[current_t] * MVN(loc=smoothed_means[current_t],
                                                                              covariance_matrix=smoothed_covs[
                                                                                  current_t]).log_prob(
                        jnp.ravel(current_param))
                    return _current_lp, None

                current_lp, _ = jax.lax.scan(_compute_posterior_lp, 0.0, jnp.arange(len(masks_flattened)))

                return None, current_lp

            _, current_lps = jax.lax.scan(_iterate_over_emissions_samples, None,
                                          jnp.arange(len(emissions_weights_samples)))

            marginal_ll -= logsumexp(current_lps)
            marginal_ll += jnp.log(len(emissions_weights_samples))

        return marginal_ll

    def filter(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None
    ) -> PosteriorGSSMFiltered:
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        trials = jnp.arange(self.num_trials, dtype=int)
        lgssm_filter_vmap = vmap(lgssm_filter, in_axes=(None, 0, 0, 0, 0))
        filters = lgssm_filter_vmap(params, emissions, inputs, masks, trials)
        return filters

    def smoother(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None
    ) -> PosteriorGSSMSmoothed:
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        trials = jnp.arange(self.num_trials, dtype=int)
        lgssm_smoother_vmap = vmap(lgssm_smoother, in_axes=(None, 0, 0, 0, 0))
        smoothers = lgssm_smoother_vmap(params, emissions, inputs, masks, trials)
        return smoothers

    def posterior_sample(
            self,
            key: PRNGKey,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None
    ) -> Float[Array, "ntime state_dim"]:
        if masks is None:
            masks = jnp.ones(emissions.shape[0], dtype=bool)
        return lgssm_posterior_sample(key, params, emissions, inputs, masks)

    def posterior_predictive(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        posterior = self.smoother(params, emissions, inputs, masks)
        H = params.emissions.weights
        d = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]

        smoothed_means = posterior.smoothed_means
        if self.standardize_states:
            if self.standardize_per_latent_dim:
                axis = (0, 1)
            else:
                axis = (0, 1, 2)
            states_mean = jnp.mean(smoothed_means, axis=axis, keepdims=True)
            states_std = jnp.std(smoothed_means, axis=axis, keepdims=True)
            smoothed_means = (smoothed_means - states_mean) / states_std

        smoothed_emissions = jnp.einsum('...lx,...yx->...ly', smoothed_means, H)
        smoothed_emissions_cov = jnp.einsum('...ya,...lab,...xb->...lyx', H, posterior.smoothed_covariances, H) + R

        if self.has_emissions_bias:
            if self.time_varying_emissions:
                smoothed_emissions += d[:, None]
            else:
                smoothed_emissions += d[None, None]

        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, :, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    def posterior_predictive_v2(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            states,
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)

        x, xp, xn = states, states[:, :-1], states[:, 1:]
        y = emissions
        N, D = y.shape[-1], x.shape[-1]
        num_trials = y.shape[0]
        Rinv = jnp.linalg.inv(params.emissions.cov)
        emissions_stats_1 = jnp.einsum('bt,bti,jk,btl->bjikl', masks, x, Rinv, x).reshape(num_trials, N * D, N * D)
        emissions_covs = jnp.linalg.inv(emissions_stats_1)
        emissions_stats_2 = jnp.einsum('bt,bti,ik,btl->bkl', masks, y, Rinv, x).reshape(num_trials, -1)
        emissions_y = jnp.einsum('bij,bj->bi', emissions_covs, emissions_stats_2)

        if self.per_column_ar_dependency:
            emissions_param_ar_dependency_cov = jnp.diag(jnp.tile(params.emissions.ar_dependency, self.emission_dim))
        else:
            emissions_param_ar_dependency_cov = jnp.eye(self.emission_dim * self.state_dim) * params.emissions.ar_dependency

        _emissions_params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                       cov=params.initial_emissions.cov),
            dynamics=ParamsLGSSMDynamics(weights=jnp.eye(N * D),
                                         bias=None,
                                         input_weights=None,
                                         cov=emissions_param_ar_dependency_cov),
            emissions=ParamsLGSSMEmissions(
                weights=jnp.eye(N * D),
                bias=None,
                input_weights=None,
                cov=emissions_covs,
                ar_dependency=None)
        )

        _emissions_smoother = lgssm_smoother_identity(_emissions_params,
                                                      emissions_y,
                                                      jnp.zeros((num_trials, 0)),
                                                      jnp.ones(num_trials, dtype=bool))

        # expand emission weights to the original shape
        H = _emissions_smoother.smoothed_means.reshape(num_trials, N, D)

        smoothed_emissions = jnp.einsum('...lx,...yx->...ly', states, H)

        return smoothed_emissions, None

    def log_joint(
            self,
            params: ParamsTVLGSSM,
            states,
            emissions,
            inputs,
            masks
    ) -> Scalar:

        """"""""
        # Double check priors for time-varying dynamics and emissions
        """"""""

        # initial state
        # lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean))
        lp = self.initial_mean_prior.log_prob(params.initial.mean).sum()
        flattened_cov = vmap(jnp.diag)(params.initial.cov)
        lp += self.initial_covariance_prior.log_prob(flattened_cov.flatten()).sum()
        lp += MVN(params.initial.mean, params.initial.cov).log_prob(states[:, 0]).sum()

        # dynamics & states
        dynamics_bias = params.dynamics.bias if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
        dynamics_matrix = jnp.column_stack(
            (params.dynamics.weights,
             params.dynamics.input_weights,
             dynamics_bias))
        lp += self.dynamics_prior.log_prob(jnp.ravel(dynamics_matrix))
        def _compute_dynamics_lp(prev_lp, current_t):
            current_state_mean = jnp.einsum('ij,rj->ri', params.dynamics.weights, states[:, current_t])
            if self.has_dynamics_bias:
                current_state_mean += params.dynamics.bias[None]
            new_lp = MVN(current_state_mean, params.dynamics.cov).log_prob(states[:, current_t + 1])
            masked_new_lp = jnp.nansum(masks[:, current_t + 1] * new_lp)
            current_lp = prev_lp + masked_new_lp
            return current_lp, None
        lp, _ = jax.lax.scan(_compute_dynamics_lp, lp, jnp.arange(self.sequence_length - 1))
        if self.update_dynamics_covariance:
            lp += self.dynamics_covariance_prior.log_prob(jnp.diag(params.dynamics.cov)).sum()

        # emissions & observations
        if self.time_varying_emissions:
            if self.per_column_ar_dependency:
                emissions_param_ar_dependency_cov = jnp.diag(
                    jnp.tile(params.emissions.ar_dependency, self.emission_dim))
                emissions_initial_emissions_cov = jnp.diag(
                    jnp.tile(params.initial_emissions.cov, self.emission_dim))
            else:
                emissions_param_ar_dependency_cov = jnp.eye(
                    self.emission_dim * (self.state_dim + self.has_emissions_bias)) * params.emissions.ar_dependency
                emissions_initial_emissions_cov = jnp.eye(
                    self.emission_dim * (self.state_dim + self.has_emissions_bias)) * params.initial_emissions.cov
            # also need to edit ar dependency variance update step.
            def _compute_emissions_lp(prev_lp, current_t):
                # current_param = params.emissions.weights[current_t]
                current_y_mean = jnp.einsum('rij,rj->ri', params.emissions.weights, states[:, current_t])
                if self.has_emissions_bias:
                    current_y_mean += params.emissions.bias
                new_lp = MVN(current_y_mean, params.emissions.cov).log_prob(emissions[:, current_t])
                masked_new_lp = jnp.nansum(masks[:, current_t] * new_lp)
                current_lp = prev_lp + masked_new_lp
                return current_lp, None

            lp, _ = jax.lax.scan(_compute_emissions_lp, lp, jnp.arange(self.sequence_length))

            def _compute_trial_emissions_lp(prev_lp, current_t):
                current_param = params.emissions.weights[current_t]
                next_param = params.emissions.weights[current_t + 1]
                if self.has_emissions_bias:
                    current_param = jnp.hstack([current_param, params.emissions.bias[current_t][:, None]])
                    next_param = jnp.hstack([next_param, params.emissions.bias[current_t+1][:, None]])
                current_lp = prev_lp + MVN(loc=jnp.ravel(current_param),
                                           covariance_matrix=emissions_param_ar_dependency_cov).log_prob(jnp.ravel(next_param))
                return current_lp, None

            lp, _ = jax.lax.scan(_compute_trial_emissions_lp, lp, jnp.arange(self.num_trials - 1))

            if self.has_emissions_bias:
                initial_emissions = jnp.concatenate([params.emissions.weights[0], params.emissions.bias[0][:, None]], axis=-1)
                initial_emissions = jnp.ravel(initial_emissions)
            else:
                initial_emissions = jnp.ravel(params.emissions.weights[0])
            lp += MVN(params.initial_emissions.mean, emissions_initial_emissions_cov).log_prob(initial_emissions)
            # lp += self.emission_prior.log_prob((params.initial_emissions.cov, params.initial_emissions.mean))
            lp += self.emission_prior.log_prob(params.initial_emissions.mean)
            # lp += self.initial_emissions_covariance_prior.log_prob(jnp.diag(params.initial_emissions.cov)).sum()

            if self.update_emissions_param_ar_dependency_variance:
                ar_dependency_lp = self.emissions_ar_dependency_prior.log_prob(params.emissions.ar_dependency)
                initial_emissions_cov_lp = self.initial_emissions_covariance_prior.log_prob(params.initial_emissions.cov)
                if self.per_column_ar_dependency:
                    ar_dependency_lp = ar_dependency_lp.sum()
                    initial_emissions_cov_lp = initial_emissions_cov_lp.sum()
                lp += ar_dependency_lp
                lp += initial_emissions_cov_lp

        else:
            emission_bias = params.emissions.bias if self.has_emissions_bias else jnp.zeros((self.emission_dim, 0))
            emission_matrix = jnp.column_stack(
                (params.emissions.weights,
                 params.emissions.input_weights,
                 emission_bias))
            lp += self.emission_prior.log_prob(jnp.ravel(emission_matrix))

            def _compute_emissions_lp(prev_lp, current_t):
                current_y_mean = jnp.einsum('ij,rj->ri', params.emissions.weights, states[:, current_t])
                if self.has_emissions_bias:
                    current_y_mean += params.emissions.bias[None]
                new_lp = MVN(current_y_mean, params.emissions.cov).log_prob(emissions[:, current_t])
                masked_new_lp = jnp.nansum(masks[:, current_t] * new_lp)
                current_lp = prev_lp + masked_new_lp
                return current_lp, None

            lp, _ = jax.lax.scan(_compute_emissions_lp, lp, jnp.arange(self.sequence_length))

        if self.update_emissions_covariance:
            lp += self.emissions_covariance_prior.log_prob(jnp.diag(params.emissions.cov)).sum()

        return lp

    def initialize_m_step_state(
            self,
            params: ParamsTVLGSSM,
            props: ParamsTVLGSSM
    ) -> Any:
        return None

    def fit_blocked_gibbs(
            self,
            key: PRNGKey,
            initial_params: ParamsTVLGSSM,
            sample_size: int,
            emissions: Float[Array, "nbatch ntime emission_dim"],
            inputs: Optional[Float[Array, "nbatch ntime input_dim"]] = None,
            return_states: bool = False,
            return_n_samples: int = 100,
            print_ll: bool = False,
            masks: jnp.array = None,
            trial_masks: jnp.array = None,
            fixed_states: jnp.array = None,
    ):
        r"""Estimate parameter posterior using block-Gibbs sampler.

        Args:
            key: random number key.
            initial_params: starting parameters.
            sample_size: how many samples to draw.
            emissions: set of observation sequences.
            inputs: optional set of input sequences.

        Returns:
            parameter object, where each field has `sample_size` copies as leading batch dimension.
        """
        num_timesteps = len(emissions)
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if trial_masks is None:
            trial_masks = jnp.ones(emissions.shape[0], dtype=bool)

        def sufficient_stats_from_sample(states, params):
            """Convert samples of states to sufficient statistics."""
            inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x, xp, xn = states, states[:, :-1], states[:, 1:]
            u, up = inputs_joint, inputs_joint[:-1]
            y = emissions

            init_stats = (x[:, 0],)

            N, D = y.shape[-1], states.shape[-1]
            # Optimized Code
            if not self.orthogonal_emissions_weights and not self.normalize_emissions:
                reshape_dim = D * (D + self.has_dynamics_bias)
                if self.has_dynamics_bias:
                    xp = jnp.pad(xp, ((0, 0), (0, 0), (0, 1)), constant_values=1)
                Qinv = jnp.linalg.inv(params.dynamics.cov + jnp.eye(params.dynamics.cov.shape[-1]) * self.EPS)
                dynamics_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', xp, Qinv, xp, masks[:, :-1]).reshape(reshape_dim, reshape_dim)
                dynamics_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', xn, Qinv, xp, masks[:, 1:]).reshape(-1)
                dynamics_stats = (dynamics_stats_1, dynamics_stats_2)
            else:
                dynamics_stats = None

            # Quantities for the emissions
            if not self.time_varying_emissions:
                reshape_dim = N * (D + self.has_emissions_bias)
                if self.has_emissions_bias:
                    x = jnp.pad(x, ((0, 0), (0, 0), (0, 1)), constant_values=1)
                Rinv = jnp.linalg.inv(params.emissions.cov)
                emissions_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', x, Rinv, x, masks).reshape(reshape_dim, reshape_dim)
                emissions_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', y, Rinv, x, masks).reshape(-1)
                emission_stats = (emissions_stats_1, emissions_stats_2)
            else:
                emission_stats = None

            return init_stats, dynamics_stats, emission_stats

        def lgssm_params_sample(rng, stats, states, params):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats
            n_splits = 7 + self.update_emissions_param_ar_dependency_variance + self.update_emissions_covariance
            rngs = iter(jr.split(rng, n_splits))

            # Sample the emission params
            if self.fix_emissions:
                H = params.emissions.weights
                d = params.emissions.bias
                D = params.emissions.input_weights
                R = params.emissions.cov
                initial_emissions_cov = params.initial_emissions.cov
                initial_emissions_mean = params.initial_emissions.mean
                emissions_ar_dependency = params.emissions.ar_dependency
            else:
                if self.time_varying_emissions:
                    x, xp, xn = states, states[:, :-1], states[:, 1:]
                    y = emissions
                    N, D = y.shape[-1], x.shape[-1]
                    Rinv = jnp.linalg.inv(params.emissions.cov)

                    reshape_dim = N * (D + self.has_emissions_bias)
                    if self.has_emissions_bias:
                        x = jnp.pad(x, ((0, 0), (0, 0), (0, 1)), constant_values=1)

                    emissions_stats_1 = jnp.einsum('bt,bti,jk,btl->bjikl', masks, x, Rinv, x).reshape(self.num_trials,
                                                                                                      reshape_dim, reshape_dim)

                    emissions_covs = jnp.linalg.inv(emissions_stats_1)
                    # emissions_covs = symmetrize(emissions_covs)
                    # emissions_covs = jnp.linalg.solve(emissions_stats_1,
                    #                                   jnp.eye(emissions_stats_1.shape[-1])[None])
                    # emissions_covs = symmetrize(emissions_covs) + jnp.eye(emissions_covs.shape[-1]) * 1e-6
                    emissions_stats_2 = jnp.einsum('bt,bti,ik,btl->bkl', masks, y, Rinv, x).reshape(self.num_trials, -1)
                    emissions_y = jnp.einsum('bij,bj->bi', emissions_covs, emissions_stats_2)
                    # emissions_y = jnp.linalg.solve(emissions_stats_1, emissions_stats_2[..., None])[..., 0]

                    if self.per_column_ar_dependency:
                        emissions_param_ar_dependency_cov = jnp.diag(
                            jnp.tile(params.emissions.ar_dependency, self.emission_dim))
                        emissions_initial_emissions_cov = jnp.diag(
                            jnp.tile(params.initial_emissions.cov, self.emission_dim))
                    else:
                        emissions_param_ar_dependency_cov = jnp.eye(
                            self.emission_dim * (self.state_dim + self.has_emissions_bias)) * params.emissions.ar_dependency
                        emissions_initial_emissions_cov = jnp.eye(self.emission_dim * (
                                    self.state_dim + self.has_emissions_bias)) * params.initial_emissions.cov

                    _emissions_params = ParamsLGSSM(
                        initial=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                                   cov=emissions_initial_emissions_cov),
                        dynamics=ParamsLGSSMDynamics(weights=jnp.eye(self.emission_dim * self.state_dim),
                                                     bias=None,
                                                     input_weights=None,
                                                     cov=emissions_param_ar_dependency_cov),
                        emissions=ParamsLGSSMEmissions(
                            weights=jnp.eye(self.emission_dim * self.state_dim),
                            bias=None,
                            input_weights=None,
                            cov=emissions_covs,
                            ar_dependency=None)
                    )

                    _emissions_weights = lgssm_posterior_sample_identity(next(rngs),
                                                                         _emissions_params,
                                                                         emissions_y,
                                                                         jnp.zeros((self.num_trials, 0)), trial_masks)

                    # expand emission weights to the original shape
                    _emissions_weights = _emissions_weights.reshape(self.num_trials,
                                                                    self.emission_dim,
                                                                    self.state_dim + self.has_emissions_bias)
                    H, d = (_emissions_weights[:, :, :-1], _emissions_weights[:, :, -1]) if self.has_emissions_bias \
                        else (_emissions_weights, None)

                    if self.orthogonal_emissions_weights:
                        svd_result = jnp.linalg.svd(H, full_matrices=False)
                        H = svd_result.U
                        x_correction = jnp.einsum('bd,bde->bde', svd_result.S, svd_result.Vh)
                        # correct the states
                        states = jnp.einsum('bij,btj->bti', x_correction, states)
                    elif self.normalize_emissions:
                        weights_norm = jnp.linalg.norm(H, ord=2, axis=1, keepdims=True)
                        H = H / weights_norm

                        # correct the states
                        states = jnp.einsum('bij,btj->btj', weights_norm, states)

                    if self.orthogonal_emissions_weights or self.normalize_emissions:
                        # recompute the initial and dynamics sufficient statistics
                        init_stats = (states[:, 0],)
                        xp, xn = states[:, :-1], states[:, 1:]
                        Qinv = jnp.linalg.inv(params.dynamics.cov)
                        dynamics_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', xp, Qinv, xp, masks[:, :-1]).reshape(D * D, D * D)
                        dynamics_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', xn, Qinv, xp, masks[:, 1:]).reshape(-1)
                        dynamics_stats = (dynamics_stats_1, dynamics_stats_2)

                    D = jnp.zeros((self.emission_dim, 0))

                    if self.has_emissions_bias:
                        initial_emissions = jnp.concatenate([H[0], d[0][:, None]], axis=-1).reshape(-1)
                    else:
                        initial_emissions = H[0].reshape(-1)

                    if self.update_init_emissions_mean:
                        # init_emissions_stats_1 = jnp.linalg.inv(emissions_initial_emissions_cov + jnp.eye(emissions_initial_emissions_cov.shape[-1]) * self.EPS)
                        init_emissions_stats_1 = jnp.linalg.inv(emissions_param_ar_dependency_cov)
                        init_emissions_stats_2 = init_emissions_stats_1 @ initial_emissions
                        init_emissions_stats = (init_emissions_stats_1, init_emissions_stats_2)

                        init_emissions_posterior = mvn_posterior_update(self.emission_prior, init_emissions_stats)
                        initial_emissions_mean = init_emissions_posterior.sample(seed=next(rngs))
                    else:
                        initial_emissions_mean = params.initial_emissions.mean

                    if self.update_init_emissions_covariance:
                        if self.per_column_ar_dependency:
                            reshape_dim = self.state_dim + self.has_emissions_bias
                            init_emissions_cov_stats_1 = self.emission_dim / 2
                            initial_emissions_reshaped = initial_emissions.reshape(self.emission_dim, reshape_dim)
                            initial_emissions_mean_reshaped = initial_emissions_mean.reshape(self.emission_dim, reshape_dim)
                            init_emissions_cov_stats_2 = initial_emissions_reshaped - initial_emissions_mean_reshaped
                            init_emissions_cov_stats_2 = jnp.nansum(jnp.square(init_emissions_cov_stats_2), axis=0) / 2
                            init_emissions_cov_stats = (init_emissions_cov_stats_1,
                                                        init_emissions_cov_stats_2)
                            init_emissions_cov_posterior = ig_posterior_update(self.initial_emissions_covariance_prior,
                                                                               init_emissions_cov_stats)
                            initial_emissions_cov = init_emissions_cov_posterior.sample(seed=next(rngs))
                            # initial_emissions_cov = jnp.where(jnp.logical_or(jnp.isnan(initial_emissions_cov),
                            #                                                  initial_emissions_cov < self.EPS),
                            #                                   self.EPS, initial_emissions_cov)
                        else:
                            init_emissions_cov_stats_1 = (self.emission_dim * (self.state_dim + self.has_emissions_bias)) / 2
                            init_emissions_cov_stats_2 = initial_emissions - initial_emissions_mean
                            init_emissions_cov_stats_2 = jnp.nansum(jnp.square(init_emissions_cov_stats_2)) / 2
                            init_emissions_cov_stats = (init_emissions_cov_stats_1,
                                                        init_emissions_cov_stats_2)
                            init_emissions_cov_posterior = ig_posterior_update(self.initial_emissions_covariance_prior,
                                                                               init_emissions_cov_stats)
                            initial_emissions_cov = init_emissions_cov_posterior.sample(seed=next(rngs))
                        # initial_emissions_cov += 1e-6
                    else:
                        initial_emissions_cov = params.initial_emissions.cov

                    if self.update_emissions_param_ar_dependency_variance:
                        if self.has_emissions_bias:
                            updated_emissions_weights = jnp.concatenate([H, d[:, :, None]], axis=-1)
                        else:
                            updated_emissions_weights = H

                        if self.per_column_ar_dependency:
                            emissions_ar_dependency_stats_1 = (self.emission_dim * (self.num_trials-1)) / 2
                            emissions_ar_dependency_stats_2 = jnp.diff(updated_emissions_weights, axis=0)
                            # emissions_ar_dependency_stats_1 = (self.emission_dim * (self.num_trials)) / 2
                            # emissions_ar_dependency_stats_2 = jnp.diff(updated_emissions_weights, axis=0,
                            #                                            prepend=initial_emissions_mean.reshape(self.emission_dim, (self.state_dim + self.has_emissions_bias))[None])

                            emissions_ar_dependency_stats_2 = jnp.nansum(jnp.square(emissions_ar_dependency_stats_2), axis=(0, 1)) / 2
                            emissions_ar_dependency_stats = (emissions_ar_dependency_stats_1,
                                                             emissions_ar_dependency_stats_2)
                            emissions_ar_dependency_posterior = ig_posterior_update(self.emissions_ar_dependency_prior,
                                                                                    emissions_ar_dependency_stats)
                            emissions_ar_dependency = emissions_ar_dependency_posterior.sample(seed=next(rngs))
                            # initial_emissions_cov = jnp.diag(jnp.tile(emissions_ar_dependency, self.emission_dim))
                            # emissions_ar_dependency = jnp.where(jnp.logical_or(jnp.isnan(emissions_ar_dependency),
                            #                                                  emissions_ar_dependency < self.EPS),
                            #                                   self.EPS, emissions_ar_dependency)
                        else:
                            emissions_ar_dependency_stats_1 = (self.emission_dim * (self.state_dim + self.has_emissions_bias) * (self.num_trials-1)) / 2
                            concatenated_emissions_weights = updated_emissions_weights.reshape(self.num_trials, -1)
                            # emissions_ar_dependency_stats_2 = jnp.diff(concatenated_emissions_weights, axis=0,
                            #                                            prepend=initial_emissions_mean.reshape(-1)[None])
                            emissions_ar_dependency_stats_2 = jnp.diff(concatenated_emissions_weights, axis=0)
                            emissions_ar_dependency_stats_2 = jnp.nansum(jnp.square(emissions_ar_dependency_stats_2)) / 2
                            emissions_ar_dependency_stats = (emissions_ar_dependency_stats_1,
                                                             emissions_ar_dependency_stats_2)
                            emissions_ar_dependency_posterior = ig_posterior_update(self.emissions_ar_dependency_prior,
                                                                                    emissions_ar_dependency_stats)
                            emissions_ar_dependency = emissions_ar_dependency_posterior.sample(seed=next(rngs))
                            # initial_emissions_cov = jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)) * emissions_ar_dependency
                        # emissions_ar_dependency += 1e-6
                    else:
                        emissions_ar_dependency = params.emissions.ar_dependency
                        # initial_emissions_cov = jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)) * params.emissions.ar_dependency

                    if self.update_emissions_covariance:
                        emissions_cov_stats_1 = jnp.ones((self.emission_dim, 1)) * (jnp.sum(masks) / 2)
                        emissions_mean = jnp.einsum('btx,byx->bty', states, H)
                        if self.has_emissions_bias:
                            emissions_mean += d[:, None]
                        sqr_err_flattened = jnp.square(emissions - emissions_mean).reshape(-1, self.emission_dim)
                        masks_flattened = masks.reshape(-1)
                        sqr_err_flattened = sqr_err_flattened * masks_flattened[:, None]
                        emissions_cov_stats_2 = jnp.nansum(sqr_err_flattened, axis=0) / 2

                        emissions_cov_stats_2 = jnp.expand_dims(emissions_cov_stats_2, -1)
                        emissions_cov_stats = (emissions_cov_stats_1, emissions_cov_stats_2)
                        emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior,
                                                                      emissions_cov_stats)
                        emissions_cov = emissions_cov_posterior.sample(seed=next(rngs))
                        emissions_cov = jnp.where(jnp.logical_or(jnp.isnan(emissions_cov), emissions_cov < self.EPS), self.EPS, emissions_cov)
                        R = jnp.diag(jnp.ravel(emissions_cov))
                    else:
                        R = params.emissions.cov

                else:
                    emission_posterior = mvn_posterior_update(self.emission_prior, emission_stats)
                    _emissions_weights = emission_posterior.sample(seed=next(rngs))
                    _emissions_weights = _emissions_weights.reshape(self.emission_dim, (self.state_dim + self.has_emissions_bias))
                    H, d = (_emissions_weights[:, :-1], _emissions_weights[:, -1]) if self.has_emissions_bias \
                        else (_emissions_weights, None)

                    if self.orthogonal_emissions_weights:
                        svd_result = jnp.linalg.svd(_emissions_weights, full_matrices=False)
                        _emissions_weights = svd_result.U
                        x_correction = jnp.einsum('d,de->de', svd_result.S, svd_result.Vh)
                        # correct the states
                        states = jnp.einsum('ij,btj->bti', x_correction, states)

                        # recompute the initial and dynamics sufficient statistics
                        init_stats = (states[:, 0],)

                        xp, xn = states[:, :-1], states[:, 1:]
                        Qinv = jnp.linalg.inv(params.dynamics.cov)

                        dynamics_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', xp, Qinv, xp, masks[:, :-1]).reshape(D * D,
                                                                                                                  D * D)
                        dynamics_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', xn, Qinv, xp, masks[:, 1:]).reshape(-1)
                        dynamics_stats = (dynamics_stats_1, dynamics_stats_2)

                    if self.update_emissions_covariance:
                        emissions_cov_stats_1 = jnp.ones((self.emission_dim, 1)) * (jnp.sum(masks) / 2)
                        emissions_mean = jnp.einsum('btx,yx->bty', states, H)
                        if self.has_emissions_bias:
                            emissions_mean += d
                        sqr_err_flattened = jnp.square(emissions - emissions_mean).reshape(-1, self.emission_dim)
                        masks_flattened = masks.reshape(-1, 1)
                        sqr_err_flattened = sqr_err_flattened * masks_flattened
                        emissions_cov_stats_2 = jnp.nansum(sqr_err_flattened, axis=0) / 2
                        emissions_cov_stats_2 = jnp.expand_dims(emissions_cov_stats_2, -1)
                        emissions_cov_stats = (emissions_cov_stats_1, emissions_cov_stats_2)
                        emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior,
                                                                      emissions_cov_stats)
                        emissions_cov = emissions_cov_posterior.sample(seed=next(rngs))
                        R = jnp.diag(jnp.ravel(emissions_cov))
                    else:
                        R = params.emissions.cov
                    D = jnp.zeros((self.emission_dim, 0))
                    initial_emissions_cov, initial_emissions_mean = None, None
                    emissions_ar_dependency = None

            # Sample the initial params
            if self.fix_initial:
                S, m = params.initial.cov, params.initial.mean
            else:
                initial_stats_1 = jnp.linalg.inv(params.initial.cov + jnp.eye(params.initial.cov.shape[-1]))
                initial_stats_2 = jnp.einsum('bij,bj->bi',
                                             initial_stats_1,
                                             init_stats[0])
                initial_stats = (initial_stats_1, initial_stats_2)

                initial_posterior = mvn_posterior_update(self.initial_mean_prior, initial_stats)
                m = initial_posterior.sample(seed=next(rngs))

                if self.update_initial_covariance:
                    init_cov_stats_1 = jnp.ones((self.num_trials * self.state_dim, 1)) / 2
                    init_cov_stats_2 = jnp.square(init_stats[0] - m) / 2
                    init_cov_stats_2 = init_cov_stats_2.flatten()
                    init_cov_stats_2 = jnp.expand_dims(init_cov_stats_2, -1)
                    init_cov_stats = (init_cov_stats_1, init_cov_stats_2)
                    init_cov_posterior = ig_posterior_update(self.initial_covariance_prior, init_cov_stats)
                    init_cov = init_cov_posterior.sample(seed=next(rngs))
                    # init_cov = jnp.where(jnp.logical_or(jnp.isnan(init_cov),
                    #                                          init_cov < self.EPS), self.EPS, init_cov)
                    init_cov = jnp.ravel(init_cov).reshape(self.num_trials, self.state_dim)
                    S = vmap(jnp.diag)(init_cov)
                else:
                    S = params.initial.cov

            # Sample the dynamics params
            if self.fix_dynamics:
                F = params.dynamics.weights
                b = params.dynamics.bias
                B = params.dynamics.input_weights
                Q = params.dynamics.cov
            else:
                xp, xn = states[:, :-1], states[:, 1:]

                dynamics_posterior = mvn_posterior_update(self.dynamics_prior, dynamics_stats)
                _dynamics_weights = dynamics_posterior.sample(seed=next(rngs))

                _dynamics_weights = _dynamics_weights.reshape(self.state_dim, self.state_dim + self.has_dynamics_bias)
                F, b = (_dynamics_weights[:, :-1], _dynamics_weights[:, -1]) if self.has_dynamics_bias \
                    else (_dynamics_weights, None)

                B = jnp.zeros((self.state_dim, 0))
                if self.update_dynamics_covariance:
                    dynamics_cov_stats_1 = jnp.ones((self.state_dim, 1)) * (masks.sum() / 2)
                    dynamics_mean = jnp.einsum('btx,yx->bty', xp, F)
                    if self.has_dynamics_bias:
                        dynamics_mean += b[None, None]
                    sqr_err_flattened = jnp.square(xn - dynamics_mean).reshape(-1, self.state_dim)
                    masks_flattened = masks[:, 1:].reshape(-1)
                    sqr_err_flattened = sqr_err_flattened * masks_flattened[:, None]
                    dynamics_cov_stats_2 = jnp.nansum(sqr_err_flattened, axis=0) / 2
                    dynamics_cov_stats_2 = jnp.expand_dims(dynamics_cov_stats_2, -1)
                    dynamics_cov_stats = (dynamics_cov_stats_1, dynamics_cov_stats_2)
                    dynamics_cov_posterior = ig_posterior_update(self.dynamics_covariance_prior,
                                                                 dynamics_cov_stats)
                    dynamics_cov = dynamics_cov_posterior.sample(seed=next(rngs))
                    # dynamics_cov = jnp.where(jnp.logical_or(jnp.isnan(dynamics_cov),
                    #                                          dynamics_cov < self.EPS), self.EPS, dynamics_cov)
                    Q = jnp.diag(jnp.ravel(dynamics_cov))
                else:
                    Q = params.dynamics.cov

            params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               ar_dependency=emissions_ar_dependency),
                initial_emissions=ParamsLGSSMInitial(mean=initial_emissions_mean, cov=initial_emissions_cov),
            )
            return params

        @jit
        def one_sample(_params, _emissions, _inputs, rng, itr_num):
            rngs = jr.split(rng, 2)

            if fixed_states is not None:
                _new_states = fixed_states
            else:
                _new_states = lgssm_posterior_sample_vmap(rngs[0], _params, _emissions,
                                                          inputs, masks, jnp.arange(self.num_trials, dtype=int))

            if self.standardize_states:
                if self.standardize_per_latent_dim:
                    axis = (0, 1)
                else:
                    axis = (0, 1, 2)
                states_mean = jnp.mean(_new_states, axis=axis, keepdims=True)
                states_std = jnp.std(_new_states, axis=axis, keepdims=True)
                _new_states = (_new_states - states_mean) / states_std
                # _new_states = _new_states / states_std

            # Sample parameters
            _stats = sufficient_stats_from_sample(_new_states, _params)
            _new_params = lgssm_params_sample(rngs[1], _stats, _new_states, _params)

            # compute the log joint
            # _ll = self.log_joint(_new_params, _states, _emissions, _inputs, masks)
            _ll = self.log_joint(_new_params, _new_states, _emissions, _inputs, masks)

            return _new_params, _new_states, _ll

        sample_of_params = []
        sample_of_states = []
        lls = []
        keys = iter(jr.split(key, sample_size + 1))
        current_params = initial_params
        lgssm_posterior_sample_vmap = vmap(lgssm_posterior_sample, in_axes=(None, None, 0, None, 0, 0))

        for sample_itr in progress_bar(range(sample_size)):
            current_params, current_states, ll = one_sample(current_params, emissions, inputs, next(keys), sample_itr)
            if sample_itr >= sample_size - return_n_samples:
                sample_of_params.append(current_params)
            if return_states and (sample_itr >= sample_size - return_n_samples):
                sample_of_states.append(current_states)
            if print_ll:
                print(jnp.isnan(current_params.emissions.weights).sum())
                # print(current_params.initial_emissions.cov)
                print(current_params.emissions.ar_dependency)
                print(jnp.diag(current_params.emissions.cov))
                print(ll)
            lls.append(ll)

        return pytree_stack(sample_of_params), lls, None, sample_of_states