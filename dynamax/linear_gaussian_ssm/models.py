from fastprogress.fastprogress import progress_bar
from functools import partial
import numpy as np
import jax
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow_probability.substrates.jax.distributions import InverseGamma as IG
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

from dynamax.ssm import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother, lgssm_posterior_sample, lgssm_posterior_sample_identity
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions, ParamsTVLGSSM
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update, mvn_posterior_update, ig_posterior_update
from dynamax.utils.utils import pytree_stack, psd_solve

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
        input_dim: int=0,
        has_dynamics_bias: bool=True,
        has_emissions_bias: bool=True
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
        key: PRNGKey =jr.PRNGKey(0),
        initial_mean: Optional[Float[Array, "state_dim"]]=None,
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
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean, params.initial.cov)

    def transition_distribution(
        self,
        params: ParamsLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
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
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
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
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Float[Array, "ntime state_dim"]:
        return lgssm_posterior_sample(key, params, emissions, inputs)

    def posterior_predictive(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
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
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
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
        inputs: Optional[Float[Array, "nbatch ntime input_dim"]]=None
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
            #print(_params.dynamics.bias)
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

        #print(sample_of_params[0])
        #print(sample_of_params[-1])
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
            input_dim: int=0,
            num_trials: int=0, # number of trials
            sequence_length: int=0, # number of timesteps per trial
            time_varying_dynamics: bool=False,
            time_varying_emissions: bool=True,
            dynamics_param_ar_dependency_variance: float=1.0,
            emissions_param_ar_dependency_variance: float=1.0,
            has_dynamics_bias: bool=False,
            has_emissions_bias: bool=False,
            fix_initial: bool=False,
            fix_dynamics: bool=False,
            fix_emissions: bool=False,
            normalize_emissions: bool=False,
            emission_weights_scale: float=1.0,
            orthogonal_emissions_weights: bool=False,
            update_emissions_param_ar_dependency_variance: bool=False,
            update_initial_covariance: bool=False,
            update_dynamics_covariance: bool=False,  # learn diagonal covariance matrix
            update_emissions_covariance: bool=False, # learn diagonal covariance matrix
            update_init_emissions_mean: bool=False,
            update_init_emissions_covariance: bool=False,
            batch_update: bool=False,
            batch_size: int=1,
            emission_per_trial: bool=False,
            scan_emissions_stats: bool=False,
            connected_dynamics: bool=False,
            **kw_priors
    ):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim,
                         has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias)

        self.num_trials = num_trials
        self.time_varying_dynamics = time_varying_dynamics
        self.time_varying_emissions = time_varying_emissions
        self.sequence_length = sequence_length

        if time_varying_dynamics:
            assert dynamics_param_ar_dependency_variance > 0.0
        if time_varying_emissions:
            assert emissions_param_ar_dependency_variance > 0.0
        self.dynamics_param_ar_dependency_variance = dynamics_param_ar_dependency_variance
        self.emissions_param_ar_dependency_variance = emissions_param_ar_dependency_variance

        self.fix_initial = fix_initial
        self.fix_dynamics = fix_dynamics
        self.fix_emissions = fix_emissions
        self.normalize_emissions = normalize_emissions
        self.emission_weights_scale = emission_weights_scale
        self.update_emissions_param_ar_dependency_variance = update_emissions_param_ar_dependency_variance
        self.update_dynamics_covariance = update_dynamics_covariance
        self.update_emissions_covariance = update_emissions_covariance
        self.update_initial_covariance = update_initial_covariance
        self.update_init_emissions_mean = update_init_emissions_mean
        self.update_init_emissions_covariance = update_init_emissions_covariance
        self.batch_update = batch_update
        self.batch_size = batch_size
        self.orthogonal_emissions_weights = orthogonal_emissions_weights
        self.emission_per_trial = emission_per_trial
        self.scan_emissions_stats = scan_emissions_stats
        self.connected_dynamics = connected_dynamics

        # Initialize prior distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_mean_prior = default_prior(
            'initial_prior',
            MVN(loc=jnp.ones(self.state_dim),
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

        if time_varying_dynamics:
            # prior on initial dynamics
            self.dynamics_prior = default_prior(
                'dynamics_prior',
                MVN(loc=jnp.zeros(self.state_dim**2), covariance_matrix=jnp.eye(self.state_dim**2))
            )
        else:
            # prior on dynamics parameters
            self.dynamics_prior = default_prior(
                'dynamics_prior',
                MVN(loc=jnp.zeros(self.state_dim**2), covariance_matrix=jnp.eye(self.state_dim**2))
            )

        if time_varying_emissions:
            # prior on initial emissions
            self.emission_prior = default_prior(
                'emission_prior',
                MVN(loc=jnp.zeros(self.emission_dim * self.state_dim),
                    covariance_matrix=jnp.eye(self.emission_dim * self.state_dim))
            )
            self.initial_emissions_covariance_prior = default_prior(
                'init_emissions_cov_prior',
                IG(concentration=1.0, scale=1.0)
            )
        else:
            # prior on emissions parameters
            self.emission_prior = default_prior(
                'emission_prior',
                MVN(loc=jnp.zeros(self.emission_dim * self.state_dim),
                    covariance_matrix=jnp.eye(self.emission_dim * self.state_dim))
            )

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return dict(inputs=(self.input_dim,)) if self.input_dim > 0 else dict()

    def initialize(
        self,
        key: PRNGKey =jr.PRNGKey(0),
        initial_mean: Optional[Float[Array, "state_dim"]]=None,
        initial_covariance=None,
        dynamics_weights=None,
        dynamics_bias=None,
        dynamics_input_weights=None,
        dynamics_covariance=None,
        stabilize_dynamics=True,
        alpha=1.0,
        emission_weights=None,
        emission_bias=None,
        emission_input_weights=None,
        emission_covariance=None,
        initial_dynamics_mean=None,
        initial_dynamics_cov=None,
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
        if self.connected_dynamics:
            _initial_mean = jnp.ones(self.state_dim)
            _initial_covariance = jnp.eye(self.state_dim)
        else:
            _initial_mean = jnp.ones((self.num_trials, self.state_dim))
            _initial_covariance = jnp.tile(jnp.eye(self.state_dim)[None], (self.num_trials, 1, 1))

        if self.time_varying_dynamics:
            # _initial_dynamics_cov = self.dynamics_param_ar_dependency_variance * jnp.eye(self.state_dim**2)
            _initial_dynamics_mean = jnp.zeros(self.state_dim**2)
            _initial_dynamics_cov = jnp.eye(self.state_dim**2)
        else:
            _initial_dynamics_cov, _initial_dynamics_mean = None, None

        if self.time_varying_emissions:
            # _initial_emissions_cov = self.emissions_param_ar_dependency_variance * jnp.eye(self.emission_dim*self.state_dim)
            _initial_emissions_mean = jnp.zeros(self.emission_dim * self.state_dim)
            _initial_emissions_cov = jnp.eye(self.emission_dim * self.state_dim)
        else:
            _initial_emissions_cov, _initial_emissions_mean = None, None

        if self.time_varying_dynamics:
            keys = jr.split(key, self.sequence_length-1)
            key = keys[-1]
            def _get_dynamics_weights(prev_weights, current_key):
                current_weights = prev_weights + jnp.sqrt(self.dynamics_param_ar_dependency_variance) \
                                  * jr.normal(current_key, shape=(self.state_dim, self.state_dim))
                return current_weights, current_weights

            key1, key = jr.split(key, 2)
            initial_dynamics_weights = jr.normal(key1, shape=(self.state_dim, self.state_dim))
            _, _dynamics_weights = jax.lax.scan(_get_dynamics_weights, initial_dynamics_weights, keys[:-1])
            _dynamics_weights = jnp.concatenate([initial_dynamics_weights[None], _dynamics_weights])
            if stabilize_dynamics:
                eigdecomp_result = np.linalg.eig(_dynamics_weights)
                eigenvalues = eigdecomp_result.eigenvalues
                eigenvectors = eigdecomp_result.eigenvectors
                eigenvectors_inv = np.linalg.inv(eigenvectors)
                normalized_eigvals = np.diag(eigenvalues / np.abs(eigenvalues))
                dynamics = np.einsum('tab,tbc,tcd->tad',
                                     eigenvectors,
                                     normalized_eigvals,
                                     eigenvectors_inv)
                _dynamics_weights = jnp.array(np.real(dynamics))
                # _dynamics_weights = _dynamics_weights / (1e-8 + alpha * np.max(np.abs(np.linalg.eigvals(_dynamics_weights))))
                #_dynamics_weights = _dynamics_weights / (1e-4 + alpha * np.max(np.abs(np.linalg.eigvals(_dynamics_weights)), axis=-1))[:, None, None]
        else:
            key1, key = jr.split(key, 2)
            _dynamics_weights = jr.normal(key1, shape=(self.state_dim, self.state_dim))
            if stabilize_dynamics:
                eigdecomp_result = np.linalg.eig(_dynamics_weights)
                eigenvalues = eigdecomp_result.eigenvalues
                eigenvectors = eigdecomp_result.eigenvectors
                dynamics = eigenvectors @ np.diag(eigenvalues / np.abs(eigenvalues)) @ np.linalg.inv(eigenvectors)
                _dynamics_weights = jnp.array(np.real(dynamics))
                # _dynamics_weights = _dynamics_weights / (1e-8 + alpha * np.max(np.abs(np.linalg.eigvals(_dynamics_weights))))

        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)

        if self.time_varying_emissions:
            emissions_param_ar_dependency_cov = jnp.eye(self.emission_dim * self.state_dim) * self.emissions_param_ar_dependency_variance
            keys = jr.split(key, self.num_trials)
            key = keys[-1]
            def _get_emission_weights(prev_weights, current_key):
                # current_weights = prev_weights + jnp.sqrt(self.emissions_param_ar_dependency_variance) \
                #                   * jr.normal(current_key, shape=(self.emission_dim, self.state_dim))
                current_weights_dist = MVN(loc=prev_weights,
                                           covariance_matrix=emissions_param_ar_dependency_cov)
                current_weights = current_weights_dist.sample(seed=current_key)
                return current_weights, current_weights

            key1, key = jr.split(key, 2)
            initial_emission_weights = jr.normal(key1, shape=(self.emission_dim * self.state_dim, ))
            _, _emission_weights = jax.lax.scan(_get_emission_weights, initial_emission_weights, keys[:-1])
            _emission_weights = jnp.concatenate([initial_emission_weights[None], _emission_weights])
            _emission_weights = _emission_weights.reshape(_emission_weights.shape[0], self.emission_dim, self.state_dim)
            if self.orthogonal_emissions_weights:
                _emission_weights = jnp.linalg.qr(_emission_weights)[0]
                _emission_weights = self.emission_weights_scale * _emission_weights
            elif self.normalize_emissions:
                _emission_weights = _emission_weights / jnp.linalg.norm(_emission_weights, ord=2, axis=-2)[:, None]
                # _emission_weights = _emission_weights / jnp.linalg.norm(_emission_weights, ord='fro', axis=(-2, -1))[:, None, None]
                _emission_weights = self.emission_weights_scale * _emission_weights
        else:
            key1, key = jr.split(key, 2)
            _emission_weights = jr.normal(key1, shape=(self.emission_dim, self.state_dim))
            if self.orthogonal_emissions_weights:
                _emission_weights = jnp.linalg.qr(_emission_weights)[0]
                _emission_weights = self.emission_weights_scale * _emission_weights
            elif self.normalize_emissions:
                _emission_weights = _emission_weights / jnp.linalg.norm(_emission_weights, ord=2, axis=-2)[None]
                # _emission_weights = _emission_weights / jnp.linalg.norm(_emission_weights, ord='fro', axis=(-2, -1))
                _emission_weights = self.emission_weights_scale * _emission_weights

        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
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
                cov=default(dynamics_covariance, _dynamics_covariance),
                ar_dependency=self.dynamics_param_ar_dependency_variance),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
                ar_dependency=self.emissions_param_ar_dependency_variance),
            initial_dynamics=ParamsLGSSMInitial(
                mean=default(initial_dynamics_mean, _initial_dynamics_mean),
                cov=default(initial_dynamics_cov, _initial_dynamics_cov)),
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
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                ar_dependency=ParameterProperties()),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                ar_dependency=ParameterProperties()),
            initial_dynamics=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            initial_emissions=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()))
            )
        return params, props,

    # All SSMs support sampling
    def sample(
        self,
        params: ParameterSet,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]]=None,
        return_signal: bool=False
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

        def _outer_step(carry, outer_args):
            key, t = outer_args
            def _step(prev_state, args):
                key, inpt, idx = args
                key1, key2 = jr.split(key, 2)
                state = self.transition_distribution(idx-1, params, prev_state, inpt).sample(seed=key2)
                emission_distribution = self.emission_distribution(t, params, state, inpt)
                emission = emission_distribution.sample(seed=key1)
                if return_signal:
                    signal = emission_distribution.mean()
                else:
                    signal = jnp.zeros_like(emission)
                return state, (state, emission, signal)

            # Sample the initial state
            key1, key2, key = jr.split(key, 3)
            initial_input = tree_map(lambda x: x[0], inputs)
            initial_state = self.initial_distribution(t, params, initial_input).sample(seed=key1)
            initial_emission_distribution = self.emission_distribution(t, params, initial_state, initial_input)
            initial_emission = initial_emission_distribution.sample(seed=key2)
            if return_signal:
                initial_signal = initial_emission_distribution.mean()
            else:
                initial_signal = jnp.zeros_like(initial_emission)

            # Sample the remaining emissions and states
            next_keys = jr.split(key, self.sequence_length - 1)
            next_inputs = tree_map(lambda x: x[1:], inputs)
            next_indices = jnp.arange(1, self.sequence_length)
            _, (next_states, next_emissions, next_signals) = lax.scan(_step, initial_state,
                                                                      (next_keys, next_inputs, next_indices))

            # Concatenate the initial state and emission with the following ones
            expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
            states = tree_map(expand_and_cat, initial_state, next_states)
            emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
            signals = tree_map(expand_and_cat, initial_signal, next_signals)

            return None, (states, emissions, signals)

        keys = jr.split(key, self.num_trials)
        _, (states, emissions, signals) = lax.scan(_outer_step, None,
                                                      (keys, jnp.arange(self.num_trials)))

        return states, emissions, signals

    def initial_distribution(
        self,
        timestep: int,
        params: ParamsLGSSM,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean[timestep], params.initial.cov[timestep])

    def transition_distribution(
        self,
        timestep: int,
        params: ParamsTVLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        if self.time_varying_dynamics:
            mean = params.dynamics.weights[timestep] @ state + params.dynamics.input_weights @ inputs
        else:
            mean = params.dynamics.weights @ state + params.dynamics.input_weights @ inputs
        if self.has_dynamics_bias:
            mean += params.dynamics.bias
        return MVN(mean, params.dynamics.cov)

    def emission_distribution(
        self,
        timestep: int,
        params: ParamsTVLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        if self.time_varying_emissions:
            mean = params.emissions.weights[timestep] @ state + params.emissions.input_weights @ inputs
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
        masks: jnp.array=None
    ) -> Scalar:
        if masks is None:
            masks = jnp.ones(emissions.shape[0], dtype=bool)
        filtered_posterior = lgssm_filter(params, emissions, inputs, masks)
        return filtered_posterior.marginal_loglik

    def filter(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        masks: jnp.array=None
    ) -> PosteriorGSSMFiltered:
        if masks is None:
            masks = jnp.ones(emissions.shape[0], dtype=bool)
        return lgssm_filter(params, emissions, inputs, masks)

    def smoother(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        masks: jnp.array=None
    ) -> PosteriorGSSMSmoothed:
        if masks is None:
            masks = jnp.ones(emissions.shape[0], dtype=bool)
        return lgssm_smoother(params, emissions, inputs, masks)

    def posterior_sample(
        self,
        key: PRNGKey,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
        masks: jnp.array=None
    ) -> Float[Array, "ntime state_dim"]:
        if masks is None:
            masks = jnp.ones(emissions.shape[0], dtype=bool)
        return lgssm_posterior_sample(key, params, emissions, inputs, masks)

    def posterior_predictive(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
        masks: jnp.array=None
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
            masks = jnp.ones(emissions.shape[0], dtype=bool)
        posterior = lgssm_smoother(params, emissions, inputs, masks)
        H = params.emissions.weights
        b = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]

        smoothed_emissions = jnp.einsum('...x,...yx->...y', posterior.smoothed_means, H)
        smoothed_emissions_cov = jnp.einsum('...ya,...ab,...xb->...yx', H, posterior.smoothed_covariances, H) + R

        if self.has_emissions_bias:
            smoothed_emissions += b

        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    def log_joint(
        self,
        params: ParamsTVLGSSM,
        trial_emissions_weights,
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
        def undiagonalize(cov):
            return jnp.diag(cov)
        flattened_cov = vmap(undiagonalize)(params.initial.cov)
        lp += self.initial_covariance_prior.log_prob(flattened_cov.flatten()).sum()
        lp += MVN(params.initial.mean, params.initial.cov).log_prob(states[:, 0]).sum()

        # dynamics & states
        if self.time_varying_dynamics:
            def _compute_dynamics_lp(prev_lp, current_t):
                current_param = params.dynamics.weights[current_t]
                next_param = params.dynamics.weights[current_t+1]
                current_lp = prev_lp + MVN(loc=jnp.ravel(current_param),
                                           covariance_matrix=jnp.eye(
                                               self.state_dim ** 2) * self.dynamics_param_ar_dependency_variance).log_prob(jnp.ravel(next_param))
                current_lp += MVN(current_param @ states[current_t], params.dynamics.cov).log_prob(states[current_t+1])
                return current_lp, None
            lp, _ = jax.lax.scan(_compute_dynamics_lp, lp, jnp.arange(self.sequence_length-2))
            lp += MVN(params.dynamics.weights[-1] @ states[-2], params.dynamics.cov).log_prob(states[-1])
            lp += MVN(params.initial_dynamics.mean, params.initial_dynamics.cov).log_prob(jnp.ravel(params.dynamics.weights[0]))
            #lp += self.dynamics_prior.log_prob((params.initial_dynamics.cov, params.initial_dynamics.mean))
            lp += self.dynamics_prior.log_prob(params.initial_dynamics.mean)
        else:
            dynamics_bias = params.dynamics.bias if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
            dynamics_matrix = jnp.column_stack(
                (params.dynamics.weights,
                 params.dynamics.input_weights,
                 dynamics_bias))
            lp += self.dynamics_prior.log_prob(jnp.ravel(dynamics_matrix))
            def _compute_dynamics_lp(prev_lp, current_t):
                current_state_mean = jnp.einsum('ij,rj->ri', params.dynamics.weights, states[:, current_t])
                new_lp = MVN(current_state_mean, params.dynamics.cov).log_prob(states[:, current_t+1])
                masked_new_lp = (masks[current_t+1] * new_lp).sum()
                current_lp = prev_lp + masked_new_lp
                return current_lp, None
            lp, _ = jax.lax.scan(_compute_dynamics_lp, lp, jnp.arange(self.sequence_length-1))

        if self.update_dynamics_covariance:
            lp += self.dynamics_covariance_prior.log_prob(jnp.diag(params.dynamics.cov)).sum()

        # emissions & observations
        if self.time_varying_emissions:

            if self.emission_per_trial:
                # also need to edit ar dependency variance update step.
                def _compute_emissions_lp(prev_lp, current_t):
                    current_param = params.emissions.weights[current_t]
                    current_y_mean = jnp.einsum('ij,rj->ri', current_param, states[:, current_t])
                    new_lp = MVN(current_y_mean, params.emissions.cov).log_prob(emissions[:, current_t])
                    masked_new_lp = (masks[current_t] * new_lp).sum()
                    current_lp = prev_lp + masked_new_lp
                    return current_lp, None

                lp, _ = jax.lax.scan(_compute_emissions_lp, lp, jnp.arange(self.sequence_length))

                def _compute_trial_emissions_lp(prev_lp, current_t):
                    current_param = trial_emissions_weights[current_t]
                    next_param = trial_emissions_weights[current_t + 1]
                    current_lp = prev_lp + MVN(loc=jnp.ravel(current_param),
                                               covariance_matrix=jnp.eye(
                                                   self.emission_dim * self.state_dim) * params.emissions.ar_dependency).log_prob(jnp.ravel(next_param))
                    return current_lp, None

                lp, _ = jax.lax.scan(_compute_trial_emissions_lp, lp, jnp.arange(self.num_trials-1))

                lp += MVN(params.initial_emissions.mean, params.initial_emissions.cov).log_prob(jnp.ravel(trial_emissions_weights[0]))
                # lp += self.emission_prior.log_prob((params.initial_emissions.cov, params.initial_emissions.mean))
                lp += self.emission_prior.log_prob(params.initial_emissions.mean)
                lp += self.init_emissions_cov_prior.log_prob(jnp.diag(params.initial_emissions.cov)).sum()

            else:
                def _compute_emissions_lp(prev_lp, current_t):
                    current_param = params.emissions.weights[current_t]
                    next_param = params.emissions.weights[current_t + 1]
                    current_lp = prev_lp + MVN(loc=jnp.ravel(current_param),
                                               covariance_matrix=jnp.eye(
                                                   self.emission_dim * self.state_dim) * params.emissions.ar_dependency).log_prob(jnp.ravel(next_param))
                    current_lp += masks[current_t] * MVN(current_param @ states[current_t], params.emissions.cov).log_prob(
                        emissions[current_t])
                    return current_lp, None

                lp, _ = jax.lax.scan(_compute_emissions_lp, lp, jnp.arange(self.sequence_length - 1))

                lp += masks[-1] * MVN(params.emissions.weights[-1] @ states[-1], params.emissions.cov).log_prob(emissions[-1])
                lp += MVN(params.initial_emissions.mean, params.initial_emissions.cov).log_prob(
                    jnp.ravel(params.emissions.weights[0]))
                # lp += self.emission_prior.log_prob((params.initial_emissions.cov, params.initial_emissions.mean))
                lp += self.emission_prior.log_prob(params.initial_emissions.mean)
                lp += self.init_emissions_cov_prior.log_prob(jnp.diag(params.initial_emissions.cov)).sum()

            if self.update_emissions_param_ar_dependency_variance:
                lp += self.emissions_ar_dependency_prior.log_prob(params.emissions.ar_dependency)

        else:
            emission_bias = params.emissions.bias if self.has_emissions_bias else jnp.zeros((self.emission_dim, 0))
            emission_matrix = jnp.column_stack(
                (params.emissions.weights,
                 params.emissions.input_weights,
                 emission_bias))
            lp += self.emission_prior.log_prob(jnp.ravel(emission_matrix))

            def _compute_emissions_lp(prev_lp, current_t):
                current_y_mean = jnp.einsum('ij,rj->ri', params.emissions.weights, states[:, current_t])
                new_lp = MVN(current_y_mean, params.emissions.cov).log_prob(emissions[:, current_t])
                masked_new_lp = (masks[current_t] * new_lp).sum()
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
            return_states: bool=False,
            return_n_samples: int=100,
            print_ll: bool=False,
            trials: jnp.array = None,
            masks: jnp.array=None,
            trial_masks: jnp.array=None,
            fixed_states: jnp.array=None,
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

            # init_stats = (x[0], jnp.outer(x[0], x[0]), 1)
            init_stats = (x[:, 0],)

            N, D = y.shape[-1], states.shape[-1]
            # Optimized Code
            if not self.time_varying_dynamics:
                Qinv = jnp.linalg.inv(params.dynamics.cov)
                dynamics_stats_1 = jnp.einsum('bti,jk,btl->jikl', xp, Qinv, xp).reshape(D*D, D*D)
                dynamics_stats_2 = jnp.einsum('bti,ik,btl->kl', xn, Qinv, xp).reshape(-1)
                dynamics_stats = (dynamics_stats_1, dynamics_stats_2)
            else:
                dynamics_stats = None

            # Quantities for the emissions
            if not self.time_varying_emissions:
                Rinv = jnp.linalg.inv(params.emissions.cov)
                emissions_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', x, Rinv, x, masks).reshape(N*D, N*D)
                emissions_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', y, Rinv, x, masks).reshape(-1)
                emission_stats = (emissions_stats_1, emissions_stats_2)
            else:
                emission_stats = None

            return init_stats, dynamics_stats, emission_stats

        def lgssm_params_sample(rng, stats, states, params):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats
            n_splits = 7 + self.update_emissions_param_ar_dependency_variance + self.update_emissions_covariance
            n_splits += self.batch_update * emissions.shape[-1]
            rngs = iter(jr.split(rng, n_splits))

            # Sample the initial params
            if self.fix_initial:
                S, m = params.initial.cov, params.initial.mean
            else:
                initial_stats_1 = jnp.linalg.inv(params.initial.cov)
                initial_stats_2 = jnp.einsum('bij,bj->bi',
                                             initial_stats_1,
                                             init_stats[0])
                initial_stats = (initial_stats_1, initial_stats_2)

                initial_posterior = mvn_posterior_update(self.initial_mean_prior, initial_stats)
                m = initial_posterior.sample(seed=next(rngs))

                if self.update_initial_covariance:
                    init_cov_stats_1 = jnp.ones((self.state_dim, 1)) / 2
                    init_cov_stats_2 = jnp.square(init_stats[0] - m) / 2
                    init_cov_stats_2 = init_cov_stats_2.flatten()
                    init_cov_stats_2 = jnp.expand_dims(init_cov_stats_2, -1)
                    init_cov_stats = (init_cov_stats_1, init_cov_stats_2)
                    init_cov_posterior = ig_posterior_update(self.initial_covariance_prior, init_cov_stats)
                    init_cov = init_cov_posterior.sample(seed=next(rngs))
                    init_cov = jnp.ravel(init_cov).reshape(self.num_trials, self.state_dim)
                    def _diagonalize(cov):
                        return jnp.diag(cov)

                    S = vmap(_diagonalize)(init_cov)
                else:
                    S = params.initial.cov

            # Sample the dynamics params
            if self.fix_dynamics:
                F = params.dynamics.weights
                b = params.dynamics.bias
                B = params.dynamics.input_weights
                Q = params.dynamics.cov
                initial_dynamics_cov = params.initial_dynamics.cov
                initial_dynamics_mean = params.initial_dynamics.mean
                dynamics_ar_dependency = params.dynamics.ar_dependency
            else:
                if self.time_varying_dynamics:
                    xp, xn = states[:-1], states[1:]

                    _dynamics_params = ParamsLGSSM(
                        initial=ParamsLGSSMInitial(mean=params.initial_dynamics.mean,
                                                   cov=params.initial_dynamics.cov),
                        dynamics=ParamsLGSSMDynamics(weights=jnp.eye(self.state_dim**2),
                                                     bias=None,
                                                     input_weights=jnp.zeros((self.state_dim**2, 0)),
                                                     cov=params.dynamics.ar_dependency*jnp.eye(self.state_dim**2),
                                                     ar_dependency=None),
                        emissions=ParamsLGSSMEmissions(weights=jnp.kron(jnp.eye(self.state_dim), jnp.expand_dims(xp, 1)), ### kron xp
                                                       bias=None,
                                                       input_weights=jnp.zeros((self.state_dim, 0)),
                                                       cov=params.dynamics.cov,
                                                       ar_dependency=None)
                    )

                    _dynamics_weights = lgssm_posterior_sample(next(rngs),
                                                               _dynamics_params,
                                                               xn,
                                                               jnp.zeros((num_timesteps-1, 0)))

                    F = _dynamics_weights.reshape(num_timesteps-1, self.state_dim, self.state_dim)

                    b = None
                    B = jnp.zeros((self.state_dim, 0))
                    # Q = params.dynamics.cov

                    # dynamics_stats = (_dynamics_weights[0], jnp.outer(_dynamics_weights[0], _dynamics_weights[0]), 1)
                    # dynamics_posterior = niw_posterior_update(self.dynamics_prior, dynamics_stats)
                    # initial_dynamics_cov, initial_dynamics_mean = dynamics_posterior.sample(seed=next(rngs))

                    dynamics_ar_dep_cov = jnp.eye(self.state_dim**2) * params.dynamics.ar_dependency
                    init_dynamics_stats_1 = jnp.linalg.inv(dynamics_ar_dep_cov)
                    # dynamics_stats_1 = jnp.linalg.inv(params.initial_dynamics.cov)
                    init_dynamics_stats_2 = init_dynamics_stats_1 @ _dynamics_weights[0]
                    init_dynamics_stats = (init_dynamics_stats_1, init_dynamics_stats_2)

                    init_dynamics_posterior = mvn_posterior_update(self.dynamics_prior, init_dynamics_stats)
                    initial_dynamics_mean = init_dynamics_posterior.sample(seed=next(rngs))
                    # initial_dynamics_cov = params.initial_dynamics.cov

                    dynamics_ar_dependency = params.dynamics.ar_dependency
                    initial_dynamics_cov = dynamics_ar_dep_cov

                    if self.update_dynamics_covariance:
                        dynamics_cov_stats_1 = jnp.ones((self.state_dim, 1)) * (num_timesteps / 2)
                        dynamics_mean = jnp.einsum('tx,tyx->ty', xp, F)
                        dynamics_cov_stats_2 = jnp.sum(jnp.square(xn-dynamics_mean), axis=0) / 2
                        dynamics_cov_stats_2 = jnp.expand_dims(dynamics_cov_stats_2, -1)
                        dynamics_cov_stats = (dynamics_cov_stats_1, dynamics_cov_stats_2)
                        dynamics_cov_posterior = ig_posterior_update(self.dynamics_covariance_prior,
                                                                      dynamics_cov_stats)
                        dynamics_cov = dynamics_cov_posterior.sample(seed=next(rngs))
                        Q = jnp.diag(jnp.ravel(dynamics_cov))
                    else:
                        Q = params.emissions.cov

                else:
                    # dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
                    # Q, FB = dynamics_posterior.sample(seed=next(rngs))
                    # F = FB[:, :self.state_dim]
                    # B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
                    #     else (FB[:, self.state_dim:], None)
                    #
                    # Q = params.dynamics.cov
                    #
                    # initial_dynamics_cov, initial_dynamics_mean = None, None
                    xp, xn = states[:, :-1], states[:, 1:]

                    dynamics_posterior = mvn_posterior_update(self.dynamics_prior, dynamics_stats)
                    _dynamics_weights = dynamics_posterior.sample(seed=next(rngs))

                    F = _dynamics_weights.reshape(self.state_dim, self.state_dim)

                    b = None
                    B = jnp.zeros((self.state_dim, 0))
                    if self.update_dynamics_covariance:
                        dynamics_cov_stats_1 = jnp.ones((self.state_dim, 1)) * (masks.sum() / 2)
                        dynamics_mean = jnp.einsum('btx,yx->bty', xp, F)
                        sqr_err_flattened = jnp.square(xn-dynamics_mean).reshape(-1, self.state_dim)
                        masks_flattend = masks.reshape(-1)
                        dynamics_cov_stats_2 = jnp.sum(sqr_err_flattened[masks_flattend], axis=0) / 2
                        dynamics_cov_stats_2 = jnp.expand_dims(dynamics_cov_stats_2, -1)
                        dynamics_cov_stats = (dynamics_cov_stats_1, dynamics_cov_stats_2)
                        dynamics_cov_posterior = ig_posterior_update(self.dynamics_covariance_prior,
                                                                      dynamics_cov_stats)
                        dynamics_cov = dynamics_cov_posterior.sample(seed=next(rngs))
                        Q = jnp.diag(jnp.ravel(dynamics_cov))
                    else:
                        Q = params.dynamics.cov

                    initial_dynamics_cov, initial_dynamics_mean = None, None
                    dynamics_ar_dependency = None

            trial_emissions_weights = None
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

                    if self.emission_per_trial:
                        assert trials is not None

                        Rinv = jnp.linalg.inv(params.emissions.cov)

                        if self.scan_emissions_stats:
                            def _compute_emissions_stats(carry, trial):
                                emissions_stats_1 = jnp.einsum('t,ti,jk,tl->jikl', trial, x, Rinv, x).reshape(N * D, N * D)
                                emissions_covs = jnp.linalg.inv(emissions_stats_1)
                                emissions_stats_2 = jnp.einsum('t,ti,ik,tl->kl', trial, y, Rinv, x).reshape(-1)
                                emissions_y = jnp.einsum('ij,j->i', emissions_covs, emissions_stats_2)
                                return None, (emissions_covs, emissions_y)

                            _, emissions_stats = lax.scan(_compute_emissions_stats, None, trials.T)
                            emissions_covs, emissions_y = emissions_stats
                        else:
                            emissions_stats_1 = jnp.einsum('bt,bti,jk,btl->bjikl', masks, x, Rinv, x).reshape(self.num_trials, N * D, N * D)
                            emissions_covs = jnp.linalg.inv(emissions_stats_1)
                            emissions_stats_2 = jnp.einsum('bt,bti,ik,btl->bkl', masks, y, Rinv, x).reshape(self.num_trials, -1)
                            emissions_y = jnp.einsum('bij,bj->bi', emissions_covs, emissions_stats_2)

                        _emissions_params = ParamsLGSSM(
                            initial=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                                       cov=params.initial_emissions.cov),
                            dynamics=ParamsLGSSMDynamics(weights=jnp.eye(self.emission_dim * self.state_dim),
                                                         bias=None,
                                                         input_weights=None,
                                                         cov=params.emissions.ar_dependency * jnp.eye(
                                                             self.emission_dim * self.state_dim),
                                                         ar_dependency=None),
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
                        # (ntrials, ND) -> (ntimesteps, ND)
                        trial_emissions_weights = _emissions_weights.reshape(self.num_trials, self.emission_dim, self.state_dim)
                        _emissions_weights = jnp.einsum('tn,ni->ti', trials, _emissions_weights)

                    else:
                        if self.batch_update:
                            D, N = x.shape[-1], emissions.shape[-1]
                            batchN = self.batch_size
                            def _update(carry, n):
                                _emissions_params = ParamsLGSSM(
                                    initial=ParamsLGSSMInitial(mean=lax.dynamic_slice(params.initial_emissions.mean, (n*D*batchN,), (D*batchN,)),
                                                               cov=lax.dynamic_slice(params.initial_emissions.cov, (n*D*batchN,n*D*batchN), (D*batchN,D*batchN))),
                                                               #params.emissions.ar_dependency * jnp.eye(D*batchN)),
                                    dynamics=ParamsLGSSMDynamics(weights=jnp.eye(D*batchN),
                                                                 bias=None,
                                                                 input_weights=jnp.zeros((D*batchN, 0)),
                                                                 cov=params.emissions.ar_dependency * jnp.eye(D*batchN),
                                                                 ar_dependency=None),
                                    emissions=ParamsLGSSMEmissions(
                                        weights=jnp.kron(jnp.eye(batchN), jnp.expand_dims(x, 1)),
                                        bias=None,
                                        input_weights=jnp.zeros((batchN, 0)),
                                        cov=lax.dynamic_slice(params.emissions.cov, (n*batchN,n*batchN), (batchN,batchN)),
                                        ar_dependency=None)
                                )

                                _emissions_weights = lgssm_posterior_sample(next(rngs),
                                                                            _emissions_params,
                                                                            lax.dynamic_slice(emissions, (0, n*batchN), (num_timesteps, batchN)),
                                                                            jnp.zeros((num_timesteps, 0)), masks)
                                return None, _emissions_weights

                            _, _emissions_weights = lax.scan(_update, None, jnp.arange(N//batchN))
                            _emissions_weights = jnp.swapaxes(_emissions_weights, 0, 1)
                            _emissions_weights = jnp.reshape(_emissions_weights, (num_timesteps, -1))

                        else:
                            _emissions_params = ParamsLGSSM(
                                initial=ParamsLGSSMInitial(mean=params.initial_emissions.mean,
                                                           cov=params.initial_emissions.cov),
                                dynamics=ParamsLGSSMDynamics(weights=jnp.eye(self.emission_dim*self.state_dim),
                                                             bias=None,
                                                             input_weights=jnp.zeros((self.emission_dim*self.state_dim, 0)),
                                                             cov=params.emissions.ar_dependency*jnp.eye(self.emission_dim*self.state_dim),
                                                             ar_dependency=None),
                                emissions=ParamsLGSSMEmissions(weights=jnp.kron(jnp.eye(self.emission_dim), jnp.expand_dims(x, 1)),
                                                               bias=None,
                                                               input_weights=jnp.zeros((self.emission_dim, 0)),
                                                               cov=params.emissions.cov,
                                                               ar_dependency=None)
                            )

                            _emissions_weights = lgssm_posterior_sample(next(rngs),
                                                                        _emissions_params,
                                                                        emissions,
                                                                        jnp.zeros((num_timesteps, 0)), masks)

                    H = _emissions_weights.reshape(num_timesteps, self.emission_dim, self.state_dim)

                    d = None
                    D = jnp.zeros((self.emission_dim, 0))

                    # emissions_stats = (_emissions_weights[0], jnp.outer(_emissions_weights[0], _emissions_weights[0]), 1)
                    # emissions_posterior = niw_posterior_update(self.emission_prior, emissions_stats)
                    # initial_emissions_cov, initial_emissions_mean = emissions_posterior.sample(seed=next(rngs))

                    # emissions_ar_dep_cov = jnp.eye(self.emission_dim * self.state_dim) * params.emissions.ar_dependency
                    # init_emissions_stats_1 = jnp.linalg.inv(emissions_ar_dep_cov)
                    if self.update_init_emissions_mean:
                        init_emissions_stats_1 = jnp.linalg.inv(params.initial_emissions.cov)
                        init_emissions_stats_2 = init_emissions_stats_1 @ _emissions_weights[0]
                        init_emissions_stats = (init_emissions_stats_1, init_emissions_stats_2)

                        init_emissions_posterior = mvn_posterior_update(self.emission_prior, init_emissions_stats)
                        initial_emissions_mean = init_emissions_posterior.sample(seed=next(rngs))
                    else:
                        initial_emissions_mean = params.initial_emissions.mean

                    if self.update_init_emissions_covariance:
                        init_emissions_cov_stats_1 = jnp.ones((self.emission_dim * self.state_dim, 1)) / 2
                        init_emissions_cov_stats_2 = jnp.square(_emissions_weights[0] - initial_emissions_mean) / 2
                        init_emissions_cov_stats_2 = jnp.expand_dims(init_emissions_cov_stats_2, -1)
                        init_emissions_cov_stats = (init_emissions_cov_stats_1,
                                                    init_emissions_cov_stats_2)
                        init_emissions_cov_posterior = ig_posterior_update(self.initial_emissions_covariance_prior,
                                                                                init_emissions_cov_stats)
                        initial_emissions_cov = init_emissions_cov_posterior.sample(seed=next(rngs))
                        initial_emissions_cov = jnp.diag(jnp.ravel(initial_emissions_cov))
                    else:
                        initial_emissions_cov = params.initial_emissions.cov

                    if self.update_emissions_param_ar_dependency_variance:
                        emissions_ar_dependency_stats_1 = (self.emission_dim * self.state_dim * num_timesteps) / 2
                        # concatenated_emissions_weights = jnp.concatenate([initial_emissions_mean[None], _emissions_weights])
                        concatenated_emissions_weights = _emissions_weights
                        emissions_ar_dependency_stats_2 = jnp.diff(concatenated_emissions_weights, axis=0)
                        emissions_ar_dependency_stats_2 = jnp.sum(jnp.square(emissions_ar_dependency_stats_2)) / 2
                        emissions_ar_dependency_stats = (emissions_ar_dependency_stats_1,
                                                         emissions_ar_dependency_stats_2)
                        emissions_ar_dependency_posterior = ig_posterior_update(self.emissions_ar_dependency_prior,
                                                                                emissions_ar_dependency_stats)
                        emissions_ar_dependency = emissions_ar_dependency_posterior.sample(seed=next(rngs))
                        # initial_emissions_cov = jnp.eye(self.emission_dim * self.state_dim) * emissions_ar_dependency
                    else:
                        emissions_ar_dependency = params.emissions.ar_dependency
                        # initial_emissions_cov = emissions_ar_dep_cov

                    if self.update_emissions_covariance:
                        emissions_cov_stats_1 = jnp.ones((self.emission_dim, 1)) * (jnp.sum(masks) / 2)
                        emissions_mean = jnp.einsum('tx,tyx->ty', states[masks], H[masks])
                        emissions_cov_stats_2 = jnp.sum(jnp.square(emissions[masks]-emissions_mean), axis=0) / 2
                        emissions_cov_stats_2 = jnp.expand_dims(emissions_cov_stats_2, -1)
                        emissions_cov_stats = (emissions_cov_stats_1, emissions_cov_stats_2)
                        emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior,
                                                                      emissions_cov_stats)
                        emissions_cov = emissions_cov_posterior.sample(seed=next(rngs))
                        R = jnp.diag(jnp.ravel(emissions_cov))
                    else:
                        R = params.emissions.cov

                else:
                    # emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
                    # R, HD = emission_posterior.sample(seed=next(rngs))
                    # H = HD[:, :self.state_dim]
                    # D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
                    #     else (HD[:, self.state_dim:], None)
                    #
                    # R = params.emissions.cov
                    #
                    # initial_emissions_cov, initial_emissions_mean = None, None

                    emission_posterior = mvn_posterior_update(self.emission_prior, emission_stats)
                    _emissions_weights = emission_posterior.sample(seed=next(rngs))

                    H = _emissions_weights.reshape(self.emission_dim, self.state_dim)

                    d = None
                    D = jnp.zeros((self.emission_dim, 0))

                    if self.update_emissions_covariance:
                        emissions_cov_stats_1 = jnp.ones((self.emission_dim, 1)) * (jnp.sum(masks) / 2)

                        emissions_mean = jnp.einsum('btx,yx->bty', states, H)
                        sqr_err_flattened = jnp.square(emissions-emissions_mean).reshape(-1, self.emission_dim)
                        masks_flattend = masks.reshape(-1)
                        emissions_cov_stats_2 = jnp.sum(sqr_err_flattened[masks_flattend], axis=0) / 2
                        emissions_cov_stats_2 = jnp.expand_dims(emissions_cov_stats_2, -1)
                        emissions_cov_stats = (emissions_cov_stats_1, emissions_cov_stats_2)
                        emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior,
                                                                      emissions_cov_stats)
                        emissions_cov = emissions_cov_posterior.sample(seed=next(rngs))
                        R = jnp.diag(jnp.ravel(emissions_cov))
                    else:
                        R = params.emissions.cov

                    initial_emissions_cov, initial_emissions_mean = None, None
                    emissions_ar_dependency = None

            params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q,
                                             ar_dependency=dynamics_ar_dependency),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               ar_dependency=emissions_ar_dependency),
                initial_dynamics=ParamsLGSSMInitial(mean=initial_dynamics_mean, cov=initial_dynamics_cov),
                initial_emissions=ParamsLGSSMInitial(mean=initial_emissions_mean, cov=initial_emissions_cov),
            )
            return params, trial_emissions_weights

        @jit
        def one_sample(_params, _states, _emissions, _inputs, rng):
            rngs = jr.split(rng, 2)
            # # Sample latent states
            # _states = lgssm_posterior_sample(rngs[0], _params, emissions, inputs)
            # # compute the log joint
            # _ll = self.log_joint(_params, states, _emissions, _inputs)

            # Sample parameters
            _stats = sufficient_stats_from_sample(_states, _params)
            _new_params, _trial_emissions_weights = lgssm_params_sample(rngs[1], _stats, _states, _params)

            if fixed_states is not None:
                _new_states = fixed_states
            else:
                _new_states = lgssm_posterior_sample_vmap(key=rngs[0],
                                                     params=_new_params,
                                                     emissions=emissions,
                                                     inputs=inputs,
                                                     masks=masks,
                                                     trial_r=jnp.arange(self.num_trials))
            # compute the log joint
            # _ll = self.log_joint(_new_params, _states, _emissions, _inputs)
            _ll = self.log_joint(_new_params, _trial_emissions_weights, _new_states, _emissions, _inputs, masks)
            return _new_params, _new_states, _ll

        sample_of_params = []
        sample_of_states = []
        lls = []
        keys = iter(jr.split(key, sample_size+1))
        current_params = initial_params

        lgssm_posterior_sample_vmap = vmap(lgssm_posterior_sample, in_axes=(None, None, 0, None, 0, 0))

        if fixed_states is not None:
            current_states = fixed_states
        else:
            current_states = lgssm_posterior_sample_vmap(key=next(keys),
                                                        params=current_params,
                                                        emissions=emissions,
                                                        inputs=inputs,
                                                        masks=masks,
                                                        trial_r=jnp.arange(self.num_trials))
        for sample_itr in progress_bar(range(sample_size)):
            current_params, current_states, ll = one_sample(current_params, current_states, emissions, inputs, next(keys))
            if sample_itr >= sample_size - return_n_samples:
                sample_of_params.append(current_params)
            if return_states and (sample_itr >= sample_size - return_n_samples):
                sample_of_states.append(current_states)
            if print_ll:
                print(ll)
            lls.append(ll)

        return pytree_stack(sample_of_params), lls, sample_of_params, sample_of_states
