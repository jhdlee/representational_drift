from fastprogress.fastprogress import progress_bar
from functools import partial
import numpy as np
import jax
from jax import jit, lax, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
import jax.nn as jnn
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
    lgssm_smoother_identity, lgssm_posterior_sample_conditional_smc
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, \
    ParamsLGSSMEmissions, ParamsTVLGSSM
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, UKFHyperParams
from dynamax.nonlinear_gaussian_ssm import unscented_kalman_posterior_sample
from dynamax.nonlinear_gaussian_ssm import extended_kalman_smoother, iterated_extended_kalman_posterior_sample
from dynamax.nonlinear_gaussian_ssm import extended_kalman_filter, iterated_extended_kalman_filter, extended_kalman_posterior_sample

from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update, mvn_posterior_update, \
    ig_posterior_update
from dynamax.utils.utils import pytree_stack, psd_solve, symmetrize, rotate_subspace

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

class GrassmannianGaussianConjugateSSM(LinearGaussianSSM):
    r"""
    Grassmannian Gaussian State Space Model with conjugate priors for the model parameters.

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
            num_trials: int = 1,  # number of trials
            sequence_length: int = 1,  # number of timesteps per trial
            num_conditions: int = 1,
            has_dynamics_bias: bool = False,
            has_emissions_bias: bool = False,
            stationary_emissions: bool = False,
            fix_initial: bool = False,
            fix_dynamics: bool = False,
            fix_emissions: bool = False,
            fix_emissions_cov: bool = False,
            fix_tau: bool = False,
            **kw_priors
    ):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim,
                         has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias)

        self.stationary_emissions = stationary_emissions

        self.dof = self.state_dim * (self.emission_dim - self.state_dim)
        self.dof_shape = (self.state_dim, (self.emission_dim - self.state_dim))

        self.num_trials = num_trials
        self.sequence_length = sequence_length
        self.num_conditions = num_conditions

        self.fix_initial = fix_initial
        self.fix_dynamics = fix_dynamics
        self.fix_emissions = fix_emissions
        self.fix_emissions_cov = fix_emissions_cov
        self.fix_tau = fix_tau

        # Initialize prior distributions
        # prior on initial distributions
        def default_prior(arg, default):
            return kw_priors[arg] if arg in kw_priors else default

        self.initial_prior = default_prior(
            'initial_prior',
            NIW(loc=jnp.zeros(self.state_dim),
                mean_concentration=1.,
                df=self.state_dim + 0.1,
                scale=jnp.eye(self.state_dim)))

        # prior on dynamics parameters
        self.dynamics_prior = default_prior(
            'dynamics_prior',
            MNIW(loc=jnp.zeros((self.state_dim, self.state_dim + self.has_dynamics_bias)),
                 col_precision=jnp.eye(self.state_dim + self.has_dynamics_bias),
                 df=self.state_dim + 0.1,
                 scale=jnp.eye(self.state_dim)))

        # prior on emissions parameters
        if self.stationary_emissions:
            self.emissions_prior = default_prior(
                'emissions_prior',
                MVN(loc=jnp.zeros(self.emission_dim * (self.state_dim + self.has_emissions_bias)),
                    covariance_matrix=jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)))
            )
        else:
            # prior on initial velocity
            self.initial_velocity_prior = default_prior(
                'initial_velocity_prior',
                NIW(loc=jnp.zeros(self.dof),
                    mean_concentration=1.,
                    df=self.dof + 0.1,
                    scale=jnp.eye(self.dof)))

            self.tau_prior = default_prior(
                'tau_prior',
                IG(concentration=1e-16, scale=1e-16)
            )

        self.emissions_covariance_prior = default_prior(
            'emissions_covariance_prior',
            IG(concentration=1.0, scale=1.0)
        )


    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def covariates_shape(self):
        return dict(inputs=(self.input_dim,)) if self.input_dim > 0 else dict()

    def initialize(
            self,
            base_subspace,
            tau,
            key=jr.PRNGKey(0),
            initial_mean=None,
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
            velocity=None,
            initial_velocity_mean=None,
            initial_velocity_cov=None,
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
        _initial_mean = jnp.zeros((self.num_conditions, self.state_dim))
        _initial_covariance = jnp.tile(jnp.eye(self.state_dim)[None], (self.num_conditions, 1, 1))

        key1, key = jr.split(key, 2)
        _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)

        if self.stationary_emissions:
            _initial_velocity_mean = None
            _initial_velocity_cov = None
            _velocity = None

            key1, key = jr.split(key, 2)
            _emission_weights = jr.normal(key1, shape=(self.emission_dim, self.state_dim))
        else:
            _initial_velocity_mean = jnp.zeros(self.dof)
            _initial_velocity_cov = jnp.eye(self.dof)

            if velocity is None:
                keys = jr.split(key, self.num_trials)
                key = keys[-1]
                _velocity_cov = jnp.diag(tau)  # per dimension / rotation tau
                def _get_velocity(prev_velocity, current_key):
                    current_velocity_dist = MVN(loc=prev_velocity, covariance_matrix=_velocity_cov)
                    current_velocity = current_velocity_dist.sample(seed=current_key)
                    return current_velocity, current_velocity

                key1, key = jr.split(key, 2)
                _initial_velocity = jr.normal(key1, shape=(self.dof,))
                _, _velocity = jax.lax.scan(_get_velocity, _initial_velocity, keys[:-1])
                _velocity = jnp.concatenate([_initial_velocity[None], _velocity])
                _velocity = _velocity.reshape((self.num_trials,) + self.dof_shape)
            else:
                _velocity = velocity

            _rotation = jnp.zeros((self.num_trials, self.emission_dim, self.emission_dim))
            _rotation = _rotation.at[:, :self.state_dim, self.state_dim:].set(_velocity)
            _rotation -= _rotation.transpose(0, 2, 1)
            _rotation = jscipy.linalg.expm(_rotation)
            _subspace = jnp.einsum('ij,rjk->rik', base_subspace, _rotation)
            _emission_weights = _subspace[:, :, :self.state_dim]

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
                cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
                tau=tau),
            initial_velocity=ParamsLGSSMInitial(
                mean=default(initial_velocity_mean, _initial_velocity_mean),
                cov=default(initial_velocity_cov, _initial_velocity_cov))
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
                tau=ParameterProperties()),
            initial_velocity=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()))
        )
        return params, props, _velocity

    # All SSMs support sampling
    def sample(
            self,
            params: ParameterSet,
            key: PRNGKey,
            num_timesteps: int,
            conditions=None,
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

        if conditions is None:
            conditions = jnp.zeros(self.num_trials, dtype=int)

        def _dynamics_outer_step(carry, outer_args):
            key, t = outer_args

            def _dynamics_step(prev_state, args):
                key, inpt, idx = args
                state = self.transition_distribution(params, prev_state, inpt).sample(seed=key)
                return state, state

            # Sample the initial state
            key1, key = jr.split(key, 2)
            initial_input = tree_map(lambda x: x[0], inputs)
            initial_state = self.initial_distribution(t, params, conditions, initial_input).sample(seed=key1)

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
            conditions,
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial.mean[conditions[timestep]],
                   params.initial.cov[conditions[timestep]])

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
        if self.stationary_emissions:
            mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
        else:
            mean = params.emissions.weights[timestep] @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias[timestep]
        return MVN(mean, params.emissions.cov)

    # this is exact for the stationary model
    def marginal_log_prob(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            conditions: jnp.array = None,
    ) -> Scalar:

        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)
        trials = jnp.arange(num_trials, dtype=int)

        def _get_marginal_ll(emission, input, mask, trial_r, condition):
            return lgssm_filter(params, emission, input, mask, trial_r, condition).marginal_loglik

        _get_marginal_ll_vmap = vmap(_get_marginal_ll, in_axes=(0, 0, 0, 0))
        marginal_lls = _get_marginal_ll_vmap(emissions, inputs, masks, trials, conditions)
        marginal_ll = marginal_lls.sum()

        return marginal_ll

    def ekf_marginal_log_prob(
            self,
            base_subspace,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            conditions: jnp.array = None,
    ) -> Scalar:

        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)

        f = self.get_f()
        h = self.get_h_v1(base_subspace, params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.initial_velocity.mean,
            initial_covariance=params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = extended_kalman_filter(NLGSSM_params, emissions,
                                                    masks, conditions=conditions,
                                                    inputs=inputs)

        return filtered_posterior.marginal_loglik

    def filter(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            conditions: jnp.array = None
    ) -> PosteriorGSSMFiltered:
        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)
        trials = jnp.arange(num_trials, dtype=int)

        lgssm_filter_vmap = vmap(lgssm_filter, in_axes=(None, 0, 0, 0, 0, 0))
        filters = lgssm_filter_vmap(params, emissions, inputs, masks, trials, conditions)
        return filters

    def ekf(
            self,
            base_subspace,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            conditions: jnp.array = None
    ):
        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)

        f = self.get_f()
        h = self.get_h_v1(base_subspace, params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.initial_velocity.mean,
            initial_covariance=params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = extended_kalman_filter(NLGSSM_params, emissions,
                                                    masks, conditions=conditions,
                                                    inputs=inputs)

        return filtered_posterior

    def smoother(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            conditions: jnp.array = None
    ) -> PosteriorGSSMSmoothed:
        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)
        trials = jnp.arange(num_trials, dtype=int)

        lgssm_smoother_vmap = vmap(lgssm_smoother, in_axes=(None, 0, 0, 0, 0, 0))
        smoothers = lgssm_smoother_vmap(params, emissions, inputs, masks, trials, conditions)
        return smoothers

    def eks(
            self,
            base_subspace,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            masks: jnp.array = None,
            conditions: jnp.array = None
    ):
        num_trials = emissions.shape[0]
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)

        f = self.get_f()
        h = self.get_h(base_subspace, params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.initial_velocity.mean,
            initial_covariance=params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        smoothed_posterior = extended_kalman_smoother(NLGSSM_params, emissions,
                                                    masks, conditions=conditions,
                                                    inputs=inputs)

        return smoothed_posterior

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
        smoothed_emissions = jnp.einsum('...lx,...yx->...ly', smoothed_means, H)
        smoothed_emissions_cov = jnp.einsum('...ya,...lab,...xb->...lyx', H, posterior.smoothed_covariances, H) + R

        if self.has_emissions_bias:
            smoothed_emissions += d[:, None]

        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, :, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    def log_joint(
            self,
            params: ParamsTVLGSSM,
            states,
            velocity,
            emissions,
            inputs,
            masks,
            conditions
    ) -> Scalar:

        """"""""
        # Double check priors for time-varying dynamics and emissions
        """"""""

        # initial state
        # lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean))
        lp = self.initial_mean_prior.log_prob(params.initial.mean).sum()
        flattened_cov = vmap(jnp.diag)(params.initial.cov)
        lp += self.initial_covariance_prior.log_prob(flattened_cov.flatten()).sum()
        lp += MVN(params.initial.mean[conditions], params.initial.cov[conditions]).log_prob(states[:, 0]).sum()

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
        lp += self.dynamics_covariance_prior.log_prob(jnp.diag(params.dynamics.cov)).sum()

        # emissions & observations
        def _compute_emissions_lp(prev_lp, current_t):
            current_y_mean = jnp.einsum('...ij,...j->...i', params.emissions.weights, states[:, current_t])
            new_lp = MVN(current_y_mean, params.emissions.cov).log_prob(emissions[:, current_t])
            masked_new_lp = jnp.nansum(masks[:, current_t] * new_lp)
            current_lp = prev_lp + masked_new_lp
            return current_lp, None
        lp, _ = jax.lax.scan(_compute_emissions_lp, lp, jnp.arange(self.sequence_length))

        if self.stationary_emissions:
            emissions_bias = params.emissions.bias if self.has_emissions_bias else jnp.zeros((self.emission_dim, 0))
            emissions_matrix = jnp.column_stack(
                (params.emissions.weights,
                 params.emissions.input_weights,
                 emissions_bias))
            lp += self.emissions_prior.log_prob(jnp.ravel(emissions_matrix))
        else:
            # tau_cov = jnp.eye(self.dof) * params.emissions.tau
            tau_cov = jnp.diag(params.emissions.tau)
            def _compute_trial_velocity_lp(prev_lp, current_t):
                current_param = velocity[current_t]
                next_param = velocity[current_t + 1]
                current_lp = prev_lp + MVN(loc=jnp.ravel(current_param),
                                           covariance_matrix=tau_cov).log_prob(jnp.ravel(next_param))
                return current_lp, None
            lp, _ = jax.lax.scan(_compute_trial_velocity_lp, lp, jnp.arange(self.num_trials - 1))

            initial_velocity = velocity[0]
            lp += MVN(params.initial_velocity.mean,
                      params.initial_velocity.cov).log_prob(initial_velocity)
            lp += self.initial_velocity_mean_prior.log_prob(params.initial_velocity.mean)
            lp += self.initial_velocity_covariance_prior.log_prob(jnp.diag(params.initial_velocity.cov)).sum()

            tau_lp = self.tau_prior.log_prob(params.emissions.tau)
            # lp += tau_lp
            lp += tau_lp.sum()

        lp += self.emissions_covariance_prior.log_prob(jnp.diag(params.emissions.cov)).sum()

        return lp

    def initialize_m_step_state(
            self,
            params: ParamsTVLGSSM,
            props: ParamsTVLGSSM
    ) -> Any:
        return None

    def get_f(self):
        def f(v):
            return v
        return f

    def get_h_v1(self, base_subspace, _params, masks):
        def h(v, obs_t, t, condition):
            C = rotate_subspace(base_subspace, self.state_dim, v)

            # new params constructed from model_params
            mu_0 = _params.initial.mean
            Sigma_0 = _params.initial.cov
            A = _params.dynamics.weights
            Q = _params.dynamics.cov
            R = _params.emissions.cov
            mu_v_0 = _params.initial_velocity.mean
            Sigma_v_0 = _params.initial_velocity.cov
            tau = _params.emissions.tau
            h_params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(
                    mean=mu_0,
                    cov=Sigma_0),
                dynamics=ParamsLGSSMDynamics(
                    weights=A,
                    bias=None,
                    input_weights=jnp.zeros((self.state_dim, 0)),
                    cov=Q),
                emissions=ParamsLGSSMEmissions(
                    weights=C,
                    bias=None,
                    input_weights=jnp.zeros((self.emission_dim, 0)),
                    cov=R,
                    tau=tau),
                initial_velocity=ParamsLGSSMInitial(mean=mu_v_0,
                                                    cov=Sigma_v_0),
            )

            filtered_posterior = lgssm_filter(h_params, obs_t, masks=masks[t], trial_r=t, condition=condition)

            # get pred means and covs
            pred_means = filtered_posterior.predicted_means
            pred_covs = filtered_posterior.predicted_covariances

            pred_obs_means = jnp.einsum('ij,tj->ti', C, pred_means)
            pred_obs_covs = jnp.einsum('ij,tjk,lk->til', C, pred_covs, C) + R
            # pred_obs_covs_sqrt = jnp.linalg.cholesky(pred_obs_covs)

            # pred_obs_means += jnp.einsum('til,tl->ti', pred_obs_covs_sqrt, eps.reshape(-1, self.emission_dim))

            return pred_obs_means.flatten(), pred_obs_covs

        return h

    def get_h_v2(self, base_subspace, _params, masks):
        def h(v, eps, obs_t, t, condition):
            C = rotate_subspace(base_subspace, self.state_dim, v)

            # new params constructed from model_params
            mu_0 = _params.initial.mean
            Sigma_0 = _params.initial.cov
            A = _params.dynamics.weights
            Q = _params.dynamics.cov
            R = _params.emissions.cov
            mu_v_0 = _params.initial_velocity.mean
            Sigma_v_0 = _params.initial_velocity.cov
            tau = _params.emissions.tau
            h_params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(
                    mean=mu_0,
                    cov=Sigma_0),
                dynamics=ParamsLGSSMDynamics(
                    weights=A,
                    bias=None,
                    input_weights=jnp.zeros((self.state_dim, 0)),
                    cov=Q),
                emissions=ParamsLGSSMEmissions(
                    weights=C,
                    bias=None,
                    input_weights=jnp.zeros((self.emission_dim, 0)),
                    cov=R,
                    tau=tau),
                initial_velocity=ParamsLGSSMInitial(mean=mu_v_0,
                                                    cov=Sigma_v_0),
            )

            filtered_posterior = lgssm_filter(h_params, obs_t, masks=masks[t], trial_r=t, condition=condition)

            # get pred means and covs
            pred_means = filtered_posterior.predicted_means
            pred_covs = filtered_posterior.predicted_covariances

            pred_obs_means = jnp.einsum('ij,tj->ti', C, pred_means)
            pred_obs_covs = jnp.einsum('ij,tjk,lk->til', C, pred_covs, C) + R
            pred_obs_covs_sqrt = jnp.linalg.cholesky(pred_obs_covs)

            pred_obs_means += jnp.einsum('til,tl->ti', pred_obs_covs_sqrt, eps.reshape(-1, self.emission_dim)) # could be optimized

            return pred_obs_means.flatten()

        return h

    def velocity_sample(self, base_subspace, _params, _emissions,
                        masks, conditions, rng, velocity_sampler='ekf'):
        f = self.get_f()
        if velocity_sampler == 'ekf':
            h = self.get_h_v1(base_subspace, _params, masks)
        elif velocity_sampler == 'ukf':
            h = self.get_h_v2(base_subspace, _params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=_params.initial_velocity.mean,
            initial_covariance=_params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(_params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        if velocity_sampler == 'ekf':
            velocity, approx_marginal_ll = extended_kalman_posterior_sample(rng, NLGSSM_params, _emissions,
                                                                            masks=masks, conditions=conditions)
        elif velocity_sampler == 'ukf':
            ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
            velocity, approx_marginal_ll = unscented_kalman_posterior_sample(rng, NLGSSM_params, _emissions,
                                                                             conditions=conditions,
                                                                             hyperparams=ukf_hyperparams)


        return velocity, approx_marginal_ll

    def velocity_smoother(self, base_subspace, _params, _emissions,
                        masks, conditions, rng, velocity_sampler='ekf'):
        f = self.get_f()
        if velocity_sampler == 'ekf':
            h = self.get_h_v1(base_subspace, _params, masks)
        elif velocity_sampler == 'ukf':
            h = self.get_h_v2(base_subspace, _params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=_params.initial_velocity.mean,
            initial_covariance=_params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(_params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        if velocity_sampler == 'ekf':
            smoother = extended_kalman_smoother(NLGSSM_params, _emissions,
                                                masks=masks, conditions=conditions)
        elif velocity_sampler == 'ukf':
            ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
            smoother = unscented_kalman_smoother(NLGSSM_params, _emissions,
                                                 conditions=conditions,
                                                 hyperparams=ukf_hyperparams)


        return smoother.smoothed_means, smoother.marginal_loglik

    def fit_blocked_gibbs(
            self,
            key: PRNGKey,
            initial_params: ParamsTVLGSSM,
            sample_size: int,
            emissions: Float[Array, "nbatch ntime emission_dim"],
            base_subspace,
            inputs: Optional[Float[Array, "nbatch ntime input_dim"]] = None,
            return_states: bool = False,
            return_n_samples: int = 100,
            print_ll: bool = False,
            masks: jnp.array = None,
            conditions: jnp.array = None,
            fixed_states: jnp.array = None,
            velocity_sampler = 'ekf',
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
        if conditions is None:
            conditions = jnp.zeros(num_timesteps, dtype=int)

        def sufficient_stats_from_sample(states, params):
            """Convert samples of states to sufficient statistics."""
            inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x, xp, xn = states, states[:, :-1], states[:, 1:]
            u, up = inputs_joint, inputs_joint[:-1]
            y = emissions

            conditions_one_hot = jnn.one_hot(conditions, self.num_conditions)
            conditions_count = jnp.sum(conditions_one_hot, axis=0)[:, None]
            init_stats_1 = jnp.einsum('bc,bi->bci',
                                      conditions_one_hot,
                                      x[:, 0])
            init_stats_1_sum = init_stats_1.sum(0)
            init_stats_1_avg = jnp.where(conditions_count > 0.0,
                                         jnp.divide(init_stats_1_sum, conditions_count),
                                         0.0) # average (C, D)
            init_stats_1_avg = jnp.einsum('bc,ci->bci',
                                          conditions_one_hot,
                                          init_stats_1_avg)
            init_stats_2 = init_stats_1 - init_stats_1_avg
            init_stats_2 = jnp.einsum('bci,bcj->cij', init_stats_2, init_stats_2)

            init_stats = (init_stats_1_sum, init_stats_2, conditions_count)

            # N, D = y.shape[-1], states.shape[-1]
            # # Optimized Code
            # reshape_dim = D * (D + self.has_dynamics_bias)
            # if self.has_dynamics_bias:
            #     xp = jnp.pad(xp, ((0, 0), (0, 0), (0, 1)), constant_values=1)
            # Qinv = jnp.linalg.inv(params.dynamics.cov)# + jnp.eye(params.dynamics.cov.shape[-1]) * self.EPS)
            # dynamics_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', xp, Qinv, xp, masks[:, 1:]).reshape(reshape_dim, reshape_dim)
            # dynamics_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', xn, Qinv, xp, masks[:, 1:]).reshape(-1)
            # dynamics_stats = (dynamics_stats_1, dynamics_stats_2)

            sum_zpzpT = jnp.einsum('bti,btl,bt->il', xp, xp, masks[:, 1:])
            sum_zpxnT = jnp.einsum('bti,btl,bt->il', xp, xn, masks[:, 1:])
            sum_xnxnT = jnp.einsum('bti,btl,bt->il', xn, xn, masks[:, 1:])
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, masks.sum() - len(emissions))

            # Quantities for the emissions
            if self.stationary_emissions:
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

        def lgssm_params_sample(rng, stats, states, params, velocity, _emission_weights):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats
            n_splits = 10
            rngs = iter(jr.split(rng, n_splits))

            # Sample the initial params
            if self.fix_initial:
                S, m = params.initial.cov, params.initial.mean
            else:
                initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
                S, m = initial_posterior.sample(seed=next(rngs))

            # Sample the dynamics params
            if self.fix_dynamics:
                F = params.dynamics.weights
                b = params.dynamics.bias
                B = params.dynamics.input_weights
                Q = params.dynamics.cov
            else:
                dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
                Q, FB = dynamics_posterior.sample(seed=next(rngs))
                F = FB[:, :self.state_dim]
                B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
                    else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

            # Sample the emission params
            if self.fix_emissions:
                H = params.emissions.weights
                d = params.emissions.bias
                D = params.emissions.input_weights
                R = params.emissions.cov
                initial_velocity_cov = params.initial_velocity.cov
                initial_velocity_mean = params.initial_velocity.mean
                tau = params.emissions.tau
            else:
                y = emissions
                D = jnp.zeros((self.emission_dim, 0))

                if self.stationary_emissions:
                    initial_velocity_cov = params.initial_velocity.cov
                    initial_velocity_mean = params.initial_velocity.mean
                    tau = params.emissions.tau

                    emission_posterior = mvn_posterior_update(self.emissions_prior, emission_stats)
                    _emissions_weights = emission_posterior.sample(seed=next(rngs))
                    _emissions_weights = _emissions_weights.reshape(self.emission_dim, (self.state_dim + self.has_emissions_bias))
                    H, d = (_emissions_weights[:, :-1], _emissions_weights[:, -1]) if self.has_emissions_bias \
                        else (_emissions_weights, None)
                else:
                    H = _emission_weights
                    d = None

                    initial_velocity_stats = (velocity[0], 0.0, 1.0)
                    initial_velocity_posterior = niw_posterior_update(self.initial_velocity_prior, initial_velocity_stats)
                    initial_velocity_cov, initial_velocity_mean = initial_velocity_posterior.sample(seed=next(rngs))

                    if self.fix_tau: # set to true during test time
                        tau = params.emissions.tau
                    else:
                        tau_stats_1 = jnp.ones((self.dof, 1)) * (self.num_trials - 1) / 2
                        tau_stats_2 = jnp.diff(velocity, axis=0)
                        tau_stats_2 = jnp.nansum(jnp.square(tau_stats_2), axis=0) / 2
                        tau_stats_2 = jnp.expand_dims(tau_stats_2, -1)
                        tau_stats = (tau_stats_1, tau_stats_2)
                        tau_posterior = ig_posterior_update(self.tau_prior, tau_stats)
                        tau = tau_posterior.sample(seed=next(rngs))
                        tau = jnp.ravel(tau)

                if self.fix_emissions_cov:
                    R = params.emissions.cov
                else:
                    emissions_cov_stats_1 = jnp.ones((self.emission_dim, 1)) * (jnp.sum(masks) / 2)
                    emissions_mean = jnp.einsum('...tx,...yx->...ty', states, H)
                    if self.has_emissions_bias:
                        emissions_mean += d[:, None]
                    sqr_err_flattened = jnp.square(y - emissions_mean).reshape(-1, self.emission_dim)
                    masks_flattened = masks.reshape(-1)
                    sqr_err_flattened = sqr_err_flattened * masks_flattened[:, None]
                    emissions_cov_stats_2 = jnp.nansum(sqr_err_flattened, axis=0) / 2

                    emissions_cov_stats_2 = jnp.expand_dims(emissions_cov_stats_2, -1)
                    emissions_cov_stats = (emissions_cov_stats_1, emissions_cov_stats_2)
                    emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior,
                                                                  emissions_cov_stats)
                    emissions_cov = emissions_cov_posterior.sample(seed=next(rngs))
                    # emissions_cov = jnp.where(jnp.logical_or(jnp.isnan(emissions_cov), emissions_cov < self.EPS),
                    #                           self.EPS, emissions_cov)
                    # emissions_cov += self.EPS
                    R = jnp.diag(jnp.ravel(emissions_cov))

            params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               tau=tau),
                initial_velocity=ParamsLGSSMInitial(mean=initial_velocity_mean,
                                                    cov=initial_velocity_cov),
            )
            return params

        @jit
        def one_sample(_params, _emissions, _inputs, rng):
            rngs = jr.split(rng, 3)

            if self.stationary_emissions:
                _new_params_emissions_updated = _params
                velocity = None
                _updated_emission_weights = None
            else:
                velocity, approx_marginal_ll = self.velocity_sample(base_subspace, _params,
                                                                    _emissions, masks, conditions,
                                                                    rngs[2], velocity_sampler)
                _updated_emission_weights = vmap(rotate_subspace, in_axes=(None, None, 0))(base_subspace,
                                                                                           self.state_dim,
                                                                                           velocity)

                mu_0 = _params.initial.mean
                Sigma_0 = _params.initial.cov
                A = _params.dynamics.weights
                Q = _params.dynamics.cov
                R = _params.emissions.cov
                mu_v_0 = _params.initial_velocity.mean
                Sigma_v_0 = _params.initial_velocity.cov
                tau = _params.emissions.tau
                _new_params_emissions_updated = ParamsTVLGSSM(
                    initial=ParamsLGSSMInitial(
                        mean=mu_0,
                        cov=Sigma_0),
                    dynamics=ParamsLGSSMDynamics(
                        weights=A,
                        bias=None,
                        input_weights=jnp.zeros((self.state_dim, 0)),
                        cov=Q),
                    emissions=ParamsLGSSMEmissions(
                        weights=_updated_emission_weights,
                        bias=None,
                        input_weights=jnp.zeros((self.emission_dim, 0)),
                        cov=R,
                        tau=tau),
                    initial_velocity=ParamsLGSSMInitial(mean=mu_v_0,
                                                        cov=Sigma_v_0),
                )

            if fixed_states is not None:
                _new_states = fixed_states
            else:
                _new_states, _approx_marginal_lls = lgssm_posterior_sample_vmap(rngs[0],
                                                                                _new_params_emissions_updated,
                                                                                _emissions,
                                                                                inputs, masks,
                                                                                jnp.arange(self.num_trials, dtype=int),
                                                                                conditions)
                if self.stationary_emissions:
                    approx_marginal_ll = _approx_marginal_lls.sum() # This is exact

            # Sample parameters
            _stats = sufficient_stats_from_sample(_new_states, _new_params_emissions_updated)
            _new_params = lgssm_params_sample(rngs[1], _stats, _new_states,
                                              _new_params_emissions_updated, velocity, _updated_emission_weights)

            # compute the log joint
            # _ll = self.log_joint(_new_params, _states, _emissions, _inputs, masks)
            _ll = self.log_joint(_new_params, _new_states, velocity,
                                 _emissions, _inputs, masks, conditions)

            return _new_params, _new_states, velocity, _ll, approx_marginal_ll

        sample_of_params = []
        sample_of_states = []
        sample_of_velocity = []
        lls = []
        marginal_lls = []
        keys = iter(jr.split(key, sample_size + 1))
        current_params = initial_params
        lgssm_posterior_sample_vmap = vmap(lgssm_posterior_sample,
                                           in_axes=(None, None, 0, None, 0, 0, 0))

        for sample_itr in progress_bar(range(sample_size)):
            current_params, current_states, current_velocity, ll, approx_marginal_ll = one_sample(current_params,
                                                                                                  emissions,
                                                                                                  inputs,
                                                                                                  next(keys))
            if sample_itr >= sample_size - return_n_samples:
                sample_of_params.append(current_params)
                sample_of_velocity.append(current_velocity)
            if return_states and (sample_itr >= sample_size - return_n_samples):
                sample_of_states.append(current_states)
            if print_ll:
                print(ll, approx_marginal_ll)
            lls.append(ll)
            marginal_lls.append(approx_marginal_ll)

        return pytree_stack(sample_of_params), sample_of_states, sample_of_velocity, lls, marginal_lls

    def fit_em(
            self,
            initial_params: ParamsTVLGSSM,
            sample_size: int,
            emissions: Float[Array, "nbatch ntime emission_dim"],
            base_subspace,
            inputs: Optional[Float[Array, "nbatch ntime input_dim"]] = None,
            return_states: bool = False,
            return_n_samples: int = 100,
            print_ll: bool = False,
            masks: jnp.array = None,
            conditions: jnp.array = None,
            fixed_states: jnp.array = None,
            velocity_sampler = 'ekf',
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
        if conditions is None:
            conditions = jnp.zeros(num_timesteps, dtype=int)

        def sufficient_stats_from_sample(states, params):
            """Convert samples of states to sufficient statistics."""
            inputs_joint = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x, xp, xn = states, states[:, :-1], states[:, 1:]
            u, up = inputs_joint, inputs_joint[:-1]
            y = emissions

            conditions_one_hot = jnn.one_hot(conditions, self.num_conditions)
            conditions_count = jnp.sum(conditions_one_hot, axis=0)[:, None]
            init_stats_1 = jnp.einsum('bc,bi->bci',
                                      conditions_one_hot,
                                      x[:, 0])
            init_stats_1_sum = init_stats_1.sum(0)
            init_stats_1_avg = jnp.where(conditions_count > 0.0,
                                         jnp.divide(init_stats_1_sum, conditions_count),
                                         0.0) # average (C, D)
            init_stats_1_avg = jnp.einsum('bc,ci->bci',
                                          conditions_one_hot,
                                          init_stats_1_avg)
            init_stats_2 = init_stats_1 - init_stats_1_avg
            init_stats_2 = jnp.einsum('bci,bcj->cij', init_stats_2, init_stats_2)

            init_stats = (init_stats_1_sum, init_stats_2, conditions_count)

            # N, D = y.shape[-1], states.shape[-1]
            # # Optimized Code
            # reshape_dim = D * (D + self.has_dynamics_bias)
            # if self.has_dynamics_bias:
            #     xp = jnp.pad(xp, ((0, 0), (0, 0), (0, 1)), constant_values=1)
            # Qinv = jnp.linalg.inv(params.dynamics.cov)# + jnp.eye(params.dynamics.cov.shape[-1]) * self.EPS)
            # dynamics_stats_1 = jnp.einsum('bti,jk,btl,bt->jikl', xp, Qinv, xp, masks[:, 1:]).reshape(reshape_dim, reshape_dim)
            # dynamics_stats_2 = jnp.einsum('bti,ik,btl,bt->kl', xn, Qinv, xp, masks[:, 1:]).reshape(-1)
            # dynamics_stats = (dynamics_stats_1, dynamics_stats_2)

            sum_zpzpT = jnp.einsum('bti,btl,bt->il', xp, xp, masks[:, 1:])
            sum_zpxnT = jnp.einsum('bti,btl,bt->il', xp, xn, masks[:, 1:])
            sum_xnxnT = jnp.einsum('bti,btl,bt->il', xn, xn, masks[:, 1:])
            dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, masks.sum() - len(emissions))

            # Quantities for the emissions
            if self.stationary_emissions:
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

        def lgssm_params_sample(stats, states, params, velocity, _emission_weights):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats

            # Sample the initial params
            if self.fix_initial:
                S, m = params.initial.cov, params.initial.mean
            else:
                initial_posterior = niw_posterior_update(self.initial_prior, init_stats)
                S, m = initial_posterior.mode()

            # Sample the dynamics params
            if self.fix_dynamics:
                F = params.dynamics.weights
                b = params.dynamics.bias
                B = params.dynamics.input_weights
                Q = params.dynamics.cov
            else:
                dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
                Q, FB = dynamics_posterior.mode()
                F = FB[:, :self.state_dim]
                B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
                    else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

            # Sample the emission params
            if self.fix_emissions:
                H = params.emissions.weights
                d = params.emissions.bias
                D = params.emissions.input_weights
                R = params.emissions.cov
                initial_velocity_cov = params.initial_velocity.cov
                initial_velocity_mean = params.initial_velocity.mean
                tau = params.emissions.tau
            else:
                y = emissions
                D = jnp.zeros((self.emission_dim, 0))

                if self.stationary_emissions:
                    initial_velocity_cov = params.initial_velocity.cov
                    initial_velocity_mean = params.initial_velocity.mean
                    tau = params.emissions.tau

                    emission_posterior = mvn_posterior_update(self.emissions_prior, emission_stats)
                    _emissions_weights = emission_posterior.mode()
                    _emissions_weights = _emissions_weights.reshape(self.emission_dim, (self.state_dim + self.has_emissions_bias))
                    H, d = (_emissions_weights[:, :-1], _emissions_weights[:, -1]) if self.has_emissions_bias \
                        else (_emissions_weights, None)
                else:
                    H = _emission_weights
                    d = None

                    initial_velocity_stats = (velocity[0], 0.0, 1.0)
                    initial_velocity_posterior = niw_posterior_update(self.initial_velocity_prior, initial_velocity_stats)
                    initial_velocity_cov, initial_velocity_mean = initial_velocity_posterior.mode()

                    if self.fix_tau: # set to true during test time
                        tau = params.emissions.tau
                    else:
                        tau_stats_1 = jnp.ones((self.dof, 1)) * (self.num_trials - 1) / 2
                        tau_stats_2 = jnp.diff(velocity, axis=0)
                        tau_stats_2 = jnp.nansum(jnp.square(tau_stats_2), axis=0) / 2
                        tau_stats_2 = jnp.expand_dims(tau_stats_2, -1)
                        tau_stats = (tau_stats_1, tau_stats_2)
                        tau_posterior = ig_posterior_update(self.tau_prior, tau_stats)
                        tau = tau_posterior.mode()
                        tau = jnp.ravel(tau)

                if self.fix_emissions_cov:
                    R = params.emissions.cov
                else:
                    emissions_cov_stats_1 = jnp.ones((self.emission_dim, 1)) * (jnp.sum(masks) / 2)
                    emissions_mean = jnp.einsum('...tx,...yx->...ty', states, H)
                    if self.has_emissions_bias:
                        emissions_mean += d[:, None]
                    sqr_err_flattened = jnp.square(y - emissions_mean).reshape(-1, self.emission_dim)
                    masks_flattened = masks.reshape(-1)
                    sqr_err_flattened = sqr_err_flattened * masks_flattened[:, None]
                    emissions_cov_stats_2 = jnp.nansum(sqr_err_flattened, axis=0) / 2

                    emissions_cov_stats_2 = jnp.expand_dims(emissions_cov_stats_2, -1)
                    emissions_cov_stats = (emissions_cov_stats_1, emissions_cov_stats_2)
                    emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior,
                                                                  emissions_cov_stats)
                    emissions_cov = emissions_cov_posterior.mode()
                    # emissions_cov = jnp.where(jnp.logical_or(jnp.isnan(emissions_cov), emissions_cov < self.EPS),
                    #                           self.EPS, emissions_cov)
                    # emissions_cov += self.EPS
                    R = jnp.diag(jnp.ravel(emissions_cov))

            params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               tau=tau),
                initial_velocity=ParamsLGSSMInitial(mean=initial_velocity_mean,
                                                    cov=initial_velocity_cov),
            )
            return params

        @jit
        def one_sample(_params, _emissions, _inputs):
            rngs = jr.split(rng, 3)

            if self.stationary_emissions:
                _new_params_emissions_updated = _params
                velocity = None
                _updated_emission_weights = None
            else:
                velocity, approx_marginal_ll = self.velocity_sample(base_subspace, _params,
                                                                    _emissions, masks, conditions,
                                                                    rngs[2], velocity_sampler)
                _updated_emission_weights = vmap(rotate_subspace, in_axes=(None, None, 0))(base_subspace,
                                                                                           self.state_dim,
                                                                                           velocity)

                mu_0 = _params.initial.mean
                Sigma_0 = _params.initial.cov
                A = _params.dynamics.weights
                Q = _params.dynamics.cov
                R = _params.emissions.cov
                mu_v_0 = _params.initial_velocity.mean
                Sigma_v_0 = _params.initial_velocity.cov
                tau = _params.emissions.tau
                _new_params_emissions_updated = ParamsTVLGSSM(
                    initial=ParamsLGSSMInitial(
                        mean=mu_0,
                        cov=Sigma_0),
                    dynamics=ParamsLGSSMDynamics(
                        weights=A,
                        bias=None,
                        input_weights=jnp.zeros((self.state_dim, 0)),
                        cov=Q),
                    emissions=ParamsLGSSMEmissions(
                        weights=_updated_emission_weights,
                        bias=None,
                        input_weights=jnp.zeros((self.emission_dim, 0)),
                        cov=R,
                        tau=tau),
                    initial_velocity=ParamsLGSSMInitial(mean=mu_v_0,
                                                        cov=Sigma_v_0),
                )

            if fixed_states is not None:
                _new_states = fixed_states
            else:
                states_smoother = lgssm_smoother_vmap(_new_params_emissions_updated,
                                                    _emissions,
                                                    inputs, masks,
                                                    jnp.arange(self.num_trials, dtype=int),
                                                    conditions)
                _new_states, _approx_marginal_lls = states_smoother.smoothed_means, states_smoother.marginal_loglik
                if self.stationary_emissions:
                    approx_marginal_ll = _approx_marginal_lls.sum() # This is exact

            # Sample parameters
            _stats = sufficient_stats_from_sample(_new_states, _new_params_emissions_updated)
            _new_params = lgssm_params_sample(_stats, _new_states,
                                              _new_params_emissions_updated, velocity, _updated_emission_weights)

            return _new_params, _new_states, velocity, approx_marginal_ll

        sample_of_params = []
        sample_of_states = []
        sample_of_velocity = []
        marginal_lls = []
        current_params = initial_params
        lgssm_smoother_vmap = vmap(lgssm_smoother, in_axes=(None, 0, None, 0, 0, 0))

        for sample_itr in progress_bar(range(sample_size)):
            current_params, current_states, current_velocity, approx_marginal_ll = one_sample(current_params,
                                                                                              emissions,
                                                                                              inputs)
            if sample_itr >= sample_size - return_n_samples:
                sample_of_params.append(current_params)
                sample_of_velocity.append(current_velocity)
            if return_states and (sample_itr >= sample_size - return_n_samples):
                sample_of_states.append(current_states)
            if print_ll:
                print(approx_marginal_ll)
            marginal_lls.append(approx_marginal_ll)

        return pytree_stack(sample_of_params), sample_of_states, sample_of_velocity, marginal_lls