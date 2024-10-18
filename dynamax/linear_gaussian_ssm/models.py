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
from dynamax.nonlinear_gaussian_ssm import unscented_kalman_posterior_sample, unscented_kalman_smoother, unscented_kalman_filter_v2
from dynamax.nonlinear_gaussian_ssm import extended_kalman_smoother, iterated_extended_kalman_posterior_sample
from dynamax.nonlinear_gaussian_ssm import extended_kalman_filter, iterated_extended_kalman_filter, \
    extended_kalman_posterior_sample, extended_kalman_filter_v1

from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalPrecision as MNP
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import mniw_posterior_update, niw_posterior_update, mvn_posterior_update, \
    ig_posterior_update, mnp_posterior_update
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
            has_dynamics_bias: bool = False,
            has_emissions_bias: bool = False,
            num_conditions: int = 1,
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias
        self.num_conditions = num_conditions

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
        _initial_mean = jnp.zeros((self.num_conditions, self.state_dim))
        _initial_covariance = jnp.tile(jnp.eye(self.state_dim)[None], (self.num_conditions, 1, 1))
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
                cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
                tau=None)
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
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                tau=ParameterProperties())
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
            emissions,
            inputs,
            masks,
            condition,
            condition_one_hot,
            trial,
    ) -> Tuple[SuffStatsLGSSM, Scalar]:
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        masks_a = jnp.expand_dims(masks, -1)
        masks_aa = jnp.expand_dims(masks_a, -1)
        # ensure masking is done properly
        emissions = emissions * masks_a

        # jax.debug.print("{x}", x=masks.shape)
        # jax.debug.print("{x}", x=trial)
        # jax.debug.print("{x}", x=condition)
        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(params, emissions, inputs, masks=masks, trial_r=trial, condition=condition)

        # shorthand
        Ex = posterior.smoothed_means * masks_a
        Exp = posterior.smoothed_means[:-1] * jnp.roll(masks_a, -1, axis=1)[:-1]
        Exn = posterior.smoothed_means[1:] * masks_a[1:]
        Vx = posterior.smoothed_covariances * masks_aa
        Vxp = posterior.smoothed_covariances[:-1] * jnp.roll(masks_aa, -1, axis=1)[:-1]
        Vxn = posterior.smoothed_covariances[1:] * masks_aa[1:]
        Expxn = posterior.smoothed_cross_covariances * jnp.roll(masks_aa, -1, axis=1)[:-1]

        # Append bias to the inputs
        inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
        up = inputs[:-1]
        u = inputs
        y = emissions

        # expected sufficient statistics for the initial tfd.Distribution
        Ex0 = posterior.smoothed_means[0]
        Ex0x0T = posterior.smoothed_covariances[0] + jnp.outer(Ex0, Ex0)
        Ex0 = jnp.einsum('c,i->ci', condition_one_hot, Ex0)
        Ex0x0T = jnp.einsum('c,ij->cij', condition_one_hot, Ex0x0T)
        init_stats = (Ex0, Ex0x0T, condition_one_hot)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, masks.sum() - 1)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                              num_timesteps - 1)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        sum_zzT = jnp.block([[Ex.T @ Ex, Ex.T @ u], [u.T @ Ex, u.T @ u]])
        sum_zzT = sum_zzT.at[:self.state_dim, :self.state_dim].add(Vx.sum(0))
        sum_zyT = jnp.block([[Ex.T @ y], [u.T @ y]])
        sum_yyT = emissions.T @ emissions
        emission_stats = (sum_zzT, sum_zyT, sum_yyT, masks.sum())
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
                 has_dynamics_bias=False,
                 has_emissions_bias=False,
                 num_conditions=1,
                 **kw_priors):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim, input_dim=input_dim,
                         has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias,
                         num_conditions=num_conditions)

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
        lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean)).sum()

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
        # def get_Sm(init_stats_c):
        #     initial_posterior = niw_posterior_update(self.initial_prior, init_stats_c)
        #     S_c, m_c = initial_posterior.mode()
        #     return S_c, m_c
        # S, m = vmap(get_Sm)(init_stats)
        S, m = params.initial.cov, params.initial.mean

        # dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
        # Q, FB = dynamics_posterior.mode()
        # F = FB[:, :self.state_dim]
        # B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
        #     else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))
        Q = params.dynamics.cov
        F = params.dynamics.weights
        b = params.dynamics.bias
        B = params.dynamics.input_weights

        emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
        R, HD = emission_posterior.mode()
        H = HD[:, :self.state_dim]
        D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
            else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))
        # R = params.emissions.cov
        # H = params.emissions.weights
        # d = params.emissions.bias
        # D = params.emissions.input_weights

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                           tau=None)
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
            fix_initial_velocity: bool = False,
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
        self.fix_initial_velocity = fix_initial_velocity

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
            MVN(loc=jnp.zeros(self.state_dim * (self.state_dim + self.has_dynamics_bias)),
                covariance_matrix=jnp.eye(self.state_dim * (self.state_dim + self.has_dynamics_bias)))
        )

        self.dynamics_covariance_prior = default_prior(
            'dynamics_covariance_prior',
            IG(concentration=1.0, scale=1.0)
        )

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
                'initial_prior',
                NIW(loc=jnp.zeros(self.dof),
                    mean_concentration=1.,
                    df=self.dof + 0.1,
                    scale=jnp.eye(self.dof)))

            self.tau_prior = default_prior(
                'tau_prior',
                IG(concentration=1e-9, scale=1e-9)
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
        _initial_covariance = 0.1 * jnp.tile(jnp.eye(self.state_dim)[None], (self.num_conditions, 1, 1))

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
        elif emission_weights is None:
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
        else:
            _initial_velocity_mean = jnp.zeros(self.dof)
            _initial_velocity_cov = jnp.eye(self.dof)
            _velocity = None
            _emission_weights = None

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

    # def log_prior(
    #         self,
    #         params
    # ) -> Scalar:
    #     lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean)).sum()

    #     # dynamics
    #     dynamics_bias = params.dynamics.bias if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
    #     dynamics_matrix = jnp.column_stack((params.dynamics.weights,
    #                                         params.dynamics.input_weights,
    #                                         dynamics_bias))
    #     lp += self.dynamics_prior.log_prob((params.dynamics.cov, dynamics_matrix))

    #     emission_bias = params.emissions.bias if self.has_emissions_bias else jnp.zeros((self.emission_dim, 0))
    #     emission_matrix = jnp.column_stack((params.emissions.weights,
    #                                         params.emissions.input_weights,
    #                                         emission_bias))
    #     lp += self.emission_prior.log_prob((params.emissions.cov, emission_matrix))
    #     return lp

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

        _get_marginal_ll_vmap = vmap(_get_marginal_ll, in_axes=(0, 0, 0, 0, 0))
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
            tau_idx: jnp.array=None,
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
            dynamics_covariance=vmap(jnp.diag)(jnp.einsum('kb,ki->bi', tau_idx, params.emissions.tau)),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = extended_kalman_filter_v1(NLGSSM_params, emissions,
                                                       masks=masks, conditions=conditions,
                                                       inputs=inputs)

        return filtered_posterior.marginal_loglik

    def ukf_marginal_log_prob(
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
        h = self.get_h_v2(base_subspace, params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.initial_velocity.mean,
            initial_covariance=params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
        filtered_posterior = unscented_kalman_filter_v2(NLGSSM_params, emissions, hyperparams=ukf_hyperparams,
                                                       masks=masks, conditions=conditions,
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
            conditions: jnp.array = None,
            tau_idx: jnp.array = None,
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
            dynamics_covariance=vmap(jnp.diag)(jnp.einsum('kb,ki->bi', tau_idx, params.emissions.tau)),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = extended_kalman_filter_v1(NLGSSM_params, emissions,
                                                       masks=masks, conditions=conditions,
                                                       inputs=inputs)

        return filtered_posterior

    def ukf(
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
        h = self.get_h_v2(base_subspace, params, masks)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.initial_velocity.mean,
            initial_covariance=params.initial_velocity.cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
        filtered_posterior = unscented_kalman_filter_v2(NLGSSM_params, emissions, hyperparams=ukf_hyperparams,
                                                        masks=masks, conditions=conditions,
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
            masks,
            conditions
    ) -> Scalar:

        """"""""
        # Double check priors for time-varying dynamics and emissions
        """"""""

        # initial state
        # lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean))
        lp = self.initial_prior.log_prob(params.initial.mean).sum()
        flattened_cov = vmap(jnp.diag)(params.initial.cov)
        lp += self.initial_covariance_prior.log_prob(flattened_cov.flatten()).sum()
        # lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean)).sum()
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

            # lp += self.initial_velocity_prior.log_prob((params.initial_velocity.cov,
            #                                             params.initial_velocity.mean))
            lp += self.initial_velocity_prior.log_prob(params.initial_velocity.mean)
            flattened_cov = jnp.diag(params.initial_velocity.cov)
            lp += self.initial_velocity_covariance_prior.log_prob(flattened_cov).sum()

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

            pred_obs_means += jnp.einsum('til,tl->ti', pred_obs_covs_sqrt,
                                         eps.reshape(-1, self.emission_dim))  # could be optimized

            return pred_obs_means.flatten()

        return h

    def get_h_v3(self, base_subspace):
        def h(v):
            C = rotate_subspace(base_subspace, self.state_dim, v)
            return C.flatten()

        return h

    def velocity_sample(self, base_subspace, params, emissions,
                        masks, conditions, rng, covs=None, velocity_sampler='ekf'):
        f = self.get_f()
        if velocity_sampler == 'ekf':
            h = self.get_h_v1(base_subspace, params, masks)
        elif velocity_sampler == 'ukf':
            h = self.get_h_v2(base_subspace, params, masks)
        elif velocity_sampler == 'ekf_v2':
            h = self.get_h_v3(base_subspace)

        if velocity_sampler in ['ekf', 'ukf']:
            NLGSSM_params = ParamsNLGSSM(
                initial_mean=params.initial_velocity.mean,
                initial_covariance=params.initial_velocity.cov,
                dynamics_function=f,
                dynamics_covariance=jnp.diag(params.emissions.tau),
                emission_function=h,
                emission_covariance=covs
            )
        elif velocity_sampler in ['ekf_v2']:
            NLGSSM_params = ParamsNLGSSM(
                initial_mean=params.initial_velocity.mean,
                initial_covariance=params.initial_velocity.cov,
                dynamics_function=f,
                dynamics_covariance=jnp.diag(params.emissions.tau),
                emission_function=h,
                emission_covariance=covs
            )

        if velocity_sampler == 'ekf':
            velocity, approx_marginal_ll = extended_kalman_posterior_sample(rng, NLGSSM_params, emissions,
                                                                            masks=masks, conditions=conditions)
        elif velocity_sampler == 'ukf':
            ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
            velocity, approx_marginal_ll = unscented_kalman_posterior_sample(rng, NLGSSM_params, emissions,
                                                                             conditions=conditions,
                                                                             hyperparams=ukf_hyperparams)
        elif velocity_sampler == 'ekf_v2':
            velocity, approx_marginal_ll = extended_kalman_posterior_sample(rng, NLGSSM_params, emissions,
                                                                            masks=masks, conditions=conditions)

        return velocity, approx_marginal_ll

    def velocity_smoother(self, base_subspace, params, emissions,
                          masks, conditions, tau_idx=None, covs=None, filtering_method='ekf_em'):
        f = self.get_f()
        if filtering_method == 'ekf':
            h = self.get_h_v1(base_subspace, params, masks)
        elif filtering_method == 'ukf':
            h = self.get_h_v2(base_subspace, params, masks)
        elif filtering_method in ['ekf_em', 'ukf_em']:
            h = self.get_h_v3(base_subspace)

        if filtering_method in ['ekf', 'ukf']:
            NLGSSM_params = ParamsNLGSSM(
                initial_mean=params.initial_velocity.mean,
                initial_covariance=params.initial_velocity.cov,
                dynamics_function=f,
                dynamics_covariance=jnp.diag(params.emissions.tau),
                emission_function=h,
                emission_covariance=covs
            )
        elif filtering_method in ['ekf_em', 'ukf_em']:
            NLGSSM_params = ParamsNLGSSM(
                initial_mean=params.initial_velocity.mean,
                initial_covariance=params.initial_velocity.cov,
                dynamics_function=f,
                dynamics_covariance=vmap(jnp.diag)(jnp.einsum('kb,ki->bi', tau_idx, params.emissions.tau)),
                emission_function=h,
                emission_covariance=covs
            )

        if filtering_method == 'ekf':
            smoother = extended_kalman_smoother(NLGSSM_params, emissions,
                                                masks=masks, conditions=conditions)
        elif filtering_method == 'ukf':
            ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
            smoother = unscented_kalman_smoother(NLGSSM_params, emissions,
                                                 conditions=conditions,
                                                 hyperparams=ukf_hyperparams)
        elif filtering_method == 'ekf_em':
            smoother = extended_kalman_smoother(NLGSSM_params, emissions,
                                                masks=masks, conditions=conditions)
        elif filtering_method == 'ukf_em':
            ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
            smoother = unscented_kalman_smoother(NLGSSM_params, emissions,
                                                 hyperparams=ukf_hyperparams)

        return smoother

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
            velocity_sampler='ekf_v2',
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
        num_trials = len(emissions)
        if inputs is None:
            inputs = jnp.zeros((num_trials, 0))
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)
        trial_idx = jnp.arange(num_trials, dtype=int)

        masks_a = jnp.expand_dims(masks, -1)
        # ensure masking is done properly
        emissions = emissions * masks_a

        conditions_one_hot = jnn.one_hot(conditions, self.num_conditions)  # B x C
        conditions_count = jnp.sum(conditions_one_hot, axis=0)  # C

        lgssm_posterior_sample_vmap = vmap(lgssm_posterior_sample,
                                           in_axes=(None, None, 0, None, 0, 0, 0))

        if not self.fix_emissions_cov:
            yyT = jnp.einsum('...tx,...ty->xy', emissions, emissions)

        def sufficient_stats_from_sample(rng, _params):
            states, marginal_ll = lgssm_posterior_sample_vmap(rng, _params, emissions, inputs,
                                                              masks, trial_idx, conditions)

            """Convert samples of states to sufficient statistics."""
            # Let xn[t] = x[t+1]          for t = 0...T-2
            x = states * masks_a
            xp = states[:, :-1] * jnp.roll(masks_a, -1, axis=1)[:, :-1]
            xn = states[:, 1:] * masks_a[:, 1:]
            y = emissions

            init_stats_1 = jnp.einsum('bc,bi->bci',  conditions_one_hot, x[:, 0])
            init_stats = (init_stats_1, )

            Qinv = jnp.linalg.inv(_params.dynamics.cov)
            reshape_dim = self.state_dim * (self.state_dim + self.has_dynamics_bias)
            if self.has_dynamics_bias:
                ones = jnp.ones(xp.shape[:2] + (1,)) * jnp.roll(masks_a, -1, axis=1)[:, :-1]
                xp = jnp.concatenate([xp, ones], axis=-1)
            dynamics_stats_1 = jnp.einsum('bti,btl->il', xp, xp)
            dynamics_stats_1 = jnp.einsum('il,jk->jikl', dynamics_stats_1, Qinv).reshape(reshape_dim, reshape_dim)
            dynamics_stats_2 = jnp.einsum('bti,btl->il', xp, xn)
            dynamics_stats_2 = jnp.einsum('il,lk->ki', dynamics_stats_2, Qinv).reshape(-1)
            dynamics_stats = (dynamics_stats_1, dynamics_stats_2)

            # sufficient statistics for the emissions
            if self.stationary_emissions:
                Rinv = jnp.linalg.inv(_params.emissions.cov)
                reshape_dim = self.emission_dim * (self.state_dim + self.has_emissions_bias)
                emissions_stats_1 = jnp.einsum('bti,btl->il', x, x)
                emissions_stats_1 = jnp.einsum('il,jk->jikl', emissions_stats_1, Rinv).reshape(reshape_dim, reshape_dim)
                emissions_stats_2 = jnp.einsum('bti,btl->il', x, y)
                emissions_stats_2 = jnp.einsum('il,lk->ki', emissions_stats_2, Rinv).reshape(-1)
                emission_stats = (emissions_stats_1, emissions_stats_2)
            else:
                Rinv = jnp.linalg.inv(_params.emissions.cov)
                reshape_dim = self.emission_dim * (self.state_dim + self.has_emissions_bias)
                emissions_stats_1 = jnp.einsum('bti,btl->bil', x, x)
                emissions_stats_1 = jnp.einsum('bil,jk->bjikl', emissions_stats_1, Rinv).reshape(num_trials, reshape_dim, reshape_dim)
                emissions_stats_1 = jnp.linalg.inv(emissions_stats_1)
                emissions_stats_2 = jnp.einsum('bti,btl->bil', x, y)
                emissions_stats_2 = jnp.einsum('bil,lk->bki', emissions_stats_2, Rinv).reshape(num_trials, -1)
                emissions_stats_2 = jnp.einsum('bij,bj->bi', emissions_stats_1, emissions_stats_2)
                emission_stats = (emissions_stats_1, emissions_stats_2)

            return (init_stats, dynamics_stats, emission_stats), marginal_ll.sum(), x

        def lgssm_params_sample(rng, _params, stats, x):
            """Sample parameters of the model given sufficient statistics from observed states and emissions."""
            init_stats, dynamics_stats, emission_stats = stats
            n_splits = 10
            rngs = iter(jr.split(rng, n_splits))

            # Sample the initial params
            if self.fix_initial:
                S, m = _params.initial.cov, _params.initial.mean
            else:
                init_stats_1 = init_stats[0]
                Sinv = jnp.linalg.inv(_params.initial.cov)
                cSinv = jnp.einsum('bc,cij->bcij', conditions_one_hot, Sinv)
                initial_mean_stats_1 = cSinv.sum(0)
                initial_mean_stats_2 = jnp.einsum('bcij,bcj->ci', cSinv, init_stats_1)
                initial_mean_stats = (initial_mean_stats_1, initial_mean_stats_2)
                initial_mean_posterior = mvn_posterior_update(self.initial_prior, initial_mean_stats)
                m = initial_mean_posterior.sample(seed=next(rngs))

                initial_cov_stats_1 = conditions_count / 2
                Exm_diff = jnp.einsum('bc,bi->bci', conditions_one_hot, x[:, 0]) - jnp.einsum('bc,ci->bci',
                                                                                              conditions_one_hot, m)
                initial_cov_stats_2 = jnp.einsum('bci,bcj->cij', Exm_diff, Exm_diff)
                initial_cov_stats_2 = vmap(jnp.diag)(initial_cov_stats_2) / 2

                def update_initial_cov(initial_cov_stats_c_1, initial_cov_stats_c_2, rngs_c):
                    def _update_initial_cov(initial_cov_stats_ci_2, rngs_ci):
                        initial_posterior = ig_posterior_update(self.initial_covariance_prior,
                                                                (initial_cov_stats_c_1, initial_cov_stats_ci_2))
                        S_ci = initial_posterior.sample(seed=rngs_ci)
                        return S_ci

                    S_c = vmap(_update_initial_cov)(initial_cov_stats_c_2, jr.split(rngs_c, self.state_dim))
                    return jnp.diag(S_c)
                S = vmap(update_initial_cov)(initial_cov_stats_1, initial_cov_stats_2,
                                             jr.split(next(rngs), self.num_conditions))

            # Sample the dynamics params
            if self.fix_dynamics:
                F = _params.dynamics.weights
                b = _params.dynamics.bias
                B = _params.dynamics.input_weights
                Q = _params.dynamics.cov
            else:
                B = _params.dynamics.input_weights

                dynamics_stats_1, dynamics_stats_2 = dynamics_stats
                dynamics_weights_posterior = mvn_posterior_update(self.dynamics_prior,
                                                                  (dynamics_stats_1, dynamics_stats_2))
                Fb = dynamics_weights_posterior.sample(seed=next(rngs))
                Fb = Fb.reshape(self.state_dim, self.state_dim + self.has_dynamics_bias)
                F, b = (Fb[:, :self.state_dim], Fb[:, -1]) if self.has_dynamics_bias else (Fb[:, :self.state_dim], None)

                def update_dynamics_cov(s1, s2, rng_d):
                    dynamics_cov_posterior = ig_posterior_update(self.dynamics_covariance_prior, (s1, s2))
                    dynamics_cov = dynamics_cov_posterior.sample(seed=rng_d)
                    return dynamics_cov

                dynamics_cov_stats_1 = (jnp.sum(masks) - num_trials) / 2
                xp = x[:, :-1] * jnp.roll(masks_a, -1, axis=1)[:, :-1]
                xn = x[:, 1:] * masks_a[:, 1:]
                if self.has_dynamics_bias:
                    ones = jnp.ones(xp.shape[:2] + (1,)) * jnp.roll(masks_a, -1, axis=1)[:, :-1]
                    xp = jnp.concatenate([xp, ones], axis=-1)
                xpxn = jnp.einsum('bti,btl->btil', xp, xn)

                FbExpxn = jnp.einsum('ij,btjk->ik', Fb, xpxn)
                ExpxpT = jnp.einsum('bti,btl->il', xp, xp)
                dynamics_cov_stats_2 = jnp.einsum('bti,btj->ij', xn, xn)
                dynamics_cov_stats_2 -= (FbExpxn + FbExpxn.T)
                dynamics_cov_stats_2 += jnp.einsum('ij,jk,kl->il', Fb, ExpxpT, Fb.T)
                dynamics_cov_stats_2 = jnp.diag(dynamics_cov_stats_2) / 2
                Q = jnp.diag(vmap(update_dynamics_cov, in_axes=(None, 0, 0))(dynamics_cov_stats_1, dynamics_cov_stats_2,
                                                                             jr.split(next(rngs), self.state_dim)))

            # Sample the emission params
            if self.fix_emissions:
                v = None
                ekf_marginal_ll = 0.0

                H = _params.emissions.weights
                d = _params.emissions.bias
                D = _params.emissions.input_weights
                R = _params.emissions.cov
                initial_velocity_cov = _params.initial_velocity.cov
                initial_velocity_mean = _params.initial_velocity.mean
                tau = _params.emissions.tau
            elif self.stationary_emissions:
                v = None
                ekf_marginal_ll = 0.0

                d = _params.emissions.bias
                D = _params.emissions.input_weights
                initial_velocity_cov = _params.initial_velocity.cov
                initial_velocity_mean = _params.initial_velocity.mean
                tau = _params.emissions.tau

                emission_posterior = mvn_posterior_update(self.emissions_prior, emission_stats)
                _emissions_weights = emission_posterior.sample(seed=next(rngs))
                H = _emissions_weights.reshape(self.emission_dim, self.state_dim)

            else:
                d = _params.emissions.bias
                D = _params.emissions.input_weights

                emission_stats_1, emission_stats2 = emission_stats
                v, ekf_marginal_ll = self.velocity_sample(base_subspace, _params,
                                                          emission_stats2, masks, conditions, next(rngs),
                                                          covs=emission_stats_1,
                                                          velocity_sampler=velocity_sampler)

                Ev0 = v[0]
                H = vmap(rotate_subspace, in_axes=(None, None, 0))(base_subspace, self.state_dim, v)

                if self.fix_initial_velocity:
                    initial_velocity_mean = _params.initial_velocity.mean
                else:
                    # VvS = velocity_smoother.smoothed_covariances[0] + _params.initial_velocity.cov
                    VvS = _params.initial_velocity.cov
                    VvSinv = jnp.linalg.inv(VvS)
                    initial_velocity_mean_stats_1 = VvSinv
                    initial_velocity_mean_stats_2 = jnp.einsum('i,ij->j', Ev0, VvSinv)
                    initial_velocity_mean_stats = (initial_velocity_mean_stats_1, initial_velocity_mean_stats_2)
                    initial_velocity_mean_posterior = mvn_posterior_update(self.initial_velocity_prior, initial_velocity_mean_stats)
                    initial_velocity_mean = initial_velocity_mean_posterior.sample(seed=next(rngs))

                initial_velocity_cov_stats_1 = 0.5
                Evm_diff = Ev0 - initial_velocity_mean
                Evm_diff_squared = jnp.outer(Evm_diff, Evm_diff)
                initial_velocity_cov_stats_2 = Evm_diff_squared
                initial_velocity_cov_stats_2 = jnp.diag(initial_velocity_cov_stats_2) / 2
                def update_initial_velocity_cov(s1, s2, rng_v):
                    initial_velocity_cov_posterior = ig_posterior_update(self.initial_velocity_covariance_prior,
                                                                         (s1, s2))
                    initial_velocity_cov_i = initial_velocity_cov_posterior.sample(seed=rng_v)
                    return initial_velocity_cov_i
                initial_velocity_cov = jnp.diag(
                    vmap(update_initial_velocity_cov, in_axes=(None, 0, 0))(initial_velocity_cov_stats_1,
                                                                         initial_velocity_cov_stats_2,
                                                                            jr.split(next(rngs), self.dof)))

                if self.fix_tau:  # set to true during test time
                    tau = _params.emissions.tau
                else:
                    tau_stats_1 = jnp.ones(self.dof) * (self.num_trials - 1) / 2

                    vp, vn = v[:-1], v[1:]
                    vpvn = jnp.einsum('ti,tl->il', vp, vn)
                    tau_stats_2 = jnp.einsum('ti,tj->ij', v[1:], v[1:])
                    tau_stats_2 -= (vpvn + vpvn.T)
                    tau_stats_2 += jnp.einsum('ti,tj->ij', v[:-1], v[:-1])
                    tau_stats_2 = jnp.diag(tau_stats_2) / 2
                    def update_tau(s1, s2, rng_tau):
                        tau_posterior = ig_posterior_update(self.tau_prior, (s1, s2))
                        tau_mode = tau_posterior.sample(seed=rng_tau)
                        return tau_mode
                    tau = vmap(update_tau)(tau_stats_1, tau_stats_2, jr.split(next(rngs), self.dof))

            if self.fix_emissions_cov:
                R = _params.emissions.cov
            else:
                emissions_cov_stats_1 = jnp.sum(masks) / 2
                Ey = jnp.einsum('...tx,...yx->...ty', x, H)
                emissions_cov_stats_2 = jnp.sum(jnp.square(emissions - Ey) * masks_a, axis=(0, 1))
                emissions_cov_stats_2 = emissions_cov_stats_2 / 2

                def update_emissions_cov(s1, s2, rng_R):
                    emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior, (s1, s2))
                    emissions_cov = emissions_cov_posterior.sample(seed=rng_R)
                    return emissions_cov

                R = jnp.diag(vmap(update_emissions_cov, in_axes=(None, 0, 0))(emissions_cov_stats_1,
                                                                              emissions_cov_stats_2,
                                                                              jr.split(next(rngs), self.emission_dim)))

            params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               tau=tau),
                initial_velocity=ParamsLGSSMInitial(mean=initial_velocity_mean,
                                                    cov=initial_velocity_cov),
            )
            return params, v, ekf_marginal_ll

        @jit
        def one_sample(_params, rng):
            rngs = jr.split(rng, 2)

            stats, marginal_ll, x = sufficient_stats_from_sample(rngs[0], _params)
            new_params, v, ekf_marginal_ll = lgssm_params_sample(rngs[1], _params, stats, x)
            ll = self.log_joint(new_params, x, v, emissions, masks, conditions)

            return new_params, x, v, ll, marginal_ll, ekf_marginal_ll

        sample_of_params = []
        sample_of_states = []
        sample_of_velocity = []
        lls = []
        marginal_lls = []
        ekf_marginal_lls = []
        keys = iter(jr.split(key, sample_size + 1))
        current_params = initial_params

        for sample_itr in progress_bar(range(sample_size)):
            current_params, current_states, current_velocity, ll, marginal_ll, ekf_marginal_ll = one_sample(current_params,
                                                                                                            next(keys))
            if sample_itr >= sample_size - return_n_samples:
                sample_of_params.append(current_params)
                sample_of_velocity.append(current_velocity)
            if return_states and (sample_itr >= sample_size - return_n_samples):
                sample_of_states.append(current_states)
            if print_ll:
                print(ll, marginal_ll, ekf_marginal_ll)
            lls.append(ll)
            marginal_lls.append(marginal_ll)
            ekf_marginal_lls.append(ekf_marginal_ll)

        return pytree_stack(sample_of_params), sample_of_states, sample_of_velocity, lls, marginal_lls, ekf_marginal_lls

    def fit_em(
            self,
            initial_params: ParamsTVLGSSM,
            num_iters: int,
            emissions: Float[Array, "nbatch ntime emission_dim"],
            base_subspace,
            inputs: Optional[Float[Array, "nbatch ntime input_dim"]] = None,
            return_states: bool = False,
            print_ll: bool = False,
            masks: jnp.array = None,
            conditions: jnp.array = None,
            tau_idx: jnp.array = None,
            filtering_method='ekf_em',
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
        num_trials = len(emissions)
        if inputs is None:
            inputs = jnp.zeros((num_trials, 0))
        if masks is None:
            masks = jnp.ones(emissions.shape[:2], dtype=bool)
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)
        if tau_idx is None:
            tau_idx = jnp.ones(num_trials, dtype=bool)
        trial_idx = jnp.arange(num_trials, dtype=int)

        masks_a = jnp.expand_dims(masks, -1)
        masks_aa = jnp.expand_dims(masks_a, -1)
        # ensure masking is done properly
        emissions = emissions * masks_a

        conditions_one_hot = jnn.one_hot(conditions, self.num_conditions)  # B x C
        conditions_count = jnp.sum(conditions_one_hot, axis=0)  # C

        lgssm_smoother_vmap = vmap(lgssm_smoother, in_axes=(None, 0, None, 0, 0, 0))

        # if not self.fix_emissions_cov:
        #     yyT = jnp.einsum('...tx,...ty->xy', emissions, emissions)
        if self.has_dynamics_bias:
            ones = jnp.ones((emissions.shape[0], emissions.shape[1]-1, 1)) * jnp.roll(masks_a, -1, axis=1)[:, :-1]

        def e_step(_params):
            states_smoother = lgssm_smoother_vmap(_params, emissions, inputs, masks, trial_idx, conditions)

            # compute sufficient statistics
            # shorthand
            y = emissions
            Ex = states_smoother.smoothed_means * masks_a
            Exp = states_smoother.smoothed_means[:, :-1] * jnp.roll(masks_a, -1, axis=1)[:, :-1]
            Exn = states_smoother.smoothed_means[:, 1:] * masks_a[:, 1:]
            Vx = states_smoother.smoothed_covariances * masks_aa
            Vxp = states_smoother.smoothed_covariances[:, :-1] * jnp.roll(masks_aa, -1, axis=1)[:, :-1]
            Vxn = states_smoother.smoothed_covariances[:, 1:] * masks_aa[:, 1:]
            Expxn = states_smoother.smoothed_cross_covariances * jnp.roll(masks_aa, -1, axis=1)[:, :-1]

            # sufficient statistics for the initial distribution
            Ex0 = states_smoother.smoothed_means[:, 0]
            Ex0x0T = states_smoother.smoothed_covariances[:, 0] + vmap(jnp.outer)(Ex0, Ex0)
            init_stats = (jnp.einsum('bc,bi->ci', conditions_one_hot, Ex0),
                          jnp.einsum('bc,bij->cij', conditions_one_hot, Ex0x0T),
                          conditions_count)

            # sufficient statistics for the dynamics
            Qinv = jnp.linalg.inv(_params.dynamics.cov)
            reshape_dim = self.state_dim * (self.state_dim + self.has_dynamics_bias)
            if self.has_dynamics_bias:
                Exp = jnp.concatenate([Exp, ones], axis=-1)
            dynamics_stats_1 = jnp.einsum('bti,btl->il', Exp, Exp)
            dynamics_stats_1 = dynamics_stats_1.at[:self.state_dim, :self.state_dim].add(jnp.einsum('btij->ij', Vxp))
            dynamics_stats_1 = jnp.einsum('il,jk->jikl', dynamics_stats_1, Qinv).reshape(reshape_dim, reshape_dim)
            dynamics_stats_2 = jnp.einsum('btij->ij', Expxn)
            if self.has_dynamics_bias:
                dynamics_stats_2 = jnp.concatenate([dynamics_stats_2, jnp.einsum('bti,btj->ij', ones, Exn)], axis=0)
            dynamics_stats_2 = jnp.einsum('il,lk->ki', dynamics_stats_2, Qinv).reshape(-1)
            dynamics_stats = (dynamics_stats_1, dynamics_stats_2)

            # sufficient statistics for the emissions
            if self.stationary_emissions:
                Rinv = jnp.linalg.inv(_params.emissions.cov)
                reshape_dim = self.emission_dim * (self.state_dim + self.has_emissions_bias)
                emissions_stats_1 = jnp.einsum('bti,btl->il', Ex, Ex)
                emissions_stats_1 += jnp.einsum('btij->ij', Vx)
                emissions_stats_1 = jnp.einsum('il,jk->jikl', emissions_stats_1, Rinv).reshape(reshape_dim, reshape_dim)
                emissions_stats_2 = jnp.einsum('bti,btl->il', Ex, y)
                emissions_stats_2 = jnp.einsum('il,lk->ki', emissions_stats_2, Rinv).reshape(-1)
                emission_stats = (emissions_stats_1, emissions_stats_2)
            else:
                Rinv = jnp.linalg.inv(_params.emissions.cov)
                reshape_dim = self.emission_dim * (self.state_dim + self.has_emissions_bias)
                emissions_stats_1 = jnp.einsum('bti,btl->bil', Ex, Ex)
                emissions_stats_1 += jnp.einsum('btij->bij', Vx)
                emissions_stats_1 = jnp.einsum('bil,jk->bjikl', emissions_stats_1, Rinv).reshape(num_trials, reshape_dim, reshape_dim)
                # emissions_stats_1 += jnp.eye(emissions_stats_1.shape[-1]) * 1e-4
                # emissions_stats_1 = jnp.linalg.inv(emissions_stats_1)
                def psd_solve_map(a):
                    return psd_solve(a, jnp.eye(a.shape[-1]))
                emissions_stats_1 = lax.map(psd_solve_map, emissions_stats_1)
                emissions_stats_2 = jnp.einsum('bti,btl->bil', Ex, y)
                emissions_stats_2 = jnp.einsum('bil,lk->bki', emissions_stats_2, Rinv).reshape(num_trials, -1)
                emissions_stats_2 = jnp.einsum('bij,bj->bi', emissions_stats_1, emissions_stats_2)
                emission_stats = (emissions_stats_1, emissions_stats_2)

            # marginal likelihood
            marginal_ll = states_smoother.marginal_loglik.sum()

            return (init_stats, dynamics_stats, emission_stats), marginal_ll, states_smoother  # also returning Ex for R

        def m_step(_params, _stats, states_smoother):
            init_stats, dynamics_stats, emission_stats = _stats

            # Update the initial params
            if self.fix_initial:
                S, m = _params.initial.cov, _params.initial.mean
            else:
                # Perform MAP estimation jointly
                def update_initial(s1, s2, s3):
                    initial_posterior = niw_posterior_update(self.initial_prior, (s1, s2, s3))
                    Sc, mc = initial_posterior.mode()
                    return Sc, mc
                S, m = vmap(update_initial)(*init_stats)

            # Update the dynamics params
            if self.fix_dynamics:
                F = _params.dynamics.weights
                b = _params.dynamics.bias
                B = _params.dynamics.input_weights
                Q = _params.dynamics.cov
            else:
                B = _params.dynamics.input_weights

                dynamics_stats_1, dynamics_stats_2 = dynamics_stats
                dynamics_weights_posterior = mvn_posterior_update(self.dynamics_prior,
                                                                  (dynamics_stats_1, dynamics_stats_2))
                Fb = dynamics_weights_posterior.mode()
                Fb = Fb.reshape(self.state_dim, self.state_dim + self.has_dynamics_bias)
                F, b = (Fb[:, :self.state_dim], Fb[:, -1]) if self.has_dynamics_bias else (Fb[:, :self.state_dim], None)

                def update_dynamics_cov(s1, s2):
                    dynamics_cov_posterior = ig_posterior_update(self.dynamics_covariance_prior, (s1, s2))
                    dynamics_cov = dynamics_cov_posterior.mode()
                    return dynamics_cov

                dynamics_cov_stats_1 = (jnp.sum(masks) - num_trials) / 2
                Exp = states_smoother.smoothed_means[:, :-1] * jnp.roll(masks_a, -1, axis=1)[:, :-1]
                Exn = states_smoother.smoothed_means[:, 1:] * masks_a[:, 1:]
                Vxp = states_smoother.smoothed_covariances[:, :-1] * jnp.roll(masks_aa, -1, axis=1)[:, :-1]
                Vxn = states_smoother.smoothed_covariances[:, 1:] * masks_aa[:, 1:]
                Expxn = states_smoother.smoothed_cross_covariances * jnp.roll(masks_aa, -1, axis=1)[:, :-1]

                if self.has_dynamics_bias:
                    ones = jnp.ones(Exp.shape[:2] + (1,)) * jnp.roll(masks_a, -1, axis=1)[:, :-1]
                    Exp = jnp.concatenate([Exp, ones], axis=-1)
                    Expxn = jnp.concatenate([Expxn,
                                             jnp.einsum('bti,btj->btij', ones, Exn)], axis=-2)

                FbExpxn = jnp.einsum('ij,btjk->ik', Fb, Expxn)
                ExpxpT = jnp.einsum('bti,btl->il', Exp, Exp)
                ExpxpT = ExpxpT.at[:self.state_dim, :self.state_dim].add(jnp.einsum('btij->ij', Vxp))
                dynamics_cov_stats_2 = jnp.einsum('bti,btj->ij', Exn, Exn) + jnp.sum(Vxn, axis=(0, 1))
                dynamics_cov_stats_2 -= (FbExpxn + FbExpxn.T)
                dynamics_cov_stats_2 += jnp.einsum('ij,jk,kl->il', Fb, ExpxpT, Fb.T)
                dynamics_cov_stats_2 = jnp.diag(dynamics_cov_stats_2) / 2
                Q = jnp.diag(vmap(update_dynamics_cov, in_axes=(None, 0))(dynamics_cov_stats_1, dynamics_cov_stats_2))

            if self.fix_emissions:
                Ev = None
                marginal_ll = 0.0

                H = _params.emissions.weights
                d = _params.emissions.bias
                D = _params.emissions.input_weights
                initial_velocity_cov = _params.initial_velocity.cov
                initial_velocity_mean = _params.initial_velocity.mean
                tau = _params.emissions.tau
            elif self.stationary_emissions:
                Ev = None
                marginal_ll = 0.0

                d = _params.emissions.bias
                D = _params.emissions.input_weights
                initial_velocity_cov = _params.initial_velocity.cov
                initial_velocity_mean = _params.initial_velocity.mean
                tau = _params.emissions.tau

                emission_posterior = mvn_posterior_update(self.emissions_prior, emission_stats)
                _emissions_weights = emission_posterior.mode()
                H = _emissions_weights.reshape(self.emission_dim, self.state_dim)
            else:
                d = _params.emissions.bias
                D = _params.emissions.input_weights

                emission_stats_1, emission_stats2 = emission_stats
                velocity_smoother = self.velocity_smoother(base_subspace, _params,
                                                           emission_stats2, masks, conditions,
                                                           tau_idx=tau_idx,
                                                           covs=emission_stats_1,
                                                           filtering_method=filtering_method)
                marginal_ll = velocity_smoother.marginal_loglik
                Ev = velocity_smoother.smoothed_means
                Ev0 = velocity_smoother.smoothed_means[0]
                Ev0v0T = velocity_smoother.smoothed_covariances[0] + jnp.outer(Ev0, Ev0)
                H = vmap(rotate_subspace, in_axes=(None, None, 0))(base_subspace, self.state_dim, Ev)

                init_velocity_stats = (Ev0, Ev0v0T, 1)

                # Perform MAP estimation jointly
                initial_velocity_posterior = niw_posterior_update(self.initial_velocity_prior, init_velocity_stats)
                initial_velocity_cov, initial_velocity_mean = initial_velocity_posterior.mode()

                if self.fix_tau:  # set to true during test time
                    tau = _params.emissions.tau
                else:
                    tau_stats_1 = tau_idx.sum(-1) / 2 #jnp.ones(self.dof) * (self.num_trials - 1) / 2

                    Vv = velocity_smoother.smoothed_covariances
                    Vvpvn_sum = jnp.einsum('kb,bij->kij', tau_idx[:, 1:], velocity_smoother.smoothed_cross_covariances)
                    tau_stats_2 = jnp.einsum('kb,bi,bj->kij', tau_idx[:, 1:], Ev[1:], Ev[1:]) + jnp.einsum('kb,bij->kij', tau_idx[:, 1:], Vv[1:])
                    tau_stats_2 -= (Vvpvn_sum + jnp.swapaxes(Vvpvn_sum, -2, -1))
                    tau_stats_2 += jnp.einsum('kb,bi,bj->kij', tau_idx[:, :-1], Ev[:-1], Ev[:-1]) + jnp.einsum('kb,bij->kij', tau_idx[:, :-1], Vv[:-1])
                    tau_stats_2 = vmap(jnp.diag)(tau_stats_2) / 2
                    def update_tau(s1, s2):
                        def _update_tau(s2i):
                            tau_posterior = ig_posterior_update(self.tau_prior, (s1, s2i))
                            tau_mode_i = tau_posterior.mode()
                            return tau_mode_i
                        tau_mode = vmap(_update_tau)(s2)
                        return tau_mode
                    tau = vmap(update_tau)(tau_stats_1, tau_stats_2)

            if self.fix_emissions_cov:
                R = _params.emissions.cov
            else:
                emissions_cov_stats_1 = jnp.sum(masks) / 2
                Ex = states_smoother.smoothed_means * masks_a
                Vx = states_smoother.smoothed_covariances * masks_aa
                Ey = jnp.einsum('...tx,...yx->...ty', Ex, H)
                emissions_cov_stats_2 = jnp.sum(jnp.square(emissions - Ey) * masks_a, axis=(0, 1))
                emissions_cov_stats_2 += jnp.diag(jnp.einsum('...ix,...txz,...jz->ij', H, Vx, H))
                emissions_cov_stats_2 = emissions_cov_stats_2 / 2

                def update_emissions_cov(s1, s2):
                    emissions_cov_posterior = ig_posterior_update(self.emissions_covariance_prior, (s1, s2))
                    emissions_cov = emissions_cov_posterior.mode()
                    return emissions_cov

                R = jnp.diag(vmap(update_emissions_cov, in_axes=(None, 0))(emissions_cov_stats_1,
                                                                           emissions_cov_stats_2))

            params = ParamsTVLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                               tau=tau),
                initial_velocity=ParamsLGSSMInitial(mean=initial_velocity_mean,
                                                    cov=initial_velocity_cov),
            )

            return params, Ev, marginal_ll

        @jit
        def em(params):
            if self.stationary_emissions:
                stats, marginal_ll, states_smoother = e_step(params)
                Ex = states_smoother.smoothed_means * masks_a
                new_params, Ev, ekf_marginal_ll = m_step(params, stats, states_smoother)
            else:
                stats, marginal_ll, states_smoother = e_step(params)
                Ex = states_smoother.smoothed_means * masks_a
                new_params, Ev, ekf_marginal_ll = m_step(params, stats, states_smoother)

            return new_params, Ex, Ev, marginal_ll, ekf_marginal_ll

        # sample_of_params = []
        sample_of_states = []
        sample_of_velocity = []
        marginal_lls = []
        ekf_marginal_lls = []
        current_params = initial_params

        for _ in progress_bar(range(num_iters)):
            current_params, current_states, current_velocity, marginal_ll, ekf_marginal_ll = em(current_params)
            # sample_of_params.append(current_params)
            sample_of_velocity.append(current_velocity)
            if return_states:
                sample_of_states.append(current_states)
            if print_ll:
                print(marginal_ll)#, ekf_marginal_ll)
            marginal_lls.append(marginal_ll)
            ekf_marginal_lls.append(ekf_marginal_ll)

        return current_params, sample_of_states, sample_of_velocity, marginal_lls, ekf_marginal_lls