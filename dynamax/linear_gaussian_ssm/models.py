from fastprogress.fastprogress import progress_bar
from functools import partial
from jax import jit, vmap
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow_probability.substrates.jax.distributions import InverseGamma as IG
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

from dynamax.ssm import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_joint_sample, lgssm_filter, lgssm_smoother, lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import (mniw_posterior_update, niw_posterior_update,
                                         mvn_posterior_update, ig_posterior_update)
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
        input_dim: int=0,
        num_conditions: int = 1,
        has_dynamics_bias: bool=True,
        has_emissions_bias: bool=False
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_conditions = num_conditions
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
        _initial_mean = jnp.zeros((self.num_conditions, self.state_dim))
        _initial_covariance = 0.1 * jnp.repeat(jnp.eye(self.state_dim)[jnp.newaxis], self.num_conditions, axis=0)
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
                cov=default(emission_covariance, _emission_covariance))
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
                cov=ParameterProperties(constrainer=RealToPSDBijector()))
            )
        return params, props

    def initial_distribution(
        self,
        params: ParamsLGSSM,
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
        condition: int=0,
    ) -> tfd.Distribution:
        return MVN(params.initial.mean[condition], params.initial.cov[condition])

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
    
    def sample(
        self,
        params: ParamsLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        condition: int = 0,
    ) -> PosteriorGSSMFiltered:
        return lgssm_joint_sample(params, key, num_timesteps, inputs, condition)

    def batch_sample(
        self,
        params: ParamsLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        conditions = None,
    ) -> PosteriorGSSMFiltered:
        keys = jr.split(key, len(conditions))
        sample_vmap = vmap(self.sample, in_axes=(None, 0, None, None, 0))
        return sample_vmap(params, keys, num_timesteps, inputs, conditions)

    def marginal_log_prob(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        condition: int=0,
    ) -> Scalar:
        filtered_posterior = lgssm_filter(params, emissions, inputs, condition)
        return filtered_posterior.marginal_loglik

    def batch_marginal_log_prob(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        conditions = None,
    ) -> Scalar:
        marginal_log_prob_vmap = vmap(self.marginal_log_prob, in_axes=(None, 0, None, 0))
        return marginal_log_prob_vmap(params, emissions, inputs, conditions).sum()

    def filter(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        condition: int=0,
    ) -> PosteriorGSSMFiltered:
        return lgssm_filter(params, emissions, inputs, condition)

    def batch_filter(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        conditions = None,
    ) -> PosteriorGSSMFiltered:
        lgssm_filter_vmap = vmap(self.filter, in_axes=(None, 0, None, 0))
        return lgssm_filter_vmap(params, emissions, inputs, conditions)

    def smoother(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        condition: int=0,
    ) -> PosteriorGSSMSmoothed:
        return lgssm_smoother(params, emissions, inputs, condition)

    def batch_smoother(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None,
        conditions = None,
    ) -> PosteriorGSSMSmoothed:
        lgssm_smoother_vmap = vmap(self.smoother, in_axes=(None, 0, None, 0))
        return lgssm_smoother_vmap(params, emissions, inputs, conditions)

    # need update
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
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
        condition: int = 0,
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        posterior = lgssm_smoother(params, emissions, inputs, condition)
        H = params.emissions.weights
        d = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + d
        smoothed_emissions_cov = H @ posterior.smoothed_covariances @ H.T + R
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    def batch_posterior_predictive(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None,
        conditions = None,
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        posterior_predictive_vmap = vmap(self.posterior_predictive, in_axes=(None, 0, None, 0))
        return posterior_predictive_vmap(params, emissions, inputs, conditions)

    # Expectation-maximization (EM) code
    def e_step(
        self,
        params: ParamsLGSSM,
        emissions: Union[Float[Array, "num_timesteps emission_dim"],
                         Float[Array, "num_batches num_timesteps emission_dim"]],
        inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
                               Float[Array, "num_batches num_timesteps input_dim"]]]=None,
        condition: int=0,
        trial_mask: bool=True,
        trial_id: int=0,
        H=None,
    ) -> Tuple[SuffStatsLGSSM, Scalar]:
        num_timesteps = emissions.shape[0]
        if inputs is None:
            inputs = jnp.zeros((num_timesteps, 0))

        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(params, emissions, inputs, condition)

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
        c = jnn.one_hot(condition, self.num_conditions)
        Ex0 = jnp.einsum('c,j->cj', c, posterior.smoothed_means[0])
        Ex0x0T = jnp.einsum('c,jk->cjk', c, posterior.smoothed_covariances[0]
                            + jnp.outer(posterior.smoothed_means[0], posterior.smoothed_means[0]))
        init_stats = (Ex0, Ex0x0T, c)

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
        Rinv = jnp.linalg.inv(params.emissions.cov)
        reshape_dim = self.emission_dim * (self.state_dim + self.has_emissions_bias)
        emissions_stats_1 = jnp.einsum('ti,tl->il', Ex, Ex)
        emissions_stats_1 += jnp.einsum('tij->ij', Vx)
        emissions_stats_1 = jnp.einsum('il,jk->jikl', emissions_stats_1, Rinv).reshape(reshape_dim, reshape_dim)
        emissions_stats_2 = jnp.einsum('ti,tl->il', Ex, y)
        emissions_stats_2 = jnp.einsum('il,lk->ki', emissions_stats_2, Rinv).reshape(-1)
        emission_stats = (emissions_stats_1, emissions_stats_2)

        return (init_stats, dynamics_stats, emission_stats), posterior.marginal_loglik, posterior

    def initialize_m_step_state(
            self,
            params: ParamsLGSSM,
            props: ParamsLGSSM
    ) -> Any:
        return None

    # need update
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
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R)
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
                 num_conditions: int = 1,
                 has_dynamics_bias=True,
                 has_emissions_bias=False,
                 **kw_priors):
        super().__init__(state_dim=state_dim, emission_dim=emission_dim,
                         input_dim=input_dim, num_conditions=num_conditions,
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
                MVN(loc=jnp.zeros(self.emission_dim * (self.state_dim + self.has_emissions_bias)),
                    covariance_matrix=jnp.eye(self.emission_dim * (self.state_dim + self.has_emissions_bias)))
            )

        self.emission_covariance_prior = default_prior(
            'emission_covariance_prior',
            IG(concentration=1.0, scale=1.0)
        )


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
        lp += self.emission_prior.log_prob(emission_matrix.flatten())
        lp += self.emission_covariance_prior.log_prob(jnp.diag(params.emissions.cov)).sum()

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
        m_step_state: Any,
        posteriors,
        emissions,
        conditions=None,
        trial_masks=None,
        session_masks=None,
        velocity_smoother=None,
        block_ids=None,
        block_masks=None,
    ):
        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats = stats

        # Perform MAP estimation jointly
        def update_initial(s1, s2, s3):
            initial_posterior = niw_posterior_update(self.initial_prior, (s1, s2, s3))
            Sc, mc = initial_posterior.mode()
            return Sc, mc
        S, m = vmap(update_initial)(*init_stats)

        dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
        Q, FB = dynamics_posterior.mode()
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

        emission_posterior = mvn_posterior_update(self.emission_prior, emission_stats)
        emission_weights = emission_posterior.mode()
        H = emission_weights.reshape(self.emission_dim, self.state_dim)

        Ex, Vx = posteriors.smoothed_means, posteriors.smoothed_covariances
        emission_cov_stats_1 = (Ex.shape[0] * Ex.shape[1]) / 2
        Ey = jnp.einsum('...tx,...yx->...ty', Ex, H)
        emission_cov_stats_2 = jnp.sum(jnp.square(emissions - Ey), axis=(0, 1))
        emission_cov_stats_2 += jnp.diag(jnp.einsum('...ix,...txz,...jz->ij', H, Vx, H))
        emission_cov_stats_2 = emission_cov_stats_2 / 2

        def update_emissions_cov(s2):
            emissions_cov_posterior = ig_posterior_update(self.emission_covariance_prior,
                                                          (emission_cov_stats_1, s2))
            emissions_cov = emissions_cov_posterior.mode()
            return emissions_cov

        R = jnp.diag(vmap(update_emissions_cov)(emission_cov_stats_2))
        D = params.emissions.input_weights
        d = params.emissions.bias

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R)
        )
        return params, m_step_state

    # need update
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
                else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

            # Sample the emission params
            emission_posterior = mniw_posterior_update(self.emission_prior, emission_stats)
            R, HD = emission_posterior.sample(seed=next(rngs))
            H = HD[:, :self.state_dim]
            D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
                else (HD[:, self.state_dim:], jnp.zeros(self.emission_dim))

            params = ParamsLGSSM(
                initial=ParamsLGSSMInitial(mean=m, cov=S),
                dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
                emissions=ParamsLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R)
            )
            return params

        @jit
        def one_sample(_params, rng):
            rngs = jr.split(rng, 2)
            # Sample latent states
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

        return pytree_stack(sample_of_params)
