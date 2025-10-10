from fastprogress.fastprogress import progress_bar
from functools import partial
import jax
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
from typing import NamedTuple

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

# class ParamsTVLGSSMEmissions(NamedTuple):
#     r"""Parameters of the emission distribution

#     $$p(y_t \mid z_t, u_t) = \mathcal{N}(y_t \mid H z_t + D u_t + d, R)$$

#     The tuple doubles as a container for the ParameterProperties.

#     :param weights: emission weights $H$
#     :param bias: emission bias $d$
#     :param input_weights: emission input weights $D$
#     :param cov: emission covariance $R$

#     """
#     weights: Union[ParameterProperties,
#     Float[Array, "emission_dim state_dim"],
#     Float[Array, "ntime emission_dim state_dim"]]

#     bias: Union[ParameterProperties,
#     Float[Array, "emission_dim"],
#     Float[Array, "ntime emission_dim"]]

#     input_weights: Union[ParameterProperties,
#     Float[Array, "emission_dim input_dim"],
#     Float[Array, "ntime emission_dim input_dim"]]

#     cov: Union[ParameterProperties,
#     Float[Array, "emission_dim emission_dim"],
#     Float[Array, "ntime emission_dim emission_dim"],
#     Float[Array, "emission_dim"],
#     Float[Array, "ntime emission_dim"],
#     Float[Array, "emission_dim_triu"]]

#     tau: Any

#     initial_emission_mean: Union[Float[Array, "emission_dim"], ParameterProperties]
#     initial_emission_covariance: Union[Float[Array, "emission_dim emission_dim"], Float[Array, "emission_dim_triu"], ParameterProperties]

# class ParamsTVLGSSM(NamedTuple):
#     r"""Parameters of a linear Gaussian SSM.

#     :param initial: initial distribution parameters
#     :param dynamics: dynamics distribution parameters
#     :param emissions: emission distribution parameters

#     """
#     initial: ParamsLGSSMInitial
#     dynamics: ParamsLGSSMDynamics
#     emissions: ParamsTVLGSSMEmissions

# class TimeVaryingLinearGaussianConjugateSSM(LinearGaussianSSM):
#     r"""
#     Linear Gaussian State Space Model with conjugate priors for the model parameters.

#     The parameters are the same as LG-SSM. The priors are as follows:

#     * p(m, S) = NIW(loc, mean_concentration, df, scale) # normal inverse wishart
#     * p([F, B, b], Q) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart
#     * p([H, D, d], R) = MNIW(loc, col_precision, df, scale) # matrix normal inverse wishart

#     :param state_dim: Dimensionality of latent state.
#     :param emission_dim: Dimensionality of observation vector.
#     :param input_dim: Dimensionality of input vector. Defaults to 0.
#     :param has_dynamics_bias: Whether model contains an offset term b. Defaults to True.
#     :param has_emissions_bias:  Whether model contains an offset term d. Defaults to True.

#     """
#     def __init__(self,
#                  state_dim,
#                  emission_dim,
#                  input_dim=0,
#                  num_trials: int = 1,
#                  num_conditions: int = 1,
#                  has_dynamics_bias=True,
#                  has_emissions_bias=False,
#                  **kw_priors):
#         super().__init__(state_dim=state_dim, emission_dim=emission_dim,
#                          input_dim=input_dim, num_conditions=num_conditions,
#                          has_dynamics_bias=has_dynamics_bias, has_emissions_bias=has_emissions_bias)

#         # Initialize prior distributions
#         def default_prior(arg, default):
#             return kw_priors[arg] if arg in kw_priors else default

#         self.emission_dof = emission_dim * state_dim

#         self.initial_prior = default_prior(
#             'initial_prior',
#             NIW(loc=jnp.zeros(self.state_dim),
#                 mean_concentration=1.,
#                 df=self.state_dim + 0.1,
#                 scale=jnp.eye(self.state_dim)))

#         self.dynamics_prior = default_prior(
#             'dynamics_prior',
#             MNIW(loc=jnp.zeros((self.state_dim, self.state_dim + self.input_dim + self.has_dynamics_bias)),
#                  col_precision=jnp.eye(self.state_dim + self.input_dim + self.has_dynamics_bias),
#                  df=self.state_dim + 0.1,
#                  scale=jnp.eye(self.state_dim)))

#         self.initial_emission_mean_prior = default_prior(
#                 'initial_emission_mean_prior',
#                 MVN(loc=jnp.zeros(self.emission_dof),
#                     covariance_matrix=1e4*jnp.eye(self.emission_dof))
#             )

#         self.initial_emission_covariance_prior = default_prior(
#             'initial_emission_covariance_prior',
#             IG(concentration=1e-4, scale=1e-4)
#         )

#         self.tau_prior = default_prior(
#             'tau_prior',
#             IG(concentration=1e-6, scale=1e-6)
#         )

#         self.emission_covariance_prior = default_prior(
#             'emission_covariance_prior',
#             IG(concentration=1.0, scale=1.0)
#         )


#     @property
#     def emission_shape(self):
#         return (self.emission_dim,)

#     @property
#     def covariates_shape(self):
#         return dict(inputs=(self.input_dim,)) if self.input_dim > 0 else dict()

#     def initialize(
#         self,
#         tau,
#         key: PRNGKey =jr.PRNGKey(0),
#         initial_mean: Optional[Float[Array, "state_dim"]]=None,
#         initial_covariance=None,
#         dynamics_weights=None,
#         dynamics_bias=None,
#         dynamics_input_weights=None,
#         dynamics_covariance=None,
#         initial_emission_mean=None,
#         initial_emission_covariance=None,
#         emission_weights=None,
#         emission_bias=None,
#         emission_input_weights=None,
#         emission_covariance=None
#     ) -> Tuple[ParamsLGSSM, ParamsLGSSM]:
#         r"""Initialize model parameters that are set to None, and their corresponding properties.

#         Args:
#             key: Random number key. Defaults to jr.PRNGKey(0).
#             initial_mean: parameter $m$. Defaults to None.
#             initial_covariance: parameter $S$. Defaults to None.
#             dynamics_weights: parameter $F$. Defaults to None.
#             dynamics_bias: parameter $b$. Defaults to None.
#             dynamics_input_weights: parameter $B$. Defaults to None.
#             dynamics_covariance: parameter $Q$. Defaults to None.
#             emission_weights: parameter $H$. Defaults to None.
#             emission_bias: parameter $d$. Defaults to None.
#             emission_input_weights: parameter $D$. Defaults to None.
#             emission_covariance: parameter $R$. Defaults to None.

#         Returns:
#             Tuple[ParamsLGSSM, ParamsLGSSM]: parameters and their properties.
#         """

#         # Arbitrary default values, for demo purposes.
#         _initial_mean = jnp.zeros((self.num_conditions, self.state_dim))
#         _initial_covariance = 0.1 * jnp.repeat(jnp.eye(self.state_dim)[jnp.newaxis], self.num_conditions, axis=0)
#         _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
#         _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
#         _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
#         _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)

#         _initial_emission_mean = jnp.zeros(self.emission_dof)
#         _initial_emission_covariance = jnp.eye(self.emission_dof)
#         _emission_weights = jr.normal(key, (self.num_trials, self.emission_dim, self.state_dim))
#         _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
#         _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
#         _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

#         # Only use the values above if the user hasn't specified their own
#         default = lambda x, x0: x if x is not None else x0

#         # Create nested dictionary of params
#         params = ParamsTVLGSSM(
#             initial=ParamsLGSSMInitial(
#                 mean=default(initial_mean, _initial_mean),
#                 cov=default(initial_covariance, _initial_covariance)),
#             dynamics=ParamsLGSSMDynamics(
#                 weights=default(dynamics_weights, _dynamics_weights),
#                 bias=default(dynamics_bias, _dynamics_bias),
#                 input_weights=default(dynamics_input_weights, _dynamics_input_weights),
#                 cov=default(dynamics_covariance, _dynamics_covariance)),
#             emissions=ParamsTVLGSSMEmissions(
#                 weights=default(emission_weights, _emission_weights),
#                 bias=default(emission_bias, _emission_bias),
#                 input_weights=default(emission_input_weights, _emission_input_weights),
#                 cov=default(emission_covariance, _emission_covariance),
#                 tau=tau,
#                 initial_emission_mean=default(initial_emission_mean, _initial_emission_mean),
#                 initial_emission_covariance=default(initial_emission_covariance, _initial_emission_covariance))
#             )

#         # The keys of param_props must match those of params!
#         props = ParamsTVLGSSM(
#             initial=ParamsLGSSMInitial(
#                 mean=ParameterProperties(),
#                 cov=ParameterProperties(constrainer=RealToPSDBijector())),
#             dynamics=ParamsLGSSMDynamics(
#                 weights=ParameterProperties(),
#                 bias=ParameterProperties(),
#                 input_weights=ParameterProperties(),
#                 cov=ParameterProperties(constrainer=RealToPSDBijector())),
#             emissions=ParamsTVLGSSMEmissions(
#                 weights=ParameterProperties(),
#                 bias=ParameterProperties(),
#                 input_weights=ParameterProperties(),
#                 cov=ParameterProperties(constrainer=RealToPSDBijector()),
#                 tau=ParameterProperties(),
#                 initial_emission_mean=ParameterProperties(),
#                 initial_emission_covariance=ParameterProperties(constrainer=RealToPSDBijector()))
#             )
#         return params, props

#     def log_prior(
#         self,
#         params: ParamsLGSSM
#     ) -> Scalar:
#         lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean)).sum()

#         # dynamics
#         dynamics_bias = params.dynamics.bias if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
#         dynamics_matrix = jnp.column_stack((params.dynamics.weights,
#                                             params.dynamics.input_weights,
#                                             dynamics_bias))
#         lp += self.dynamics_prior.log_prob((params.dynamics.cov, dynamics_matrix))

#         lp += self.initial_emission_mean_prior.log_prob(params.emissions.initial_emission_mean)
#         lp += self.initial_emission_covariance_prior.log_prob(jnp.diag(params.emissions.initial_emission_covariance)).sum()
#         lp += self.tau_prior.log_prob(params.emissions.tau).sum()
#         lp += self.emission_covariance_prior.log_prob(jnp.diag(params.emissions.cov)).sum()

#         return lp

#     def initialize_m_step_state(
#         self,
#         params: ParamsTVLGSSM,
#         props: ParamsTVLGSSM
#     ) -> Any:
#         return None

#     def e_step(
#         self,
#         params: ParamsTVLGSSM,
#         emissions: Union[Float[Array, "num_timesteps emission_dim"],
#                          Float[Array, "num_batches num_timesteps emission_dim"]],
#         inputs: Optional[Union[Float[Array, "num_timesteps input_dim"],
#                                Float[Array, "num_batches num_timesteps input_dim"]]]=None,
#         condition: int=0,
#         trial_mask: bool=True,
#         trial_id: int=0,
#         H=None,
#     ) -> Tuple[SuffStatsLGSSM, Scalar]:
#         num_timesteps = emissions.shape[0]
#         if inputs is None:
#             inputs = jnp.zeros((num_timesteps, 0))

#         # Run the smoother to get posterior expectations
#         posterior = lgssm_smoother(params, emissions, inputs, condition, trial_id)

#         # shorthand
#         Ex = posterior.smoothed_means
#         Exp = posterior.smoothed_means[:-1]
#         Exn = posterior.smoothed_means[1:]
#         Vx = posterior.smoothed_covariances
#         Vxp = posterior.smoothed_covariances[:-1]
#         Vxn = posterior.smoothed_covariances[1:]
#         Expxn = posterior.smoothed_cross_covariances

#         # Append bias to the inputs
#         inputs = jnp.concatenate((inputs, jnp.ones((num_timesteps, 1))), axis=1)
#         up = inputs[:-1]
#         u = inputs
#         y = emissions

#         # expected sufficient statistics for the initial tfd.Distribution
#         c = trial_mask * jnn.one_hot(condition, self.num_conditions)
#         Ex0 = jnp.einsum('c,j->cj', c, posterior.smoothed_means[0])
#         Ex0x0T = jnp.einsum('c,jk->cjk', c, posterior.smoothed_covariances[0]
#                             + jnp.outer(posterior.smoothed_means[0], posterior.smoothed_means[0]))
#         init_stats = (Ex0, Ex0x0T, c)

#         # expected sufficient statistics for the dynamics tfd.Distribution
#         # let zp[t] = [x[t], u[t]] for t = 0...T-2
#         # let xn[t] = x[t+1]          for t = 0...T-2
#         sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
#         sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
#         sum_zpzpT = trial_mask * sum_zpzpT

#         sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
#         sum_zpxnT = trial_mask * sum_zpxnT

#         sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
#         sum_xnxnT = trial_mask * sum_xnxnT

#         dynamics_counts = trial_mask * (num_timesteps - 1)
#         dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, dynamics_counts)
#         if not self.has_dynamics_bias:
#             dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT, dynamics_counts)

#         # more expected sufficient statistics for the emissions
#         # let z[t] = [x[t], u[t]] for t = 0...T-1
#         # Rinv = jnp.linalg.inv(h_params.emissions.cov)
#         # Assumes that Rinv is diagonal
#         emissions_stats_1 = jnp.einsum('ti,tj->ij', Ex, Ex)
#         emissions_stats_1 += jnp.einsum('tij->ij', Vx)
#         emissions_stats_2 = jnp.einsum('ti,tn->in', Ex, y)
#         emission_stats = (trial_mask * emissions_stats_1, trial_mask * emissions_stats_2)

#         return (init_stats, dynamics_stats, emission_stats), trial_mask * posterior.marginal_loglik, posterior

#     def emission_weights_smoother(self, params, emission_stats_1, emission_stats_2, block_masks):
        
#         # emission_weights_smoother_params = ParamsTVLGSSM(

#         return None

#     def m_step(
#         self,
#         params: ParamsLGSSM,
#         props: ParamsLGSSM,
#         batch_stats: SuffStatsLGSSM,
#         m_step_state: Any,
#         posteriors,
#         emissions,
#         conditions=None,
#         trial_masks=None,
#         velocity_smoother=None,
#         block_ids=None,
#         block_masks=None,
#     ):

#         num_blocks = block_ids.shape[0]

#         # Sum the statistics across all batches
#         stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
#         init_stats, dynamics_stats, emission_stats = stats

#         # Perform MAP estimation jointly
#         def update_initial(s1, s2, s3):
#             initial_posterior = niw_posterior_update(self.initial_prior, (s1, s2, s3))
#             Sc, mc = initial_posterior.mode()
#             return Sc, mc
#         S, m = vmap(update_initial)(*init_stats)

#         dynamics_posterior = mniw_posterior_update(self.dynamics_prior, dynamics_stats)
#         Q, FB = dynamics_posterior.mode()
#         F = FB[:, :self.state_dim]
#         B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
#             else (FB[:, self.state_dim:], jnp.zeros(self.state_dim))

#         o_emission_stats_1, o_emission_stats_2 = emission_stats
#         Rinv_d = 1.0/jnp.diag(params.emissions.cov)
#         emission_stats_1 = jnp.einsum('bij,n->bnij', o_emission_stats_1, Rinv_d)
#         emission_stats_1 = jnp.einsum('bkij,lb->lkij', emission_stats_1, block_ids)
#         emission_stats_1 = jnp.linalg.inv(emission_stats_1)
#         emission_stats_2 = jnp.einsum('bin,n->bni', o_emission_stats_2, Rinv_d)
#         emission_stats_2 = jnp.einsum('...ni,l...->lni', emission_stats_2, block_ids)
#         emission_stats_2 = jnp.einsum('lnji,lni->lnj', emission_stats_1, emission_stats_2)
#         emission_weights_smoother = self.emission_weights_smoother(params, emission_stats_1, emission_stats_2, block_masks)
#         E_emission_weights = emission_weights_smoother.smoothed_means
#         emission_weights = E_emission_weights.reshape((num_blocks, self.emission_dim, self.state_dim))
#         H = jnp.einsum('bij,bk->kij', emission_weights, block_ids)

#         # MAP estimation for the initial emission mean
#         initial_emission_cov = params.emissions.initial_emission_covariance
#         initial_emission_cov_inv = 1/jnp.diag(initial_emission_cov)#jnp.linalg.inv(initial_velocity_cov)
#         E_emission_weights_0 = emission_weights_smoother.smoothed_means[0]
#         init_emission_stats = (jnp.diag(initial_emission_cov_inv), initial_emission_cov_inv * E_emission_weights_0)
#         initial_emission_posterior = mvn_posterior_update(self.initial_emission_prior, init_emission_stats)
#         initial_emission_mean = initial_emission_posterior.mode()

#         # MAP estimation for the initial velocity variance
#         initial_emission_cov_stats_1 = 0.5
#         initial_emission_cov_stats_2 = jnp.diag(emission_weights_smoother.smoothed_covariances_0 + jnp.outer(E_emission_weights_0, E_emission_weights_0))
#         initial_emission_cov_stats_2 -= 2 * initial_emission_mean * E_emission_weights_0
#         initial_emission_cov_stats_2 += initial_emission_mean ** 2
#         def update_initial_emission_cov(s2):
#             initial_emission_cov_posterior = ig_posterior_update(self.initial_emission_covariance_prior, 
#                                                                 (initial_emission_cov_stats_1, s2))
#             initial_emission_cov = initial_emission_cov_posterior.mode()
#             return initial_emission_cov
#         initial_emission_cov = vmap(update_initial_emission_cov)(initial_emission_cov_stats_2)
#         initial_emission_cov = jnp.diag(initial_emission_cov)

#         tau_stats_1 = jnp.ones(self.emission_dof) * (num_blocks - 1) / 2
#         Vvpvn_sum = emission_weights_smoother.smoothed_cross_covariances
#         tau_stats_2 = jnp.einsum('ti,tj->ij', E_emission_weights[1:], E_emission_weights[1:]) + emission_weights_smoother.smoothed_covariances_n
#         tau_stats_2 -= (Vvpvn_sum + Vvpvn_sum.T)
#         tau_stats_2 += jnp.einsum('ti,tj->ij', E_emission_weights[:-1], E_emission_weights[:-1]) + emission_weights_smoother.smoothed_covariances_p
#         tau_stats_2 = jnp.diag(tau_stats_2) / 2
#         def update_tau(s1, s2):
#             tau_posterior = ig_posterior_update(self.tau_prior, (s1, s2))
#             tau_mode = tau_posterior.mode()
#             return tau_mode
#         tau = vmap(update_tau)(tau_stats_1, tau_stats_2)
#         tau = jnp.clip(tau, max=self.max_tau)

#         Ex, Vx = posteriors.smoothed_means, posteriors.smoothed_covariances
#         emission_cov_stats_1 = (Ex.shape[0] * Ex.shape[1]) / 2
#         Ey = jnp.einsum('...tx,...yx->...ty', Ex, H)
#         emission_cov_stats_2 = jnp.sum(jnp.square(emissions - Ey), axis=(0, 1))
#         emission_cov_stats_2 += jnp.diag(jnp.einsum('...ix,...txz,...jz->ij', H, Vx, H))
#         emission_cov_stats_2 = emission_cov_stats_2 / 2

#         def update_emissions_cov(s2):
#             emissions_cov_posterior = ig_posterior_update(self.emission_covariance_prior,
#                                                           (emission_cov_stats_1, s2))
#             emissions_cov = emissions_cov_posterior.mode()
#             return emissions_cov

#         R = jnp.diag(vmap(update_emissions_cov)(emission_cov_stats_2))
#         D = params.emissions.input_weights
#         d = params.emissions.bias

#         params = ParamsTVLGSSM(
#             initial=ParamsLGSSMInitial(mean=m, cov=S),
#             dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
#             emissions=ParamsTVLGSSMEmissions(weights=H, bias=d, input_weights=D, cov=R,
#                                              initial_emission_mean=initial_emission_mean,
#                                              initial_emission_covariance=initial_emission_cov,
#                                              tau=tau)
#         )
#         return params, m_step_state


class WeightSpaceGaussianProcess():
    r"""
    Weight-space Gaussian Process prior for matrix-valued random functions
        A_ij(u) = \sum_l w^{(ij)} \phi_l(u),       w^{(ij)} ~ N(0, 1)
    where w are the weights and \phi_l are the basis functions.
    
    Constants:
        L: number of basis functions
        D1: input dimension
        D2: output dimension
        M: dimension of u, the "conditions"
    """
    def __init__(self, basis_funcs: list, D1: int=1, D2: int=1):
        self.basis_funcs = basis_funcs
        self.L = len(basis_funcs)
        self.D1 = D1
        self.D2 = D2

    def __call__(self, 
            weights: Float[Array, "L D1 D2"], 
            conditions: Float[Array, "M"]
        ) -> Float[Array, "T D1 D2"]:
        r"""
        Evaluate A_ij(u) = \sum_l w^{(ij)} \phi_l(u) at the M-dimensional points u in `conditions`
        with `weights` w^{(ij)} and basis functions \phi_l.
        """
        PhiX = self.evaluate_basis(conditions)
        return jnp.einsum('lij,l->ij', weights, PhiX)
    
    def sample_weights(self, key: jr.PRNGKey) -> Float[Array, "L D1 D2"]:
        return jr.normal(key, shape=(self.L, self.D1, self.D2))
    
    def evaluate_basis(self, u: Float[Array, "M"]) -> Float[Array, "L"]:
        return jnp.array([f(u) for f in self.basis_funcs]).T

    def sample(self, key: jr.PRNGKey, conditions: Float[Array, "T M"]) -> Float[Array, "T D1 D2"]:
        """
        Sample from the GP prior at the points `conditions`
        """
        weights = self.sample_weights(key)
        # PhiX = self.evaluate_basis(conditions)
        return self.__call__(weights, conditions)
    
    def log_prob(self, conditions: Float[Array, "T M"], fs: Float[Array, "T D1 D2"]) -> Float[Array, "D1 D2"]:
        """
        Compute the log probability of the GP draws at the points `conditions`
        """
        # Check dimensions
        if fs.ndim == 2:
            assert (self.D1 == 1) ^ (self.D2 == 1), 'Incorrect dimensions'
            fs = fs.reshape(-1, self.D1, self.D2)
        assert fs.shape[1] == self.D1 and fs.shape[2] == self.D2, 'Incorrect dimensions'
        
        # Compute log prob
        T = len(fs)
        Phi = self.evaluate_basis(conditions) # T x L
        cov = jnp.dot(Phi, Phi.T)   # T x T
        # return jax.vmap(lambda _f: logprob_analytic(_f, jnp.zeros(T), cov), in_axes=(1))(fs.reshape(T, -1)).reshape(self.D1, self.D2)

        model_dist = tfd.MultivariateNormalFullCovariance(loc=jnp.zeros(T), covariance_matrix=cov)
        return model_dist.log_prob(fs.reshape(T,-1).T).reshape(self.D1, self.D2)

    def log_prob_weights(self, weights: Float[Array, "L D1 D2"]) -> float:
        """
        Standard Gaussian prior N(0,1) on the weights
        """
        return -0.5 * jnp.sum(weights**2) - 0.5 * jnp.asarray(weights.shape).prod() * jnp.log(2*jnp.pi)

class ConditionallyLinearGaussianSSM(SSM):
    r"""
    Conditionally Linear Gaussian State Space Model.

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
        torus_basis_funcs,
        input_dim: int=0,
        num_trials: int = 1,  # number of trials
        num_conditions: int = 1,
        has_dynamics_bias: bool=True,
        has_emissions_bias: bool=False,
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_trials = num_trials
        self.num_conditions = num_conditions
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias

        self.wpgs_C = WeightSpaceGaussianProcess(torus_basis_funcs, D1=emission_dim, D2=state_dim)

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
        _emission_weights = self.wpgs_C.sample_weights(key) #jr.normal(key, (self.num_trials, self.emission_dim, self.state_dim))
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

        _params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=params.initial.mean, cov=params.initial.cov),
            dynamics=ParamsLGSSMDynamics(weights=params.dynamics.weights, bias=params.dynamics.bias, 
                input_weights=params.dynamics.input_weights, cov=params.dynamics.cov),
            emissions=ParamsLGSSMEmissions(weights=self.wpgs_C(params.emissions.weights, trial_id), bias=params.emissions.bias, 
                input_weights=params.emissions.input_weights, cov=params.emissions.cov)
        )

        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(_params, emissions, inputs, condition)

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
        c = trial_mask * jnn.one_hot(condition, self.num_conditions)
        Ex0 = jnp.einsum('c,j->cj', c, posterior.smoothed_means[0])
        Ex0x0T = jnp.einsum('c,jk->cjk', c, posterior.smoothed_covariances[0]
                            + jnp.outer(posterior.smoothed_means[0], posterior.smoothed_means[0]))
        init_stats = (Ex0, Ex0x0T, c)

        # expected sufficient statistics for the dynamics tfd.Distribution
        # let zp[t] = [x[t], u[t]] for t = 0...T-2
        # let xn[t] = x[t+1]          for t = 0...T-2
        sum_zpzpT = jnp.block([[Exp.T @ Exp, Exp.T @ up], [up.T @ Exp, up.T @ up]])
        sum_zpzpT = sum_zpzpT.at[:self.state_dim, :self.state_dim].add(Vxp.sum(0))
        sum_zpzpT = trial_mask * sum_zpzpT

        sum_zpxnT = jnp.block([[Expxn.sum(0)], [up.T @ Exn]])
        sum_zpxnT = trial_mask * sum_zpxnT

        sum_xnxnT = Vxn.sum(0) + Exn.T @ Exn
        sum_xnxnT = trial_mask * sum_xnxnT

        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, trial_mask * (num_timesteps - 1))
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT,
                                trial_mask * (num_timesteps - 1))

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        # Rinv = jnp.linalg.inv(params.emissions.cov)
        # reshape_dim = self.emission_dim * (self.state_dim + self.has_emissions_bias)
        # emissions_stats_1 = jnp.einsum('ti,tl->il', Ex, Ex)
        # emissions_stats_1 += jnp.einsum('tij->ij', Vx)
        # emissions_stats_1 = jnp.einsum('il,jk->jikl', emissions_stats_1, Rinv).reshape(reshape_dim, reshape_dim)
        # emissions_stats_2 = jnp.einsum('ti,tl->il', Ex, y)
        # emissions_stats_2 = jnp.einsum('il,lk->ki', emissions_stats_2, Rinv).reshape(-1)
        # emission_stats = (emissions_stats_1, emissions_stats_2)

        y = emissions
        sum_xxT = Vx.sum(0) + Ex.T @ Ex
        sum_xyT = Ex.T @ y
        sum_yyT = emissions.T @ emissions
        emission_stats = (trial_mask * sum_xxT, trial_mask * sum_xyT, trial_mask * sum_yyT, trial_mask * num_timesteps)

        def weightspace_stats(
                XTX: Float[Array, 'D2 D2'], 
                XTY: Float[Array, 'D2 D1'], 
                wgp_prior: WeightSpaceGaussianProcess,
                ) -> tuple:
            '''
            Compute the expected sufficient statistics for the weight-space GP prior. 
            Provide the sufficient stats X^T X and X^T Y for the problem Y = A(C)X + noise.
            This returns the expanded stats Phi @ X^T X @ Phi^T and Phi @ X^T Y for the basis functions Phi(C).
            '''
            _Phi = wgp_prior.evaluate_basis(trial_id)

            ZTZ = jnp.einsum('k,ij,l->ikjl', _Phi, XTX, _Phi)
            ZTY = jnp.einsum('k,im->ikm', _Phi, XTY)

            ZTZ = ZTZ.reshape(wgp_prior.L * wgp_prior.D2, wgp_prior.L * wgp_prior.D2)
            ZTY = ZTY.reshape(wgp_prior.L * wgp_prior.D2, wgp_prior.D1)
            return (ZTZ, ZTY)

        wgpC_stats = weightspace_stats(sum_xxT, sum_xyT, self.wpgs_C)

        wgpC_sylvester_stats = (trial_mask * wgpC_stats[0], trial_mask * params.emissions.cov, 
                                trial_mask * wgpC_stats[1], trial_mask * num_timesteps)

        return (init_stats, dynamics_stats, emission_stats, wgpC_sylvester_stats), trial_mask * posterior.marginal_loglik, posterior

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

        def fit_gplinear_regression(ZTZ, ZTY, wgp_prior):
            # Solve a linear regression in weight-space given sufficient statistics
            weights = jax.scipy.linalg.solve(
                ZTZ + jnp.eye(wgp_prior.L * wgp_prior.D2), ZTY, 
                assume_a='pos'
                )
            weights = weights.reshape(wgp_prior.D2, wgp_prior.L, wgp_prior.D1).transpose(1,2,0)
            return weights

        # def fit_gplinear_regression_sylvester(ZTZ, Sigma, ZTY, wgp_prior):
        #     # Solve a linear regression in weight-space given sufficient statistics
        #     # weights = utils.jax_solve_sylvester(B, ZTZ, ZTY, assume_a='pos')
        #     weights = utils.jax_solve_sylvester_BS(ZTZ, Sigma, ZTY)
        #     weights = weights.reshape(wgp_prior.D2, wgp_prior.L, wgp_prior.D1).transpose(1,2,0)
        #     return weights

        # Sum the statistics across all batches
        stats = tree_map(partial(jnp.sum, axis=0), batch_stats)
        init_stats, dynamics_stats, emission_stats, wgpC_sylvester_stats = stats

        # Perform MLE estimation jointly
        sum_x0, sum_x0x0T, N = init_stats
        S = (sum_x0x0T - jnp.outer(sum_x0, sum_x0)) / N
        m = sum_x0 / N

        FB, Q = fit_linear_regression(*dynamics_stats)
        F = FB[:, :self.state_dim]
        B, b = (FB[:, self.state_dim:-1], FB[:, -1]) if self.has_dynamics_bias \
            else (FB[:, self.state_dim:], None)

        # HD, R = fit_linear_regression(*emission_stats)
        # H = HD[:, :self.state_dim]
        # D, d = (HD[:, self.state_dim:-1], HD[:, -1]) if self.has_emissions_bias \
        #     else (HD[:, self.state_dim:], None)

        _, R = fit_linear_regression(*emission_stats)
        wgpC_stats = (wgpC_sylvester_stats[0], wgpC_sylvester_stats[2])
        W_C = fit_gplinear_regression(*wgpC_stats, self.wpgs_C)
        # W_C = fit_gplinear_regression_sylvester(
        #         wgpC_sylvester_stats[0], wgpC_sylvester_stats[1] / wgpC_sylvester_stats[3], wgpC_sylvester_stats[2],
        #         wgp_prior=self.wpgs_C
        #         )

        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsLGSSMEmissions(weights=W_C, bias=params.emissions.bias, input_weights=params.emissions.input_weights, cov=R)
        )
        return params, m_step_state