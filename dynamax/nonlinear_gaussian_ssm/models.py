from fastprogress.fastprogress import progress_bar
from functools import partial
import jax
from jax import jit, vmap, lax, jacfwd
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax.tree_util import tree_map
from jaxtyping import Array, Float, PyTree
from typing import NamedTuple, Optional, Union, Callable
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from tensorflow_probability.substrates.jax.distributions import InverseGamma as IG
from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

from dynamax.ssm import SSM
from dynamax.linear_gaussian_ssm.inference import lgssm_filter, lgssm_smoother
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial, ParamsLGSSMDynamics
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

from dynamax.nonlinear_gaussian_ssm.inference_ekf import ParamsNLGSSM
from dynamax.nonlinear_gaussian_ssm.inference_ekf import (extended_kalman_filter, extended_kalman_smoother,
                                                          smc_ekf_proposal_augmented_state,
                                                          extended_kalman_filter_x_marginalized,
                                                          extended_kalman_filter_augmented_state,
                                                          smc_ekf_proposal_x_marginalized,
                                                          extended_kalman_smoother_marginal_log_prob)
from dynamax.nonlinear_gaussian_ssm.inference_ukf import (unscented_kalman_filter, unscented_kalman_smoother,
                                                          unscented_kalman_filter_x_marginalized,
                                                          UKFHyperParams)

from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.distributions import MatrixNormalInverseWishart as MNIW
from dynamax.utils.distributions import NormalInverseWishart as NIW
from dynamax.utils.distributions import (mniw_posterior_update, niw_posterior_update,
                                         mvn_posterior_update, ig_posterior_update)
from dynamax.utils.utils import pytree_stack, psd_solve, symmetrize, rotate_subspace

tfd = tfp.distributions
tfb = tfp.bijectors

class SuffStatsLGSSM(Protocol):
    """A :class:`NamedTuple` with sufficient statistics for LGSSM parameter estimation."""
    pass

class ParamsSMDSEmissions(NamedTuple):
    r"""Parameters of the emission distribution

    $$p(y_t \mid z_t, u_t) = \mathcal{N}(y_t \mid H z_t + D u_t + d, R)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: emission weights $H$
    :param bias: emission bias $d$
    :param input_weights: emission input weights $D$
    :param cov: emission covariance $R$

    """
    weights: Union[ParameterProperties,
    Float[Array, "emission_dim state_dim"],
    Float[Array, "ntime emission_dim state_dim"]]

    bias: Union[ParameterProperties,
    Float[Array, "emission_dim"],
    Float[Array, "ntime emission_dim"]]

    input_weights: Union[ParameterProperties,
    Float[Array, "emission_dim input_dim"],
    Float[Array, "ntime emission_dim input_dim"]]

    cov: Union[ParameterProperties,
    Float[Array, "emission_dim emission_dim"],
    Float[Array, "ntime emission_dim emission_dim"],
    Float[Array, "emission_dim"],
    Float[Array, "ntime emission_dim"],
    Float[Array, "emission_dim_triu"]]

    base_subspace: Any

    tau: Any

    initial_velocity_mean: Union[Float[Array, "state_dim"], ParameterProperties]
    initial_velocity_cov: Union[Float[Array, "state_dim state_dim"], Float[Array, "state_dim_triu"], ParameterProperties]

class ParamsSMDS(NamedTuple):
    r"""Parameters of a linear Gaussian SSM.

    :param initial: initial distribution parameters
    :param dynamics: dynamics distribution parameters
    :param emissions: emission distribution parameters

    """
    initial: ParamsLGSSMInitial
    dynamics: ParamsLGSSMDynamics
    emissions: ParamsSMDSEmissions

class NonlinearGaussianSSM(SSM):
    """
    Nonlinear Gaussian State Space Model.

    The model is defined as follows

    $$p(z_t | z_{t-1}, u_t) = N(z_t | f(z_{t-1}, u_t), Q_t)$$
    $$p(y_t | z_t) = N(y_t | h(z_t, u_t), R_t)$$
    $$p(z_1) = N(z_1 | m, S)$$

    where the model parameters are

    * $z_t$ = hidden variables of size `state_dim`,
    * $y_t$ = observed variables of size `emission_dim`
    * $u_t$ = input covariates of size `input_dim` (defaults to 0).
    * $f$ = dynamics (transition) function
    * $h$ = emission (observation) function
    * $Q$ = covariance matrix of dynamics (system) noise
    * $R$ = covariance matrix for emission (observation) noise
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state


    These parameters of the model are stored in a separate object of type :class:`ParamsNLGSSM`.
    """


    def __init__(self, state_dim: int, emission_dim: int, input_dim: int = 0):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = 0

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    @property
    def inputs_shape(self):
        return (self.input_dim,) if self.input_dim > 0 else None

    def initial_distribution(
        self,
        params: ParamsNLGSSM,
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        return MVN(params.initial_mean, params.initial_covariance)

    def transition_distribution(
        self,
        params: ParamsNLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
    ) -> tfd.Distribution:
        f = params.dynamics_function
        if inputs is None:
            mean = f(state)
        else:
            mean = f(state, inputs)
        return MVN(mean, params.dynamics_covariance)

    def emission_distribution(
        self,
        params: ParamsNLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "input_dim"]] = None
     ) -> tfd.Distribution:
        h = params.emission_function
        if inputs is None:
            mean = h(state)
        else:
            mean = h(state, inputs)
        return MVN(mean, params.emission_covariance)

class StiefelManifoldSSM(SSM):
    r"""
    StiefelManifold State Space Model with conjugate priors for the model parameters.

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
            num_conditions: int = 1,
            has_dynamics_bias: bool = True,
            has_emissions_bias: bool = False,
            tau_per_dim: bool = False,
            tau_per_axis: bool = False,
            max_tau: float = 1e-4,
            fix_initial: bool = False,
            fix_dynamics: bool = False,
            fix_emissions: bool = False,
            fix_emissions_cov: bool = False,
            fix_tau: bool = False,
            fix_initial_velocity: bool = False,
            emissions_cov_eps: float = 0.0,
            velocity_smoother_method: str = 'ekf',
            ekf_mode: str='hybrid',
            ekf_num_iters: int = 1,
            **kw_priors
    ):
        self.state_dim = state_dim
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        self.num_conditions = num_conditions
        self.has_dynamics_bias = has_dynamics_bias
        self.has_emissions_bias = has_emissions_bias
        self.tau_per_dim = tau_per_dim
        self.tau_per_axis = tau_per_axis
        self.max_tau = max_tau

        self.dof = self.state_dim * (self.emission_dim - self.state_dim)
        self.dof_shape = (self.state_dim, (self.emission_dim - self.state_dim))

        self.num_trials = num_trials
        self.num_conditions = num_conditions

        self.fix_initial = fix_initial
        self.fix_dynamics = fix_dynamics
        self.fix_emissions = fix_emissions
        self.fix_emissions_cov = fix_emissions_cov
        self.fix_tau = fix_tau
        self.fix_initial_velocity = fix_initial_velocity
        self.emissions_cov_eps = emissions_cov_eps

        self.velocity_smoother_method = velocity_smoother_method
        self.ekf_mode = ekf_mode
        self.ekf_num_iters = ekf_num_iters
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

        # prior on initial velocity
        self.initial_velocity_prior = default_prior(
            'initial_velocity_prior',
            NIW(loc=jnp.zeros(self.dof),
                mean_concentration=1.,
                df=self.dof + 0.1,
                scale=jnp.eye(self.dof)))

        self.tau_prior = default_prior(
            'tau_prior',
            IG(concentration=1e-9, scale=1e-9)
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
        params: ParamsSMDS
    ) -> Scalar:
        lp = self.initial_prior.log_prob((params.initial.cov, params.initial.mean)).sum()

        # dynamics
        dynamics_bias = params.dynamics.bias if self.has_dynamics_bias else jnp.zeros((self.state_dim, 0))
        dynamics_matrix = jnp.column_stack((params.dynamics.weights,
                                            params.dynamics.input_weights,
                                            dynamics_bias))
        lp += self.dynamics_prior.log_prob((params.dynamics.cov, dynamics_matrix))

        lp += self.initial_velocity_prior.log_prob((params.emissions.initial_velocity_cov,
                                                    params.emissions.initial_velocity_mean))
        if not self.fix_tau:
            if self.tau_per_dim:
                if self.tau_per_axis:
                    tau_lp = self.tau_prior.log_prob(params.emissions.tau.reshape(self.dof_shape)[:, 0])
                else:
                    tau_lp = self.tau_prior.log_prob(params.emissions.tau)
                tau_lp = tau_lp.sum()
            else:
                tau_lp = self.tau_prior.log_prob(params.emissions.tau[0])
            lp += tau_lp
        lp += self.emission_covariance_prior.log_prob(jnp.diag(params.emissions.cov)).sum()

        return lp

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
            velocity=None,
            initial_velocity_mean=None,
            initial_velocity_cov=None,
            emission_weights=None,
            emission_bias=None,
            emission_input_weights=None,
            emission_covariance=None,
    ) -> Tuple[ParamsSMDS, ParamsSMDS]:
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

        _emission_weights = vmap(rotate_subspace, in_axes=(None, None, 0))(base_subspace, self.state_dim, _velocity)
        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # Create nested dictionary of params
        params = ParamsSMDS(
            initial=ParamsLGSSMInitial(
                mean=default(initial_mean, _initial_mean),
                cov=default(initial_covariance, _initial_covariance)),
            dynamics=ParamsLGSSMDynamics(
                weights=default(dynamics_weights, _dynamics_weights),
                bias=default(dynamics_bias, _dynamics_bias),
                input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=ParamsSMDSEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance),
                base_subspace=base_subspace,
                tau=tau,
                initial_velocity_mean=default(initial_velocity_mean, _initial_velocity_mean),
                initial_velocity_cov=default(initial_velocity_cov, _initial_velocity_cov)),
        )

        # The keys of param_props must match those of params!
        props = ParamsSMDS(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            dynamics=ParamsLGSSMDynamics(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            emissions=ParamsSMDSEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()),
                base_subspace=ParameterProperties(),
                tau=ParameterProperties(),
                initial_velocity_mean=ParameterProperties(),
                initial_velocity_cov=ParameterProperties(constrainer=RealToPSDBijector())),
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
            initial_state = self.initial_distribution(t, params, initial_input, conditions).sample(seed=key1)

            # Sample the remaining emissions and states
            next_keys = jr.split(key, num_timesteps - 1)
            next_inputs = tree_map(lambda x: x[1:], inputs)
            next_indices = jnp.arange(1, num_timesteps)
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
                emission_distribution = self.emission_distribution(params, x, inpt, t)
                emission = emission_distribution.sample(seed=key)
                signal = emission_distribution.mean()
                return None, (emission, signal)

            # Sample the remaining emissions and states
            next_keys = jr.split(key, num_timesteps)
            next_states = tree_map(lambda x: x, state)
            next_inputs = tree_map(lambda x: x, inputs)
            next_indices = jnp.arange(num_timesteps)
            _, (next_emissions, next_signals) = lax.scan(_emissions_step, None, (next_keys, next_states,
                                                                                 next_inputs, next_indices))
            return None, (next_emissions, next_signals)

        keys = jr.split(keys[-1], self.num_trials)
        _, (emissions, signals) = lax.scan(_emissions_outer_step, None, (keys, states, jnp.arange(self.num_trials)))

        return states, emissions, signals

    def initial_distribution(
            self,
            timestep: int,
            params: ParamsSMDS,
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            conditions=None,
    ) -> tfd.Distribution:
        return MVN(params.initial.mean[conditions[timestep]],
                   params.initial.cov[conditions[timestep]])

    def transition_distribution(
            self,
            params: ParamsSMDS,
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
            params: ParamsSMDS,
            state: Float[Array, "state_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None,
            timestep: int=0,
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights[timestep] @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias[timestep]
        return MVN(mean, params.emissions.cov)

    def marginal_log_prob(
            self,
            params: ParamsSMDS,
            emissions: Float[Array, "ntime emission_dim"],
            conditions: jnp.array = None,
            block_masks: jnp.array = None,
            method: int = 0,
            num_iters: int = 1,
            num_particles: int = 100,
            key: jr.PRNGKey = jr.PRNGKey(0),
    ) -> Scalar:

        num_blocks = emissions.shape[0]
        if conditions is None:
            conditions = jnp.zeros(emissions.shape[:2], dtype=int)
        if block_masks is None:
            block_masks = jnp.ones(num_blocks, dtype=bool)

        f = self.get_f()
        if method == 0:
            h = self.get_h_x_marginalized(params)
            filtering_function = partial(extended_kalman_filter_x_marginalized, num_iters=num_iters)
        elif method == 1:
            h = self.get_h_augmented(params.emissions.base_subspace)
            filtering_function = partial(extended_kalman_filter_augmented_state, num_iters=num_iters)
        elif method == 2:
            h = self.get_h_augmented(params.emissions.base_subspace)
            filtering_function = partial(smc_ekf_proposal_augmented_state, num_particles=num_particles, key=key)
        elif method == 3:
            h = self.get_h_x_marginalized(params)
            filtering_function = partial(smc_ekf_proposal_x_marginalized, num_particles=num_particles, key=key)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.emissions.initial_velocity_mean,
            initial_covariance=params.emissions.initial_velocity_cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = filtering_function(params=NLGSSM_params, model_params=params, emissions=emissions,
                                                conditions=conditions, block_masks=block_masks)

        return filtered_posterior.marginal_loglik

    def filter(
            self,
            params: ParamsSMDS,
            emissions: Float[Array, "ntime emission_dim"],
            conditions: jnp.array = None,
            trial_masks: jnp.array = None,
            method: int = 0,
    ):
        num_trials = emissions.shape[0]
        if conditions is None:
            conditions = jnp.zeros(num_trials, dtype=int)
        if trial_masks is None:
            trial_masks = jnp.ones(num_trials, dtype=bool)

        f = self.get_f()
        if method == 0:
            h = self.get_h_x_marginalized(params)
            filtering_function = extended_kalman_filter_x_marginalized
        elif method == 1:
            h = self.get_h_augmented(params.emissions.base_subspace)
            filtering_function = extended_kalman_filter_augmented_state

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.emissions.initial_velocity_mean,
            initial_covariance=params.emissions.initial_velocity_cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = filtering_function(NLGSSM_params, params, emissions,
                                                conditions=conditions, trial_masks=trial_masks)

        return filtered_posterior

    def smoother(
            self,
            params: ParamsSMDS,
            emissions: Float[Array, "ntime emission_dim"],
            conditions: jnp.array = None,
            block_masks: jnp.array = None,
            method: int = 0,
    ):
        num_blocks = emissions.shape[0]
        if conditions is None:
            conditions = jnp.zeros(emissions.shape[:2], dtype=int)
        if block_masks is None:
            block_masks = jnp.ones(num_blocks, dtype=bool)

        f = self.get_f()
        if method == 0:
            h = self.get_h_x_marginalized(params)
            filtering_function = partial(extended_kalman_filter_x_marginalized, num_iters=self.ekf_num_iters)
        elif method == 1:
            h = self.get_h_augmented(params.emissions.base_subspace)
            filtering_function = partial(extended_kalman_filter_augmented_state, num_iters=self.ekf_num_iters)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.emissions.initial_velocity_mean,
            initial_covariance=params.emissions.initial_velocity_cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=None
        )

        filtered_posterior = filtering_function(params=NLGSSM_params, model_params=params, emissions=emissions,
                                                conditions=conditions, block_masks=block_masks)
        smoothed_posterior = extended_kalman_smoother(NLGSSM_params, emissions,
                                                      filtered_posterior=filtered_posterior)

        return smoothed_posterior

    def initialize_m_step_state(
            self,
            params: ParamsSMDS,
            props: ParamsSMDS
    ) -> Any:
        return None

    def get_h_x_marginalized(self, params):

        def h(v, obs_t, condition, eps=None):
            C = rotate_subspace(params.emissions.base_subspace, self.state_dim, v)

            # new params constructed from model_params
            mu_0 = params.initial.mean
            Sigma_0 = params.initial.cov
            A = params.dynamics.weights
            b = params.dynamics.bias
            Q = params.dynamics.cov
            R = params.emissions.cov
            mu_v_0 = params.emissions.initial_velocity_mean
            Sigma_v_0 = params.emissions.initial_velocity_cov
            tau = params.emissions.tau
            h_params = ParamsSMDS(
                initial=ParamsLGSSMInitial(
                    mean=mu_0,
                    cov=Sigma_0),
                dynamics=ParamsLGSSMDynamics(
                    weights=A,
                    bias=b,
                    input_weights=jnp.zeros((self.state_dim, 0)),
                    cov=Q),
                emissions=ParamsSMDSEmissions(
                    weights=C,
                    bias=None,
                    input_weights=jnp.zeros((self.emission_dim, 0)),
                    cov=R,
                    base_subspace=params.emissions.base_subspace,
                    tau=tau,
                    initial_velocity_mean=mu_v_0,
                    initial_velocity_cov=Sigma_v_0)
            )

            filtered_posterior = lgssm_filter(h_params, obs_t, condition=condition)

            # get pred means and covs
            pred_means = filtered_posterior.predicted_means
            pred_covs = filtered_posterior.predicted_covariances

            pred_obs_means = jnp.einsum('ij,tj->ti', C, pred_means)
            pred_obs_covs = jnp.einsum('ij,tjk,lk->til', C, pred_covs, C) + R

            if self.velocity_smoother_method == 'ekf':
                return pred_obs_means, (pred_obs_means, pred_obs_covs)
            elif self.velocity_smoother_method == 'ukf':
                pred_obs_covs_sqrt = jnp.linalg.cholesky(pred_obs_covs)
                pred_obs_means += jnp.einsum('til,tl->ti', pred_obs_covs_sqrt,
                                             eps.reshape(-1, self.emission_dim))
                return pred_obs_means.flatten()

        return h

    def get_f(self):
        def f(v):
            return v
        return f

    def get_h(self, base_subspace):
        def h(v):
            C = rotate_subspace(base_subspace, self.state_dim, v)
            return C.flatten()

        return h
    
    def get_h_augmented(self, base_subspace):
        def h_augmented(u):
            x, v = jnp.split(u, [self.state_dim])
            C = rotate_subspace(base_subspace, self.state_dim, v)
            y_pred = C @ x
            return y_pred, y_pred
        return h_augmented

    def velocity_smoother(self, params, covs, emissions, trial_masks):
        f = self.get_f()
        h = self.get_h(params.emissions.base_subspace)

        NLGSSM_params = ParamsNLGSSM(
            initial_mean=params.emissions.initial_velocity_mean,
            initial_covariance=params.emissions.initial_velocity_cov,
            dynamics_function=f,
            dynamics_covariance=jnp.diag(params.emissions.tau),
            emission_function=h,
            emission_covariance=covs
        )

        if self.velocity_smoother_method == 'ekf':
            smoother = extended_kalman_smoother(NLGSSM_params, emissions, trial_masks=trial_masks,
                                                mode=self.ekf_mode, num_iters=self.ekf_num_iters)
        else:
            ukf_hyperparams = UKFHyperParams(alpha=1e-3, beta=2, kappa=0)
            smoother = unscented_kalman_smoother(NLGSSM_params, emissions,
                                                 hyperparams=ukf_hyperparams, trial_masks=trial_masks)

        return smoother

    def e_step(
        self,
        params: ParamsSMDS,
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

        mu_0 = params.initial.mean
        Sigma_0 = params.initial.cov
        A = params.dynamics.weights
        b = params.dynamics.bias
        Q = params.dynamics.cov
        R = params.emissions.cov
        mu_v_0 = params.emissions.initial_velocity_mean
        Sigma_v_0 = params.emissions.initial_velocity_cov
        tau = params.emissions.tau
        h_params = ParamsSMDS(
            initial=ParamsLGSSMInitial(
                mean=mu_0,
                cov=Sigma_0),
            dynamics=ParamsLGSSMDynamics(
                weights=A,
                bias=b,
                input_weights=jnp.zeros((self.state_dim, 0)),
                cov=Q),
            emissions=ParamsSMDSEmissions(
                weights=params.emissions.weights if H is None else H,
                bias=None,
                input_weights=jnp.zeros((self.emission_dim, 0)),
                cov=R,
                base_subspace=params.emissions.base_subspace,
                tau=tau,
                initial_velocity_mean=mu_v_0,
                initial_velocity_cov=Sigma_v_0)
        )

        # Run the smoother to get posterior expectations
        posterior = lgssm_smoother(h_params, emissions, inputs, condition, trial_id)

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

        dynamics_counts = trial_mask * (num_timesteps - 1)
        dynamics_stats = (sum_zpzpT, sum_zpxnT, sum_xnxnT, dynamics_counts)
        if not self.has_dynamics_bias:
            dynamics_stats = (sum_zpzpT[:-1, :-1], sum_zpxnT[:-1, :], sum_xnxnT, dynamics_counts)

        # more expected sufficient statistics for the emissions
        # let z[t] = [x[t], u[t]] for t = 0...T-1
        Rinv = jnp.linalg.inv(h_params.emissions.cov)
        # Assumes that Rinv is diagonal
        Rinv_d = jnp.diag(Rinv)
        emissions_stats_1 = jnp.einsum('ti,tj->ij', Ex, Ex)
        emissions_stats_1 += jnp.einsum('tij->ij', Vx)
        emissions_stats_1 = jnp.einsum('ij,k->kij', emissions_stats_1, Rinv_d)
        emissions_stats_2 = jnp.einsum('ti,tj->ij', Ex, y)
        emissions_stats_2 = jnp.einsum('ij,j->ji', emissions_stats_2, Rinv_d)
        emission_stats = (emissions_stats_1, emissions_stats_2)

        return (init_stats, dynamics_stats, emission_stats), trial_mask * posterior.marginal_loglik, posterior

    def m_step(
        self,
        params: ParamsSMDS,
        props: ParamsSMDS,
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
        
        num_blocks = block_ids.shape[0]

        trial_masks_a = jnp.expand_dims(trial_masks, -1)
        trial_masks_aa = jnp.expand_dims(trial_masks_a, -1)

        # Sum the statistics across all batches
        init_stats_, dynamics_stats_, emission_stats = batch_stats
        stats = tree_map(partial(jnp.sum, axis=0), (init_stats_, dynamics_stats_))
        init_stats, dynamics_stats = stats

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

        # EKF to infer Vs
        emission_stats_1, emission_stats_2 = emission_stats
        if velocity_smoother is None:
            emission_stats_1 = jnp.einsum('bkij,lb->lkij', emission_stats_1, block_ids)
            if self.ekf_mode == 'cov':
                emission_stats_1 = jnp.linalg.inv(emission_stats_1)
            emission_stats_2 = jnp.einsum('bki,lb->lki', emission_stats_2, block_ids)
            if self.ekf_mode == 'cov':
                emission_stats_2 = jnp.einsum('lkij,lkj->lki', emission_stats_1, emission_stats_2)
            velocity_smoother = self.velocity_smoother(params, emission_stats_1, emission_stats_2, block_masks)
        Ev = velocity_smoother.smoothed_means
        H = vmap(rotate_subspace, in_axes=(None, None, 0))(params.emissions.base_subspace, self.state_dim, Ev)
        H = jnp.einsum('bij,bk->kij', H, block_ids)

        Ev0 = velocity_smoother.smoothed_means[0]
        Ev0v0T = velocity_smoother.smoothed_covariances_0 + jnp.outer(Ev0, Ev0)
        init_velocity_stats = (Ev0, Ev0v0T, 1)
        initial_velocity_posterior = niw_posterior_update(self.initial_velocity_prior, init_velocity_stats)
        initial_velocity_cov, initial_velocity_mean = initial_velocity_posterior.mode()

        if self.fix_tau:
            tau = params.emissions.tau
        else:
            if self.tau_per_dim:
                if self.tau_per_axis:
                    tau_stats_1 = jnp.ones(self.state_dim) * ((num_blocks - 1) * (self.emission_dim - self.state_dim)) / 2
                    Vvpvn_sum = velocity_smoother.smoothed_cross_covariances
                    tau_stats_2 = jnp.einsum('ti,tj->ij', Ev[1:], Ev[1:]) + velocity_smoother.smoothed_covariances_n
                    tau_stats_2 -= (Vvpvn_sum + Vvpvn_sum.T)
                    tau_stats_2 += jnp.einsum('ti,tj->ij', Ev[:-1], Ev[:-1]) + velocity_smoother.smoothed_covariances_p
                    tau_stats_2 = jnp.diag(tau_stats_2).reshape(self.dof_shape).sum(1) / 2
                    def update_tau(s1, s2):
                        tau_posterior = ig_posterior_update(self.tau_prior, (s1, s2))
                        tau_mode = tau_posterior.mode()
                        return tau_mode
                    tau = vmap(update_tau)(tau_stats_1, tau_stats_2)
                    tau = jnp.repeat(tau, self.emission_dim - self.state_dim)
                else:
                    tau_stats_1 = jnp.ones(self.dof) * (num_blocks - 1) / 2
                    Vvpvn_sum = velocity_smoother.smoothed_cross_covariances
                    tau_stats_2 = jnp.einsum('ti,tj->ij', Ev[1:], Ev[1:]) + velocity_smoother.smoothed_covariances_n
                    tau_stats_2 -= (Vvpvn_sum + Vvpvn_sum.T)
                    tau_stats_2 += jnp.einsum('ti,tj->ij', Ev[:-1], Ev[:-1]) + velocity_smoother.smoothed_covariances_p
                    tau_stats_2 = jnp.diag(tau_stats_2) / 2
                    def update_tau(s1, s2):
                        tau_posterior = ig_posterior_update(self.tau_prior, (s1, s2))
                        tau_mode = tau_posterior.mode()
                        return tau_mode
                    tau = vmap(update_tau)(tau_stats_1, tau_stats_2)
            else:
                tau_stats_1 = self.dof * (num_blocks - 1) / 2
                Vvpvn_sum = velocity_smoother.smoothed_cross_covariances
                tau_stats_2 = jnp.einsum('ti,tj->ij', Ev[1:], Ev[1:]) + velocity_smoother.smoothed_covariances_n
                tau_stats_2 -= (Vvpvn_sum + Vvpvn_sum.T)
                tau_stats_2 += jnp.einsum('ti,tj->ij', Ev[:-1], Ev[:-1]) + velocity_smoother.smoothed_covariances_p
                tau_stats_2 = jnp.diag(tau_stats_2).sum() / 2
                tau_posterior = ig_posterior_update(self.tau_prior, (tau_stats_1, tau_stats_2))
                tau_mode = tau_posterior.mode()
                tau = jnp.ones(self.dof) * tau_mode
            tau = jnp.clip(tau, max=self.max_tau)

        Ex, Vx = posteriors.smoothed_means, posteriors.smoothed_covariances
        emission_cov_stats_1 = (trial_masks.sum() * Ex.shape[1]) / 2
        Ey = jnp.einsum('...tx,...yx->...ty', Ex, H)
        emission_cov_stats_2 = jnp.sum(jnp.square(emissions - Ey) * trial_masks_aa, axis=(0, 1))
        emission_cov_stats_2 += jnp.diag(jnp.einsum('...,...ix,...txz,...jz->ij', trial_masks, H, Vx, H))
        emission_cov_stats_2 = emission_cov_stats_2 / 2
        def update_emissions_cov(s2):
            emissions_cov_posterior = ig_posterior_update(self.emission_covariance_prior,
                                                          (emission_cov_stats_1, s2))
            emissions_cov = emissions_cov_posterior.mode()
            return emissions_cov
        # R = vmap(update_emissions_cov)(emission_cov_stats_2) + self.emissions_cov_eps
        R = vmap(update_emissions_cov)(emission_cov_stats_2)
        R = jnp.clip(R, min=self.emissions_cov_eps)
        R = jnp.diag(R)

        # H, R = params.emissions.weights, params.emissions.cov
        # tau = params.emissions.tau
        # initial_velocity_mean, initial_velocity_cov = params.emissions.initial_velocity_mean, params.emissions.initial_velocity_cov
        D = params.emissions.input_weights
        d = params.emissions.bias

        params = ParamsSMDS(
            initial=ParamsLGSSMInitial(mean=m, cov=S),
            dynamics=ParamsLGSSMDynamics(weights=F, bias=b, input_weights=B, cov=Q),
            emissions=ParamsSMDSEmissions(weights=H, bias=d, input_weights=D, cov=R,
                                          base_subspace=params.emissions.base_subspace,
                                          tau=tau,
                                          initial_velocity_mean=initial_velocity_mean,
                                          initial_velocity_cov=initial_velocity_cov)
        )
        return params, m_step_state