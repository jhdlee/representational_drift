from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
import tensorflow_probability.substrates.jax as tfp
from jax import lax, vmap
from jaxtyping import Array, Float

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_filter
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered
from dynamax.types import PRNGKey, Scalar
from dynamax.utils.utils import rotate_subspace

tfd = tfp.distributions


class SMCResult(NamedTuple):
    states: Array
    log_weights: Float[Array, "num_steps num_particles"]
    ancestors: Float[Array, "num_steps num_particles"]
    log_Z_hat: Scalar
    resampled: Float[Array, " num_steps"]
    log_increments: Float[Array, "num_steps num_particles"]


def ess_criterion(log_weights: Array, unused_t: int) -> Array:
    """Resample when the effective sample size falls below half the particles."""
    del unused_t
    num_particles = log_weights.shape[0]
    log_ess = 2 * jscipy.special.logsumexp(log_weights) - jscipy.special.logsumexp(2 * log_weights)
    return log_ess <= jnp.log(num_particles / 2.0)


def never_resample_criterion(log_weights: Array, t: int) -> Array:
    del log_weights, t
    return jnp.array(False)


def always_resample_criterion(log_weights: Array, t: int) -> Array:
    del log_weights, t
    return jnp.array(True)


def multinomial_resampling(key: PRNGKey, log_weights: Array, states):
    """Resample a pytree of particle states with multinomial resampling."""
    num_particles = log_weights.shape[0]
    parents = tfd.Categorical(logits=log_weights).sample(sample_shape=(num_particles,), seed=key)
    parents = parents.astype(jnp.int32)
    return jax.tree_util.tree_map(lambda item: item[parents], states), parents


def stratified_resampling(key: PRNGKey, log_weights: Array, states):
    """Resample a pytree of particle states with stratified resampling."""
    num_particles = log_weights.shape[0]
    us = (jnp.arange(num_particles) + jr.uniform(key, shape=(num_particles,))) / num_particles
    normalized_weights = jnp.exp(log_weights - jscipy.special.logsumexp(log_weights))
    bins = jnp.cumsum(normalized_weights)
    parents = jnp.digitize(us, bins).astype(jnp.int32)
    return jax.tree_util.tree_map(lambda item: item[parents], states), parents


def smc(
    key: PRNGKey,
    initial_states,
    transition_fn: Callable,
    num_steps: int,
    num_particles: int,
    observations=None,
    resampling_criterion: Callable = ess_criterion,
    resampling_fn: Callable = multinomial_resampling,
    resample_on_zero_increment: bool = False,
) -> SMCResult:
    """Run SMC and estimate the log normalizer using FIVO-style accumulation.

    The transition function must return ``(new_state, incremental_log_weight)``.
    ``resample_on_zero_increment=False`` is useful for masked observations: skipped
    steps preserve the drift process but do not create artificial resampling events.
    """

    def resample(args):
        subkey, log_weights, states = args
        resampled_states, parents = resampling_fn(subkey, log_weights, states)
        return resampled_states, parents, jnp.zeros_like(log_weights)

    def dont_resample(args):
        _, log_weights, states = args
        return states, jnp.arange(num_particles, dtype=jnp.int32), log_weights

    if observations is None:
        observations = jnp.arange(num_steps)

    def step(carry, step_args):
        key, states, log_weights, log_Z_hat = carry
        key, transition_key, resampling_key = jr.split(key, 3)
        t, observation = step_args

        particle_keys = jr.split(transition_key, num_particles)
        new_states, incr_log_weights = vmap(transition_fn, in_axes=(0, 0, None, None))(
            particle_keys, states, observation, t
        )
        updated_log_weights = log_weights + incr_log_weights

        log_p_hat = jscipy.special.logsumexp(updated_log_weights) - jnp.log(num_particles)
        has_increment = jnp.any(incr_log_weights != 0.0)
        may_resample = jnp.logical_or(resample_on_zero_increment, has_increment)
        should_resample = jnp.logical_and(resampling_criterion(updated_log_weights, t), may_resample)

        resampled_states, parents, next_log_weights = lax.cond(
            should_resample,
            resample,
            dont_resample,
            (resampling_key, updated_log_weights, new_states),
        )
        next_log_Z_hat = log_Z_hat + jnp.where(should_resample, log_p_hat, 0.0)

        return (
            key,
            resampled_states,
            next_log_weights,
            next_log_Z_hat,
        ), (
            new_states,
            updated_log_weights,
            parents,
            should_resample,
            log_p_hat,
            incr_log_weights,
        )

    initial_log_weights = jnp.zeros((num_particles,))
    initial_log_Z_hat = jnp.array(0.0)
    (_, _, final_log_weights, log_Z_hat), outputs = lax.scan(
        step,
        (key, initial_states, initial_log_weights, initial_log_Z_hat),
        (jnp.arange(num_steps), observations),
    )
    states, log_weights, ancestors, resampled, _, log_increments = outputs

    final_log_p_hat = jscipy.special.logsumexp(final_log_weights) - jnp.log(num_particles)
    log_Z_hat = log_Z_hat + jnp.where(resampled[num_steps - 1], 0.0, final_log_p_hat)

    return SMCResult(
        states=states,
        log_weights=log_weights,
        ancestors=ancestors,
        log_Z_hat=log_Z_hat,
        resampled=resampled,
        log_increments=log_increments,
    )


def _select_trial_param(param, absolute_trial_id: int, default):
    if param is None:
        return default
    if getattr(param, "ndim", 0) == 2:
        return param[absolute_trial_id]
    return param


def rb_smc_block_loglik(
    model_params,
    base_subspace: Float[Array, "emission_dim emission_dim"],
    state_dim: int,
    velocity: Float[Array, " drift_dim"],
    block_emissions: Float[Array, "block_size num_timesteps emission_dim"],
    block_conditions: Float[Array, " block_size"],
    trial_mask: Float[Array, " block_size"],
    block_id: int = 0,
    block_size: Optional[int] = None,
) -> Scalar:
    """Compute ``log p(y_block | velocity)`` with neural states marginalized.

    Each observed trial is a condition-specific LGSSM with shared emission matrix
    ``C(velocity)``. Masked trials contribute exactly zero.
    """

    if block_size is None:
        block_size = block_emissions.shape[0]

    C = rotate_subspace(base_subspace, state_dim, velocity)
    R = model_params.emissions.cov
    A = model_params.dynamics.weights
    Q = model_params.dynamics.cov
    dynamics_bias = model_params.dynamics.bias
    dynamics_input_weights = model_params.dynamics.input_weights
    emissions_input_weights = model_params.emissions.input_weights
    emission_bias_default = jnp.zeros((C.shape[0],))
    emission_scale_default = jnp.ones((state_dim,))

    def trial_loglik(trial_emissions, condition, is_observed, trial_id):
        absolute_trial_id = block_id * block_size + trial_id
        emissions_bias = _select_trial_param(model_params.emissions.bias, absolute_trial_id, emission_bias_default)
        emissions_scale = _select_trial_param(
            model_params.emissions.scale, absolute_trial_id, emission_scale_default
        )
        lgssm_params = make_lgssm_params(
            initial_mean=model_params.initial.mean,
            initial_cov=model_params.initial.cov,
            dynamics_weights=A,
            dynamics_cov=Q,
            emissions_weights=C * emissions_scale[jnp.newaxis, :],
            emissions_cov=R,
            dynamics_bias=dynamics_bias,
            dynamics_input_weights=dynamics_input_weights,
            emissions_bias=emissions_bias,
            emissions_input_weights=emissions_input_weights,
        )

        def observed(args):
            y, c = args
            return lgssm_filter(lgssm_params, y, condition=c).marginal_loglik

        def skipped(args):
            del args
            return jnp.array(0.0)

        return lax.cond(is_observed, observed, skipped, (trial_emissions, condition))

    trial_ids = jnp.arange(block_emissions.shape[0])
    trial_lls = vmap(trial_loglik)(block_emissions, block_conditions, trial_mask, trial_ids)
    return jnp.sum(trial_lls)


def rb_smc_marginal_log_prob(
    key: PRNGKey,
    model_params,
    emissions: Float[Array, "num_blocks block_size num_timesteps emission_dim"],
    conditions: Float[Array, "num_blocks block_size"],
    block_masks: Float[Array, " num_blocks"],
    trial_masks: Float[Array, "num_blocks block_size"],
    num_particles: int = 100,
    resampling_criterion: Callable = ess_criterion,
    resampling_fn: Callable = multinomial_resampling,
):
    """Rao-Blackwellized bootstrap SMC for SMDS marginal log likelihood.

    The particle state is the full SMDS velocity vector, including the
    within-manifold component. Neural states are analytically marginalized by
    LGSSM filtering conditional on each particle's velocity.
    """

    num_blocks, block_size = emissions.shape[:2]
    state_dim = model_params.initial.mean.shape[-1]
    drift_dim = model_params.emissions.initial_velocity_mean.shape[-1]
    tau_diag = jnp.asarray(model_params.emissions.tau)
    tau_diag = jnp.broadcast_to(tau_diag, (drift_dim,))
    tau = jnp.diag(tau_diag)
    base_subspace = model_params.emissions.base_subspace

    initial_dist = tfd.MultivariateNormalFullCovariance(
        loc=model_params.emissions.initial_velocity_mean,
        covariance_matrix=model_params.emissions.initial_velocity_cov,
    )
    initial_states = jnp.tile(
        model_params.emissions.initial_velocity_mean[jnp.newaxis],
        (num_particles, 1),
    )

    transition_dist = partial(tfd.MultivariateNormalFullCovariance, covariance_matrix=tau)

    def transition_fn(key, prev_velocity, block_args, block_id):
        block_emissions, block_conditions, block_mask, block_trial_masks = block_args
        velocity = lax.cond(
            block_id == 0,
            lambda _: initial_dist.sample(seed=key),
            lambda _: transition_dist(loc=prev_velocity).sample(seed=key),
            operand=None,
        )
        observed_trial_masks = jnp.logical_and(block_mask, block_trial_masks)
        block_loglik = rb_smc_block_loglik(
            model_params,
            base_subspace,
            state_dim,
            velocity,
            block_emissions,
            block_conditions,
            observed_trial_masks,
            block_id,
            block_size,
        )
        return velocity, block_loglik

    result = smc(
        key,
        initial_states,
        transition_fn,
        num_blocks,
        num_particles,
        observations=(emissions, conditions, block_masks, trial_masks),
        resampling_criterion=resampling_criterion,
        resampling_fn=resampling_fn,
    )

    return PosteriorGSSMFiltered(marginal_loglik=result.log_Z_hat)
