import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jscipy
import tensorflow_probability.substrates.jax as tfp
from jax import vmap

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_filter
from dynamax.nonlinear_gaussian_ssm.inference_smc import (
    always_resample_criterion,
    rb_smc_block_loglik,
    smc,
)
from dynamax.nonlinear_gaussian_ssm.models import StiefelManifoldSSM
from dynamax.utils.utils import rotate_subspace

tfd = tfp.distributions


def build_small_smds(num_blocks=4, block_size=2, num_timesteps=5, key=0):
    state_dim = 2
    emission_dim = 6
    num_conditions = 2
    num_trials = num_blocks * block_size
    model = StiefelManifoldSSM(
        state_dim=state_dim,
        emission_dim=emission_dim,
        num_trials=num_trials,
        num_conditions=num_conditions,
        has_dynamics_bias=True,
        has_emissions_bias=False,
        fix_scale=True,
    )
    drift_dim = model.dof
    base_subspace = jnp.eye(emission_dim)
    block_velocities = jnp.zeros((num_blocks, drift_dim))
    block_velocities = block_velocities.at[:, 0].set(jnp.linspace(0.0, 0.05, num_blocks))
    block_velocities = block_velocities.at[:, 1].set(jnp.linspace(0.0, -0.03, num_blocks))
    trial_velocities = jnp.repeat(block_velocities, block_size, axis=0)
    conditions = jnp.tile(jnp.arange(num_conditions), num_trials)[:num_trials]
    params, _, _ = model.initialize(
        base_subspace=base_subspace,
        tau=jnp.ones((drift_dim,)) * 1e-4,
        key=jr.PRNGKey(key + 1),
        initial_mean=jnp.zeros((num_conditions, state_dim)),
        initial_velocity_mean=jnp.zeros((drift_dim,)),
        initial_velocity_cov=1e-4 * jnp.eye(drift_dim),
        dynamics_weights=0.9 * jnp.eye(state_dim),
        dynamics_covariance=0.05 * jnp.eye(state_dim),
        emission_covariance=0.1 * jnp.eye(emission_dim),
        velocity=trial_velocities,
        scale=jnp.ones((num_trials, state_dim)),
    )
    _, emissions, _ = model.sample(params, jr.PRNGKey(key + 2), num_timesteps, conditions=conditions)
    emissions = emissions.reshape(num_blocks, block_size, num_timesteps, emission_dim)
    conditions = conditions.reshape(num_blocks, block_size)
    return model, params, emissions, conditions, block_velocities


def test_smc_normalizer_matches_gaussian_integral():
    q_std = 1.25
    num_steps = 5
    num_particles = 1500

    def transition_fn(key, state, obs, t):
        del state, obs, t
        q_dist = tfd.Normal(0.0, q_std)
        x = q_dist.sample(seed=key)
        log_q = q_dist.log_prob(x)
        log_p = -jnp.square(x) / 2.0
        return x, log_p - log_q

    result = smc(
        jr.PRNGKey(0),
        jnp.zeros((num_particles,)),
        transition_fn,
        num_steps,
        num_particles,
        resampling_criterion=always_resample_criterion,
    )
    expected = (num_steps / 2.0) * (jnp.log(2.0) + jnp.log(jnp.pi))
    assert jnp.allclose(result.log_Z_hat, expected, atol=8e-2)


def test_rb_smc_all_masked_returns_zero():
    model, params, emissions, conditions, _ = build_small_smds()
    ll = model.marginal_log_prob(
        params,
        emissions,
        conditions=conditions,
        block_masks=jnp.zeros((emissions.shape[0],), dtype=bool),
        trial_masks=jnp.ones(conditions.shape, dtype=bool),
        method="rb_smc",
        key=jr.PRNGKey(1),
        num_particles=32,
    )
    assert jnp.allclose(ll, 0.0)


def test_rb_smc_block_masked_emissions_are_ignored():
    model, params, emissions, conditions, _ = build_small_smds()
    block_masks = jnp.array([True, False, True, False])
    trial_masks = jnp.ones(conditions.shape, dtype=bool)
    changed = jnp.where(block_masks[:, None, None, None], emissions, emissions + 1000.0)
    kwargs = dict(
        conditions=conditions,
        block_masks=block_masks,
        trial_masks=trial_masks,
        method="rb_smc",
        key=jr.PRNGKey(2),
        num_particles=32,
    )
    ll = model.marginal_log_prob(params, emissions, **kwargs)
    changed_ll = model.marginal_log_prob(params, changed, **kwargs)
    assert jnp.allclose(ll, changed_ll)


def test_rb_smc_trial_masked_emissions_are_ignored():
    model, params, emissions, conditions, _ = build_small_smds()
    block_masks = jnp.ones((emissions.shape[0],), dtype=bool)
    trial_masks = jnp.ones(conditions.shape, dtype=bool).at[1, 0].set(False)
    changed = emissions.at[1, 0].add(1000.0)
    kwargs = dict(
        conditions=conditions,
        block_masks=block_masks,
        trial_masks=trial_masks,
        method="rb_smc",
        key=jr.PRNGKey(3),
        num_particles=32,
    )
    ll = model.marginal_log_prob(params, emissions, **kwargs)
    changed_ll = model.marginal_log_prob(params, changed, **kwargs)
    assert jnp.allclose(ll, changed_ll)


def test_fixed_velocity_block_loglik_equals_direct_lgssm_filters():
    model, params, emissions, conditions, block_velocities = build_small_smds()
    block_id = 2
    trial_mask = jnp.array([True, False])
    velocity = block_velocities[block_id]
    helper_ll = rb_smc_block_loglik(
        params,
        params.emissions.base_subspace,
        model.state_dim,
        velocity,
        emissions[block_id],
        conditions[block_id],
        trial_mask,
        block_id=block_id,
        block_size=emissions.shape[1],
    )

    C = rotate_subspace(params.emissions.base_subspace, model.state_dim, velocity)
    lgssm_params = make_lgssm_params(
        params.initial.mean,
        params.initial.cov,
        params.dynamics.weights,
        params.dynamics.cov,
        C,
        params.emissions.cov,
        dynamics_bias=params.dynamics.bias,
        dynamics_input_weights=params.dynamics.input_weights,
        emissions_input_weights=params.emissions.input_weights,
    )

    def direct_trial_ll(y, condition, observed):
        return jax.lax.cond(
            observed,
            lambda args: lgssm_filter(lgssm_params, args[0], condition=args[1]).marginal_loglik,
            lambda args: jnp.array(0.0),
            (y, condition),
        )

    direct_ll = jnp.sum(vmap(direct_trial_ll)(emissions[block_id], conditions[block_id], trial_mask))
    assert jnp.allclose(helper_ll, direct_ll)


def test_cayley_geometry_uses_full_within_and_out_of_manifold_velocity():
    model, _, _, _, _ = build_small_smds()
    assert model.dof == 9
    base_subspace = jnp.eye(model.emission_dim)
    v0 = jnp.zeros((model.dof,))
    v_within = v0.at[0].set(0.2)
    v_out = v0.at[1].set(0.2)
    C0 = rotate_subspace(base_subspace, model.state_dim, v0)
    C_within = rotate_subspace(base_subspace, model.state_dim, v_within)
    C_out = rotate_subspace(base_subspace, model.state_dim, v_out)
    assert not jnp.allclose(C0, C_within)
    assert not jnp.allclose(C0, C_out)
