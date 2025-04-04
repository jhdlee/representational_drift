import jax.numpy as jnp
from sklearn.metrics import r2_score
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother
from jax import vmap
from typing import Dict, Tuple, Optional, Any, Union, List
from dynamax.utils.wandb_utils import log_evaluation_metrics
from dynamax.utils.utils import rotate_subspace
def compute_lds_test_marginal_ll(test_model, test_params, test_obs, test_conditions):
    return test_model.batch_marginal_log_prob(test_params, test_obs, conditions=test_conditions)

def compute_lds_test_r2(test_model, test_params, test_obs, test_conditions):
    C = test_params.emissions.weights
    smoother = test_model.batch_smoother(test_params, test_obs, conditions=test_conditions)
    smoothed_states = smoother.smoothed_means

    prediction = jnp.einsum('bti,ji->btj', smoothed_states, C)
    return r2_score(test_obs.flatten(), prediction.flatten())

def compute_lds_test_cosmoothing(test_model, test_params, test_obs, test_conditions, cosmoothing_mask):
    mu_0 = test_params.initial.mean
    Sigma_0 = test_params.initial.cov
    A = test_params.dynamics.weights
    b = test_params.dynamics.bias
    Q = test_params.dynamics.cov
    C_held_in = test_params.emissions.weights[cosmoothing_mask]
    C_held_out = test_params.emissions.weights[~cosmoothing_mask]
    R_held_in = test_params.emissions.cov[cosmoothing_mask][:, cosmoothing_mask]

    held_in_test_params = make_lgssm_params(mu_0,
                                            Sigma_0,
                                            A,
                                            Q,
                                            C_held_in,
                                            R_held_in,
                                            dynamics_bias=b)
    held_in_test_obs = test_obs[:, :, cosmoothing_mask]

    smoother = test_model.batch_smoother(held_in_test_params, held_in_test_obs, conditions=test_conditions)
    smoothed_states = smoother.smoothed_means

    prediction = jnp.einsum('bti,ji->btj', smoothed_states, C_held_out)
    held_out_test_obs = test_obs[:, :, ~cosmoothing_mask]

    return r2_score(held_out_test_obs.flatten(), prediction.flatten())

def compute_smds_test_marginal_ll(test_model, test_params, obs, conditions, block_masks, method):
    xy_ekf_marginal_ll = test_model.marginal_log_prob(test_params, obs, conditions=conditions, block_masks=jnp.ones(len(obs), dtype=bool), method=method)
    y_ekf_marginal_ll = test_model.marginal_log_prob(test_params, obs, conditions=conditions, block_masks=block_masks, method=method)
    test_ekf_marginal_ll = xy_ekf_marginal_ll - y_ekf_marginal_ll
    return test_ekf_marginal_ll

def compute_smds_test_r2(test_model, Hs, test_params, test_obs, test_conditions):
    mu_0 = test_params.initial.mean
    Sigma_0 = test_params.initial.cov
    A = test_params.dynamics.weights
    b = test_params.dynamics.bias
    Q = test_params.dynamics.cov
    R = test_params.emissions.cov

    def batch_smoothed_xs(H, ys, condition):
        trial_test_params = make_lgssm_params(mu_0,
                                              Sigma_0,
                                              A,
                                              Q,
                                              H,
                                              R,
                                              dynamics_bias=b)
    
        posterior = lgssm_smoother(trial_test_params, ys, None, condition)
        return posterior.smoothed_means

    smds_xs = vmap(batch_smoothed_xs)(Hs,
                                      test_obs,
                                      test_conditions)

    prediction = jnp.einsum('bti,bji->btj', smds_xs, Hs)
    return r2_score(test_obs.flatten(), prediction.flatten())

def compute_smds_test_cosmoothing(test_model, Hs, test_params, test_obs, test_conditions, cosmoothing_mask):
    mu_0 = test_params.initial.mean
    Sigma_0 = test_params.initial.cov
    A = test_params.dynamics.weights
    b = test_params.dynamics.bias
    Q = test_params.dynamics.cov
    Hs_held_in = Hs[:, cosmoothing_mask]
    Hs_held_out = Hs[:, ~cosmoothing_mask]
    R_held_in = test_params.emissions.cov[cosmoothing_mask][:, cosmoothing_mask]

    held_in_test_obs = test_obs[:, :, cosmoothing_mask]

    def batch_smoothed_xs(H, ys, condition):
        trial_test_params = make_lgssm_params(mu_0,
                                              Sigma_0,
                                              A,
                                              Q,
                                              H,
                                              R_held_in,
                                              dynamics_bias=b)
    
        posterior = lgssm_smoother(trial_test_params, ys, None, condition)
        return posterior.smoothed_means

    smds_xs = vmap(batch_smoothed_xs)(Hs_held_in,
                                      held_in_test_obs,
                                      test_conditions)

    prediction = jnp.einsum('bti,bji->btj', smds_xs, Hs_held_out)
    held_out_test_obs = test_obs[:, :, ~cosmoothing_mask]

    return r2_score(held_out_test_obs.flatten(), prediction.flatten())

def evaluate_smds_model(
    model,
    params,
    train_obs,
    test_obs,
    conditions,
    test_conditions,
    num_blocks,
    block_size,
    sequence_length,
    emission_dim,
    state_dim,
    block_ids,
    trial_masks,
    block_masks,
    cosmoothing_mask,
    wandb_run=None
):
    """Evaluate an SMDS model on test data.
    
    Args:
        model: The SMDS model
        params: Model parameters
        test_emissions: Test emissions data
        test_inputs: Optional test inputs
        test_conditions: Optional test conditions (default: zeros)
        test_masks: Optional test masks (default: ones)
        block_masks: Optional block masks for SMDS
        method: Method for marginal log likelihood computation ("marginal" or "joint")
        compute_r2: Whether to compute R² scores
        compute_cosmoothing: Whether to compute co-smoothing metrics
        cosmoothing_mask: Mask for held-in dimensions for co-smoothing
        wandb_run: Optional wandb run for logging metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    velocity_smoother0 = model.smoother(params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
                                       conditions.reshape(num_blocks, block_size), block_masks,
                                       method=0)

    velocity_smoother1 = model.smoother(params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
                                       conditions.reshape(num_blocks, block_size), block_masks,
                                       method=1)
    
    Ev0 = velocity_smoother0.smoothed_means
    Hs0 = vmap(rotate_subspace, in_axes=(None, None, 0))(params.emissions.base_subspace, state_dim, Ev0)
    Hs0 = jnp.einsum('bij,bk->kij', Hs0, block_ids)[~trial_masks]

    Ev1 = velocity_smoother1.smoothed_means
    Hs1 = vmap(rotate_subspace, in_axes=(None, None, 0))(params.emissions.base_subspace, state_dim, Ev1)
    Hs1 = jnp.einsum('bij,bk->kij', Hs1, block_ids)[~trial_masks]

    Hs = params.emissions.weights[~trial_masks]

    test_ll_sum_0 = compute_smds_test_marginal_ll(model, params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
                                                  conditions.reshape(num_blocks, block_size), block_masks, 0)
    test_ll_sum_1 = compute_smds_test_marginal_ll(model, params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
                                                  conditions.reshape(num_blocks, block_size), block_masks, 1)

    test_r2 = compute_smds_test_r2(model, Hs, params, test_obs, test_conditions)
    test_cosmoothing = compute_smds_test_cosmoothing(model, Hs, params, test_obs, test_conditions, cosmoothing_mask)

    test_r2_0 = compute_smds_test_r2(model, Hs0, params, test_obs, test_conditions)
    test_cosmoothing_0 = compute_smds_test_cosmoothing(model, Hs0, params, test_obs, test_conditions, cosmoothing_mask)

    test_r2_1 = compute_smds_test_r2(model, Hs1, params, test_obs, test_conditions)
    test_cosmoothing_1 = compute_smds_test_cosmoothing(model, Hs1, params, test_obs, test_conditions, cosmoothing_mask)

    # Initialize metrics dictionary
    metrics = {
        'test_log_likelihood_0': float(test_ll_sum_0),
        'test_log_likelihood_1': float(test_ll_sum_1),
        'test_r2': float(test_r2),
        'test_cosmoothing': float(test_cosmoothing),
        'test_r2_0': float(test_r2_0),
        'test_cosmoothing_0': float(test_cosmoothing_0),
        'test_r2_1': float(test_r2_1),
        'test_cosmoothing_1': float(test_cosmoothing_1),
    }
    
    # Log metrics to wandb if provided
    if wandb_run is not None:
        log_evaluation_metrics(wandb_run, metrics)
    
    return metrics