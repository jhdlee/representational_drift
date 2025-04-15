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
    r2_score_1 = r2_score(test_obs.flatten(), prediction.flatten())
    r2_score_2 = r2_score(test_obs.reshape(-1, test_obs.shape[-1]), prediction.reshape(-1, prediction.shape[-1]),
                          multioutput='uniform_average')
    r2_score_3 = r2_score(test_obs.reshape(-1, test_obs.shape[-1]), prediction.reshape(-1, prediction.shape[-1]), 
                          multioutput='variance_weighted')
    r2_score_4 = r2_score(test_obs.swapaxes(0, 1).reshape(test_obs.shape[1], -1), 
                          prediction.swapaxes(0, 1).reshape(prediction.shape[1], -1),
                          multioutput='uniform_average')
    r2_score_5 = r2_score(test_obs.swapaxes(0, 1).reshape(test_obs.shape[1], -1), 
                          prediction.swapaxes(0, 1).reshape(prediction.shape[1], -1),
                          multioutput='variance_weighted')
    return r2_score_1, r2_score_2, r2_score_3, r2_score_4, r2_score_5

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

def compute_lds_test_condition_averaged_r2(test_model, test_params, test_obs, test_conditions):
    C = test_params.emissions.weights
    smoother = test_model.batch_smoother(test_params, test_obs, conditions=test_conditions)
    smoothed_states = smoother.smoothed_means

    prediction = jnp.einsum('bti,ji->btj', smoothed_states, C)

    conditions = jnp.unique(test_conditions)
    condition_averaged_true_psths = []
    condition_averaged_pred_psths = []
    for i, condition in enumerate(conditions):
        condition_averaged_true_psth = jnp.mean(test_obs[test_conditions == condition], axis=0)
        condition_averaged_pred_psth = jnp.mean(prediction[test_conditions == condition], axis=0)
        condition_averaged_true_psths.append(condition_averaged_true_psth)
        condition_averaged_pred_psths.append(condition_averaged_pred_psth)

    condition_averaged_true_psths = jnp.concatenate(condition_averaged_true_psths, axis=0).flatten()
    condition_averaged_pred_psths = jnp.concatenate(condition_averaged_pred_psths, axis=0).flatten()

    return jnp.corrcoef(condition_averaged_true_psths, condition_averaged_pred_psths)[0, 1]

def compute_smds_test_marginal_ll(test_model, test_params, obs, conditions, block_masks, method, num_iters):
    xy_ekf_marginal_ll = test_model.marginal_log_prob(test_params, obs, conditions=conditions, 
                                                      block_masks=jnp.ones(len(obs), dtype=bool), 
                                                      method=method, num_iters=num_iters)
    y_ekf_marginal_ll = test_model.marginal_log_prob(test_params, obs, conditions=conditions, 
                                                     block_masks=block_masks, 
                                                     method=method, num_iters=num_iters)
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

    # enumerate all possible r2 scores
    r2_score_1 = r2_score(test_obs.flatten(), prediction.flatten())
    r2_score_2 = r2_score(test_obs.reshape(-1, test_obs.shape[-1]), prediction.reshape(-1, prediction.shape[-1]),
                          multioutput='uniform_average')
    r2_score_3 = r2_score(test_obs.reshape(-1, test_obs.shape[-1]), prediction.reshape(-1, prediction.shape[-1]), 
                          multioutput='variance_weighted')
    r2_score_4 = r2_score(test_obs.swapaxes(0, 1).reshape(test_obs.shape[1], -1), 
                          prediction.swapaxes(0, 1).reshape(prediction.shape[1], -1),
                          multioutput='uniform_average')
    r2_score_5 = r2_score(test_obs.swapaxes(0, 1).reshape(test_obs.shape[1], -1), 
                          prediction.swapaxes(0, 1).reshape(prediction.shape[1], -1),
                          multioutput='variance_weighted')
    
    # compute per channel r2 score
    r2_score_6 = jnp.array([r2_score(test_obs[:, :, i].flatten(), prediction[:, :, i].flatten()) 
                            for i in range(test_obs.shape[-1])])

    return r2_score_1, r2_score_2, r2_score_3, r2_score_4, r2_score_5, r2_score_6

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

def compute_smds_test_condition_averaged_r2(test_model, Hs, test_params, test_obs, test_conditions):
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

    conditions = jnp.unique(test_conditions)
    condition_averaged_true_psths = []
    condition_averaged_pred_psths = []
    for i, condition in enumerate(conditions):
        condition_averaged_true_psth = jnp.mean(test_obs[test_conditions == condition], axis=0)
        condition_averaged_pred_psth = jnp.mean(prediction[test_conditions == condition], axis=0)
        condition_averaged_true_psths.append(condition_averaged_true_psth)
        condition_averaged_pred_psths.append(condition_averaged_pred_psth)

    condition_averaged_true_psths = jnp.concatenate(condition_averaged_true_psths, axis=0).flatten()
    condition_averaged_pred_psths = jnp.concatenate(condition_averaged_pred_psths, axis=0).flatten()

    condition_averaged_r2 = jnp.corrcoef(condition_averaged_true_psths, condition_averaged_pred_psths)[0, 1]

    return condition_averaged_r2

def evaluate_lds_model(
    model,
    params,
    test_obs,
    test_conditions,
    cosmoothing_mask,
    wandb_run=None
):
    """Evaluate an LDS model on test data.
    
    Args:
        model: The LDS model
        params: Model parameters
        test_obs: Test observations
        test_conditions: Conditions for test observations
        cosmoothing_mask: Mask for held-in dimensions for co-smoothing
        wandb_run: Optional wandb run for logging metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    test_data_size = test_obs.shape[0] * test_obs.shape[1] * test_obs.shape[2]

    test_marginal_ll = compute_lds_test_marginal_ll(model, params, test_obs, test_conditions)
    # test_marginal_ll = test_marginal_ll / test_data_size
    test_r2_1, test_r2_2, test_r2_3, test_r2_4, test_r2_5 = compute_lds_test_r2(model, params, test_obs, test_conditions)
    # test_cosmoothing = compute_lds_test_cosmoothing(model, params, test_obs, test_conditions, cosmoothing_mask)

    condition_averaged_r2 = compute_lds_test_condition_averaged_r2(model, params, test_obs, test_conditions)

    metrics = {
        'test_log_likelihood': float(test_marginal_ll),
        'test_r2_1': float(test_r2_1),
        'test_r2_2': float(test_r2_2),
        'test_r2_3': float(test_r2_3),
        'test_r2_4': float(test_r2_4),
        'test_r2_5': float(test_r2_5),
        'test_condition_averaged_r2': float(condition_averaged_r2),
        # 'test_cosmoothing': float(test_cosmoothing),
        # 'test_log_likelihood_0': float(test_marginal_ll),
        # 'test_r2_0': float(test_r2),
        # 'test_cosmoothing_0': float(test_cosmoothing),
        # 'test_log_likelihood_1': float(test_marginal_ll),
        # 'test_r2_1': float(test_r2),
        # 'test_cosmoothing_1': float(test_cosmoothing),
    }

    if wandb_run is not None:
        log_evaluation_metrics(wandb_run, metrics)
    
    return metrics

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
    ekf_num_iters,
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

    test_data_size = test_obs.shape[0] * test_obs.shape[1] * test_obs.shape[2]

    # velocity_smoother0 = model.smoother(params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
    #                                    conditions.reshape(num_blocks, block_size), jnp.ones(num_blocks, dtype=bool),
    #                                    method=0, num_iters=ekf_num_iters)
    # Ev0 = velocity_smoother0.smoothed_means
    # del velocity_smoother0

    velocity_smoother1 = model.smoother(params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
                                       conditions.reshape(num_blocks, block_size), jnp.ones(num_blocks, dtype=bool),
                                       method=1, num_iters=ekf_num_iters)
    Ev1 = velocity_smoother1.smoothed_means
    del velocity_smoother1

    # Hs0 = vmap(rotate_subspace, in_axes=(None, None, 0))(params.emissions.base_subspace, state_dim, Ev0)
    # Hs0 = jnp.einsum('bij,bk->kij', Hs0, block_ids)[~trial_masks]

    Hs1 = vmap(rotate_subspace, in_axes=(None, None, 0))(params.emissions.base_subspace, state_dim, Ev1)
    Hs1 = jnp.einsum('bij,bk->kij', Hs1, block_ids)[~trial_masks]

    Hs = params.emissions.weights[~trial_masks]

    # test_ll_sum_0 = compute_smds_test_marginal_ll(model, params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
    #                                               conditions.reshape(num_blocks, block_size), block_masks, 0, ekf_num_iters)
    test_ll_sum_1 = compute_smds_test_marginal_ll(model, params, train_obs.reshape(num_blocks, block_size, sequence_length, emission_dim), 
                                                  conditions.reshape(num_blocks, block_size), block_masks, 1, ekf_num_iters)
    # test_ll_sum_0 = test_ll_sum_0 / test_data_size
    # test_ll_sum_1 = test_ll_sum_1 / test_data_size

    # test_r2 = compute_smds_test_r2(model, Hs, params, test_obs, test_conditions)
    # test_cosmoothing = compute_smds_test_cosmoothing(model, Hs, params, test_obs, test_conditions, cosmoothing_mask)

    # test_r2_0 = compute_smds_test_r2(model, Hs0, params, test_obs, test_conditions)
    # test_cosmoothing_0 = compute_smds_test_cosmoothing(model, Hs0, params, test_obs, test_conditions, cosmoothing_mask)

    (test_r2_1, test_r2_2, test_r2_3, 
     test_r2_4, test_r2_5, test_r2_6) = compute_smds_test_r2(model, Hs1, params, test_obs, test_conditions)
    # test_cosmoothing_1 = compute_smds_test_cosmoothing(model, Hs1, params, test_obs, test_conditions, cosmoothing_mask)

    # compute condition averaged pearson r2
    condition_averaged_r2 = compute_smds_test_condition_averaged_r2(model, Hs1, params, test_obs, test_conditions)

    # Initialize metrics dictionary
    metrics = {
        # 'test_r2': float(test_r2),
        # 'test_cosmoothing': float(test_cosmoothing),
        # 'test_log_likelihood_0': float(test_ll_sum_0),
        # 'test_r2_0': float(test_r2_0),
        # 'test_cosmoothing_0': float(test_cosmoothing_0),
        # 'test_log_likelihood_0': float(test_ll_sum_0),
        'test_log_likelihood': float(test_ll_sum_1),
        'test_r2_1': float(test_r2_1),
        'test_r2_2': float(test_r2_2),
        'test_r2_3': float(test_r2_3),
        'test_r2_4': float(test_r2_4),
        'test_r2_5': float(test_r2_5),
        'test_condition_averaged_r2': float(condition_averaged_r2),
        # 'test_cosmoothing_1': float(test_cosmoothing_1),
    }

    for i in range(test_obs.shape[-1]):
        metrics[f'test_r2_channel_{i}'] = float(test_r2_6[i])
    
    # Log metrics to wandb if provided
    if wandb_run is not None:
        log_evaluation_metrics(wandb_run, metrics)
    
    return metrics