import jax.numpy as jnp
from sklearn.metrics import r2_score
from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_smoother
from jax import vmap

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