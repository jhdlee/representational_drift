#!/usr/bin/env python

import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import numpy as np
import pickle as pkl
from sklearn.decomposition import PCA
from dynamax.nonlinear_gaussian_ssm.models import StiefelManifoldSSM
from dynamax.linear_gaussian_ssm.models import LinearGaussianConjugateSSM
from dynamax.nonlinear_gaussian_ssm.inference_ekf import ParamsNLGSSM
from dynamax.utils.wandb_utils import init_wandb, save_model
from dynamax.utils.eval_utils import evaluate_smds_model, evaluate_lds_model
from dynamax.utils.utils import gram_schmidt, rotate_subspace, random_rotation
from dynamax.utils.distributions import IG, MVN

def split_data(emissions, conditions, block_size, seed):
    """Split data into train/test sets"""
    num_conditions = len(np.unique(conditions))
    num_blocks = len(emissions) // block_size
    num_trials = num_blocks * block_size
    emissions = emissions[:num_trials]
    conditions = conditions[:num_trials]

    block_masks = jnp.ones(num_blocks, dtype=bool)
    trial_masks = jnp.ones(len(emissions), dtype=bool)
    num_test_blocks = num_blocks // 6
    key = jr.PRNGKey(seed)
    test_idx = jr.choice(key, jnp.arange(2, num_blocks-2, dtype=int), shape=(num_test_blocks,), replace=False)
    block_masks = block_masks.at[test_idx].set(False)
    num_train_blocks = block_masks.sum()
    block_ids = jnp.repeat(jnp.eye(num_blocks), block_size, axis=1)
    trial_masks = jnp.repeat(block_masks, block_size)

    train_obs = emissions
    _, sequence_length, emission_dim = train_obs.shape
    test_obs = train_obs[~trial_masks]

    train_conditions = conditions[trial_masks]
    test_conditions = conditions[~trial_masks]

    return (emissions, conditions,train_obs, test_obs, train_conditions, test_conditions, block_ids, 
            trial_masks, block_masks, sequence_length, emission_dim, num_conditions, num_blocks)

@hydra.main(version_base=None, config_path="../configs", config_name="simulated_config")
def main(config: DictConfig):
    """Main function to train and evaluate SMDS model"""
    # Print config if needed
    print(OmegaConf.to_yaml(config))

    # Initialize model
    model_config = config.model
    training_config = config.training
    eval_config = config.eval
    seed = config.seed

    data_config = config.data
    true_state_dim = data_config.state_dim
    emission_dim = data_config.emission_dim
    num_trials = data_config.num_trials
    num_conditions = data_config.num_conditions
    num_timesteps = data_config.num_timesteps
    block_size = data_config.block_size
    dof = true_state_dim * (emission_dim - true_state_dim)
    dof_shape = (true_state_dim, emission_dim - true_state_dim)

    data_dir = model_dir = f'/oak/stanford/groups/swl1/hdlee/smds/simulated_{true_state_dim}x{emission_dim}/'
    model_name = f"{model_config.type}_D.{model_config.state_dim}"
    if model_config.type == 'smds':
        model_name += f"_ekfmode.{model_config.ekf_mode}_base.{model_config.base_subspace_type}_ivc.{model_config.initial_velocity_cov}"
        model_name += f"_itau.{model_config.init_tau}_mtau.{model_config.max_tau}_ekf_num_iters.{training_config.ekf_num_iters}"
        model_name += f"_tau_concentration.{model_config.tau_concentration}_tau_scale.{model_config.tau_scale}"
    model_name = f"{model_name}_seed.{seed}"
    
    # Check for evaluation-only mode
    eval_only = config.get('eval_only', False)
    pretrained_model = config.get('pretrained_model', None)
    
    # Initialize wandb
    use_wandb = config.get('use_wandb', True)
    if use_wandb:
        # Convert the config to a dictionary
        args_as_dict = OmegaConf.to_container(config)
        wandb_configs = {}
        wandb_configs['config'] = args_as_dict
        wandb_configs['name'] = model_name
        wandb_configs['project'] = config.project
        wandb_run, _ = init_wandb(**wandb_configs)
    else:
        wandb_run = None
    
    # Generate data from smds if not already generated
    data_name = f'emissions_seed.{seed}.npy'
    condition_name = f'conditions_seed.{seed}.npy'
    states_name = f'states_seed.{seed}.npy'
    params_name = f'params_seed.{seed}.pkl'
    if not os.path.exists(os.path.join(data_dir, data_name)):
        key = jr.PRNGKey(seed)
        dynamics = random_rotation(seed=key, n=true_state_dim, theta=jnp.pi / 5)
        true_model = StiefelManifoldSSM(state_dim=true_state_dim, 
                                        emission_dim=emission_dim,
                                        num_trials=num_trials,
                                        num_conditions=num_conditions)

        key, key_root = jr.split(key)
        true_base_subspace = jr.orthogonal(key_root, emission_dim)
        key, key_root = jr.split(key)
        true_tau = jr.uniform(key_root, shape=(dof,), minval=1e-10, maxval=1e-4)

        _velocity_cov = jnp.diag(true_tau)
        def _get_velocity(prev_velocity, current_key):
            current_velocity_dist = MVN(loc=prev_velocity, covariance_matrix=_velocity_cov)
            current_velocity = current_velocity_dist.sample(seed=current_key)
            return current_velocity, current_velocity

        keys = jr.split(key, num_trials)
        key = keys[-1]
        key, key_root = jr.split(key)
        _initial_velocity = jnp.zeros(dof)
        _, _velocity = jax.lax.scan(_get_velocity, _initial_velocity, keys[:-1])
        _velocity = jnp.concatenate([_initial_velocity[None], _velocity])
        _velocity = _velocity.reshape((num_trials,) + dof_shape)

        key, key_root = jr.split(key)
        true_params, param_props, true_velocity = true_model.initialize(tau=true_tau,
                                                                        base_subspace=true_base_subspace,
                                                                        key=key, 
                                                                        initial_mean=jnp.sqrt(emission_dim/true_state_dim)*jr.normal(key_root, shape=(num_conditions, true_state_dim)),
                                                                        dynamics_weights=dynamics,
                                                                        dynamics_covariance=jnp.eye(true_state_dim)*1e-2,
                                                                        emission_covariance=jnp.eye(emission_dim)*1e-2,
                                                                        velocity=_velocity,
                                                                        )

        conditions = jnp.tile(jnp.arange(num_conditions), num_trials)[:num_trials]
        key, key_root = jr.split(key)
        true_states, emissions, _ = true_model.sample(true_params, key, num_timesteps, conditions=conditions)
        jnp.save(os.path.join(data_dir, data_name), emissions)
        jnp.save(os.path.join(data_dir, condition_name), conditions)
        jnp.save(os.path.join(data_dir, states_name), true_states)
        pkl.dump(true_params, open(os.path.join(data_dir, params_name), 'wb'))
    else:
        emissions = jnp.load(os.path.join(data_dir, data_name))
        conditions = jnp.load(os.path.join(data_dir, condition_name))
        true_states = jnp.load(os.path.join(data_dir, states_name))
        true_params = pkl.load(open(os.path.join(data_dir, params_name), 'rb'))

    (emissions, conditions, train_obs, test_obs, 
        train_conditions, test_conditions, block_ids,
        trial_masks, block_masks, sequence_length,
        emission_dim, num_conditions, num_blocks) = split_data(emissions, conditions, block_size, seed)
    sorted_var_idx = jnp.argsort(train_obs[~trial_masks].var(axis=(0, 1)))[::-1]
    held_out_idx = sorted_var_idx[:5]
    cosmoothing_mask = jnp.ones(emission_dim, dtype=bool)
    cosmoothing_mask = cosmoothing_mask.at[held_out_idx].set(False)
    
    if model_config.type == 'smds':
        model = StiefelManifoldSSM(
            state_dim=model_config.state_dim,
            emission_dim=emission_dim,
            num_trials=len(train_obs),
            num_conditions=num_conditions,
            has_dynamics_bias=model_config.has_dynamics_bias,
            tau_per_dim=model_config.tau_per_dim,
            fix_tau=model_config.fix_tau,
            emissions_cov_eps=model_config.emissions_cov_eps,
            velocity_smoother_method=training_config.velocity_smoother_method,
            ekf_mode=model_config.ekf_mode,
            max_tau=model_config.max_tau,
            ekf_num_iters=training_config.ekf_num_iters,
            tau_prior=IG(concentration=model_config.tau_concentration, scale=model_config.tau_scale),
        )
    elif model_config.type == 'lds':
        model = LinearGaussianConjugateSSM(
            state_dim=model_config.state_dim,
            emission_dim=emission_dim,
            num_conditions=num_conditions,
            has_dynamics_bias=model_config.has_dynamics_bias,
        )
    
    if eval_only and pretrained_model:
        # Load pretrained model
        print(f"Loading pretrained model from {pretrained_model}")
        params = pkl.load(open(pretrained_model, 'rb'))
    else:
        # Initialize parameters
        D = model_config.state_dim
        N = emission_dim

        if model_config.type == 'smds':
            ddof = D * (N - D)
            key = jr.PRNGKey(seed)

            if model_config.base_subspace_type == 'pca':
                base_subspace = PCA(n_components=N).fit(train_obs[trial_masks].reshape(-1, N)).components_.T
                emission_weights = jnp.tile(base_subspace[:, :D][None], (len(train_obs), 1, 1))
            else:
                key, key_root = jr.split(key)
                key, key_root = jr.split(key)
                random_rotation_matrix = jr.orthogonal(key_root, D)
                emission_weights = PCA(n_components=N).fit(train_obs[trial_masks].reshape(-1, N)).components_.T[:, :D] @ random_rotation_matrix
                base_subspace = gram_schmidt(jnp.concatenate([emission_weights, jr.normal(key_root, shape=(N, N-D))], axis=-1))
                emission_weights = jnp.tile(emission_weights[None], (len(train_obs), 1, 1))

            params, props, _ = model.initialize(base_subspace=base_subspace, 
                                            emission_weights=emission_weights,
                                            tau=jnp.ones(ddof) * model_config.init_tau,
                                            initial_velocity_cov=jnp.eye(ddof) * model_config.initial_velocity_cov,
                                            key=key)
        
            # Train model
            best_params, train_lps = model.fit_em(
                params=params,
                props=props,
                emissions=train_obs,
                conditions=conditions,
                trial_masks=trial_masks,
                block_ids=block_ids,
                block_masks=block_masks,
                num_iters=training_config.num_iters,
                run_velocity_smoother=training_config.run_velocity_smoother,
                print_ll=training_config.print_ll,
                use_wandb=use_wandb,
                wandb_run=wandb_run if use_wandb else None,
            )
        elif model_config.type == 'lds':
            key = jr.PRNGKey(seed)
            params, props = model.initialize(key=key)
            best_params, train_lps = model.fit_em(
                params=params,
                props=props,
                emissions=train_obs[trial_masks],
                conditions=train_conditions,
                num_iters=training_config.num_iters,
                use_wandb=use_wandb,
                wandb_run=wandb_run if use_wandb else None,
            )

        if use_wandb:
            wandb.log({"train_log_posteriors_min_increase": jnp.diff(jnp.array(train_lps)[1:]).min()})
            save_model(wandb_run, best_params, model_dir, model_name)
        else:
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, f"{model_name}.pkl")
            pkl.dump(best_params, open(model_save_path, 'wb'))
    
    # Evaluate on test data
    print("Evaluating on test data...")
    if model_config.type == 'smds':
        # Run evaluation
        metrics = evaluate_smds_model(
            model=model,
            params=best_params,
            train_obs=train_obs,
            test_obs=test_obs,
            conditions=conditions,
            test_conditions=test_conditions,
            num_blocks=num_blocks,
            block_size=block_size,
            sequence_length=sequence_length,
            emission_dim=emission_dim,
            state_dim=model_config.state_dim,
            block_ids=block_ids,
            trial_masks=trial_masks,
            block_masks=block_masks,
            cosmoothing_mask=cosmoothing_mask,
            ekf_num_iters=eval_config.ekf_num_iters,
            wandb_run=wandb_run if use_wandb else None
        )
    elif model_config.type == 'lds':
        metrics = evaluate_lds_model(
            model=model,
            params=best_params,
            test_obs=test_obs,
            test_conditions=test_conditions,
            cosmoothing_mask=cosmoothing_mask,
            wandb_run=wandb_run if use_wandb else None
        )
    
    # Print evaluation results
    print("\nEvaluation Results:")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    
    # Finish wandb run
    if use_wandb:
        wandb_run.finish()

if __name__ == "__main__":
    main() 
