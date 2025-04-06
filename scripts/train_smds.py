#!/usr/bin/env python

import os
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
from dynamax.nonlinear_gaussian_ssm.inference_ekf import ParamsNLGSSM
from dynamax.utils.wandb_utils import init_wandb, save_model
from dynamax.utils.eval_utils import evaluate_smds_model
from dynamax.utils.utils import gram_schmidt, rotate_subspace

def load_data(data_path):
    """Load data from npy file"""
    emissions_path = os.path.join(data_path, 'emissions.npy')
    conditions_path = os.path.join(data_path, 'conditions.npy')

    emissions = jnp.load(emissions_path)
    conditions = jnp.load(conditions_path, allow_pickle=True)

    return emissions, conditions

def split_and_standardize_data(emissions, conditions, block_size, seed):
    """Split data into train/test sets and standardize"""
    num_conditions = len(np.unique(conditions))
    num_blocks = len(emissions) // block_size
    num_trials = num_blocks * block_size
    emissions = emissions[:num_trials]
    conditions = conditions[:num_trials]

    block_masks = jnp.ones(num_blocks, dtype=bool)
    trial_masks = jnp.ones(len(emissions), dtype=bool)
    num_test_blocks = num_blocks // 6
    key = jr.PRNGKey(seed)
    test_idx = jr.choice(key, jnp.arange(5, num_blocks-5, dtype=int), shape=(num_test_blocks,), replace=False)
    block_masks = block_masks.at[test_idx].set(False)
    num_train_blocks = block_masks.sum()
    block_ids = jnp.repeat(jnp.eye(num_blocks), block_size, axis=1)

    trial_masks = jnp.repeat(block_masks, block_size)

    train_obs_ = emissions[trial_masks]
    train_obs_mean = jnp.mean(train_obs_, axis=(0, 1), keepdims=True)
    train_obs_std = jnp.std(train_obs_, axis=(0, 1), keepdims=True)
    train_obs = (emissions - train_obs_mean) / train_obs_std
    _, sequence_length, emission_dim = train_obs.shape
    test_obs = train_obs[~trial_masks]

    train_conditions = conditions[trial_masks]
    test_conditions = conditions[~trial_masks]

    return (train_obs, test_obs, train_conditions, test_conditions, block_ids, 
            trial_masks, block_masks, sequence_length, emission_dim, num_conditions, num_blocks)

@hydra.main(version_base=None, config_path="../configs", config_name="smds_config")
def main(config: DictConfig):
    """Main function to train and evaluate SMDS model"""
    # Print config if needed
    print(OmegaConf.to_yaml(config))

    # Initialize model
    model_config = config.model
    training_config = config.training
    seed = config.seed

    model_dir = '/oak/stanford/groups/swl1/hdlee/crcns/'
    model_name = f"smds_model_{model_config.state_dim}_{model_config.ekf_mode}_{model_config.fix_tau}_{model_config.base_subspace_type}_{model_config.initial_velocity_cov}_{model_config.init_tau}_{model_config.max_tau}_{training_config.ekf_num_iters}"
    model_name = f"{model_name}_{seed}"
    
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
    
    # Load data
    data_path = config.data.path
    block_size = config.data.block_size
    
    emissions, conditions = load_data(data_path)
    (train_obs, test_obs, train_conditions, test_conditions, block_ids,
        trial_masks, block_masks, sequence_length,
        emission_dim, num_conditions, num_blocks) = split_and_standardize_data(emissions, conditions, block_size, seed)
    sorted_var_idx = jnp.argsort(train_obs[~trial_masks].var(axis=(0, 1)))[::-1]
    held_out_idx = sorted_var_idx[:5]
    cosmoothing_mask = jnp.ones(emission_dim, dtype=bool)
    cosmoothing_mask = cosmoothing_mask.at[held_out_idx].set(False)
    
    smds = StiefelManifoldSSM(
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
    )
    
    if eval_only and pretrained_model:
        # Load pretrained model
        print(f"Loading pretrained model from {pretrained_model}")
        params = pkl.load(open(pretrained_model, 'rb'))
    else:
        # Initialize parameters
        D = model_config.state_dim
        N = emission_dim
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

        params, props, _ = smds.initialize(base_subspace=base_subspace, 
                                           emission_weights=emission_weights,
                                           tau=jnp.ones(ddof) * model_config.init_tau,
                                           initial_velocity_cov=jnp.eye(ddof) * model_config.initial_velocity_cov,
                                           key=key)
        
        # Train model
        best_params, train_lps = smds.fit_em(
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

        if use_wandb:
            wandb.log({"train_log_posteriors_min_increase": jnp.diff(jnp.array(train_lps)).min()})
        
        # Save model
        if use_wandb:
            save_model(wandb_run, best_params, model_dir, model_name)
        else:
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, f"{model_name}.pkl")
            pkl.dump(best_params, open(model_save_path, 'wb'))
    
    # Evaluate on test data
    print("Evaluating on test data...")
    # Run evaluation
    metrics = evaluate_smds_model(
        model=smds,
        params=params,
        train_obs=train_obs,
        test_obs=test_obs,
        conditions=conditions,
        test_conditions=test_conditions,
        num_blocks=num_blocks,
        block_size=block_size,
        sequence_length=sequence_length,
        emission_dim=emission_dim,
        block_ids=block_ids,
        trial_masks=trial_masks,
        block_masks=block_masks,
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
