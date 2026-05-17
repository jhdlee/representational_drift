#!/usr/bin/env python

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

from dynamax.linear_gaussian_ssm.inference import make_lgssm_params, lgssm_filter
from dynamax.nonlinear_gaussian_ssm.models import StiefelManifoldSSM
from dynamax.utils.eval_utils import compute_smds_test_marginal_ll
from dynamax.utils.utils import random_dynamics_weights, rotate_subspace


def parse_csv(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def append_csv(path, fieldnames, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_csv_path(path, fieldnames):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return path
    with path.open(newline="") as f:
        reader = csv.reader(f)
        existing_header = next(reader, None)
    if existing_header == fieldnames:
        return path

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{timestamp}_{counter}{path.suffix}")
        counter += 1
    print(
        f"[output] existing {path} has a different schema; writing new rows to {candidate}",
        flush=True,
    )
    return candidate


def as_float(value):
    return float(jax.device_get(value))


def summarize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def logmeanexp(log_values, axis=None):
    log_values = np.asarray(log_values, dtype=float)
    if axis is None:
        max_value = np.max(log_values)
        return float(max_value + np.log(np.mean(np.exp(log_values - max_value))))
    max_value = np.max(log_values, axis=axis, keepdims=True)
    return np.squeeze(max_value, axis=axis) + np.log(np.mean(np.exp(log_values - max_value), axis=axis))


def bootstrap_logmeanexp_ratio_se(all_lls, train_lls, num_bootstrap=1000, seed=0):
    all_lls = np.asarray(all_lls, dtype=float)
    train_lls = np.asarray(train_lls, dtype=float)
    num_replicates = len(all_lls)
    if num_replicates <= 1 or num_bootstrap <= 1:
        return 0.0

    rng = np.random.default_rng(seed)
    all_indices = rng.integers(0, num_replicates, size=(num_bootstrap, num_replicates))
    train_indices = rng.integers(0, num_replicates, size=(num_bootstrap, num_replicates))
    estimates = logmeanexp(all_lls[all_indices], axis=1) - logmeanexp(train_lls[train_indices], axis=1)
    return float(np.std(estimates, ddof=1))


def summarize_smc_log_replicates(all_lls, train_lls, num_bootstrap=1000, bootstrap_seed=0):
    all_lls = np.asarray(all_lls, dtype=float)
    train_lls = np.asarray(train_lls, dtype=float)
    conditional_lls = all_lls - train_lls
    mean_log_conditional, mean_log_conditional_se = summarize(conditional_lls)
    all_logmeanexp = logmeanexp(all_lls)
    train_logmeanexp = logmeanexp(train_lls)
    conditional_logmeanexp = all_logmeanexp - train_logmeanexp
    conditional_bootstrap_se = bootstrap_logmeanexp_ratio_se(
        all_lls,
        train_lls,
        num_bootstrap=num_bootstrap,
        seed=bootstrap_seed,
    )
    return {
        "all_logmeanexp": all_logmeanexp,
        "train_logmeanexp": train_logmeanexp,
        "conditional_logmeanexp": conditional_logmeanexp,
        "conditional_bootstrap_se": conditional_bootstrap_se,
        "mean_log_conditional": mean_log_conditional,
        "mean_log_conditional_se": mean_log_conditional_se,
    }


def principal_angle_summary(emission_weights):
    angles = []
    for t in range(len(emission_weights) - 1):
        singular_values = np.linalg.svd(
            np.asarray(emission_weights[t]).T @ np.asarray(emission_weights[t + 1]),
            compute_uv=False,
        )
        singular_values = np.clip(singular_values, -1.0, 1.0)
        angles.extend(np.degrees(np.arccos(singular_values)))
    angles = np.asarray(angles)
    return float(angles.mean()), float(angles.max())


def sample_velocity_path(key, num_blocks, drift_dim, tau, initial_velocity_cov):
    key_init, key_steps = jr.split(key)
    initial_velocity = jr.multivariate_normal(
        key_init,
        jnp.zeros((drift_dim,)),
        initial_velocity_cov * jnp.eye(drift_dim),
    )
    step_keys = jr.split(key_steps, max(num_blocks - 1, 1))
    step_cov = tau * jnp.eye(drift_dim)

    def step(prev_velocity, step_key):
        velocity = jr.multivariate_normal(step_key, prev_velocity, step_cov)
        return velocity, velocity

    if num_blocks == 1:
        return initial_velocity[jnp.newaxis]
    _, velocities = jax.lax.scan(step, initial_velocity, step_keys[: num_blocks - 1])
    return jnp.concatenate([initial_velocity[jnp.newaxis], velocities], axis=0)


def make_dataset(args, tau, data_seed):
    key = jr.PRNGKey(data_seed)
    (
        key_base,
        key_velocity,
        key_dynamics,
        key_initial,
        key_sample,
        key_mask,
        key_init_model,
    ) = jr.split(key, 7)

    num_trials = args.num_blocks * args.block_size
    model = StiefelManifoldSSM(
        state_dim=args.state_dim,
        emission_dim=args.emission_dim,
        num_trials=num_trials,
        num_conditions=args.num_conditions,
        has_dynamics_bias=True,
        has_emissions_bias=False,
        fix_scale=True,
    )
    drift_dim = model.dof
    tau_vec = jnp.ones((drift_dim,)) * tau
    base_subspace = jr.orthogonal(key_base, args.emission_dim)
    block_velocities = sample_velocity_path(
        key_velocity,
        args.num_blocks,
        drift_dim,
        tau,
        args.initial_velocity_cov,
    )
    trial_velocities = jnp.repeat(block_velocities, args.block_size, axis=0)

    dynamics_weights = random_dynamics_weights(key_dynamics, args.state_dim, args.num_dynamics_rotations)
    initial_mean = np.sqrt(args.emission_dim / args.state_dim) * jr.normal(
        key_initial, shape=(args.num_conditions, args.state_dim)
    )
    scale = jnp.ones((num_trials, args.state_dim))

    params, _, _ = model.initialize(
        base_subspace=base_subspace,
        tau=tau_vec,
        key=key_init_model,
        initial_mean=initial_mean,
        initial_velocity_mean=jnp.zeros((drift_dim,)),
        initial_velocity_cov=args.initial_velocity_cov * jnp.eye(drift_dim),
        dynamics_weights=dynamics_weights,
        dynamics_covariance=args.dynamics_cov * jnp.eye(args.state_dim),
        emission_covariance=args.emissions_cov * jnp.eye(args.emission_dim),
        velocity=trial_velocities,
        scale=scale,
    )

    conditions = jnp.tile(jnp.arange(args.num_conditions), num_trials)[:num_trials]
    _, emissions, _ = model.sample(params, key_sample, args.num_timesteps, conditions=conditions)
    obs = emissions.reshape(args.num_blocks, args.block_size, args.num_timesteps, args.emission_dim)
    block_conditions = conditions.reshape(args.num_blocks, args.block_size)

    num_test_blocks = min(
        max(1, int(round(args.heldout_fraction * args.num_blocks))),
        args.num_blocks - 1,
    )
    test_idx = jr.choice(key_mask, jnp.arange(args.num_blocks), shape=(num_test_blocks,), replace=False)
    block_masks = jnp.ones((args.num_blocks,), dtype=bool).at[test_idx].set(False)
    trial_masks = jnp.ones((args.num_blocks, args.block_size), dtype=bool)

    emission_weights_by_block = vmap(rotate_subspace, in_axes=(None, None, 0))(
        base_subspace, args.state_dim, block_velocities
    )
    mean_angle, max_angle = principal_angle_summary(emission_weights_by_block)

    return model, params, obs, block_conditions, block_masks, trial_masks, mean_angle, max_angle


def run_one_ekf(args, model, params, obs, conditions, block_masks):
    start = time.perf_counter()
    conditional_ll, all_ll, train_ll = compute_smds_test_marginal_ll(
        model,
        params,
        obs,
        conditions,
        block_masks,
        method=1,
        num_iters=args.ekf_num_iters,
        return_components=True,
    )
    conditional_ll, all_ll, train_ll = map(as_float, (conditional_ll, all_ll, train_ll))
    return conditional_ll, all_ll, train_ll, time.perf_counter() - start


def run_one_smc(args, model, params, obs, conditions, block_masks, particle_count, smc_seed):
    start = time.perf_counter()
    conditional_ll, all_ll, train_ll = compute_smds_test_marginal_ll(
        model,
        params,
        obs,
        conditions,
        block_masks,
        method="rb_smc",
        num_iters=args.ekf_num_iters,
        key=jr.PRNGKey(smc_seed),
        num_particles=particle_count,
        return_components=True,
    )
    conditional_ll, all_ll, train_ll = map(as_float, (conditional_ll, all_ll, train_ll))
    return conditional_ll, all_ll, train_ll, time.perf_counter() - start


def _select_trial_param(param, absolute_trial_id, default):
    if param is None:
        return default
    if getattr(param, "ndim", 0) == 2:
        return param[absolute_trial_id]
    return param


def _flatten_fixed_cs(fixed_Cs, num_blocks, block_size):
    fixed_Cs = jnp.asarray(fixed_Cs)
    if fixed_Cs.ndim == 2:
        return jnp.broadcast_to(fixed_Cs, (num_blocks * block_size,) + fixed_Cs.shape)
    if fixed_Cs.ndim == 4:
        return fixed_Cs.reshape(num_blocks * block_size, fixed_Cs.shape[-2], fixed_Cs.shape[-1])
    return fixed_Cs


def fixed_c_loglik(params, emissions, conditions, block_masks, trial_masks, fixed_Cs):
    """Exact log p(observed emissions | fixed C, theta_x) via LGSSM filtering."""

    num_blocks, block_size = emissions.shape[:2]
    if conditions is None:
        conditions = jnp.zeros((num_blocks, block_size), dtype=int)
    if block_masks is None:
        block_masks = jnp.ones((num_blocks,), dtype=bool)
    if trial_masks is None:
        trial_masks = jnp.ones((num_blocks, block_size), dtype=bool)

    fixed_Cs = _flatten_fixed_cs(fixed_Cs, num_blocks, block_size)
    A = params.dynamics.weights
    Q = params.dynamics.cov
    R = params.emissions.cov
    dynamics_bias = params.dynamics.bias
    dynamics_input_weights = params.dynamics.input_weights
    emissions_input_weights = params.emissions.input_weights
    emission_bias_default = jnp.zeros((emissions.shape[-1],))
    emission_scale_default = jnp.ones((fixed_Cs.shape[-1],))

    def trial_loglik(trial_emissions, condition, block_observed, trial_observed, block_id, trial_id):
        absolute_trial_id = block_id * block_size + trial_id
        C = fixed_Cs[absolute_trial_id]
        emissions_bias = _select_trial_param(params.emissions.bias, absolute_trial_id, emission_bias_default)
        emissions_scale = _select_trial_param(params.emissions.scale, absolute_trial_id, emission_scale_default)
        lgssm_params = make_lgssm_params(
            initial_mean=params.initial.mean,
            initial_cov=params.initial.cov,
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

        return jax.lax.cond(block_observed & trial_observed, observed, skipped, (trial_emissions, condition))

    def block_loglik(block_emissions, block_conditions, block_observed, block_trial_masks, block_id):
        trial_ids = jnp.arange(block_size)
        block_ids = jnp.full((block_size,), block_id)
        block_observed = jnp.full((block_size,), block_observed)
        trial_lls = vmap(trial_loglik)(
            block_emissions,
            block_conditions,
            block_observed,
            block_trial_masks,
            block_ids,
            trial_ids,
        )
        return jnp.sum(trial_lls)

    block_ids = jnp.arange(num_blocks)
    block_lls = vmap(block_loglik)(emissions, conditions, block_masks, trial_masks, block_ids)
    return jnp.sum(block_lls)


def compute_fixed_c_conditional_mll(params, emissions, conditions, block_masks, fixed_Cs, trial_masks=None):
    """Compute log p(all observed | fixed C) - log p(train observed | fixed C)."""

    all_block_masks = jnp.ones((emissions.shape[0],), dtype=bool)
    all_ll = fixed_c_loglik(params, emissions, conditions, all_block_masks, trial_masks, fixed_Cs)
    train_ll = fixed_c_loglik(params, emissions, conditions, block_masks, trial_masks, fixed_Cs)
    return all_ll - train_ll, all_ll, train_ll


def infer_train_fixed_cs(model, params, emissions, conditions, block_masks, trial_masks=None, num_iters=1):
    """Infer a block-level drift path using training blocks only, then freeze C."""

    smoother = model.smoother(
        params,
        emissions,
        conditions=conditions,
        block_masks=block_masks,
        trial_masks=trial_masks,
        method=1,
        num_iters=num_iters,
    )
    block_Cs = vmap(rotate_subspace, in_axes=(None, None, 0))(
        params.emissions.base_subspace,
        model.state_dim,
        smoother.smoothed_means,
    )
    trial_Cs = jnp.repeat(block_Cs, emissions.shape[1], axis=0)
    return trial_Cs


def run_fixed_c_oracle(params, obs, conditions, block_masks, trial_masks):
    start = time.perf_counter()
    conditional_ll, all_ll, train_ll = compute_fixed_c_conditional_mll(
        params,
        obs,
        conditions,
        block_masks,
        params.emissions.weights,
        trial_masks=trial_masks,
    )
    conditional_ll, all_ll, train_ll = map(as_float, (conditional_ll, all_ll, train_ll))
    return conditional_ll, all_ll, train_ll, time.perf_counter() - start


def run_fixed_c_train_inferred(args, model, params, obs, conditions, block_masks, trial_masks):
    start = time.perf_counter()
    fixed_Cs = infer_train_fixed_cs(
        model,
        params,
        obs,
        conditions,
        block_masks,
        trial_masks=trial_masks,
        num_iters=args.ekf_num_iters,
    )
    conditional_ll, all_ll, train_ll = compute_fixed_c_conditional_mll(
        params,
        obs,
        conditions,
        block_masks,
        fixed_Cs,
        trial_masks=trial_masks,
    )
    conditional_ll, all_ll, train_ll = map(as_float, (conditional_ll, all_ll, train_ll))
    return conditional_ll, all_ll, train_ll, time.perf_counter() - start


def make_figure(summary_rows, output_dir):
    if not summary_rows:
        return None
    output_dir = Path(output_dir)
    particle_counts = sorted({int(row["num_particles"]) for row in summary_rows})
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for particle_count in particle_counts:
        rows = [row for row in summary_rows if int(row["num_particles"]) == particle_count]
        xs = np.asarray([float(row["mean_angle_deg"]) for row in rows])
        ys = np.asarray([float(row["ekf_minus_smc_per_entry"]) for row in rows])
        yerr = np.asarray([float(row["smc_conditional_se_per_entry"]) for row in rows])
        order = np.argsort(xs)
        ax.errorbar(
            xs[order],
            ys[order],
            yerr=yerr[order],
            marker="o",
            linewidth=1.5,
            capsize=3,
            label=f"{particle_count} particles",
        )
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("Mean consecutive principal angle (deg)")
    ax.set_ylabel("EKF - SMC held-out MLL / entry")
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = output_dir / "ekf_minus_smc_mll.png"
    pdf_path = output_dir / "ekf_minus_smc_mll.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def make_fixed_c_figure(summary_rows, output_dir):
    if not summary_rows:
        return None
    output_dir = Path(output_dir)
    particle_count = max(int(row["num_particles"]) for row in summary_rows)
    rows = [row for row in summary_rows if int(row["num_particles"]) == particle_count]
    xs = np.asarray([float(row["mean_angle_deg"]) for row in rows])
    entries = np.asarray([float(row["test_entries"]) for row in rows])
    order = np.argsort(xs)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(
        xs[order],
        (np.asarray([float(row["ekf_conditional_ll"]) for row in rows]) / entries)[order],
        marker="o",
        linewidth=1.5,
        label="EKF marginal",
    )
    ax.errorbar(
        xs[order],
        (np.asarray([float(row["smc_conditional_mean"]) for row in rows]) / entries)[order],
        yerr=(np.asarray([float(row["smc_conditional_se"]) for row in rows]) / entries)[order],
        marker="o",
        linewidth=1.5,
        capsize=3,
        label=f"SMC marginal ({particle_count} particles)",
    )
    ax.plot(
        xs[order],
        (np.asarray([float(row["fixed_c_oracle_conditional_ll"]) for row in rows]) / entries)[order],
        marker="o",
        linewidth=1.5,
        label="Oracle fixed C",
    )
    ax.plot(
        xs[order],
        (np.asarray([float(row["fixed_c_train_inferred_conditional_ll"]) for row in rows]) / entries)[order],
        marker="o",
        linewidth=1.5,
        label="Train-inferred fixed C",
    )
    ax.set_xlabel("Mean consecutive principal angle (deg)")
    ax.set_ylabel("Held-out conditional MLL / entry")
    ax.legend(frameon=False)
    fig.tight_layout()
    png_path = output_dir / "fixed_c_mll_comparison.png"
    pdf_path = output_dir / "fixed_c_mll_comparison.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def build_parser():
    parser = argparse.ArgumentParser(description="Compare EKF and SMC SMDS held-out marginal log likelihood.")
    parser.add_argument("--output_dir", default="results/ekf_smc_mll_validation")
    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--emission_dim", type=int, default=6)
    parser.add_argument("--num_conditions", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=20)
    parser.add_argument("--heldout_fraction", type=float, default=0.25)
    parser.add_argument("--tau_values", default="1e-7,3e-7,1e-6,3e-6,1e-5")
    parser.add_argument("--data_seeds", default="0,1")
    parser.add_argument("--particle_counts", default="256,1024")
    parser.add_argument("--smc_replicates", type=int, default=4)
    parser.add_argument("--smc_bootstrap_samples", type=int, default=1000)
    parser.add_argument("--smc_seed", type=int, default=1000)
    parser.add_argument("--ekf_num_iters", type=int, default=1)
    parser.add_argument("--initial_velocity_cov", type=float, default=1e-8)
    parser.add_argument("--dynamics_cov", type=float, default=1e-1)
    parser.add_argument("--emissions_cov", type=float, default=1e-1)
    parser.add_argument("--num_dynamics_rotations", type=int, default=16)
    return parser


def main():
    args = build_parser().parse_args()
    tau_values = parse_csv(args.tau_values, float)
    data_seeds = parse_csv(args.data_seeds, int)
    particle_counts = parse_csv(args.particle_counts, int)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replicate_csv = output_dir / "replicate_results.csv"
    summary_csv = output_dir / "summary_results.csv"

    print("[config]", vars(args), flush=True)

    replicate_fields = [
        "tau",
        "tau_index",
        "data_seed",
        "num_particles",
        "replicate",
        "smc_seed",
        "mean_angle_deg",
        "max_angle_deg",
        "ekf_all_ll",
        "ekf_train_ll",
        "ekf_conditional_ll",
        "fixed_c_oracle_all_ll",
        "fixed_c_oracle_train_ll",
        "fixed_c_oracle_conditional_ll",
        "fixed_c_oracle_runtime_sec",
        "fixed_c_train_inferred_all_ll",
        "fixed_c_train_inferred_train_ll",
        "fixed_c_train_inferred_conditional_ll",
        "fixed_c_train_inferred_runtime_sec",
        "smc_all_ll",
        "smc_train_ll",
        "smc_conditional_ll",
        "smc_runtime_sec",
        "ekf_minus_smc",
        "ekf_minus_smc_per_entry",
        "ekf_minus_fixed_c_oracle",
        "ekf_minus_fixed_c_oracle_per_entry",
        "ekf_minus_fixed_c_train_inferred",
        "ekf_minus_fixed_c_train_inferred_per_entry",
        "fixed_c_oracle_minus_train_inferred",
        "fixed_c_oracle_minus_train_inferred_per_entry",
        "test_entries",
    ]
    summary_fields = [
        "tau",
        "tau_index",
        "data_seed",
        "num_particles",
        "mean_angle_deg",
        "max_angle_deg",
        "ekf_all_ll",
        "ekf_train_ll",
        "ekf_conditional_ll",
        "ekf_runtime_sec",
        "fixed_c_oracle_all_ll",
        "fixed_c_oracle_train_ll",
        "fixed_c_oracle_conditional_ll",
        "fixed_c_oracle_runtime_sec",
        "fixed_c_train_inferred_all_ll",
        "fixed_c_train_inferred_train_ll",
        "fixed_c_train_inferred_conditional_ll",
        "fixed_c_train_inferred_runtime_sec",
        "smc_all_logmeanexp",
        "smc_train_logmeanexp",
        "smc_conditional_logmeanexp",
        "smc_conditional_mean",
        "smc_conditional_se",
        "smc_conditional_se_per_entry",
        "smc_conditional_mean_log",
        "smc_conditional_mean_log_se",
        "smc_conditional_mean_log_se_per_entry",
        "smc_bootstrap_samples",
        "ekf_minus_smc",
        "ekf_minus_smc_per_entry",
        "ekf_minus_fixed_c_oracle",
        "ekf_minus_fixed_c_oracle_per_entry",
        "ekf_minus_fixed_c_train_inferred",
        "ekf_minus_fixed_c_train_inferred_per_entry",
        "fixed_c_oracle_minus_train_inferred",
        "fixed_c_oracle_minus_train_inferred_per_entry",
        "test_entries",
    ]
    replicate_csv = resolve_csv_path(replicate_csv, replicate_fields)
    summary_csv = resolve_csv_path(summary_csv, summary_fields)
    print(f"[output] replicate_csv={replicate_csv}", flush=True)
    print(f"[output] summary_csv={summary_csv}", flush=True)

    summary_rows = []

    for tau_index, tau in enumerate(tau_values):
        for data_seed in data_seeds:
            print(f"[dataset] tau_index={tau_index} tau={tau:g} data_seed={data_seed}", flush=True)
            dataset_start = time.perf_counter()
            model, params, obs, conditions, block_masks, trial_masks, mean_angle, max_angle = make_dataset(
                args, tau, data_seed
            )
            test_trial_count = int(np.sum((~np.asarray(block_masks))[:, None] & np.asarray(trial_masks)))
            test_entries = test_trial_count * args.num_timesteps * args.emission_dim
            print(
                f"[dataset] built in {time.perf_counter() - dataset_start:.2f}s "
                f"mean_angle={mean_angle:.4f}deg max_angle={max_angle:.4f}deg test_entries={test_entries}",
                flush=True,
            )

            ekf_cond, ekf_all, ekf_train, ekf_runtime = run_one_ekf(args, model, params, obs, conditions, block_masks)
            print(
                f"[ekf] all={ekf_all:.4f} train={ekf_train:.4f} conditional={ekf_cond:.4f} "
                f"runtime={ekf_runtime:.2f}s",
                flush=True,
            )

            fixed_oracle_cond, fixed_oracle_all, fixed_oracle_train, fixed_oracle_runtime = run_fixed_c_oracle(
                params,
                obs,
                conditions,
                block_masks,
                trial_masks,
            )
            print(
                f"[fixed-c/oracle] all={fixed_oracle_all:.4f} train={fixed_oracle_train:.4f} "
                f"conditional={fixed_oracle_cond:.4f} runtime={fixed_oracle_runtime:.2f}s",
                flush=True,
            )

            (
                fixed_train_cond,
                fixed_train_all,
                fixed_train_train,
                fixed_train_runtime,
            ) = run_fixed_c_train_inferred(args, model, params, obs, conditions, block_masks, trial_masks)
            print(
                f"[fixed-c/train-inferred] all={fixed_train_all:.4f} train={fixed_train_train:.4f} "
                f"conditional={fixed_train_cond:.4f} runtime={fixed_train_runtime:.2f}s",
                flush=True,
            )

            for particle_count in particle_counts:
                smc_alls = []
                smc_trains = []
                print(f"[smc] starting num_particles={particle_count}", flush=True)
                for replicate in range(args.smc_replicates):
                    smc_seed = args.smc_seed + 100000 * tau_index + 1000 * data_seed + 17 * replicate + particle_count
                    smc_cond, smc_all, smc_train, smc_runtime = run_one_smc(
                        args,
                        model,
                        params,
                        obs,
                        conditions,
                        block_masks,
                        particle_count,
                        smc_seed,
                    )
                    smc_alls.append(smc_all)
                    smc_trains.append(smc_train)
                    running_summary = summarize_smc_log_replicates(
                        smc_alls,
                        smc_trains,
                        num_bootstrap=args.smc_bootstrap_samples,
                        bootstrap_seed=smc_seed,
                    )
                    print(
                        f"[smc] particles={particle_count} rep={replicate + 1}/{args.smc_replicates} "
                        f"seed={smc_seed} all={smc_all:.4f} train={smc_train:.4f} conditional={smc_cond:.4f} "
                        f"runtime={smc_runtime:.2f}s "
                        f"running_logmeanexp={running_summary['conditional_logmeanexp']:.4f} "
                        f"running_bootstrap_se={running_summary['conditional_bootstrap_se']:.4f} "
                        f"running_mean_log={running_summary['mean_log_conditional']:.4f}",
                        flush=True,
                    )
                    append_csv(
                        replicate_csv,
                        replicate_fields,
                        {
                            "tau": tau,
                            "tau_index": tau_index,
                            "data_seed": data_seed,
                            "num_particles": particle_count,
                            "replicate": replicate,
                            "smc_seed": smc_seed,
                            "mean_angle_deg": mean_angle,
                            "max_angle_deg": max_angle,
                            "ekf_all_ll": ekf_all,
                            "ekf_train_ll": ekf_train,
                            "ekf_conditional_ll": ekf_cond,
                            "fixed_c_oracle_all_ll": fixed_oracle_all,
                            "fixed_c_oracle_train_ll": fixed_oracle_train,
                            "fixed_c_oracle_conditional_ll": fixed_oracle_cond,
                            "fixed_c_oracle_runtime_sec": fixed_oracle_runtime,
                            "fixed_c_train_inferred_all_ll": fixed_train_all,
                            "fixed_c_train_inferred_train_ll": fixed_train_train,
                            "fixed_c_train_inferred_conditional_ll": fixed_train_cond,
                            "fixed_c_train_inferred_runtime_sec": fixed_train_runtime,
                            "smc_all_ll": smc_all,
                            "smc_train_ll": smc_train,
                            "smc_conditional_ll": smc_cond,
                            "smc_runtime_sec": smc_runtime,
                            "ekf_minus_smc": ekf_cond - smc_cond,
                            "ekf_minus_smc_per_entry": (ekf_cond - smc_cond) / test_entries,
                            "ekf_minus_fixed_c_oracle": ekf_cond - fixed_oracle_cond,
                            "ekf_minus_fixed_c_oracle_per_entry": (ekf_cond - fixed_oracle_cond) / test_entries,
                            "ekf_minus_fixed_c_train_inferred": ekf_cond - fixed_train_cond,
                            "ekf_minus_fixed_c_train_inferred_per_entry": (
                                ekf_cond - fixed_train_cond
                            ) / test_entries,
                            "fixed_c_oracle_minus_train_inferred": fixed_oracle_cond - fixed_train_cond,
                            "fixed_c_oracle_minus_train_inferred_per_entry": (
                                fixed_oracle_cond - fixed_train_cond
                            ) / test_entries,
                            "test_entries": test_entries,
                        },
                    )

                smc_summary = summarize_smc_log_replicates(
                    smc_alls,
                    smc_trains,
                    num_bootstrap=args.smc_bootstrap_samples,
                    bootstrap_seed=args.smc_seed + 100000 * tau_index + 1000 * data_seed + particle_count,
                )
                smc_mean = smc_summary["conditional_logmeanexp"]
                smc_se = smc_summary["conditional_bootstrap_se"]
                row = {
                    "tau": tau,
                    "tau_index": tau_index,
                    "data_seed": data_seed,
                    "num_particles": particle_count,
                    "mean_angle_deg": mean_angle,
                    "max_angle_deg": max_angle,
                    "ekf_all_ll": ekf_all,
                    "ekf_train_ll": ekf_train,
                    "ekf_conditional_ll": ekf_cond,
                    "ekf_runtime_sec": ekf_runtime,
                    "fixed_c_oracle_all_ll": fixed_oracle_all,
                    "fixed_c_oracle_train_ll": fixed_oracle_train,
                    "fixed_c_oracle_conditional_ll": fixed_oracle_cond,
                    "fixed_c_oracle_runtime_sec": fixed_oracle_runtime,
                    "fixed_c_train_inferred_all_ll": fixed_train_all,
                    "fixed_c_train_inferred_train_ll": fixed_train_train,
                    "fixed_c_train_inferred_conditional_ll": fixed_train_cond,
                    "fixed_c_train_inferred_runtime_sec": fixed_train_runtime,
                    "smc_all_logmeanexp": smc_summary["all_logmeanexp"],
                    "smc_train_logmeanexp": smc_summary["train_logmeanexp"],
                    "smc_conditional_logmeanexp": smc_mean,
                    "smc_conditional_mean": smc_mean,
                    "smc_conditional_se": smc_se,
                    "smc_conditional_se_per_entry": smc_se / test_entries,
                    "smc_conditional_mean_log": smc_summary["mean_log_conditional"],
                    "smc_conditional_mean_log_se": smc_summary["mean_log_conditional_se"],
                    "smc_conditional_mean_log_se_per_entry": (
                        smc_summary["mean_log_conditional_se"] / test_entries
                    ),
                    "smc_bootstrap_samples": args.smc_bootstrap_samples,
                    "ekf_minus_smc": ekf_cond - smc_mean,
                    "ekf_minus_smc_per_entry": (ekf_cond - smc_mean) / test_entries,
                    "ekf_minus_fixed_c_oracle": ekf_cond - fixed_oracle_cond,
                    "ekf_minus_fixed_c_oracle_per_entry": (ekf_cond - fixed_oracle_cond) / test_entries,
                    "ekf_minus_fixed_c_train_inferred": ekf_cond - fixed_train_cond,
                    "ekf_minus_fixed_c_train_inferred_per_entry": (ekf_cond - fixed_train_cond) / test_entries,
                    "fixed_c_oracle_minus_train_inferred": fixed_oracle_cond - fixed_train_cond,
                    "fixed_c_oracle_minus_train_inferred_per_entry": (
                        fixed_oracle_cond - fixed_train_cond
                    ) / test_entries,
                    "test_entries": test_entries,
                }
                summary_rows.append(row)
                append_csv(summary_csv, summary_fields, row)
                print(
                    f"[checkpoint] wrote particles={particle_count} tau={tau:g} data_seed={data_seed} "
                    f"summary to {summary_csv}",
                    flush=True,
                )

    figure_paths = make_figure(summary_rows, output_dir)
    fixed_c_figure_paths = make_fixed_c_figure(summary_rows, output_dir)
    print(f"[done] summary_csv={summary_csv}", flush=True)
    if figure_paths is not None:
        print(f"[done] figure_png={figure_paths[0]}", flush=True)
        print(f"[done] figure_pdf={figure_paths[1]}", flush=True)
    if fixed_c_figure_paths is not None:
        print(f"[done] fixed_c_figure_png={fixed_c_figure_paths[0]}", flush=True)
        print(f"[done] fixed_c_figure_pdf={fixed_c_figure_paths[1]}", flush=True)


if __name__ == "__main__":
    main()
