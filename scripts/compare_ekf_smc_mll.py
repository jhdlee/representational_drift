#!/usr/bin/env python

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

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


def as_float(value):
    return float(jax.device_get(value))


def summarize(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


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


def build_parser():
    parser = argparse.ArgumentParser(description="Compare EKF and RB-SMC SMDS held-out marginal log likelihood.")
    parser.add_argument("--output_dir", default="results/ekf_smc_mll_validation")
    parser.add_argument("--state_dim", type=int, default=2)
    parser.add_argument("--emission_dim", type=int, default=6)
    parser.add_argument("--num_conditions", type=int, default=2)
    parser.add_argument("--num_blocks", type=int, default=24)
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=12)
    parser.add_argument("--heldout_fraction", type=float, default=0.25)
    parser.add_argument("--tau_values", default="1e-7,3e-7,1e-6,3e-6,1e-5")
    parser.add_argument("--data_seeds", default="0,1")
    parser.add_argument("--particle_counts", default="256,1024")
    parser.add_argument("--smc_replicates", type=int, default=4)
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
    print(f"[output] replicate_csv={replicate_csv}", flush=True)
    print(f"[output] summary_csv={summary_csv}", flush=True)

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
        "smc_all_ll",
        "smc_train_ll",
        "smc_conditional_ll",
        "smc_runtime_sec",
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
        "smc_conditional_mean",
        "smc_conditional_se",
        "smc_conditional_se_per_entry",
        "ekf_minus_smc",
        "ekf_minus_smc_per_entry",
        "test_entries",
    ]
    summary_rows = []

    for tau_index, tau in enumerate(tau_values):
        for data_seed in data_seeds:
            print(f"[dataset] tau_index={tau_index} tau={tau:g} data_seed={data_seed}", flush=True)
            dataset_start = time.perf_counter()
            model, params, obs, conditions, block_masks, _, mean_angle, max_angle = make_dataset(args, tau, data_seed)
            test_entries = int((~np.asarray(block_masks)).sum() * args.block_size * args.num_timesteps * args.emission_dim)
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

            for particle_count in particle_counts:
                smc_conditionals = []
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
                    smc_conditionals.append(smc_cond)
                    running_mean, running_se = summarize(smc_conditionals)
                    print(
                        f"[smc] particles={particle_count} rep={replicate + 1}/{args.smc_replicates} "
                        f"seed={smc_seed} all={smc_all:.4f} train={smc_train:.4f} conditional={smc_cond:.4f} "
                        f"runtime={smc_runtime:.2f}s running_mean={running_mean:.4f} running_se={running_se:.4f}",
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
                            "smc_all_ll": smc_all,
                            "smc_train_ll": smc_train,
                            "smc_conditional_ll": smc_cond,
                            "smc_runtime_sec": smc_runtime,
                        },
                    )

                smc_mean, smc_se = summarize(smc_conditionals)
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
                    "smc_conditional_mean": smc_mean,
                    "smc_conditional_se": smc_se,
                    "smc_conditional_se_per_entry": smc_se / test_entries,
                    "ekf_minus_smc": ekf_cond - smc_mean,
                    "ekf_minus_smc_per_entry": (ekf_cond - smc_mean) / test_entries,
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
    print(f"[done] summary_csv={summary_csv}", flush=True)
    if figure_paths is not None:
        print(f"[done] figure_png={figure_paths[0]}", flush=True)
        print(f"[done] figure_pdf={figure_paths[1]}", flush=True)


if __name__ == "__main__":
    main()
