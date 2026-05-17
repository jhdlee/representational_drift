#!/usr/bin/env python

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


def parse_csv(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(row, key, default=np.nan):
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return float(value)


def to_int(row, key, default=0):
    value = row.get(key, "")
    if value is None or value == "":
        return default
    return int(float(value))


def logmeanexp(log_values, axis=None):
    log_values = np.asarray(log_values, dtype=float)
    if axis is None:
        max_value = np.max(log_values)
        return float(max_value + np.log(np.mean(np.exp(log_values - max_value))))
    max_value = np.max(log_values, axis=axis, keepdims=True)
    return np.squeeze(max_value, axis=axis) + np.log(np.mean(np.exp(log_values - max_value), axis=axis))


def bootstrap_logmeanexp_ratio_se(all_lls, train_lls, num_bootstrap, seed):
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


def condition_key(row):
    return (
        row.get("tau", ""),
        row.get("tau_index", ""),
        row.get("data_seed", ""),
        row.get("num_particles", ""),
    )


def stable_seed_offset(key):
    text = "|".join(str(item) for item in key)
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value % 1_000_000


def dataset_key(row):
    return (
        row.get("tau", ""),
        row.get("tau_index", ""),
        row.get("data_seed", ""),
    )


def smc_conditional(row):
    if "smc_conditional_logmeanexp" in row and row["smc_conditional_logmeanexp"] != "":
        return to_float(row, "smc_conditional_logmeanexp")
    return to_float(row, "smc_conditional_mean")


def matches_tau(row, tau):
    row_tau = to_float(row, "tau")
    return np.isclose(row_tau, tau, rtol=1e-6, atol=1e-12)


def recompute_smc_from_replicates(summary_rows, replicate_rows, num_bootstrap, seed):
    if not replicate_rows:
        return summary_rows

    summary_by_key = {condition_key(row): dict(row) for row in summary_rows}
    grouped = defaultdict(list)
    for row in replicate_rows:
        grouped[condition_key(row)].append(row)

    for key, rows in grouped.items():
        base = summary_by_key.get(key, dict(rows[0]))
        all_lls = np.asarray([to_float(row, "smc_all_ll") for row in rows])
        train_lls = np.asarray([to_float(row, "smc_train_ll") for row in rows])
        conditionals = all_lls - train_lls
        smc_all = logmeanexp(all_lls)
        smc_train = logmeanexp(train_lls)
        smc_cond = smc_all - smc_train
        smc_se = bootstrap_logmeanexp_ratio_se(all_lls, train_lls, num_bootstrap, seed + stable_seed_offset(key))
        test_entries = to_float(base, "test_entries", default=1.0)

        base["smc_all_logmeanexp"] = smc_all
        base["smc_train_logmeanexp"] = smc_train
        base["smc_conditional_logmeanexp"] = smc_cond
        base["smc_conditional_mean"] = smc_cond
        base["smc_conditional_se"] = smc_se
        base["smc_conditional_se_per_entry"] = smc_se / test_entries
        base["smc_conditional_mean_log"] = float(np.mean(conditionals))
        base["smc_conditional_mean_log_se"] = (
            float(np.std(conditionals, ddof=1) / np.sqrt(len(conditionals))) if len(conditionals) > 1 else 0.0
        )
        base["smc_conditional_mean_log_se_per_entry"] = to_float(base, "smc_conditional_mean_log_se") / test_entries
        base["smc_bootstrap_samples"] = num_bootstrap

        ekf_cond = to_float(base, "ekf_conditional_ll")
        oracle_cond = to_float(base, "fixed_c_oracle_conditional_ll")
        train_fixed_cond = to_float(base, "fixed_c_train_inferred_conditional_ll")
        base["ekf_minus_smc"] = ekf_cond - smc_cond
        base["ekf_minus_smc_per_entry"] = (ekf_cond - smc_cond) / test_entries
        base["ekf_minus_fixed_c_oracle"] = ekf_cond - oracle_cond
        base["ekf_minus_fixed_c_oracle_per_entry"] = (ekf_cond - oracle_cond) / test_entries
        base["ekf_minus_fixed_c_train_inferred"] = ekf_cond - train_fixed_cond
        base["ekf_minus_fixed_c_train_inferred_per_entry"] = (ekf_cond - train_fixed_cond) / test_entries
        base["fixed_c_oracle_minus_train_inferred"] = oracle_cond - train_fixed_cond
        base["fixed_c_oracle_minus_train_inferred_per_entry"] = (oracle_cond - train_fixed_cond) / test_entries
        summary_by_key[key] = base

    return list(summary_by_key.values())


def sorted_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            to_float(row, "mean_angle_deg"),
            to_float(row, "tau"),
            to_int(row, "data_seed"),
            to_int(row, "num_particles"),
        ),
    )


def rows_at_largest_particles(rows):
    if not rows:
        return []
    max_particles = max(to_int(row, "num_particles") for row in rows)
    return [row for row in rows if to_int(row, "num_particles") == max_particles]


def convergence_rows(rows):
    by_dataset = defaultdict(list)
    for row in rows:
        by_dataset[dataset_key(row)].append(row)

    out = []
    for _, dataset_rows in by_dataset.items():
        if not dataset_rows:
            continue
        ref_row = max(dataset_rows, key=lambda row: to_int(row, "num_particles"))
        ref_smc = smc_conditional(ref_row)
        for row in dataset_rows:
            test_entries = to_float(row, "test_entries", default=1.0)
            delta = smc_conditional(row) - ref_smc
            out.append(
                {
                    "tau": row.get("tau", ""),
                    "tau_index": row.get("tau_index", ""),
                    "data_seed": row.get("data_seed", ""),
                    "num_particles": row.get("num_particles", ""),
                    "reference_num_particles": ref_row.get("num_particles", ""),
                    "mean_angle_deg": to_float(row, "mean_angle_deg"),
                    "delta_to_reference": delta,
                    "delta_to_reference_per_entry": delta / test_entries,
                    "abs_delta_to_reference_per_entry": abs(delta) / test_entries,
                }
            )
    return sorted_rows(out)


def metric_summary(rows):
    largest_rows = rows_at_largest_particles(rows)
    conv_rows = convergence_rows(rows)
    convergence_by_particles = defaultdict(list)
    for row in conv_rows:
        convergence_by_particles[to_int(row, "num_particles")].append(to_float(row, "abs_delta_to_reference_per_entry"))

    convergence_summary = []
    for particle_count in sorted(convergence_by_particles):
        values = np.asarray(convergence_by_particles[particle_count], dtype=float)
        convergence_summary.append(
            {
                "num_particles": particle_count,
                "median_abs_delta_to_largest_particles_per_entry": float(np.median(values)),
                "p90_abs_delta_to_largest_particles_per_entry": float(np.percentile(values, 90)),
                "max_abs_delta_to_largest_particles_per_entry": float(np.max(values)),
            }
        )

    angles = np.asarray([to_float(row, "mean_angle_deg") for row in largest_rows], dtype=float)
    ekf_minus_smc = np.asarray([to_float(row, "ekf_minus_smc_per_entry") for row in largest_rows], dtype=float)
    abs_ekf_minus_smc = np.abs(ekf_minus_smc)
    oracle_minus_smc = np.asarray(
        [
            (to_float(row, "fixed_c_oracle_conditional_ll") - smc_conditional(row))
            / to_float(row, "test_entries", default=1.0)
            for row in largest_rows
        ],
        dtype=float,
    )

    if len(angles) >= 2 and np.std(angles) > 0 and np.std(abs_ekf_minus_smc) > 0:
        corr_abs_error_angle = float(np.corrcoef(angles, abs_ekf_minus_smc)[0, 1])
        slope_abs_error_angle = float(np.polyfit(angles, abs_ekf_minus_smc, 1)[0])
    else:
        corr_abs_error_angle = np.nan
        slope_abs_error_angle = np.nan

    if len(abs_ekf_minus_smc) >= 3:
        order = np.argsort(angles)
        third = max(1, len(order) // 3)
        low_rotation_abs_error = float(np.mean(abs_ekf_minus_smc[order[:third]]))
        high_rotation_abs_error = float(np.mean(abs_ekf_minus_smc[order[-third:]]))
    else:
        low_rotation_abs_error = np.nan
        high_rotation_abs_error = np.nan

    claim_rows = [
        {
            "metric": "num_largest_particle_conditions",
            "value": len(largest_rows),
        },
        {
            "metric": "ekf_minus_smc_abs_mean_per_entry_largest_particles",
            "value": float(np.mean(abs_ekf_minus_smc)) if len(abs_ekf_minus_smc) else np.nan,
        },
        {
            "metric": "ekf_minus_smc_abs_max_per_entry_largest_particles",
            "value": float(np.max(abs_ekf_minus_smc)) if len(abs_ekf_minus_smc) else np.nan,
        },
        {
            "metric": "corr_abs_ekf_minus_smc_with_mean_angle",
            "value": corr_abs_error_angle,
        },
        {
            "metric": "slope_abs_ekf_minus_smc_vs_mean_angle",
            "value": slope_abs_error_angle,
        },
        {
            "metric": "low_rotation_abs_ekf_minus_smc_mean_per_entry",
            "value": low_rotation_abs_error,
        },
        {
            "metric": "high_rotation_abs_ekf_minus_smc_mean_per_entry",
            "value": high_rotation_abs_error,
        },
        {
            "metric": "oracle_fixed_c_minus_smc_min_per_entry",
            "value": float(np.min(oracle_minus_smc)) if len(oracle_minus_smc) else np.nan,
        },
        {
            "metric": "oracle_fixed_c_minus_smc_median_per_entry",
            "value": float(np.median(oracle_minus_smc)) if len(oracle_minus_smc) else np.nan,
        },
        {
            "metric": "oracle_fixed_c_minus_smc_positive_fraction",
            "value": float(np.mean(oracle_minus_smc > 0)) if len(oracle_minus_smc) else np.nan,
        },
    ]

    return convergence_summary, claim_rows


def savefig(fig, output_dir, stem):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_smc_convergence(rows, output_dir):
    conv = convergence_rows(rows)
    if not conv:
        return None

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    particle_counts = sorted({to_int(row, "num_particles") for row in conv})
    for particle_count in particle_counts:
        values = np.asarray(
            [
                to_float(row, "abs_delta_to_reference_per_entry")
                for row in conv
                if to_int(row, "num_particles") == particle_count
            ],
            dtype=float,
        )
        xs = np.full_like(values, particle_count, dtype=float)
        ax.scatter(xs, values, color="0.6", s=18, alpha=0.45)

    medians = []
    lows = []
    highs = []
    for particle_count in particle_counts:
        values = np.asarray(
            [
                to_float(row, "abs_delta_to_reference_per_entry")
                for row in conv
                if to_int(row, "num_particles") == particle_count
            ],
            dtype=float,
        )
        medians.append(float(np.median(values)))
        lows.append(float(np.percentile(values, 25)))
        highs.append(float(np.percentile(values, 75)))
    medians = np.asarray(medians)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    ax.errorbar(
        particle_counts,
        medians,
        yerr=np.vstack([medians - lows, highs - medians]),
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="black",
        label="median [IQR]",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("SMC particles")
    ax.set_ylabel("|SMC - largest-particle SMC| / entry")
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, output_dir, "smc_particle_convergence")


def select_smc_path_rows(rows, tau_values, max_data_seeds):
    selected = []
    for tau in tau_values:
        tau_rows = [row for row in rows if matches_tau(row, tau)]
        data_seeds = sorted({to_int(row, "data_seed") for row in tau_rows})[:max_data_seeds]
        for data_seed in data_seeds:
            seed_rows = [row for row in tau_rows if to_int(row, "data_seed") == data_seed]
            selected.extend(sorted(seed_rows, key=lambda row: to_int(row, "num_particles")))
    return selected


def plot_smc_logmeanexp_particle_paths(rows, output_dir, tau_values, max_data_seeds):
    num_panels = len(tau_values)
    if num_panels == 0:
        return None, []

    fig, axes = plt.subplots(1, num_panels, figsize=(4.1 * num_panels, 3.4), sharey=True)
    if num_panels == 1:
        axes = [axes]

    selected_rows = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, tau in zip(axes, tau_values):
        tau_rows = [row for row in rows if matches_tau(row, tau)]
        data_seeds = sorted({to_int(row, "data_seed") for row in tau_rows})[:max_data_seeds]
        if not data_seeds:
            ax.text(0.5, 0.5, f"tau={tau:g}\nnot found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        for seed_index, data_seed in enumerate(data_seeds):
            seed_rows = sorted(
                [row for row in tau_rows if to_int(row, "data_seed") == data_seed],
                key=lambda row: to_int(row, "num_particles"),
            )
            selected_rows.extend(seed_rows)
            xs = np.asarray([to_int(row, "num_particles") for row in seed_rows], dtype=float)
            entries = np.asarray([to_float(row, "test_entries", default=1.0) for row in seed_rows], dtype=float)
            ys = np.asarray([smc_conditional(row) for row in seed_rows], dtype=float) / entries
            yerr = np.asarray(
                [to_float(row, "smc_conditional_se_per_entry", default=0.0) for row in seed_rows],
                dtype=float,
            )
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                linewidth=1.6,
                capsize=3,
                color=colors[seed_index % len(colors)],
                label=f"data seed {data_seed}",
            )

        ax.set_xscale("log", base=2)
        ax.set_title(f"tau={tau:g}")
        ax.set_xlabel("SMC particles")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel("RB-SMC logmeanexp conditional MLL / entry")
    fig.tight_layout()
    paths = savefig(fig, output_dir, "smc_logmeanexp_vs_particles_by_tau")
    return paths, selected_rows


def plot_ekf_degradation(rows, output_dir):
    rows = sorted_rows(rows_at_largest_particles(rows))
    if not rows:
        return None
    xs = np.asarray([to_float(row, "mean_angle_deg") for row in rows])
    ys = np.asarray([to_float(row, "ekf_minus_smc_per_entry") for row in rows])
    yerr = np.asarray([to_float(row, "smc_conditional_se_per_entry", default=0.0) for row in rows])
    order = np.argsort(xs)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True)
    axes[0].errorbar(xs[order], ys[order], yerr=yerr[order], marker="o", linewidth=1.5, capsize=3)
    axes[0].axhline(0, color="black", linewidth=1, linestyle=":")
    axes[0].set_ylabel("EKF - RB-SMC / entry")
    axes[0].set_xlabel("Mean rotation angle (deg)")

    abs_y = np.abs(ys)
    axes[1].scatter(xs, abs_y, s=32)
    if len(xs) >= 2 and np.std(xs) > 0:
        grid = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
        coef = np.polyfit(xs, abs_y, 1)
        axes[1].plot(grid, coef[0] * grid + coef[1], color="black", linewidth=1.5)
    axes[1].set_ylabel("|EKF - RB-SMC| / entry")
    axes[1].set_xlabel("Mean rotation angle (deg)")
    fig.tight_layout()
    return savefig(fig, output_dir, "ekf_smc_rotation_degradation")


def plot_oracle_advantage(rows, output_dir):
    rows = sorted_rows(rows_at_largest_particles(rows))
    if not rows:
        return None
    xs = np.asarray([to_float(row, "mean_angle_deg") for row in rows])
    entries = np.asarray([to_float(row, "test_entries", default=1.0) for row in rows])
    smc = np.asarray([smc_conditional(row) for row in rows])
    oracle = np.asarray([to_float(row, "fixed_c_oracle_conditional_ll") for row in rows])
    train_fixed = np.asarray([to_float(row, "fixed_c_train_inferred_conditional_ll") for row in rows])
    order = np.argsort(xs)

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.plot(xs[order], ((oracle - smc) / entries)[order], marker="o", linewidth=1.5, label="Oracle fixed C - RB-SMC")
    ax.plot(
        xs[order],
        ((train_fixed - smc) / entries)[order],
        marker="o",
        linewidth=1.5,
        label="Train-inferred fixed C - RB-SMC",
    )
    ax.axhline(0, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("Mean rotation angle (deg)")
    ax.set_ylabel("Conditional MLL advantage / entry")
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, output_dir, "fixed_c_advantage_over_smc")


def write_markdown_summary(path, rows, convergence_summary, claim_rows, figure_paths):
    metrics = {row["metric"]: row["value"] for row in claim_rows}
    particle_counts = sorted({to_int(row, "num_particles") for row in rows})
    tau_values = sorted({to_float(row, "tau") for row in rows})
    data_seeds = sorted({to_int(row, "data_seed") for row in rows})

    lines = [
        "# EKF/SMC MLL Result Summary",
        "",
        f"- Conditions: {len(rows)} rows",
        f"- Particle counts: {particle_counts}",
        f"- Tau values: {tau_values}",
        f"- Data seeds: {data_seeds}",
        "",
        "## Claim 1: SMC Convergence",
        "",
        "Absolute difference from the largest-particle SMC estimate, per held-out entry:",
        "",
        "| particles | median | p90 | max |",
        "|---:|---:|---:|---:|",
    ]
    for row in convergence_summary:
        lines.append(
            f"| {row['num_particles']} | "
            f"{row['median_abs_delta_to_largest_particles_per_entry']:.6g} | "
            f"{row['p90_abs_delta_to_largest_particles_per_entry']:.6g} | "
            f"{row['max_abs_delta_to_largest_particles_per_entry']:.6g} |"
        )

    lines.extend(
        [
            "",
            "## Claim 2: EKF Matches SMC, Then Degrades With Rotation",
            "",
            f"- Mean |EKF - RB-SMC| / entry at largest particle count: "
            f"{metrics['ekf_minus_smc_abs_mean_per_entry_largest_particles']:.6g}",
            f"- Max |EKF - RB-SMC| / entry at largest particle count: "
            f"{metrics['ekf_minus_smc_abs_max_per_entry_largest_particles']:.6g}",
            f"- Correlation of |EKF - RB-SMC| with mean rotation angle: "
            f"{metrics['corr_abs_ekf_minus_smc_with_mean_angle']:.6g}",
            f"- Low-rotation mean absolute error / entry: "
            f"{metrics['low_rotation_abs_ekf_minus_smc_mean_per_entry']:.6g}",
            f"- High-rotation mean absolute error / entry: "
            f"{metrics['high_rotation_abs_ekf_minus_smc_mean_per_entry']:.6g}",
            "",
            "## Claim 3: Oracle Fixed-C Conditional LL Is Higher",
            "",
            f"- Min oracle fixed-C minus RB-SMC / entry: "
            f"{metrics['oracle_fixed_c_minus_smc_min_per_entry']:.6g}",
            f"- Median oracle fixed-C minus RB-SMC / entry: "
            f"{metrics['oracle_fixed_c_minus_smc_median_per_entry']:.6g}",
            f"- Fraction of largest-particle conditions with oracle fixed-C > RB-SMC: "
            f"{metrics['oracle_fixed_c_minus_smc_positive_fraction']:.3f}",
            "",
            "## Figures",
            "",
        ]
    )
    for paths in figure_paths:
        if paths is None:
            continue
        lines.append(f"- {paths[0]}")
        lines.append(f"- {paths[1]}")
    Path(path).write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Summarize EKF/RB-SMC/fixed-C MLL validation results.")
    parser.add_argument("--results_dir", default="results/ekf_smc_mll_validation")
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--replicate_csv", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--convergence_tau_values", default="1e-5,1e-4,1e-3")
    parser.add_argument("--convergence_num_data_seeds", type=int, default=2)
    parser.add_argument("--no_recompute_smc_from_replicates", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    convergence_tau_values = parse_csv(args.convergence_tau_values, float)
    results_dir = Path(args.results_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else results_dir / "summary_results.csv"
    replicate_csv = Path(args.replicate_csv) if args.replicate_csv else results_dir / "replicate_results.csv"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "claim_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[input] summary_csv={summary_csv}", flush=True)
    print(f"[input] replicate_csv={replicate_csv}", flush=True)
    print(f"[output] output_dir={output_dir}", flush=True)

    rows = read_csv(summary_csv)
    replicate_rows = read_csv(replicate_csv) if replicate_csv.exists() else []
    if replicate_rows and not args.no_recompute_smc_from_replicates:
        rows = recompute_smc_from_replicates(
            rows,
            replicate_rows,
            num_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        recomputed_csv = output_dir / "summary_results_logmeanexp_recomputed.csv"
        fieldnames = sorted({key for row in rows for key in row.keys()})
        write_csv(recomputed_csv, fieldnames, rows)
        print(f"[output] recomputed_summary_csv={recomputed_csv}", flush=True)

    rows = sorted_rows(rows)
    convergence_summary, claim_rows = metric_summary(rows)
    write_csv(
        output_dir / "smc_particle_convergence_table.csv",
        [
            "num_particles",
            "median_abs_delta_to_largest_particles_per_entry",
            "p90_abs_delta_to_largest_particles_per_entry",
            "max_abs_delta_to_largest_particles_per_entry",
        ],
        convergence_summary,
    )
    write_csv(output_dir / "claim_metrics.csv", ["metric", "value"], claim_rows)

    smc_path_figure, smc_path_rows = plot_smc_logmeanexp_particle_paths(
        rows,
        output_dir,
        convergence_tau_values,
        args.convergence_num_data_seeds,
    )
    if smc_path_rows:
        write_csv(
            output_dir / "smc_logmeanexp_vs_particles_by_tau_rows.csv",
            [
                "tau",
                "tau_index",
                "data_seed",
                "num_particles",
                "mean_angle_deg",
                "max_angle_deg",
                "smc_conditional_logmeanexp",
                "smc_conditional_logmeanexp_per_entry",
                "smc_conditional_se_per_entry",
                "test_entries",
            ],
            [
                {
                    "tau": row.get("tau", ""),
                    "tau_index": row.get("tau_index", ""),
                    "data_seed": row.get("data_seed", ""),
                    "num_particles": row.get("num_particles", ""),
                    "mean_angle_deg": row.get("mean_angle_deg", ""),
                    "max_angle_deg": row.get("max_angle_deg", ""),
                    "smc_conditional_logmeanexp": smc_conditional(row),
                    "smc_conditional_logmeanexp_per_entry": (
                        smc_conditional(row) / to_float(row, "test_entries", default=1.0)
                    ),
                    "smc_conditional_se_per_entry": row.get("smc_conditional_se_per_entry", ""),
                    "test_entries": row.get("test_entries", ""),
                }
                for row in smc_path_rows
            ],
        )

    figure_paths = [
        smc_path_figure,
        plot_smc_convergence(rows, output_dir),
        plot_ekf_degradation(rows, output_dir),
        plot_oracle_advantage(rows, output_dir),
    ]
    write_markdown_summary(output_dir / "claim_summary.md", rows, convergence_summary, claim_rows, figure_paths)

    print(f"[output] claim_summary={output_dir / 'claim_summary.md'}", flush=True)
    print(f"[output] claim_metrics={output_dir / 'claim_metrics.csv'}", flush=True)
    for paths in figure_paths:
        if paths is not None:
            print(f"[output] figure_png={paths[0]}", flush=True)
            print(f"[output] figure_pdf={paths[1]}", flush=True)


if __name__ == "__main__":
    main()
