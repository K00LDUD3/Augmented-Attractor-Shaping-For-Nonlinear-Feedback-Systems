"""
replot_visible.py
-----------------
Regenerates all static PNG visualizations for a specific run using
white-background-friendly colors. Reads all data from the run's saved JSON
logs — no re-simulation is performed.

Target run: 6c_gali_both_batch/2026-04-14_13-34-07_13a78838c9

Usage:
    python replot_visible.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Color palette (visible on white backgrounds) ─────────────────────────────
C_PID        = '#1f4e79'   # Dark navy   — PID baseline solid
C_PID_LIGHT  = '#a9c4e4'   # Steel blue  — PID baseline dashed / secondary
C_RL         = '#c00000'   # Dark red    — RL / Hybrid solid
C_RL_LIGHT   = '#f4a58a'   # Salmon      — RL Oscillator 2 dashed / secondary
C_ACTION     = '#d07000'   # Dark orange — RL residual thrust
C_TOTAL      = '#5a2d82'   # Deep purple — Total hybrid signal
C_TARGET     = '#cc0000'   # Red star    — Origin target
C_START_O1   = '#217a21'   # Forest green — O1 start point
C_START_O2   = '#8b008b'   # Dark magenta — O2 start point
# ─────────────────────────────────────────────────────────────────────────────

RUN_DIR = os.path.dirname(os.path.abspath(__file__))

def load_trajectories():
    """Load pre-saved trajectory data from the run's log files."""
    master_ics_path = os.path.join(RUN_DIR, "logs", "master_fixed_ics.json")
    pid_baseline_path = os.path.join(RUN_DIR, "logs", "pid_baseline.json")
    eval_history_path = os.path.join(RUN_DIR, "logs", "eval_history.json")
    metrics_path = os.path.join(RUN_DIR, "logs", "post_run_individual_metrics.json")

    with open(master_ics_path, 'r') as f:
        master_data = json.load(f)
    ics = np.array(master_data["master_ics"])

    with open(metrics_path, 'r') as f:
        individual_metrics = json.load(f)

    n_eval = len(individual_metrics)
    print(f"Found {n_eval} evaluated ICs from post_run_individual_metrics.json")

    return ics, n_eval, individual_metrics


def load_numerical_summaries():
    """Load the per-IC windowed interval data from the Numerical_info folder."""
    num_dir = os.path.join(RUN_DIR, "logs", "visualizations", "Numerical_info")
    if not os.path.exists(num_dir):
        print(f"  [WARN] Numerical_info directory not found at {num_dir}. Skipping interval-based plots.")
        return None

    summaries = []
    # Only load the _summary.json files — the raw numerical_ic_XX.json files have a different schema
    files = sorted([f for f in os.listdir(num_dir) if f.endswith('_summary.json')])
    for fname in files:
        with open(os.path.join(num_dir, fname), 'r') as f:
            summaries.append(json.load(f))
    print(f"Loaded {len(summaries)} numerical IC summaries.")
    return summaries



def plot_metrics_distribution(metrics, out_dir):
    """Boxplots for total_effort, final_error, itwae — white-bg friendly."""
    keys = ["total_effort", "final_error", "itwae"]
    labels = ["Total Effort", "Final Error", "ITWAE"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (key, label) in enumerate(zip(keys, labels)):
        pid_vals = [m['Metrics_PID'][key] for m in metrics]
        rl_vals  = [m['Metrics_RL'][key]  for m in metrics]

        bp = axes[i].boxplot(
            [pid_vals, rl_vals],
            tick_labels=["Pure PID", "Hybrid RL"],
            patch_artist=True,
            widths=0.5,
        )
        bp['boxes'][0].set_facecolor('#d0e4f7')
        bp['boxes'][0].set_edgecolor(C_PID)
        bp['boxes'][1].set_facecolor('#ffd6cc')
        bp['boxes'][1].set_edgecolor(C_RL)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(2)

        axes[i].set_title(f"Distribution: {label}", fontsize=13)
        axes[i].set_ylabel(key, fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(bottom=0)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "metrics_distribution_replot.png")
    plt.savefig(out_path, dpi=500, facecolor='white')
    plt.close()
    print(f"  -> Saved: {out_path}")


def plot_interval_diagnostics(summaries, out_dir):
    """Per-IC time-windowed error and effort line plots from Numerical_info JSONs."""
    diag_dir = os.path.join(out_dir, "diagnostic_trajectories")
    os.makedirs(diag_dir, exist_ok=True)

    for s in summaries:
        ic_idx = s["IC_Index"]
        intervals = s["One_Second_Intervals"]

        time_labels  = [iv["Interval_S"].split('-')[0] for iv in intervals]
        pid_error    = [iv["Avg_Error"]["PID"]    for iv in intervals]
        rl_error     = [iv["Avg_Error"]["RL"]     for iv in intervals]
        pid_effort   = [iv["Effort_Sum"]["PID"]   for iv in intervals]
        hyb_effort   = [iv["Effort_Sum"]["Hybrid"] for iv in intervals]
        rl_resid     = [iv["Effort_Sum"]["RL_Resid"] for iv in intervals]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Windowed Diagnostics (IC {ic_idx})", fontsize=14)

        # Error panel
        axes[0].plot(time_labels, pid_error, color=C_PID,   linewidth=2, marker='o', label='PID Baseline')
        axes[0].plot(time_labels, rl_error,  color=C_RL,    linewidth=2, marker='s', label='Hybrid RL')
        axes[0].set_title("Average Tracking Error (1s windows)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Mean Abs Error")
        axes[0].set_ylim(bottom=0)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)

        # Effort panel
        axes[1].plot(time_labels, pid_effort,  color=C_PID,    linewidth=2, marker='o', label='PID Signal')
        axes[1].plot(time_labels, hyb_effort,  color=C_TOTAL,  linewidth=2, marker='s', label='Total Hybrid Signal')
        axes[1].plot(time_labels, rl_resid,    color=C_ACTION, linewidth=1.5, marker='^', linestyle='--', label='RL Residual Only')
        axes[1].set_title("Actuator Effort (1s windows)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Effort Sum")
        axes[1].set_ylim(bottom=0)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        out_path = os.path.join(diag_dir, f"diagnostic_ic_{ic_idx}_replot.png")
        plt.savefig(out_path, dpi=300, facecolor='white')
        plt.close()

    print(f"  -> Saved {len(summaries)} diagnostic replot(s) to {diag_dir}")






def plot_per_ic_delta_bars(metrics, out_dir):
    """Bar chart of per-IC effort reduction percentages, sorted."""
    deltas  = [m['Deltas_pct']['total_effort'] for m in metrics]
    ic_idxs = [m['IC_index'] for m in metrics]

    sorted_pairs = sorted(zip(deltas, ic_idxs), key=lambda x: x[0])
    sorted_deltas = [p[0] for p in sorted_pairs]
    sorted_labels = [f"IC {p[1]}" for p in sorted_pairs]

    colors = [C_RL if d < 0 else C_ACTION for d in sorted_deltas]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_deltas) * 0.4), 6))
    bars = ax.bar(range(len(sorted_deltas)), sorted_deltas, color=colors, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1.2)
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=7)
    ax.set_title("Per-IC Effort Reduction (% Delta: RL vs PID)", fontsize=13)
    ax.set_ylabel("% Change in Total Effort (negative = improvement)")
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "per_ic_effort_delta_replot.png")
    plt.savefig(out_path, dpi=300, facecolor='white')
    plt.close()
    print(f"  -> Saved: {out_path}")


if __name__ == "__main__":
    viz_dir = os.path.join(RUN_DIR, "logs", "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"\n=== Replotting visualizations for run: {RUN_DIR} ===\n")

    ics, n_eval, individual_metrics = load_trajectories()

    print("\n[1/4] Plotting metrics distribution boxplots...")
    plot_metrics_distribution(individual_metrics, viz_dir)

    print("\n[2/4] Plotting per-IC effort delta bar chart...")
    plot_per_ic_delta_bars(individual_metrics, viz_dir)

    summaries = load_numerical_summaries()
    if summaries:
        print("\n[3/3] Plotting windowed diagnostic timelines...")
        plot_interval_diagnostics(summaries, viz_dir)
    else:
        print("\n[3/3] Skipping interval diagnostics (no Numerical_info data found).")

    print("\n=== All replots complete ===")
    print("NOTE: For kinematic 3D trajectory plots, re-run post_run_analysis.py with the color-fixed version.")
