import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
metrics_file = os.path.join(base_dir, "logs", "post_run_individual_metrics.json")
output_file = os.path.join(base_dir, "logs", "visualizations", "effort_distribution_zero_scaled.png")

print(f"Reading metrics from: {metrics_file}")
with open(metrics_file, 'r') as f:
    data = json.load(f)

pid_efforts = [d['Metrics_PID']['total_effort'] for d in data]
rl_efforts = [d['Metrics_RL']['total_effort'] for d in data]

# Calculate and print means & delta
pid_mean = np.mean(pid_efforts)
rl_mean = np.mean(rl_efforts)
delta_pct = ((rl_mean - pid_mean) / pid_mean) * 100

print(f"\n--- EFFORT STATISTICS ---")
print(f"PID Mean Effort: {pid_mean:.2f}")
print(f"RL Mean Effort:  {rl_mean:.2f}")
print(f"Delta (RL vs PID): {delta_pct:+.2f}%\n")

# Recreate the exact Matplotlib Style from post_run_analysis.py
fig, ax = plt.subplots(figsize=(6, 6))

ax.boxplot([pid_efforts, rl_efforts], tick_labels=["Pure PID", "Hybrid RL"])
ax.set_title("Distribution: Total Effort")
ax.set_ylabel("total_effort")
ax.grid(True, alpha=0.3)

# Start the y-axis at 0 as requested
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(output_file, dpi=500)
print(f"Saved new zero-scaled effort plot to: {output_file}")
plt.close()

# --- EXPORT DETAILED STATISTICS TO JSON ---
def get_stats(arr):
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "mean": float(np.mean(arr)),
        "median": float(median),
        "std_dev": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "q1_25th": float(q1),
        "q3_75th": float(q3),
        "iqr_spread": float(q3 - q1),
        "total_range": float(np.max(arr) - np.min(arr))
    }

pid_stats = get_stats(pid_efforts)
rl_stats = get_stats(rl_efforts)

def calc_delta(val_rl, val_pid):
    return float(((val_rl - val_pid) / val_pid) * 100) if val_pid != 0 else 0.0

comparison_stats = {
    "PID_Effort_Statistics": pid_stats,
    "RL_Effort_Statistics": rl_stats,
    "Deltas_Pct_RL_vs_PID": {
        "mean_delta_pct": calc_delta(rl_stats["mean"], pid_stats["mean"]),
        "median_delta_pct": calc_delta(rl_stats["median"], pid_stats["median"]),
        "max_outlier_reduction_pct": calc_delta(rl_stats["max"], pid_stats["max"]),
        "iqr_spread_reduction_pct": calc_delta(rl_stats["iqr_spread"], pid_stats["iqr_spread"]),
        "std_dev_reduction_pct": calc_delta(rl_stats["std_dev"], pid_stats["std_dev"])
    }
}

json_out_file = os.path.join(base_dir, "logs", "visualizations", "effort_statistics.json")
with open(json_out_file, 'w') as f:
    json.dump(comparison_stats, f, indent=4)
    
print(f"Exported detailed effort statistics to: {json_out_file}")
