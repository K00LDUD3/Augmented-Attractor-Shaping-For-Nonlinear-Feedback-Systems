import os
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
metrics_file = os.path.join(base_dir, "logs", "post_run_individual_metrics.json")

print(f"Reading individual metrics from: {metrics_file}")
with open(metrics_file, 'r') as f:
    data = json.load(f)

# 1. Calculate the mathematical average of the individual percentage deltas
deltas = [d['Deltas_pct']['total_effort'] for d in data]
avg_of_deltas = sum(deltas) / len(deltas)

print(f"\n--- STATISTICAL DIFFERENCE ---")
print(f"The 'Average of the Individual Percentage Deltas' is: {avg_of_deltas:+.2f}%")
print(f"Note: This is mathematically different from the 'Percentage Delta of the Averages' (-22.10%).")
print(f"      (Jensen's Inequality: The expectation of a ratio is not the ratio of the expectations).")

# 2. Sort to find the best performing individual runs (most negative delta)
sorted_runs = sorted(data, key=lambda x: x['Deltas_pct']['total_effort'])

TOP_N = 10
print(f"\n--- TOP {TOP_N} BEST EFFORT REDUCTIONS (Individual ICs) ---")
for i in range(TOP_N):
    run = sorted_runs[i]
    print(f"IC Index {run['IC_index']}:")
    print(f"  PID Effort: {run['Metrics_PID']['total_effort']:>10.2f}")
    print(f"  RL Effort:  {run['Metrics_RL']['total_effort']:>10.2f}")
    print(f"  Reduction:  {run['Deltas_pct']['total_effort']:>+10.2f}%")
    print("-" * 35)
