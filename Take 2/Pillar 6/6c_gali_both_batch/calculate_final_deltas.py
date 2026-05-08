import json

with open("/home/tanush/Repos/Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems/Take 2/Pillar 6/6c_gali_both_batch/2026-04-14_13-34-07_13a78838c9/logs/post_run_individual_metrics_del-13.1.json", 'r') as f:
    data = json.load(f)

effort_deltas = [d["Deltas_pct"]["total_effort"] for d in data]
error_deltas = [d["Deltas_pct"]["final_error"] for d in data]

mean_effort = sum(effort_deltas) / len(effort_deltas)
mean_error = sum(error_deltas) / len(error_deltas)

print(f"Mean Effort Delta: {mean_effort:.2f}%")
print(f"Mean Error Delta: {mean_error:.2f}%")
