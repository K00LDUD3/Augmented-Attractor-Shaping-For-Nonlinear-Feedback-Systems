import os
import json
import numpy as np

def generate_summary():
    root = r'Pillar 4\experiments\Drive_Upload_PID_KP30'
    if not os.path.exists(root):
        print(f"Error: Path {root} does not exist.")
        return

    runs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    all_metrics = []

    for run in runs:
        m_path = os.path.join(run, 'metrics.json')
        if os.path.exists(m_path):
            try:
                with open(m_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_metrics.append({'uid': os.path.basename(run), **data})
            except Exception as e:
                print(f"Skipping {run}: {e}")

    if not all_metrics:
        print("Error: No valid metrics found.")
        return

    errors = [m['final_error'] for m in all_metrics]
    efforts = [m['total_effort'] for m in all_metrics]

    stats = {
        'total_runs': len(all_metrics),
        'avg_error': float(np.mean(errors)),
        'max_error': float(np.max(errors)),
        'avg_effort': float(np.mean(efforts)),
        'success_rate': float(np.sum(np.array(errors) < 0.2) / len(errors)) * 100.0
    }

    # Generate Plaintext Markdown summary
    md = [
        "# Pillar 4 Stress Test Batch Summary (Kp=30.0)",
        f"Location: {root}",
        "",
        "## Aggregate Statistics",
        f"- Total Randomized Runs: {stats['total_runs']}",
        f"- Average Settling Error: {stats['avg_error']:.6f}",
        f"- Maximum Deviation: {stats['max_error']:.6f}",
        f"- Average Control Effort: {stats['avg_effort']:.2f}",
        f"- Stabilization Success Rate: {stats['success_rate']:.1f}% (Threshold < 0.2)",
        "",
        "## Run Breakdown",
        "| Run UID | Final Error | Actuator Effort |",
        "| :--- | :--- | :--- |"
    ]
    
    for m in all_metrics:
        uid = m['uid']
        err = m['final_error']
        eff = m['total_effort']
        md.append(f"| {uid} | {err:.6f} | {eff:.2f} |")

    with open(os.path.join(root, 'batch_summary.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(md))

    with open(os.path.join(root, 'batch_summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'summary': stats, 'runs': all_metrics}, f, indent=4)

    print(f"SUCCESS: Batch summary generated at {os.path.join(root, 'batch_summary.md')}")

if __name__ == "__main__":
    generate_summary()
