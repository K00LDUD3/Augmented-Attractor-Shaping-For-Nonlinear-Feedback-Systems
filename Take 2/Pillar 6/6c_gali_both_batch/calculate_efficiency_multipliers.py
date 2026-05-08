import os
import json
import numpy as np
import pandas as pd

def calculate_efficiency_stats(run_dir):
    num_info_dir = os.path.join(run_dir, "logs", "visualizations", "Numerical_info")
    if not os.path.exists(num_info_dir):
        print(f"Error: Numerical_info directory not found at {num_info_dir}")
        return

    # Find all summary JSON files
    json_files = sorted([f for f in os.listdir(num_info_dir) if f.endswith("_summary.json")])
    
    ic_results = []

    print(f"Processing {len(json_files)} IC summaries...")

    for fname in json_files:
        with open(os.path.join(num_info_dir, fname), 'r') as f:
            data = json.load(f)
        
        ic_idx = data["IC_Index"]
        eps = data["Episode_Summary"]
        intervals = data["One_Second_Intervals"]

        # 1. Episode-wide Multiplier
        pid_total_eff = eps["PID_Baseline"]["total_effort"]
        hybrid_total_eff = eps["Hybrid_RL"]["total_effort"]
        rl_total_resid = eps["Hybrid_RL"]["total_rl_effort"]

        # Calculation: (PID_Baseline - Hybrid_Total) / RL_Residual
        if rl_total_resid > 1e-6:
            episode_em = (pid_total_eff - hybrid_total_eff) / rl_total_resid
        else:
            episode_em = 0.0

        # 2. Steady-State Multiplier (Final Interval: 9.0-10.0s)
        last_iv = intervals[-1]
        pid_ss_eff = last_iv["Effort_Sum"]["PID"]
        hybrid_ss_eff = last_iv["Effort_Sum"]["Hybrid"]
        rl_ss_resid = last_iv["Effort_Sum"]["RL_Resid"]

        if rl_ss_resid > 1e-6:
            steady_state_em = (pid_ss_eff - hybrid_ss_eff) / rl_ss_resid
        else:
            steady_state_em = 0.0

        ic_results.append({
            "IC_Index": ic_idx,
            "Episode_Efficiency_Multiplier": episode_em,
            "Steady_State_Efficiency_Multiplier": steady_state_em,
            "PID_Total_Effort": pid_total_eff,
            "Hybrid_Total_Effort": hybrid_total_eff,
            "RL_Residual_Total": rl_total_resid
        })

    # Create DataFrame for aggregation
    df = pd.DataFrame(ic_results)
    
    # Calculate Aggregates
    summary_stats = {
        "Episode_EM": {
            "mean": df["Episode_Efficiency_Multiplier"].mean(),
            "std": df["Episode_Efficiency_Multiplier"].std(),
            "median": df["Episode_Efficiency_Multiplier"].median(),
            "min": df["Episode_Efficiency_Multiplier"].min(),
            "max": df["Episode_Efficiency_Multiplier"].max()
        },
        "Steady_State_EM": {
            "mean": df["Steady_State_Efficiency_Multiplier"].mean(),
            "std": df["Steady_State_Efficiency_Multiplier"].std(),
            "median": df["Steady_State_Efficiency_Multiplier"].median(),
            "min": df["Steady_State_Efficiency_Multiplier"].min(),
            "max": df["Steady_State_Efficiency_Multiplier"].max()
        }
    }

    # Display Results
    print("\n" + "="*50)
    print("EFFICIENCY MULTIPLIER AGGREGATE RESULTS")
    print("="*50)
    print(f"Episode EM      : {summary_stats['Episode_EM']['mean']:.4f} \u00b1 {summary_stats['Episode_EM']['std']:.4f}")
    print(f"Steady-State EM : {summary_stats['Steady_State_EM']['mean']:.4f} \u00b1 {summary_stats['Steady_State_EM']['std']:.4f}")
    print(f"Max Episode EM  : {summary_stats['Episode_EM']['max']:.4f}")
    print(f"Min Episode EM  : {summary_stats['Episode_EM']['min']:.4f}")
    print("-" * 50)
    
    # Save to files
    df.to_csv(os.path.join(run_dir, "logs", "efficiency_multipliers_per_ic.csv"), index=False)
    with open(os.path.join(run_dir, "logs", "efficiency_multipliers_summary.json"), 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    print(f"Detailed CSV saved to: logs/efficiency_multipliers_per_ic.csv")
    print(f"Summary JSON saved to: logs/efficiency_multipliers_summary.json")

if __name__ == "__main__":
    # Path to the specific run directory
    RUN_DIR = os.path.dirname(os.path.abspath(__file__))
    calculate_efficiency_stats(RUN_DIR)
