import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze():
    # Path to the 325k production dataset
    csv_path = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_dataset.csv"
    output_img = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_distribution.png"
    
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Transformation to Log Stability Space
    log_gali = np.log10(df['gali2'] + 1e-15)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(log_gali, bins=100, color='#2c3e50', alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Highlight the SSM Chaos Edge [0.1, 0.4] in Log Space
    # log10(0.1) = -1.0, log10(0.4) = -0.39
    plt.axvspan(np.log10(0.1), np.log10(0.4), color='orange', alpha=0.3, label='SSM Transition Zone')
    
    plt.title(f"GALI Stability Distribution (N={len(df):,})", fontsize=14, weight='bold')
    plt.xlabel("Log10(GALI2) - Stability Index", fontsize=12)
    plt.ylabel("Point Density", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()
    
    plt.savefig(output_img, dpi=300)
    print(f"Histogram saved to: {output_img}")
    
    # Statistical Breakdown
    total = len(df)
    divergent = (df['gali2'] <= 0.1).sum()
    chaotic = ((df['gali2'] > 0.1) & (df['gali2'] < 0.4)).sum()
    stable = (df['gali2'] >= 0.4).sum()
    
    print("\n--- Stability Population Statistics ---")
    print(f"Total Samples: {total:,}")
    print(f"Divergent/Collapsed Layer (<0.1) : {divergent:,} ({divergent/total*100:.2f}%)")
    print(f"SSM Transition Zone (0.1-0.4)   : {chaotic:,} ({chaotic/total*100:.2f}%)")
    print(f"Strict Stable/Periodic (>0.4)  : {stable:,} ({stable/total*100:.2f}%)")

if __name__ == "__main__":
    analyze()
