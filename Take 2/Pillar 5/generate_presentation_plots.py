import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_dataset_v2 import compute_gali2_vm, worker

# Professional, High-Contrast Palette compatible with transparency
COLORS = {
    'chaotic': '#D63031',    # Strong Red
    'transition': '#E17055', # Orange-Red
    'stable': '#00B894',     # Mint/Green
    'log': '#0984E3',        # Clear Blue
    'text': '#2D3436'        # Dark Grey (Visible on typical slides)
}

def generate_minimal_presentation_suite():
    exp_path = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42"
    df = pd.read_csv(os.path.join(exp_path, "gali_dataset.csv"))
    
    # Global plot params for transparency cleanliness
    plt.rcParams.update({
        "axes.facecolor": (0,0,0,0),
        "figure.facecolor": (0,0,0,0),
        "savefig.facecolor": (0,0,0,0),
        "axes.edgecolor": "#636E72",
        "xtick.color": "#2D3436",
        "ytick.color": "#2D3436",
        "axes.labelcolor": "#2D3436"
    })

    print("Refining Plot 1: Logarithm Justification...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear Scale (Refined - No Arrows)
    ax1.hist(df['gali2'], bins=100, color=COLORS['chaotic'], alpha=0.6)
    ax1.set_title("GALI Distribution (Linear)", fontsize=14)
    ax1.set_xlabel("SALI/GALI Value")
    ax1.set_ylabel("Frequency")
    
    # Log Scale (Refined - No Arrows)
    log_gali = np.log10(df['gali2'] + 1e-15)
    ax2.hist(log_gali, bins=100, color=COLORS['log'], alpha=0.6)
    ax2.set_title("GALI Distribution (Log10)", fontsize=14)
    ax2.set_xlabel("Log10(GALI2)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_path, "presentation_log_justification.png"), dpi=300, transparent=True)

    print("Refining Plot 2: Log10 Evolution...")
    # Identify 3 points
    p_stable = df[df['gali2'] > 0.6].iloc[0][['x1','y1','z1','x2','y2','z2']].values
    p_trans = df[(df['gali2'] > 0.1) & (df['gali2'] < 0.4)].iloc[0][['x1','y1','z1','x2','y2','z2']].values
    p_chaos = df[df['gali2'] < 1e-10].iloc[0][['x1','y1','z1','x2','y2','z2']].values
    
    def compute_gali_history(x0, t_max=12.0, steps=100):
        from scipy.integrate import solve_ivp
        from generate_dataset_v2 import coupled_lorenz_vm
        vecs = np.random.randn(2, 6)
        q, _ = np.linalg.qr(vecs.T)
        state = np.zeros(18)
        state[0:6] = x0
        state[6:12], state[12:18] = q[:, 0], q[:, 1]
        dt_step = t_max / steps
        history = []
        for _ in range(steps):
            sol = solve_ivp(coupled_lorenz_vm, (0, dt_step), state, method='RK45')
            state = sol.y[:, -1]
            w1 = state[6:12] / np.linalg.norm(state[6:12])
            w2 = state[12:18] / np.linalg.norm(state[12:18])
            sali = min(np.linalg.norm(w1 + w2), np.linalg.norm(w1 - w2))
            history.append(sali)
            state[6:12], state[12:18] = w1, w2
        return history

    h_stable = compute_gali_history(p_stable)
    h_trans = compute_gali_history(p_trans)
    h_chaos = compute_gali_history(p_chaos)
    
    plt.figure(figsize=(10, 6))
    time = np.linspace(0, 12, len(h_stable))
    plt.plot(time, h_stable, color=COLORS['stable'], label='Stable Region', linewidth=2.5)
    plt.plot(time, h_trans, color=COLORS['transition'], label='Transition Region', linewidth=2.5)
    plt.plot(time, h_chaos, color=COLORS['chaotic'], label='Chaotic Region', linewidth=2.5)
    
    plt.yscale('log')
    plt.title("Log10(GALI) evolution for different initial GALI classes", fontsize=14)
    plt.xlabel("Integration Time (Seconds)")
    plt.ylabel("GALI (Log Scale)")
    plt.axhline(1e-12, color='#636E72', linestyle='--', alpha=0.5, label='Numerical Floor')
    plt.legend(frameon=False)
    plt.grid(alpha=0.15)
    plt.savefig(os.path.join(exp_path, "presentation_min_gali_justification.png"), dpi=300, transparent=True)

    print("Refining Plot 3: Distribution Topography...")
    plt.figure(figsize=(10, 6))
    plt.hist(log_gali, bins=100, density=True, color='#636E72', alpha=0.2, histtype='stepfilled')
    plt.axvspan(np.log10(0.1), np.log10(0.4), color=COLORS['transition'], alpha=0.4, label='Target Refinement Interval')
    plt.title("Distribution of GALI stability signatures", fontsize=14)
    plt.xlabel("Stability Signature (Log10 GALI)")
    plt.ylabel("Systemic Density")
    plt.grid(alpha=0.1)
    plt.legend(frameon=False)
    plt.savefig(os.path.join(exp_path, "presentation_interval_justification.png"), dpi=300, transparent=True)

    print("Aesthetic refinements complete. Plots overridden.")

if __name__ == "__main__":
    generate_minimal_presentation_suite()
