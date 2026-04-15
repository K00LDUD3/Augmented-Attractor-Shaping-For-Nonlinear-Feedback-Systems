import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_manifold():
    csv_path = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\gali_dataset.csv"
    output_img = r"Pillar 5\experiments\2026-04-06_02-10-39_0a402e42\stability_manifold_3d.png"
    
    print(f"Loading dataset for 3D render...")
    df = pd.read_csv(csv_path)
    
    # Sample 10,000 points for clear visualization without clutter
    plot_df = df.sample(10000)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use Log10(GALI) for color intensity
    sali_c = np.log10(plot_df['gali2'] + 1e-15)
    
    sc = ax.scatter(plot_df['x1'], plot_df['y1'], plot_df['z1'], 
                    c=sali_c, cmap='magma', s=5, alpha=0.6, 
                    edgecolor='none', antialiased=True)
    
    cbar = plt.colorbar(sc, pad=0.1, shrink=0.5)
    cbar.set_label('Log10(GALI2) - Stability Signature', fontsize=12)
    
    # Modern transparent aesthetics
    ax.set_title("3D GALI Stability Manifold (2-Coupled Lorenz)", fontsize=16, weight='bold', pad=20)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X1 (Primary Oscillator)', fontsize=10)
    ax.set_ylabel('Y1', fontsize=10)
    ax.set_zlabel('Z1', fontsize=10)
    
    # Rotate for most dramatic view of the butterfly wings
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(output_img, dpi=300, transparent=False, bbox_inches='tight')
    print(f"3D Manifold saved to: {output_img}")

if __name__ == "__main__":
    generate_manifold()
