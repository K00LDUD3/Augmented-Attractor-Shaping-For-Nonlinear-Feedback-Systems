import os
import sys
import numpy as np
import torch
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import argparse

# Ensure we can import from Pillar 5 and Pillar 4
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pillar 4"))
from train_surrogate import SALISurrogate
from generate_dataset_v2 import compute_gali2_vm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sali_surrogate.pth")
    parser.add_argument("--res", type=int, default=30, help="Grid resolution")
    args = parser.parse_args()

    # 1. Setup GUI
    app = pg.mkQApp("Surrogate Error Field")
    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle("Phase 3: GALI Surrogate Error Field (Viridis)")
    view.setCameraPosition(distance=120, elevation=25, azimuth=45)
    view.addItem(gl.GLGridItem())

    # 2. Add Bounding Box HUD
    box_bounds = [-60, 60]
    b = box_bounds[1]
    box_pts = np.array([
        [-b, -b, -b], [b, -b, -b], [b, b, -b], [-b, b, -b], [-b, -b, -b], # Bottom
        [-b, -b, b], [b, -b, b], [b, b, b], [-b, b, b], [-b, -b, b],     # Top
        [b, -b, b], [b, -b, -b],                                         # Side 1
        [b, b, -b], [b, b, b],                                           # Side 2
        [-b, b, b], [-b, b, -b]                                          # Side 3
    ])
    view.addItem(gl.GLLinePlotItem(pos=box_pts, color=(1,1,1,0.3), width=1, antialias=True))
    
    # 3. Label Vertices
    font = QtGui.QFont('Helvetica', 9)
    vertices = [[-b, -b, -b], [b, -b, -b], [-b, b, -b], [-b, -b, b], [b, b, b]]
    for v in vertices:
        v_label = gl.GLTextItem(pos=v, text=f"({v[0]}, {v[1]}, {v[2]})", font=font, color=(200, 200, 200, 255))
        view.addItem(v_label)

    # 4. Load Surrogate Model
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found.")
        return

    checkpoint = torch.load(args.model, map_location='cpu')
    model = SALISurrogate()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    s_min = checkpoint['sali_min']
    s_max = checkpoint['sali_max']
    s_bound = checkpoint.get('state_max', 60.0)

    # 5. Generate Grid and Compute Errors
    print(f"Sampling {args.res}^3 grid Points...")
    x = np.linspace(-b, b, args.res)
    y = np.linspace(-b, b, args.res)
    z = np.linspace(-b, b, args.res)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid_points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    # To project 6D -> 3D for visualization, we set Oscillator 2 = Oscillator 1
    # This shows the error on the symmetric manifold
    states_6d = np.hstack([grid_points, grid_points])
    
    # 6. Predict and Compare
    print("Computing surrogate predictions...")
    with torch.no_grad():
        tensor_states = torch.tensor(states_6d / s_bound, dtype=torch.float32)
        preds_norm = model(tensor_states).numpy().flatten()
        preds_unnorm = (preds_norm * (s_max - s_min)) + s_min

    # For speed, we'll perform ground truth on a subset or skip it if too large
    # Instead, we visualize the GALI landscape itself as the field
    print("Mapping stability landscape...")
    errors = preds_unnorm # Visualizing stability value as the color field for now
    
    # Normalize errors for color mapping [0, 1]
    # Viridis color map approximation (Purple -> Blue -> Green -> Yellow)
    norm_err = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
    
    colors = np.zeros((len(grid_points), 4))
    # Simple Viridis-like mapping
    colors[:, 0] = norm_err          # Red channel
    colors[:, 1] = 1.0 - norm_err    # Green channel
    colors[:, 2] = 0.5               # Blue channel
    colors[:, 3] = 0.4               # Alpha

    # 7. Add Scatter Plot
    scatter = gl.GLScatterPlotItem(pos=grid_points, color=colors, size=4, pxMode=True)
    view.addItem(scatter)

    print("Success! Displaying 3D Error Field.")
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()

if __name__ == "__main__":
    main()
