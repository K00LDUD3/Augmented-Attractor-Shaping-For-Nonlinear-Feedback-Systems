import os
import sys
import numpy as np
import json
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

# os.environ["PYQTGRAPH_QT_LIB"] = "PyQt6"

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from baseline_pid_lorenz2 import CoupledLorenzEnv, PIDController

# --- CONFIGURATION ---
SHOW_GRID = True  # Set to False to hide the unit-size 3D mesh
# ---------------------

def main():
    # 1. Load the constrained gains from Pillar 4
    gains_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ideal_pid_gains.json")
    if os.path.exists(gains_path):
        with open(gains_path, 'r') as f:
            gains = json.load(f)
        kp, ki, kd = gains['Kp'], gains['Ki'], gains['Kd']
        # kp, ki, kd = 27.0, 0.02, 0.0
        print(f"Loaded Robust Gains: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
    else:
        kp, ki, kd = 35.0, 0.02, 0.0
        print("Warning: Gains file not found, using default soft-start values.")

    # 2. Setup Environment
    dt = 0.005
    env = CoupledLorenzEnv(k=2.5, dt=dt)
    target = np.zeros(6)

    # 3. Setup GUI
    app = pg.mkQApp("Live PID Simulation")
    view = gl.GLViewWidget()
    view.show()
    view.setWindowTitle("2-Oscillator Lorenz: Soft PID Homing (Looping)")
    view.setCameraPosition(distance=90, elevation=25, azimuth=45)

    # Add background grid and explicit axes
    view.addItem(gl.GLGridItem())
    # X-Red, Y-Green, Z-Blue
    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], [30,0,0]]), color=(1,0,0,1), width=2))
    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,30,0]]), color=(0,1,0,1), width=2))
    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,30]]), color=(0,0,1,1), width=2))

    if SHOW_GRID:
        # Unit-size 1x1x1 mesh grid (barely visible)
        grid_color = (255, 255, 255, 15) # Very faint white
        
        # XY Plane (Bottom)
        gz = gl.GLGridItem()
        gz.setSize(80, 80)
        gz.setSpacing(1, 1)
        gz.setColor(grid_color)
        view.addItem(gz)
        
        # YZ Plane (Side)
        gx = gl.GLGridItem()
        gx.setSize(80, 80)
        gx.setSpacing(1, 1)
        gx.rotate(90, 0, 1, 0)
        gx.translate(-40, 0, 0)
        gx.setColor(grid_color)
        view.addItem(gx)
        
        # XZ Plane (Side)
        gy = gl.GLGridItem()
        gy.setSize(80, 80)
        gy.setSpacing(1, 1)
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, -40, 0)
        gy.setColor(grid_color)
        view.addItem(gy)

    # Phase Space Bounding Box [-40, 40]
    box_pts = np.array([
        [-40, -40, -40], [40, -40, -40], [40, 40, -40], [-40, 40, -40], [-40, -40, -40], # Bottom
        [-40, -40, 40], [40, -40, 40], [40, 40, 40], [-40, 40, 40], [-40, -40, 40],     # Top
        [40, -40, 40], [40, -40, -40],                                                  # Side 1
        [40, 40, -40], [40, 40, 40],                                                    # Side 2
        [-40, 40, 40], [-40, 40, -40]                                                   # Side 3
    ])
    view.addItem(gl.GLLinePlotItem(pos=box_pts, color=(1,1,1,0.2), width=1, antialias=True))

    # Path curves for the two oscillators (Cyan and Orange)
    curve1 = gl.GLLinePlotItem(width=1, antialias=True, color=(1.0, 0.6, 0.0, 0.9)) 
    curve2 = gl.GLLinePlotItem(width=1, antialias=True, color=(0.0, 0.7, 1.0, 0.9)) 
    view.addItem(curve1)
    view.addItem(curve2)

    # Text Labels for starting points (Smaller font)
    font = QtGui.QFont('Helvetica', 8)
    label1 = gl.GLTextItem(color=(255, 150, 0, 255), font=font) 
    label2 = gl.GLTextItem(color=(0, 200, 255, 255), font=font) 
    view.addItem(label1)
    view.addItem(label2)

    # Static Dots for start positions
    start_dots = gl.GLScatterPlotItem(size=8, pxMode=True)
    view.addItem(start_dots)

    # --- SIMULATION STATE ---
    sim_data = {
        "state": np.zeros(6),
        "history": [], 
        "controllers": [
            PIDController(kp, ki, kd, dt, u_max=250.0),
            PIDController(kp, ki, kd, dt, u_max=250.0)
        ],
        "steps_since_reset": 0
    }

    def reset_sim():
        print("Homed or Timed out. Generating new random starting state...")
        # Random initial state in [-40, 40]
        sim_data["state"] = np.random.uniform(-40, 40, size=6)
        sim_data["history"] = [sim_data["state"].copy()]
        sim_data["steps_since_reset"] = 0
        # Reset PID internals for new run
        for pid in sim_data["controllers"]:
            pid.integral = 0.0
            pid.prev_error = 0.0
            
        # Update Labels and Dots
        s1 = sim_data["state"][0:3]
        s2 = sim_data["state"][3:6]
        label1.setData(pos=s1, text=f"  ({s1[0]:.1f}, {s1[1]:.1f}, {s1[2]:.1f})")
        label2.setData(pos=s2, text=f"  ({s2[0]:.1f}, {s2[1]:.1f}, {s2[2]:.1f})")
        
        # Set persistent dots at start positions
        dot_pos = np.array([s1, s2])
        dot_colors = np.array([[1.0, 0.6, 0.0, 1.0], [0.0, 0.7, 1.0, 1.0]])
        start_dots.setData(pos=dot_pos, color=dot_colors)
        
        label1.setVisible(True)
        label2.setVisible(True)

    reset_sim()

    def update():
        # Run 5 physics steps per update to match real-time (40fps * 5 * 0.005s = 1.0x speed)
        for _ in range(5):
            state = sim_data["state"]
            
            # Calc Errors relative to origin (Y-coord only for Lorenz control)
            e1 = target[1] - state[1]
            e2 = target[4] - state[4]
            
            # Pure PID actions
            u1 = sim_data["controllers"][0].get_action(e1)
            u2 = sim_data["controllers"][1].get_action(e2)
            
            # Step environment
            state = env.step(state, [u1, u2])
            sim_data["state"] = state
            sim_data["history"].append(state.copy())
            sim_data["steps_since_reset"] += 1

        # Keep a much smaller trail (last 80 integration steps)
        if len(sim_data["history"]) > 20:
            sim_data["history"].pop(0)

        # Update 3D Curves
        hist = np.array(sim_data["history"])
        curve1.setData(pos=hist[:, 0:3])
        curve2.setData(pos=hist[:, 3:6])

        # Labels now stay persistent until reset as requested

        # Reset conditions: distance to origin is small OR simulation has run too long
        dist = np.linalg.norm(sim_data["state"])
        if dist < 0.25 or sim_data["steps_since_reset"] > 6000: # 30 seconds max per run
             reset_sim()

    # Driving timer (40 FPS)
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(25) 

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()

if __name__ == "__main__":
    main()
