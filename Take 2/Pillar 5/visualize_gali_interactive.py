import sys
import os
import pandas as pd
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QHBoxLayout, QCheckBox
from PyQt6.QtCore import Qt

class GALIAtlasVisualizer(QMainWindow):
    def __init__(self, csv_path):
        super().__init__()
        self.setWindowTitle(f"Interactive GALI Atlas Visualizer - {os.path.basename(csv_path)}")
        self.resize(1200, 800)

        # 1. Load Data
        print(f"Loading {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # 2. Setup UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 3. Setup 3D View
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 150
        self.layout.addWidget(self.view)

        # Add Grid
        gz = gl.GLGridItem()
        gz.translate(0, 0, -40)
        self.view.addItem(gz)

        # 4. Controls
        self.control_layout = QHBoxLayout()
        
        # GALI Threshold Slider
        self.slider_label = QLabel("GALI2 Threshold: -15.00")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(-1500, 0)
        self.slider.setValue(-1500)
        self.slider.valueChanged.connect(self.update_plot)
        
        # Core/Shell Checkboxes
        self.cb_core = QCheckBox("Show Core ([-40, 40])")
        self.cb_core.setChecked(True)
        self.cb_core.stateChanged.connect(self.update_plot)
        
        self.cb_shell = QCheckBox("Show Shell ([-60, 60])")
        self.cb_shell.setChecked(True)
        self.cb_shell.stateChanged.connect(self.update_plot)

        self.control_layout.addWidget(self.slider_label)
        self.control_layout.addWidget(self.slider)
        self.control_layout.addWidget(self.cb_core)
        self.control_layout.addWidget(self.cb_shell)
        self.layout.addLayout(self.control_layout)

        # 5. Initial Plot
        self.scatter = gl.GLScatterPlotItem()
        self.view.addItem(self.scatter)
        self.update_plot()

    def update_plot(self):
        threshold = self.slider.value() / 100.0
        self.slider_label.setText(f"GALI2 Threshold (log10): {threshold:.2f}")

        # Filter data
        mask = np.log10(self.df['gali2'] + 1e-15) >= threshold
        
        if not self.cb_core.isChecked():
            mask &= (self.df['is_core'] == 0)
        if not self.cb_shell.isChecked():
            mask &= (self.df['is_core'] == 1)
            
        filtered_df = self.df[mask].sample(min(sum(mask), 50000)) # Cap for smooth interaction
        
        if len(filtered_df) == 0:
            self.scatter.setData(pos=np.empty((0,3)))
            return

        # Prepare positions (Projecting to x1, y1, z1)
        pos = filtered_df[['x1', 'y1', 'z1']].values
        
        # Colors based on GALI2
        gali_vals = np.log10(filtered_df['gali2'] + 1e-15)
        # Normalize to [0, 1] for colormap
        norm_gali = (gali_vals - gali_vals.min()) / (gali_vals.max() - gali_vals.min() + 1e-8)
        
        # Viridis-like mapping (approximate)
        colors = np.zeros((len(filtered_df), 4))
        colors[:, 0] = 1 - norm_gali # R
        colors[:, 1] = norm_gali     # G
        colors[:, 2] = 0.5           # B
        colors[:, 3] = 0.6           # A
        
        self.scatter.setData(pos=pos, color=colors, size=2, pxMode=True)

def main():
    if len(sys.argv) < 2:
        # Try to find latest run if no path provided
        exp_dir = "experiments"
        if not os.path.exists(exp_dir):
            exp_dir = "../experiments"
            
        if os.path.exists(exp_dir):
            runs = sorted([d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))], reverse=True)
            for run in runs:
                p = os.path.join(exp_dir, run, "gali_dataset.csv")
                if os.path.exists(p):
                    csv_path = p
                    break
            else:
                print("No dataset found.")
                return
        else:
            print("No experiments directory found.")
            return
    else:
        csv_path = sys.argv[1]

    app = QApplication(sys.argv)
    window = GALIAtlasVisualizer(csv_path)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
