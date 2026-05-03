# Augmented Attractor Shaping for Nonlinear Feedback Systems
### Project Navigation & Codebase Overview

Welcome to the repository for **Augmented Attractor Shaping**. This project explores a highly novel intersection of Chaos Theory and Deep Reinforcement Learning (RL), specifically focusing on optimizing transient trajectories using a Residual Soft Actor-Critic (SAC) architecture acting on top of a baseline PID controller. The RL agent is guided by a high-speed neural surrogate of Generalized Alignment Indices (GALI) acting as an offline topological atlas of chaos.

This repository is structured into distinct "Pillars", representing the chronological and architectural evolution of the system.

---

## Project Roadmap (The Pillars)

### `Pillar 1/` & `Pillar 2/` : Theoretical Foundations & Environment Construction
- **Scope:** Defines the core mathematical environment. Here we established the physical state-space equations for the **2-Coupled Lorenz System** and implemented the high-fidelity SciPy RK45 numerical integrations.
- **Key Files:** Initial system simulations and the formulation of the baseline PID controller.

### `Pillar 3/` & `Pillar 4/` : Stability Metrics & Constrained Baselines
- **Scope:** Evaluated traditional 1D chaos indicators (like the Maximal Lyapunov Exponent - MLE) against multi-dimensional GALI. We identified that GALI is necessary to map the non-Markovian topological spaces required for transient stability optimization.
- **Key Files:** Tuning of the baseline constrained PID gain ($K_p$), which establishes the foundational limit cycle the RL agent must optimize.

### `Pillar 5/` : Manifold Extraction & Surrogate Intelligence (The GALI Atlas)
- **Scope:** Because computing exact GALI values requires simulating an 18-dimensional variational system (which takes ~72ms per step), it strictly violates the latency requirements for 500 Hz closed-loop control. This Pillar focuses on building a sub-millisecond neural surrogate.
- **Key Files:**
  - `generate_dataset_v2.py` / `generate_dataset.py`: Multi-resolution adaptive sampling (Strategy C) scripts used to generate the 325,000-sample stability atlas.
  - `train_surrogate_v4.py`: The final production surrogate employing **Inverse-Density Distribution Matching** to resolve thin stable manifolds without mode collapse.
  - `experiments/`: Tracker registries holding the surrogate iteration data.
- **Outcome:** An ONNX-exported neural model capable of evaluating the identical stability landscape in just 0.086 ms.

### `Pillar 6/` : Residual RL Control (Hybrid Architecture)
- **Scope:** The construction and tuning of the Soft Actor-Critic (SAC) agent that acts residually ($u(t) = u_{PID}(t) + u_{RL}(t)$). 
- **Key Features:**
  - **Multi-Timescale Decoupling:** The PID handles micro-stabilization at 500 Hz, while the RL agent updates at 50 Hz to allow the physical system time to manifest perturbations.
  - **LayerNorm Standardization:** Restoring gradient sensitivity to micro-actions ($10\% - 20\%$ of $U_{max}$).
  - **Control-Augmented Spectral Submanifold (caSSM) Encoder:** Aligning the encoder's latent space using Trace Overlap Loss and SOAP optimizers.
- **Key Files:**
  - `lorenz_env.py`: The custom Gym environment marrying the PID, RL, and ONNX GALI surrogate.
  - `train_sac.py`: The main RL training loop orchestrating the multi-timescale updates and reward logic ($L_1$ absolute effort penalty + proximity convergence).
  - `hyperparameter_deltas.md`: The rigorous log of hyperparameter economics and physical reasoning behind the 15 tuned deltas.

---

## How to Read the Code
For the physics and control theory:
1. Start with the **LaTeX Project Report** (`project_report.tex` / `.pdf`) at the root. It provides the comprehensive theoretical derivations, surrogate evaluations, and future directions.
2. Review **Pillar 5** (`pillar5_report.md` and `train_surrogate_v4.py`) to understand how we bypassed the computational bottleneck of the 18D variational equations using Inverse-Density neural networks.
3. Review **Pillar 6** (`lorenz_env.py` and `hyperparameter_deltas.md`) to see the final reward function economics and the multi-timescale agent implementation.

## Tech Stack
- **Deep Learning / RL:** PyTorch, ONNXRuntime
- **Numerical Integration:** SciPy (solve_ivp RK45)
- **Data Generation:** Latin Hypercube Sampling (LHS), Multi-processing (`concurrent.futures`)
- **Analysis:** NumPy, Pandas, Matplotlib
