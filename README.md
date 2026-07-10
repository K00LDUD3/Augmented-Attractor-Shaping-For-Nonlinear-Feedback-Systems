# Augmented Attractor Shaping For Nonlinear Feedback Systems
The report containing a full description of the problem statement, methodology and results can be found [here](https://github.com/K00LDUD3/Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems/blob/main/Report.pdf)
## Take 2: Project Pillars

### Pillar 2: Nonlinear Dynamics Analysis
Modeling and analysis of nonlinear dynamical systems including Van der Pol oscillators, Duffing oscillators, and Lorenz attractors. Notebooks implement system behavior, stability properties, and state-space dynamics, using both Jupyter and Wolfram notebooks.

### Pillar 3: Control System Formulation  
Analysis of coupled and independent Lorenz systems to develop baseline formulations for feedback control. 

### Pillar 4: Classical PID Tuning & Validation
Develops and stress-tests PID controllers on the Lorenz environment. Includes optimal gain calculation, parameter tuning scripts, batch simulations, and comprehensive stress testing to establish PID baseline performance across randomized conditions.

### Pillar 5: Neural Surrogate Model Development
Trains deep neural network surrogates to approximate system dynamics. Handles dataset generation, model architecture, training/validation loops, and produces learned surrogate models for accelerated policy learning and testing.

### Pillar 6: RL Integration with Surrogate Guidance
Implements Soft Actor-Critic (SAC) agent trained in the 2-coupled Lorenz environment (other chaotic systems queued) using the neural surrogate as reward shaping guidance. Combines classical control insights with modern RL to learn adaptive feedback policies. Includes replay buffers, post-run analysis, and performance evaluation.

## Tech Stack
**Core ML & Dynamics:**
- **PyTorch** – Deep learning framework for neural surrogates and SAC agent implementation
- **NumPy** – Numerical computations and array operations
- **Matplotlib** – Visualization of dynamics and training results
- **OpenAI Gymnasium** - Custom environment & dynamics setup for SAC agent implementation

**Supporting Libraries:**
- **Pandas** – Data handling and analysis
- **scikit-learn** (implicit) – Metrics and statistical analysis

## Quick Start - Pending
Navigate to individual Pillar directories to run experiments. Each pillar is progressively integrated, with later pillars building on earlier analyses and models.

*Note: Cloning & running python files will throw multiple errors due to absolute path integration. Once the [issue](https://github.com/K00LDUD3/Augmented-Attractor-Shaping-For-Nonlinear-Feedback-Systems/issues/1) is fixed, this readme will be updated.* 
