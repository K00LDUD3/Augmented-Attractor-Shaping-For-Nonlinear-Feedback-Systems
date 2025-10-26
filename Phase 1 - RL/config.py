from torch.cuda import is_available


# Manual Overrides 
DEVICE_Override = False
run_num = "1.4"
exclusions = ["wandb"]

DEVICE = 'cuda' if (is_available() and not DEVICE_Override) else 'cpu'

class Environment:
    name = "LunarLander-v3"
    continuous = True
    gravity = -10
    enable_wind = False
    turbulence_power = 0
    wind_power = 0

class SAC:
    policy = 'MlpPolicy'
    lr = 3e-4
    buffer_size = 1_000_000
    learning_starts = 10000       # Steps before learning begins
    batch_size=256              # Samples per gradient step
    tau=0.005                   # Target smoothing coefficient
    gamma=0.99                  # Discount factor
    train_freq=(1, "step")      # Learn after every step
    gradient_steps=1            # How many gradient updates per step
    ent_coef="auto"      

class Wandb:
    run = f"forge-{run_num}" # Run name
    save_freq = 10000 # save checkpoint every 10k steps
    log_interval = 5

class config:
    n_envs = 1
    algo = "SAC"
    total_timesteps = 500_000
    device = DEVICE
    env = Environment  # Simple assignment
    sac = SAC
    wandb = Wandb

def class_to_dict(cls, exclude = []):
    """Recursively converts a class (and nested class attributes) to a dict."""
    result = {}
    for key, value in cls.__dict__.items():
        if key.startswith("__") or key in exclude:
            continue  # Skip built-in attributes and methods and manual exclusions
        if isinstance(value, type):  # If attribute is a class itself
            result[key] = class_to_dict(value)
        else:
            result[key] = value
    return result

config_dict = class_to_dict(config, exclusions)


"""
continuous determines if discrete or continuous actions (corresponding to the throttle of the engines) will be used
 with the action space being Discrete(4) or Box(-1, +1, (2,), dtype=np.float32) respectively. 
 For continuous actions, the first coordinate of an action determines the throttle of the main engine, 
 while the second coordinate specifies the throttle of the lateral boosters. 
 Given an action np.array([main, lateral]), the main engine will be turned off completely if main < 0 and
 the throttle scales affinely from 50% to 100% for 0 <= main <= 1 (in particular, the main engine doesn’t 
 work with less than 50% power). Similarly, if -0.5 < lateral < 0.5, the lateral boosters will not fire at all. 
 If lateral < -0.5, the left booster will fire, and if lateral > 0.5, the right booster will fire. 
 Again, the throttle scales affinely from 50% to 100% between -1 and -0.5 (and 0.5 and 1, respectively).

 SOURCED from https://gymnasium.farama.org/environments/box2d/lunar_lander/
"""
