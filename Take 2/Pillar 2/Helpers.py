import os
import json
import pandas as pd

def read_json_config(config_path: str, normalize: bool = False):
    """
    Reads a JSON config file and returns it as a dictionary.
    
    Args:
        config_path (str): Path to the JSON config file.
        normalize (bool): If True, returns a flattened pandas DataFrame.

    Returns:
        dict | pandas.DataFrame
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError("JSON config must contain a single dictionary at the top level.")

    if normalize:
        return pd.json_normalize(config)

    return config

# print(read_json_config("DVPParams.config"))

def time_format(seconds):
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    # :02 ensures at least 2 digits with a leading zero if needed
    return f"{hours:02} hrs {minutes:02} mins {secs:02.2f} secs"  # Output: 01:23:20
    