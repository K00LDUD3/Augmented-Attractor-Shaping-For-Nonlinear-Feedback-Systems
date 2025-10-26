import json
import os

def save_config_to_json(config_dict, filename):
    """
    Saves a configuration dictionary to a JSON file.

    Args:
        config_dict (dict): The configuration dictionary.
        filename (str): The desired output filename (e.g. 'config.json').
    """
    # Ensure .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    filename = "configs//"+ filename
    
    
    # Optionally ensure directory exists
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Save in human-readable format
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(f"Config saved to '{filename}'")
