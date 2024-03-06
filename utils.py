import json
import numpy as np

def read_config(config_file):
    """
    Read input file path from a JSON configuration file.

    Parameters:
    ------------
        - config_file (str): The path to the JSON configuration file.

    Returns:
    ------------
    tuple: A tuple containing the input and output file paths:
        - input_file (str): The path to the input npz file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('input_file')