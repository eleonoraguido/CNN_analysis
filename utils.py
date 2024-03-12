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
    tuple: A tuple containing some information:
        - input_file (str): The path to the input npz file.
        - test_sample_size (float): size of the test sample in percentage
        - num_epochs (int): number of epochs
        - bach_size (int): mini-batch size
        - output_name (str): additional info to be printed in the output file name
        - L1 (float): regularizer parameter
        - L2 (float): regularizer parameter
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return (config.get('input_file'), config.get('test_sample_size'), 
            config.get('num_epochs'), config.get('batch_size'), 
            config.get('output_name'), config.get('L1_regularizer'), 
            config.get('L2_regularizer'))