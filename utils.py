"""
Author: Eleonora Guido
Last modification date: 05.2024
Photon search with a CNN
"""

import sys, os
import json
import numpy as np
from keras.models import model_from_json
from tensorflow import keras as K
from config import PDF_SAVE_PATH, LOSS_FUNCTION, OPTIMIZER
from create_model import compile_model


def read_config_type(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('config_type')



def read_config_data(config_file):
    """
    Read input file path from a JSON configuration file.

    Parameters:
    ------------
        - config_file (str): The path to the JSON configuration file.

    Returns:
    ------------
    tuple: A tuple containing some information:
        - input_file (str): The path to the input npz file.
        - partitioning_method (str): Method used to partition the dataset ("k_fold" or "split").
        - partitioning_param (float): Parameter associated with the partitioning method (number of folds or size of test data set).
        - sel_threshold (float): Chosen threshold for selection.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    input_file = config.get('input_file')
    partitioning_method = config.get('partitioning_method')
    partitioning_param = config.get('partitioning_param')
    sel_threshold = config.get('sel_threshold')

    return input_file, partitioning_method, partitioning_param, sel_threshold




def read_config_settings(config_file):
    """
    Read input file path from a JSON configuration file.

    Parameters:
    ------------
        - config_file (str): The path to the JSON configuration file.

    Returns:
    --------
    tuple
        A tuple containing the following information:
        - num_epochs : int
            Number of epochs for training.
        - batch_size : int
            Batch size for training.
        - output_name : str
            Output name for saving trained model.
        - L1_regularizer : float
            L1 regularization parameter.
        - L2_regularizer : float
            L2 regularization parameter.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return (config.get('num_epochs'), config.get('batch_size'), config.get('output_name'), 
            config.get('L1_regularizer'), config.get('L2_regularizer'))




def load_model(config_file, fold=None):
    """
    Read input file path from a JSON configuration file.

    Parameters:
    ------------
        - config_file (str): The path to the JSON configuration file.
        - fold (int, optional): The fold number (only in the case case "k_fold"). Defaults to None.

    Returns:
    ------------
    loaded_model : tf.keras.Model
        The loaded compiled Keras model.

    Raises:
    -------
    FileNotFoundError:
        If either the JSON or H5 file does not exist.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    input_arg = config.get('model_name')

    # Construct file names with fold number, if provided
    if fold is not None:
        input_arg = input_arg.replace('*', str(fold))

    h5_file_name = os.path.join(PDF_SAVE_PATH, input_arg + ".h5")
    json_file_name = os.path.join(PDF_SAVE_PATH, input_arg + ".json")

    print("\nStart loading the model files:")
    # Check if the .h5 file exists
    if os.path.exists(h5_file_name):
        # If the .h5 file exists, proceed to check for the .json file
        if os.path.exists(json_file_name):
            # If both files exist, proceed with processing
            with open(json_file_name, 'r') as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            # Load weights into the new model
            loaded_model.load_weights(h5_file_name)
            print(f"JSON file '{json_file_name}'.json has been read")
            print(f"H5 file '{h5_file_name}'.h5 has been read")
            print("----> Model is loaded from disk.")
            # Compile the model
            compile_model(loaded_model, loss = LOSS_FUNCTION, optimizer = OPTIMIZER)
            print("----> Model is compiled.\n")
            return loaded_model
        else:
            print(f"Error: JSON file '{json_file_name}' does not exist.")
            sys.exit(1)
    else:
        print(f"Error: H5 file '{h5_file_name}' does not exist.")
        sys.exit(1)  



def compute_bin_edges(x, num_bins):
    """
    Compute the edges of bins for equal binning (same number of events in each bin).

    Parameters:
    -------------
        x (array-like): Input data array.
        num_bins (int): Number of bins.

    Returns:
    -------------
        list: List containing the edges of the bins.
    """
    sorted_indices = np.argsort(x)
    sorted_x = x[sorted_indices]
    bin_size = len(sorted_x) // num_bins
    bin_edges = [sorted_x[i * bin_size] for i in range(num_bins)]
    
    # Add a small fraction of the maximum value to the last bin edge
    max_value = sorted_x[-1]
    bin_edges.append(max_value + max_value * 0.01)  # Adjust the fraction as needed
    return bin_edges    