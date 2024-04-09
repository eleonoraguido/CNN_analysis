import sys, os
import json
import numpy as np
from keras.models import model_from_json
from tensorflow import keras as K


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
        - test_sample_size (float): size of the test sample in percentage.
        - sel_threshold (float): chosen threshold for selection.
        - k_cross_validation (int): number of folds used for cross validation procedure (no cross validation if 0)
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return (config.get('input_file'), config.get('test_sample_size'), config.get('sel_threshold'), config.get('k_cross_validation'))





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




def load_model(config_file):
    """
    Read input file path from a JSON configuration file.

    Parameters:
    ------------
        - config_file (str): The path to the JSON configuration file.

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

    # Construct file names
    h5_file_name = input_arg + ".h5"
    json_file_name = input_arg + ".json"

    print("\nStart loading the model files:")
    # Check if the .h5 file exists
    if os.path.exists(h5_file_name):
        # If the .h5 file exists, proceed to check for the .json file
        if os.path.exists(json_file_name):
            # If both files exist, proceed with processing
            json_file = open(json_file_name, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(h5_file_name)
            print(f"JSON file '{json_file_name}'.json has been read")
            print(f"H5 file '{h5_file_name}'.h5 has been read")
            print("----> Model is loaded from disk.")
            adam_optimizer = K.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5)
            rmsprop_optimizer = K.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, clipnorm=5)
            adagrad_optimizer = K.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-08, clipnorm=5)
            loaded_model.compile(loss='bce', optimizer=adam_optimizer, metrics=["accuracy"])
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