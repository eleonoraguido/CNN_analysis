
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from typing import NamedTuple, Tuple
from collections import namedtuple


def load_file(input_file: str)-> dict:
    """
    Load the compressed NumPy file and return its content.

    Parameters:
    -----------
    input_file : str
        Directory path where the dataset file is located and name of the dataset file.

    Returns:
    -----------
    dict
        Dictionary containing the loaded arrays.
    """
    # Load the compressed NumPy file
    data = np.load(input_file)

    print("Start loading the file: \n " + input_file)
    # Check the keys of the loaded data
    print(" \t Keys in the compressed file:  ", data.files)

    # Access individual arrays by their keys
    loaded_data = {}
    loaded_data['traces'] = data['traces']
    loaded_data['dist'] = data['dist']
    loaded_data['Stot'] = data['Stot']
    loaded_data['azimuthSP'] = data['azimuthSP']
    loaded_data['info_event'] = data['info_event']
    loaded_data['labels'] = data['labels']
    
    data.close()

    return loaded_data



DataSet = namedtuple('DataSet', ['event', 'traces', 'dist', 'azimuth', 'Stot', 'label'])

class DataSets:
    def __init__(self, train: DataSet = None, validation: DataSet = None, test: DataSet = None):
        self.train = train
        self.validation = validation
        self.test = test


def split_data(input_dict: dict, test_size: float = 0.2, random_state: int = 0) -> DataSets:
    """
    Prepare the data for training, validation, and testing.

    Parameters:
    -----------
    input_dict : dict
        Dictionary containing the loaded arrays.
    test_size : float, optional
        The proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed used for shuffling the data.

    Returns:
    -----------
    DataSets
        Object containing training, validation, and testing data.

    Notes:
    -----------
    The returned DataSets object contains the following attributes:
    
    - train: DataSet
    - validation: DataSet
    - test: DataSet

        Each tuple containing the corresponding dataset. It consists of the following arrays:
        - event: Event-level information (S1000, theta, Nstat).
        - traces: 3 Traces.
        - dist: 3 Distances .
        - azimuth: 3 Azimuths.
        - Stot: 3 Stot.
        - label: Labels.
    """
    input1 = input_dict['traces']
    input2 = input_dict['dist']
    input3 = input_dict['info_event']
    input4 = input_dict['azimuthSP']
    input5 = input_dict['Stot']
    labels = input_dict['labels']

    # Splitting data into trainval and test sets
    x_trainval, x_test, y_trainval, y_test = train_test_split(input1, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_dist, x_test_dist, _, _ = train_test_split(input2, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_event, x_test_event, _, _ = train_test_split(input3, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_azimuth, x_test_azimuth, _, _ = train_test_split(input4, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_Stot, x_test_Stot, _, _ = train_test_split(input5, labels, test_size=test_size, shuffle=True, random_state=random_state)

    val_size = len(x_test) / len(x_trainval)

    # Further splitting trainval into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
    x_train_dist, x_val_dist, _, _ = train_test_split(x_trainval_dist, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
    x_train_event, x_val_event, _, _ = train_test_split(x_trainval_event, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
    x_train_azimuth, x_val_azimuth, _, _ = train_test_split(x_trainval_azimuth, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
    x_train_Stot, x_val_Stot, _, _ = train_test_split(x_trainval_Stot, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)

    print("\nThe train sample contains", len(x_train), "events, of which", "{:.2f}".format(sum(y_train)/len(y_train)*100), "% are signals (label 1)")
    print("The validation sample contains", len(x_val), "events, of which", "{:.2f}".format(sum(y_val)/len(y_val)*100), "% are signals")
    print("The test sample contains", len(x_test), "events, of which", "{:.2f}".format(sum(y_test)/len(y_test)*100), "% are signals")

    # Create named tuples for train, validation, and test datasets
    train_set = DataSet(traces=x_train, dist=x_train_dist, event=x_train_event, azimuth=x_train_azimuth, Stot=x_train_Stot, label=y_train)
    test_set = DataSet(traces=x_test, dist=x_test_dist, event=x_test_event, azimuth=x_test_azimuth, Stot=x_test_Stot, label=y_test)
    val_set = DataSet(traces=x_val, dist=x_val_dist, event=x_val_event, azimuth=x_val_azimuth, Stot=x_val_Stot, label=y_val)

    return DataSets(train=train_set, validation=val_set, test=test_set)