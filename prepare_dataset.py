
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from typing import NamedTuple, Tuple
from collections import namedtuple
from collections import Counter


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

    print("\n\nStart loading the simulation file: \n " + input_file)
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

     # Print total number of events and count of events with labels 1 and 0
    total_events = len(labels)
    label_counts = Counter(labels)
    label_1_count = label_counts[1]
    label_0_count = label_counts[0]
    label_1_percentage = (label_1_count / total_events) * 100
    label_0_percentage = (label_0_count / total_events) * 100
    
    print("Total number of events:", total_events)
    print("Number of events with label 1:", label_1_count, "(", label_1_percentage, "%)")
    print("Number of events with label 0:", label_0_count, "(", label_0_percentage, "%)")

    # Splitting data into trainval and test sets
    x_trainval, x_test, y_trainval, y_test = train_test_split(input1, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_dist, x_test_dist, _, _ = train_test_split(input2, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_event, x_test_event, _, _ = train_test_split(input3, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_azimuth, x_test_azimuth, _, _ = train_test_split(input4, labels, test_size=test_size, shuffle=True, random_state=random_state)
    x_trainval_Stot, x_test_Stot, _, _ = train_test_split(input5, labels, test_size=test_size, shuffle=True, random_state=random_state)

    test_set = DataSet(traces=x_test, dist=x_test_dist, event=x_test_event, azimuth=x_test_azimuth, Stot=x_test_Stot, label=y_test)

    val_size = len(x_test) / len(x_trainval)
    if(test_size > 0.5):
        val_set = DataSet(traces=[], dist=[], event=[], azimuth=[], Stot=[], label=[])
        train_set = DataSet(traces=x_trainval, dist=x_trainval_dist, event=x_trainval_event, azimuth=x_trainval_azimuth, Stot=x_trainval_Stot, label=y_trainval)
        print("\nThe train sample contains", len(x_trainval), "events, of which", "{:.2f}".format(sum(y_trainval)/len(y_trainval)*100), "% are signals (label 1)")
        print("The test sample contains", len(x_test), "events, of which", "{:.2f}".format(sum(y_test)/len(y_test)*100), "% are signals")


    else:
        # Further splitting trainval into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
        x_train_dist, x_val_dist, _, _ = train_test_split(x_trainval_dist, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
        x_train_event, x_val_event, _, _ = train_test_split(x_trainval_event, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
        x_train_azimuth, x_val_azimuth, _, _ = train_test_split(x_trainval_azimuth, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)
        x_train_Stot, x_val_Stot, _, _ = train_test_split(x_trainval_Stot, y_trainval, test_size=val_size, shuffle=True, random_state=random_state)

        # Create named tuples for train, validation datasets
        train_set = DataSet(traces=x_train, dist=x_train_dist, event=x_train_event, azimuth=x_train_azimuth, Stot=x_train_Stot, label=y_train)
        val_set = DataSet(traces=x_val, dist=x_val_dist, event=x_val_event, azimuth=x_val_azimuth, Stot=x_val_Stot, label=y_val)

        print("\nThe train sample contains", len(x_train), "events, of which", "{:.2f}".format(sum(y_train)/len(y_train)*100), "% are signals (label 1)")
        print("The validation sample contains", len(x_val), "events, of which", "{:.2f}".format(sum(y_val)/len(y_val)*100), "% are signals")
        print("The test sample contains", len(x_test), "events, of which", "{:.2f}".format(sum(y_test)/len(y_test)*100), "% are signals")

        # Print percentage of data used for train, validation, and test
        train_percentage = (len(train_set.label) / total_events) * 100
        val_percentage = (len(val_set.label) / total_events) * 100
        test_percentage = (len(test_set.label) / total_events) * 100

        print("Percentage of data used for training:", "{:.2f}".format(train_percentage), "%")
        print("Percentage of data used for validation:", "{:.2f}".format(val_percentage), "%")
        print("Percentage of data used for testing:", "{:.2f}".format(test_percentage), "%")


    return DataSets(train=train_set, validation=val_set, test=test_set)




def split_data_kfold(input_dict: Dict[str, Any], k_fold: int, seed: int = 7) -> List[Dict[str, DataSet]]:
    """
    Prepare the data for k-fold cross-validation.

    Parameters:
    -----------
    input_dict : dict
        Dictionary containing the loaded arrays.
    k_fold : int
        Number of folds for cross-validation.
    seed : int, optional
        Random seed used for shuffling the data.

    Returns:
    -----------
    List[Dict[str, DataSet]]
        List of datasets, each representing one fold of the cross-validation. 
        Each dataset contains training, validation, and test sets.

    Notes:
    -----------
    The returned datasets list contains dictionaries, each representing one fold of the cross-validation.
    Each dictionary has keys 'train', 'validation', and 'test', with corresponding DataSet objects.
    Each DataSet object consists of arrays for traces, distances, event information, azimuth, Stot, and labels.
    """

    input1 = input_dict['traces']
    input2 = input_dict['dist']
    input3 = input_dict['info_event']
    input4 = input_dict['azimuthSP']
    input5 = input_dict['Stot']
    labels = input_dict['labels']

    np.random.seed(seed) 
    kfold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    
    datasets = []  # Array to store datasets for each fold

    for i, (train_idx, test_idx) in enumerate(kfold.split(input1, labels)):
        print(f"\nFold: {i+1}")

        # Split data into train and test sets
        x_train, x_test = input1[train_idx], input1[test_idx]
        x_train_dist, x_test_dist = input2[train_idx], input2[test_idx]
        x_train_event, x_test_event = input3[train_idx], input3[test_idx]
        x_train_azimuth, x_test_azimuth = input4[train_idx], input4[test_idx]
        x_train_Stot, x_test_Stot = input5[train_idx], input5[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Further splitting train into train and validation sets
        x_train, x_val, x_train_dist, x_val_dist, x_train_event, x_val_event, x_train_azimuth, x_val_azimuth, x_train_Stot, x_val_Stot, y_train, y_val = train_test_split(
            x_train, x_train_dist, x_train_event, x_train_azimuth, x_train_Stot, y_train,
            test_size=len(x_test) / len(x_train), shuffle=True, random_state=seed)

        # Create DataSet objects
        train_set = DataSet(traces=x_train, dist=x_train_dist, event=x_train_event, azimuth=x_train_azimuth, Stot=x_train_Stot, label=y_train)
        val_set = DataSet(traces=x_val, dist=x_val_dist, event=x_val_event, azimuth=x_val_azimuth, Stot=x_val_Stot, label=y_val)
        test_set = DataSet(traces=x_test, dist=x_test_dist, event=x_test_event, azimuth=x_test_azimuth, Stot=x_test_Stot, label=y_test)

         # Print information about the datasets
        print(f"Number of events in train set: {len(train_set.label)}, Percentage: {len(train_set.label)/len(labels) * 100:.2f}%")
        print(f"Number of events in validation set: {len(val_set.label)}, Percentage: {len(val_set.label)/len(labels) * 100:.2f}%")
        print(f"Number of events in test set: {len(test_set.label)}, Percentage: {len(test_set.label)/len(labels) * 100:.2f}%")


        datasets.append({'train': train_set, 'validation': val_set, 'test': test_set})

    return datasets