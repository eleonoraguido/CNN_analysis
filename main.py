"""
Author: Eleonora Guido
Last modification date: 05.2024
Photon search with a CNN
"""

import sys
import utils, prepare_dataset
import part_methods

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    config_type = utils.read_config_type(config_file)

    input_file, part_method, part_param, thres_set = utils.read_config_data(config_file)

    data = prepare_dataset.load_file(input_file)    #load the data set
    print("\nInput file is "+input_file)

    print(part_method)
    models = {}  #dictionary to store the models 

    if part_method == "split":
        models, datasets = part_methods.run_split_method(part_param, data, config_type, config_file)
    elif part_method == "k_fold":
        models, datasets = part_methods.run_k_fold_method(part_param, data, config_type, config_file)
    else:
        print("Error: Unknown partitioning method:", part_method, " -> choose between 'split' and 'k_fold'.")
        sys.exit(1)

    part_methods.evaluate_and_plot(models, datasets, thres_set)


if __name__ == "__main__":
    main()