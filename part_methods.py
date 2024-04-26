import sys
import utils
import prepare_dataset
import create_model
import apply_model
from config import OPTIMIZER, LOSS_FUNCTION

def run_split_method(part_param, data, config_type, config_file):
    """
    - Perform data splitting to train one model. 
    - The fraction of data not ot be used for training is passed as argument. 
    - The model is trained.

    Parameters:
    --------------
    - part_param : float
        The fraction of data that are not used for training.
    - data : object
        The dataset object.
    - config_type : str
        Type of configuration file.
    - config_file : str
        Configuration file path.

    Returns:
    --------------
    tuple
        A tuple containing dictionaries of trained models and datasets.
        - The first element is a dictionary of trained models, where the keys represent the model names.
        - The second element is a dictionary of datasets, where the keys represent dataset names.
        - Here there is just one model, so the key is a placeholder called "model".
    """
    print("\nNo cross validation is performed, only one model is trained.") 

    if isinstance(part_param, int):
        print("Error: For 'split' case, partitioning_param should be a float between 0 and 1.")
        sys.exit(1)
    
    try:
        split_size = float(part_param)
        if not 0 < split_size < 1:
            raise ValueError("Split size should be between 0 and 1.")
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)
    
    print("Test data set is "+str(split_size*100)+"% of the total data set." )

    model_key = "model"   #the key for the dictionaries (just one item in this case)

    dataset = prepare_dataset.split_data(data, split_size)     #split it into training, testing, validation; return a dictionary
    
    # Access the training, validation, and test datasets from the dictionary
    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    datasets = {}
    datasets[model_key] = dataset

    models = {}  #dictionary to store the model
    metrics_list = {}

    # Check the type of configuration file
    if config_type == "create_model":
        epochs, batch_size, output_name, l1, l2 = utils.read_config_settings(config_file)

        print("Number of epochs: "+str(epochs))
        print("Mini-batch size: "+str(batch_size))
        print("Regularization parameter L1: "+str(l1))
        print("Regularization parameter L2: "+str(l2)+"\n\n")

        CNN = create_model.create_CNN_model(l1, l2, loss = LOSS_FUNCTION, optimizer = OPTIMIZER)
        trained_CNN, metrics = create_model.simple_CNN_training(epochs, batch_size, train_dataset, val_dataset, CNN)
        create_model.save_CNN_model(trained_CNN, epochs, batch_size, output_name, l1, l2)
        metrics_list[model_key] = metrics   #create a dictionary with just one key
        models[model_key] = trained_CNN  # Store the trained model in the dictionary
        create_model.plot_loss_and_accuracy(metrics_list)

    elif config_type == "load_model":
        trained_CNN = utils.load_model(config_file)
        models[model_key] = trained_CNN  # Store the loaded trained model in the dictionary
    else:
        print("\nUnknown config_type specified in the config file. Exiting.")
        sys.exit(1)

    return models, datasets




def run_k_fold_method(part_param, data, config_type, config_file):
    """
    - Perform k-fold cross-validation method for model training. 
    - The data set is divided into k subsets and k models are trained, each one without one fold.
    - Each model is evaluated on the fold not included for training.

    Parameters:
    --------------
    - part_param : int
        The number of folds k.
    - data : object
        The dataset object.
    - config_type : str
        Type of configuration file.
    - config_file : str
        Configuration file path.

    Returns:
    --------------
    tuple
        A tuple containing dictionaries of trained models and datasets.
        - The first element is a dictionary of trained models, where the keys represent the model names.
        - The second element is a dictionary of datasets, where the keys represent dataset names.
    """
    print("\nk-fold cross validation is performed")
    try:
        k_fold = int(part_param)
        if k_fold <= 2:
            raise ValueError("Error: For 'k_fold' case, partitioning_param should be an integer greater than 2.")
    except ValueError as e:
        print("Error:", e)
        sys.exit(1)
    print("The number of folds is k = "+str(k_fold)+".")

    datasets = prepare_dataset.split_data_kfold(data, k_fold)

    models = {}  #dictionary to store the models
    metrics_list = {}

    # Iterate over each fold's dataset in the datasets dictionary
    for fold, (fold_name, fold_data) in enumerate(datasets.items(), start=1):
        print(f"\n{fold_name}:")
        train_dataset = fold_data['train']
        val_dataset = fold_data['validation']

        # Check the type of configuration file
        if config_type == "create_model":
            epochs, batch_size, output_name, l1, l2 = utils.read_config_settings(config_file)

            print("Number of epochs: " + str(epochs))
            print("Mini-batch size: " + str(batch_size))
            print("Regularization parameter L1: " + str(l1))
            print("Regularization parameter L2: " + str(l2) + "\n\n")

            CNN = create_model.create_CNN_model(l1, l2, loss = LOSS_FUNCTION, optimizer = OPTIMIZER)
            trained_CNN, metrics = create_model.simple_CNN_training(epochs, batch_size, train_dataset, val_dataset, CNN)
            models[fold_name] = trained_CNN  # Store the trained model in the dictionary
            metrics_list[fold_name] = metrics
            create_model.save_CNN_model(trained_CNN, epochs, batch_size, output_name + f"_{fold_name}", l1, l2)

        elif config_type == "load_model":
            # Get the number of keys in datasets
            num_folds = len(datasets)
            # Check if the number of keys matches k_fold
            if num_folds != k_fold:
                print(f"Error: Mismatch between k_fold value ({k_fold}) and the number of folds in datasets ({num_folds}).")
                sys.exit(1) 
            # Load the model for the current fold
            trained_CNN = utils.load_model(config_file, fold)
            # Store the trained model in the dictionary
            models[fold_name] = trained_CNN
        else:
            print("\nUnknown config_type specified in the config file. Exiting.")
            sys.exit(1)

    if config_type == "create_model":
        create_model.plot_loss_and_accuracy(metrics_list)

    return models, datasets
    



def evaluate_and_plot(trained_models, datasets, thres_set):
    _ = apply_model.evaluate_model(trained_models, datasets)

    y_values = apply_model.return_labels(trained_models, datasets)
    apply_model.plot_labels(y_values)
    apply_model.plot_confusion_matrix(y_values)
    percentage_signal = None
    # Iterate over datasets
    for (data_key, dataset) in datasets.items():
        # Calculate percentage_signal for the first dataset encountered
        test_dataset = dataset['test']
        percentage_signal = sum(test_dataset.label) / len(test_dataset.label)
        break
    else:
        print("No datasets found")
        sys.exit(1)
    
    if (percentage_signal > 0.01 and percentage_signal < 0.99):    #do not plot ROC curve if all the data set belong to one class
        roc_results, _ = apply_model.plot_ROC(y_values)
        thres_set = apply_model.plot_confusion_matrix_50sigeff(y_values, roc_results)
        apply_model.plot_tpr_vs_S1000(y_values, datasets)
        apply_model.plot_tpr_vs_theta(y_values, datasets)

        fp_indices = apply_model.get_background_rejection(y_values, thres_set)
        
        if len(fp_indices) != 0:
            apply_model.print_events_info(fp_indices, datasets, "fp_events_thres"+"{:.{}f}".format(thres_set['model'], 3)+".pdf")
    
    apply_model.plot_fpr_vs_S1000(y_values, datasets)
    apply_model.plot_fpr_vs_theta(y_values, datasets)
