import sys, os
import utils, prepare_dataset, create_model, apply_model

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    config_type = utils.read_config_type(config_file)

    input_file, split_size = utils.read_config_data(config_file)
    data = prepare_dataset.load_file(input_file)    #load the data set
    datasets = prepare_dataset.split_data(data, split_size)     #split it into training, testing, validation
    train_dataset = datasets.train
    val_dataset = datasets.validation
    test_dataset = datasets.test

    # Check if the input argument is a file
    if config_type == "create_model":
        epochs, batch_size, output_name, l1, l2 = utils.read_config_settings(config_file)

        print("\nInput file is "+input_file)
        print("Number of epochs: "+str(epochs))
        print("Mini-batch size: "+str(batch_size))
        print("Regularization parameter L1: "+str(l1))
        print("Regularization parameter L2: "+str(l2)+"\n\n")

        CNN = create_model.create_CNN_model(l1, l2)
        trained_CNN = create_model.simple_CNN_training(epochs, batch_size, train_dataset, val_dataset, CNN)
        create_model.save_CNN_model(trained_CNN, epochs, batch_size, output_name, l1, l2)
        
    elif config_type == "load_model":
        # If it's not a file, treat it as a string
        trained_CNN = utils.load_model(config_file)
    
    else:
        print("\nUnknown config_type specified in the config file. Exiting.")
        sys.exit(1)
    
    scores = apply_model.evaluate_model(trained_CNN, test_dataset)
    y_pred, y_true = apply_model.return_labels(trained_CNN, test_dataset)
    apply_model.plot_labels(y_true, y_pred)
    apply_model.plot_confusion_matrix(y_true, y_pred)


    percentage_signal = sum(test_dataset.label)/len(test_dataset.label)
    if (percentage_signal > 0.01 and percentage_signal < 0.99):    #do not plot ROC curve if all the data set belong to one class
        tpr, threshold = apply_model.plot_ROC(y_true, y_pred)
        apply_model.plot_confusion_matrix_50sigeff(y_true, y_pred, tpr, threshold)
        

if __name__ == "__main__":
    main()