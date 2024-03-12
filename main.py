import sys
import utils, prepare_dataset, create_model

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    input_file, split_size, epochs, batch_size, output_name, l1, l2 = utils.read_config(config_file)

    data = prepare_dataset.load_file(input_file)    #load the data set
    datasets = prepare_dataset.split_data(data, split_size)     #split it into training, testing, validation

    print("\nInput file is "+input_file)
    print("Number of epochs: "+str(epochs))
    print("Mini-batch size: "+str(batch_size))
    print("Regularization parameter L1: "+str(l1))
    print("Regularization parameter L2: "+str(l2)+"\n\n")

    CNN = create_model.create_CNN_model(l1, l2)
    train_dataset = datasets.train
    val_dataset = datasets.validation
    test_dataset = datasets.test
    trained_CNN = create_model.simple_CNN_training(epochs, batch_size, train_dataset, val_dataset, CNN)
    create_model.evaluate_model(trained_CNN, test_dataset)

    create_model.save_CNN_model(trained_CNN, epochs, batch_size, output_name, l1, l2)

if __name__ == "__main__":
    main()