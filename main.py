import sys
import utils, prepare_dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    input_file, split_size = utils.read_config(config_file)
    data = prepare_dataset.load_file(input_file)    #load the data set
    print(split_size)
    datasets = prepare_dataset.split_data(data, split_size)     #split it into training, testing, validation

    

if __name__ == "__main__":
    main()