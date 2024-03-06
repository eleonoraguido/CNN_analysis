import sys
import utils

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    input_file = utils.read_config(config_file)
    print(input_file)
    

if __name__ == "__main__":
    main()