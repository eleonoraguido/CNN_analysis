- Author: Eleonora Guido
- Last modification date: 05.2024
- Photon search with a CNN

Introduction
-----------

This program reads a .npz file, previously created by merging together the data of interest. Typically the data are from Offline simulations of two different particle species (generally photons and protons labelled as 1 and 0 respectively).
- The .npz file can be created starting from Offline simulations with the script in the directory *Read_simulations*.

Two different tasks can be performed depending on the **configuration file type**:
1) A CNN model is defined, trained and tested.
2) An already existent CNN model is loaded and tested.


Task 1: Define, train and test a model
-----------

Use the *config.json* file.

**Input**: pre-created .npz file, containing selected information from Offline simulations.

**Output**: All the outputs will be saved in the working directory (see below).
1) Trained model configuration (.json file(s)) and weights (.h5 file(s)). 
    - The output files will look like: *CNN_model_<...>L1_<...>L2_<output_name>_<...>_E<...>_B<...>.json*. 
    - All the gaps will be filled according to variables set in this configuration file (*L1_regularizer, L2_regularizer, partitioning_method, num_epochs, batch_size*). 
    - Additionally, in "output_name" generally the particle species, the maximum theta and the energy range of the simualtions are reported.
2) Results of training and testing.
3) Plots.

-------------
Some variables can be set in the configuration file:
- "config_type": string. The type of configuration file. It has to be set to *create_model*.
- "input_file": string. Absolute path to the input data file + input .npz file name. The files are located in */data/auger8/DNN_photon_analysis_input_files/dataset/* 
- "output_name": string. It contains some information about the data set: the particle species, the maximum theta and the energy range. You have to follow this format: *"output_name" : "phot_prot_maxtheta60_energy_185_195"*. When the model is saved, this information will be printed in the name of the saved model to identify it and it will replace <output_name>. 
- "partitioning_method": string. Two possible types: 
    1) "k_fold" : data are splitted into k subsets. A k-fold cross validation is performed by training k models. For each one a different fold is held out and used for testing.
    2) "split" : data are splitted into three subsets: training, validation, testing data sets. Only one model is trained and tested. 
- "partitioning_param": int/float. 
    - If "partitioning_method" : "k_fold", it is an int, the number of folds k. 
    - If "partitioning_method" : "split", it is a float, the fraction of simulations used for testing (= number of simulations used for validation).
- "num_epochs": int. The number of epochs.
- "batch_size": int. The mini-batch size.
- "L1_regularizer": float. Regularizer parameter L1 (penalty applied to absolute value of weights, 0-0.1)
- "L2_regularizer": float. Regularizer parameter L2 (penalty applied to square of weights, 0-0.1)
- "sel_threshold": float. The selected threshold (between 0 and 1) is the value used to compute the background rejection (it is just a placeholder for possible future tests, it will be computed again at 50% signal efficiency by default).


Task 2: Load and test an already existent model
-----------

Use the *config_load_model.json* file.

**Input**: 
1) Pre-created .npz file, containing selected information from Offline simulations.
2) Trained model to be loaded, located in the working directory (see below).

**Output**: All the outputs will be saved in the working directory (see below) and overwritten if already existent.
1) Results of testing.
2) Plots.

-------------
Some variables can be set in the configuration file:
- "config_type": string. The type of configuration file. It has to be set to *load_model*.
- "input_file": string. Absolute path to the input data file + input .npz file name. The files are located in */data/auger8/DNN_photon_analysis_input_files/dataset/* 
- "model_name": string. The complete name of the model that you want to load (located in the working directory), without extension. 
    - In case you want to load just one model, for example: 
    
        ```"model_name" : "CNN_model_0.0001L1_0.0001L2_phot_prot_maxtheta60_energy_185_195_E50_B64"```.
    - In case you want to load more than one model (k-fold cross validation case), you have to use a "\*" to replace the number of fold, for example: 
    
         ```"model_name" : "CNN_model_0.0001L1_0.0001L2_phot_prot_maxtheta60_energy_185_195_fold_*_E50_B64" ```
- "partitioning_method": string. Two possible types: 
    1) "k_fold" : you want to load k models. Data are splitted into k subsets. For each one a different fold is used for testing.
    2) "split" : you want to load just one model. Data are splitted into three subsets: training, validation, testing data sets. The test data set is used for testing. 
- "partitioning_param": int/float. You have to be sure to use the same value that was used for training.
    - If "partitioning_method" : "k_fold", it is the number of folds. 
    - If "partitioning_method" : "split", it is the fraction of simulations used for testing (= number of simulations used for validation).
- "sel_threshold": float. The selected threshold (between 0 and 1) is the value used to compute the background rejection. 
    - If in the test sample there are signal and background events, the variable is overwritten to have a 50% signal efficiency. 
    - If in the test sample there are only background events, the value set here is used. For example, a value obtained to have a 50% signal efficiency in another test sample can be set for comparison.
    


The working directory
-----------------------------
- The working directory is NOT the directory where the scripts are located, as the scripts are also backed up in git repository.
    - Note that the pdfs that are produced when the script is run can be very big (both if the model is created or loaded), then they should NOT be saved in the git repository but in a different working directory.
- It can be set in the *config.py* file. The variable *PDF_SAVE_PATH* contains the relative path to the working directory.
- The working directory is where all the outputs are created and, in the case we are using the config file *config_load_model.json* is also where the already existent model to load has to be located.
- Note that the simulations used for training and testing can be located in a different directory, which is set in the configuration files with the variable "input_file" through its absolute path.


Additional options
-----------------------------
Other options that can be set for the compilation of the model (both if the model is created and if it is loaded, as compilation options are not included in the saved model):
- Optimizer: optimizer to use when the model is compiled. Options are 'adam', 'rmsprop', and 'adagrad'. Default is 'adam'.
- Loss function: the one used when compiling the model. Default function: binary cross-entropy ('bce').

Both can be set in config.py (OPTIMIZER, LOSS_FUNCTION).

**IMPORTANT**: the model is always saved without information about the options to compile it. In order to reproduce the results when loading it, be sure to set the same loss function and optimizer that were used to train it.


Notes
---------------------------
- When the k-fold cross validation is performed, k files are created with both the extensions .json and .h5 . It is not actually necessary to create k .json files, as the model architecture is the same and only the weights are different (as they depend on the folds used for training). However, the files are very small and this is the simplest extension of the case with just one model.

-----------------------------
How to run the program locally:
----------------------------
```source activate_env.sh```

```python3 main.py <config_file>```




Where the benchmark models are stored
------------------------------
The trained benchmark models (reference results) are located in *CNN_analysis_output/benchmark_models/*.



Documentation
------------------------------
https://keras.io/api/layers/convolution_layers/
https://www.tensorflow.org/api_docs/python/tf
https://scikit-learn.org/stable/user_guide.html