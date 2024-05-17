"""
Author: Eleonora Guido
Last modification date: 05.2024
Photon search with a CNN
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tensorflow import keras as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import *
import CNN_model
from config import PDF_SAVE_PATH


def create_CNN_model(l1=0.0001, l2=0.0001, loss='bce', optimizer='adam', num_stat=3):
    """
    Create a Convolutional Neural Network (CNN) model for particle classification.

    Parameters:
    -----------
    - L1 : float, optional. 
        Regularization parameter L1 (penalty applied to the absolute values of the weights).
    - L2 : float, optional.
        Regularization parameter L2 (penalty applied to the square of the weights).
    - loss : str, optional.
        Loss function used to wuantify the goodness of predictions. Default option is binary cross-entropy.
    - optimizer : str, optional.
        Optimizer to use. Options are 'adam', 'rmsprop', and 'adagrad'. Default is 'adam'.
    - num_stat : int, optional.
        Number of stations, default is 3.

    Returns:
    --------
    tf.keras.Model
        Compiled CNN model ready for training.
    """
    # Define model hyperparameters
    kernel_regularizer = regularizers.l1_l2(l1=l1, l2=l2)
    reduce_learning_rate = callbacks.ReduceLROnPlateau(monitor='val_Distance', factor=0.4, 
                                                       patience=17, verbose=1, mode='min', 
                                                       min_delta=1, cooldown=0, min_lr=0.0)

    # Choose the input shape
    input_traces = Input(shape=[num_stat, 150, 1])  # 3 cumulative traces (for 3 SD stations), 150 time bins
    input_distances = Input(shape=[num_stat, 1])  # distances of each SD station from the shower core
    input_event = Input(shape=[3])  # theta, S1000, Nstat of the event
    input_azimuths = Input(shape=[num_stat, 1])  # azimuth of each SD station wrt the SP
    input_Stot = Input(shape=[num_stat, 1])  # total signal in each station

    CNN_mod = CNN_model.define_layers(kernel_regularizer, input_traces, input_distances, input_event, input_azimuths, input_Stot)
    
    compile_model(CNN_mod, loss= loss, optimizer=optimizer)

    return CNN_mod


def compile_model(model, loss='bce', optimizer='adam', metrics=["accuracy"]):
    """
    Compile the given model with specified loss function, optimizer, and metrics.

    Parameters:
    -----------
    model : tf.keras.Model
        The model to be compiled.
    loss : str, optional
        The loss function to use. Default is 'bce'.
    optimizer : str, optional
        The optimizer to use. Default is 'adam'.
    metrics : list of str, optional
        List of metrics to evaluate the model. Default is ["accuracy"].

    Returns:
    --------
    None
    """
    # Choose optimizer
    if optimizer == 'adam':
        chosen_optimizer = K.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5)
    elif optimizer == 'rmsprop':
        chosen_optimizer = K.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, clipnorm=5)
    elif optimizer == 'adagrad':
        chosen_optimizer = K.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-08, clipnorm=5)
    else:
        raise ValueError("Invalid optimizer. Choose 'adam', 'rmsprop', or 'adagrad'.")

    model.compile(loss=loss, optimizer=chosen_optimizer, metrics=metrics)



def simple_CNN_training(epochs, batch_size, data_train, data_val, model):
    """
    Train a given CNN model using the provided training data.

    Parameters:
    -----------
    epochs : int
        Number of epochs for training.
    batch_size : int
        Batch size for training.
    data_train : tuple
        Tuple containing training data (traces, distances, event info, azimuth, Stot) and labels.
    data_val : tuple
        Tuple containing validation data (traces, distances, event info, azimuth, Stot) and labels.
    model : tf.keras.Model
        Compiled CNN model.

    Returns:
    --------
    model : tf.keras.Model
        Trained CNN model.
    metrics_list : dict
        Dictionary containing metrics for the trained model.
        Keys are model names and values are tuples of the form
        (train_loss, val_loss, train_accuracy, val_accuracy).
    """
    metrics = model.fit([data_train.traces, data_train.dist, data_train.event, data_train.azimuth, data_train.Stot], data_train.label, epochs=epochs, batch_size=batch_size, verbose=1,  
                    validation_data=([data_val.traces, data_val.dist, data_val.event, data_val.azimuth, data_val.Stot], data_val.label) )
    
    train_loss = metrics.history['loss']
    val_loss = metrics.history['val_loss']
    train_accuracy = metrics.history['accuracy']
    val_accuracy = metrics.history['val_accuracy']

    metrics_list = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }

    return model, metrics_list



def plot_loss_and_accuracy(models_metrics, output_file = "loss_accuracy_plot.pdf", save_path=PDF_SAVE_PATH):
    """
    Plot the training and validation loss, as well as accuracy for multiple models (or one model) and save it to a PDF file.

    Parameters:
    -----------
    models_metrics : dict
        Dictionary containing metrics for each model. Keys are model names and values are tuples of the form
        (train_loss, val_loss, train_accuracy, val_accuracy).
    output_file : str, optional
        Path to the output PDF file. Default is "loss_accuracy_plot.pdf".
    save_path : str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.
    """

    # Generate PDF and save it to the specified path
    full_path = os.path.join(save_path, output_file)

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    
    axs[0].grid(True)
    axs[1].grid(True)

    axs[0].set_ylim(0.05, 0.5)
    axs[1].set_ylim(0.86, 0.99)
    axs[0].set_ylabel('loss', fontsize=13)
    axs[0].set_xlabel('epoch', fontsize=13)
    axs[1].set_ylabel('accuracy', fontsize=13)
    axs[1].set_xlabel('epoch', fontsize=13)

    if len(models_metrics) == 1:  # Check if there is only one model
        model_name, metrics = next(iter(models_metrics.items()))
        
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        train_accuracy = metrics['train_accuracy']
        val_accuracy = metrics['val_accuracy']

        color = 'black'
        axs[0].plot(train_loss, alpha=0.8, color=color, label = 'training')
        axs[1].plot(train_accuracy, alpha=0.8, color=color, label = 'training')
        axs[0].plot(val_loss, linestyle='dotted', alpha=0.8, color=color, label = 'validation')
        axs[1].plot(val_accuracy, linestyle='dotted', alpha=0.8, color=color, label = 'validation')

    else:   # There are more than one model in the dictionary
        color_map = plt.cm.viridis  # Use Viridis colormap for multiple models
        colors = color_map(np.linspace(0, 1, len(models_metrics)))  # Generate colors for multiple models
        for i, (model_name, metrics) in enumerate(models_metrics.items()):
            train_loss = metrics['train_loss']
            val_loss = metrics['val_loss']
            train_accuracy = metrics['train_accuracy']
            val_accuracy = metrics['val_accuracy']
            #print(train_loss)
            #print(val_accuracy)
            color = colors[i]  # Use different color for each model
            axs[0].plot(train_loss, alpha=0.8, color=color, label=f'Training, model {model_name}')
            axs[1].plot(train_accuracy, alpha=0.8, color=color, label=f'Training, model {model_name}')
            axs[0].plot(val_loss, linestyle='dotted', alpha=0.8, color=color, label=f'Validation, model {model_name}')
            axs[1].plot(val_accuracy, linestyle='dotted', alpha=0.8, color=color, label=f'Validation, model {model_name}')

    # Create legend entries without plotting any actual data
    legend_entries = [Line2D([0], [0], linestyle='--', color='black', label='Validation'),
                  Line2D([0], [0], linestyle='-', color='black', label='Training')]

    # Add legend
    axs[0].legend(handles=legend_entries, loc='upper right')
    axs[1].legend(handles=legend_entries, loc='upper left')

    plt.tight_layout()
    plt.savefig(full_path)
    plt.close()




def save_CNN_model(model, epochs, batch_size, output_name, l1, l2, save_path=PDF_SAVE_PATH):
    """
    Save a trained CNN model and its weights.

    Parameters:
    -----------
    model : tf.keras.Model
        Trained CNN model.
    epochs : int
        Number of epochs the model was trained for.
    batch_size : int
        Batch size used during training.
    output_name : str
        Name used to identify the saved model files.
    save_path: str, optional
        Path to the directory where the plot will be saved. If not provided,
        the global variable PDF_SAVE_PATH will be used.

    """
    # serialize model to JSON
    model_json = model.to_json()
    model_name = f"CNN_model_{l1}L1_{l2}L2_{output_name}_E{epochs}_B{batch_size}"
    with open(save_path+model_name+".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(save_path+model_name+".h5")

    print("The model will be saved in the directory "+save_path)
    print("The trained model has been saved as "+model_name+".json")
    print("The weights have been saved in the corresponding .h5 file \n")

