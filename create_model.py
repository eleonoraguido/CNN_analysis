import matplotlib.pyplot as plt
from tensorflow import keras as K
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import *
import CNN_model



def create_CNN_model(l1 = 0.0001, l2 = 0.0001, num_stat = 3):
    """
    Create a Convolutional Neural Network (CNN) model for particle classification.

    Parameters:
    -----------
    L1 : float, optional 
        Regularization parameter L1 (penalty applied to the absolute values of the weights).
    L2 : float, optional
        Regularization parameter L2 (penalty applied to the square of the weights).
    num_stat : int, optional
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
    input_traces = Input (shape =[num_stat, 150, 1])  # 3 comulative traces (for 3 SD stations), 150 time bins
    input_distances = Input (shape =[num_stat, 1]) # distances of each SD station from the shower core
    input_event = Input (shape =[3]) # theta, S1000, Nstat of the event
    input_azimuths = Input (shape =[num_stat, 1]) # azimuth of each SD station wrt the SP
    input_Stot = Input (shape =[num_stat, 1]) # total signal in each station

    CNN_mod = CNN_model.define_layers(kernel_regularizer, input_traces, input_distances, input_event, input_azimuths, input_Stot)
    
    adam_optimizer = K.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=5)
    rmsprop_optimizer = K.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, clipnorm=5)
    adagrad_optimizer = K.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1, epsilon=1e-08, clipnorm=5)
    CNN_mod.compile(loss='bce', optimizer=adam_optimizer, metrics=["accuracy"])

    return CNN_mod


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
    tf.keras.Model
        Trained CNN model.
    """
    metrics = model.fit([data_train.traces, data_train.dist, data_train.event, data_train.azimuth, data_train.Stot], data_train.label, epochs=epochs, batch_size=batch_size, verbose=1,  
                    validation_data=([data_val.traces, data_val.dist, data_val.event, data_val.azimuth, data_val.Stot], data_val.label) )
    
    plt.figure(1, (12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(metrics.history['loss'], "k")
    plt.plot(metrics.history['val_loss'], "k--")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(metrics.history['accuracy'], "k")
    plt.plot(metrics.history['val_accuracy'], "k--")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    # Save the plots
    plt.savefig('loss_accuracy_plot.pdf')

    return model


def save_CNN_model(model, epochs, batch_size, output_name, l1, l2):
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
    """
    # serialize model to JSON
    model_json = model.to_json()
    model_name = f"CNN_model_{l1}L1_{l2}L2_{output_name}_E{epochs}_B{batch_size}"
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_name+".h5")

    print("The trained model has been saved as "+model_name+".json")
    print("Correpsoding weights are saved in the corresponding .h5 file")



#def cross_validation_training(epochs, batch_size, n_fold, data_train, data_val, model):