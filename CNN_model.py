"""
Author: Eleonora Guido
Last modification date: 05.2024
Photon search with a CNN
"""

from tensorflow import keras as K
from tensorflow.keras.layers import *

def DenselyConnectedSepConv(input_tensor, num_filters, **kwargs):
    """
    Implement Densely Connected Separable Convolution.

    Parameters:
    -----------
    input_tensor : Tensor
        Input tensor for convolution operation.
    num_filters : int
        Number of filters for convolution operation.
    **kwargs : dict
        Additional keyword arguments for convolution operation.

    Returns:
    -----------
    Tensor
        Output tensor after Densely Connected Separable Convolution.
    """
    conv = SeparableConv1D(num_filters, (1), padding='same', depth_multiplier=1, **kwargs)(input_tensor)
    return concatenate([input_tensor, conv], axis=-1)


def feature_extractor_from_traces(SD_input_1, kernel_regularizer):
    """
    Define the part of the model where the traces features are extracted.

    Parameters:
    -----------
    SD_input_1 : Tensor
        Input tensor representing the traces.
    kernel_regularizer : Regularizer
        Regularizer for kernel weights.

    Returns:
    -----------
    Tensor
        Output tensor after feature extraction.
    """
    x = Conv2D(64, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(SD_input_1)
    x = Conv2D(64, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(32, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(32, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = Conv2D(16, kernel_size=(1, 10), strides=(1, 1), padding="valid", kernel_regularizer=kernel_regularizer)(x)
    x = Reshape((3, 16))(x)
    return x


def feature_extractor_stations(x, kernel_regularizer):
    """
    Define the feature extraction part of the model, considering all the variables at the station level.

    Parameters:
    -----------
    x : Tensor
        Input tensor representing the variables of each stations.
    kernel_regularizer : Regularizer
        Regularizer for kernel weights.

    Returns:
    -----------
    Tensor
        Output tensor after feature extraction for stations.
    """
    num_filters = 17
    for i in range(3):
        x = DenselyConnectedSepConv(x, num_filters, kernel_regularizer=kernel_regularizer)
        num_filters *= 2

    x = Conv1D(64, kernel_size=(3,), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = Conv1D(128, kernel_size=(3,), padding="same", kernel_regularizer=kernel_regularizer)(x)
    return x

def classification_layer(x, kernel_regularizer):
    """
    Define the classification part of the model.

    Parameters:
    -----------
    x : Tensor
        Input tensor representing the extracted features and the information at the event level.
    kernel_regularizer : Regularizer
        Regularizer for kernel weights.

    Returns:
    -----------
    Single value
        Final output.
    """
    #x = Dropout(0.2)(x)
    x = Dense(128, kernel_regularizer=kernel_regularizer)(x)
    x = Dropout(0.1)(x)
    x = Dense(64, kernel_regularizer=kernel_regularizer)(x)
    x = Dense(10, kernel_regularizer=kernel_regularizer)(x)
    #x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)
    return output


def define_layers(kernel_regularizer, SD_input_1, SD_input_2, SD_input_3, SD_input_4, SD_input_5):
    """
    Define the CNN model architecture.

    Parameters:
    -----------
    kernel_regularizer : Regularizer
        Regularization applied to the kernel weights.
    SD_input_1 : Tensor
        Input tensor for SD_input_1 (traces).
    SD_input_2 : Tensor
        Input tensor for SD_input_2 (distances).
    SD_input_3 : Tensor
        Input tensor for SD_input_3 (event-level info).
    SD_input_4 : Tensor
        Input tensor for SD_input_4 (azimuths).
    SD_input_5 : Tensor
        Input tensor for SD_input_5 (Stot values).

    Returns:
    -----------
    Model
        CNN model architecture.
    """

    #The actual model:
    x = feature_extractor_from_traces(SD_input_1, kernel_regularizer)

    x = concatenate([x, SD_input_2], axis=-1)
    x = concatenate([x, SD_input_4], axis=-1)
    x = concatenate([x, SD_input_5], axis=-1)

    x = feature_extractor_stations(x, kernel_regularizer)
    
    x = Flatten()(x)
    x = concatenate([x, SD_input_3], axis=-1)

    output = classification_layer(x, kernel_regularizer)

    model = K.models.Model([SD_input_1, SD_input_2, SD_input_3, SD_input_4, SD_input_5], [output])
    print(model.summary())

    return model
