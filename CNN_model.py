from keras import models
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import regularizers

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
    conv = layers.SeparableConv1D(num_filters, (1), padding='same', depth_multiplier=1, **kwargs)(input_tensor)
    return layers.concatenate([input_tensor, conv], axis=-1)


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
    x = layers.Conv2D(64, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(SD_input_1)
    x = layers.Conv2D(64, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = layers.Conv2D(32, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = layers.Conv2D(32, kernel_size=(1, 7), strides=(1, 2), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = layers.Conv2D(16, kernel_size=(1, 10), strides=(1, 1), padding="valid", kernel_regularizer=kernel_regularizer)(x)
    x = layers.Reshape((3, 16))(x)
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

    x = layers.Conv1D(12, kernel_size=(3,), padding="same", kernel_regularizer=kernel_regularizer)(x)
    x = layers.Conv1D(128, kernel_size=(3,), padding="same", kernel_regularizer=kernel_regularizer)(x)
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
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(128, kernel_regularizer=kernel_regularizer)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, kernel_regularizer=kernel_regularizer)(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    return output


def define_model(SD_input_1, SD_input_2, SD_input_3, SD_input_4, SD_input_5):
    """
    Define the CNN model architecture.

    Parameters:
    -----------
    SD_input_1 : Tensor
        Input tensor for SD_input_1.
    SD_input_2 : Tensor
        Input tensor for SD_input_2.
    SD_input_3 : Tensor
        Input tensor for SD_input_3.
    SD_input_4 : Tensor
        Input tensor for SD_input_4.
    SD_input_5 : Tensor
        Input tensor for SD_input_5.

    Returns:
    -----------
    Model
        CNN model architecture.
    """
    # Define model hyperparameters
    kernel_regularizer = regularizers.l1_l2(l1=0.0001, l2=0.0001)
    adam_optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=5)
    reduce_learning_rate = callbacks.ReduceLROnPlateau(monitor='val_Distance', factor=0.4, patience=17, verbose=1, mode='min', min_delta=1, cooldown=0, min_lr=0.0)

    #The actual model:
    x = feature_extractor_from_traces(SD_input_1, kernel_regularizer)

    x = layers.concatenate([x, SD_input_2], axis=-1)
    x = layers.concatenate([x, SD_input_4], axis=-1)
    x = layers.concatenate([x, SD_input_5], axis=-1)

    x = feature_extractor_stations(x, kernel_regularizer)
    
    x = layers.Flatten()(x)
    x = layers.concatenate([x, SD_input_3], axis=-1)

    output = classification_layer(x, kernel_regularizer)

    model = models.Model([SD_input_1, SD_input_2, SD_input_3, SD_input_4, SD_input_5], [output])
    print(model.summary())

    model.compile(loss='bce', optimizer=adam_optimizer, metrics=["accuracy"])
    return model