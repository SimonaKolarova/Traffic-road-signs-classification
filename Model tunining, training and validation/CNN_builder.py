import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def build_model(img_height, img_width, n_classes, 
                conv_pool_pairs, conv_kernel, pool_kernel, conv_filters, conv_filters_rate,
                hidden_dropout_pairs, hidden_layer_size, hidden_layer_rate, dropout_rate, summary):
    """
    Builds and compiles a convolutional neural network model 
    given the required parameters.
    """

    # Create a sequential neural network
    model = tf.keras.models.Sequential()

    # Convolutional and max pooling layers pairs
    for i in range(conv_pool_pairs):
        if i == 0:
            # First convolutional layer 
            model.add(tf.keras.layers.Conv2D( 
                conv_filters, 
                (conv_kernel, conv_kernel), 
                activation="relu", 
                padding = "same", 
                input_shape=(img_height, img_width, 3)))
        else:
            # Additional convolutional layers 
            model.add(tf.keras.layers.Conv2D(
                conv_filters*conv_filters_rate*(conv_pool_pairs-1), 
                (conv_kernel, conv_kernel), 
                activation="relu", 
                padding = "same"))

        # Max-pooling layer 
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(pool_kernel, pool_kernel)))


    # Flatten units
    model.add(tf.keras.layers.Flatten())

    # Hidden and dropoutlayers
    for i in range(hidden_dropout_pairs):
        
        if i == 0:
            # First hidden layers
            model.add(tf.keras.layers.Dense(
                hidden_layer_size, 
                activation="relu"))
        else:
            # Additional hidden layers
            model.add(tf.keras.layers.Dense(
                hidden_layer_size*hidden_layer_rate*(hidden_dropout_pairs-1), 
                activation="relu"))

        if dropout_rate != 0:
            # Dropout layer
            model.add(tf.keras.layers.Dropout(dropout_rate))

    # Output layer with an output unit for each image category
    model.add(tf.keras.layers.Dense(
        n_classes, 
        activation="softmax"))

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    if summary == "True":
        model.summary()

    return model


def train_model(images, labels, test_size, model, epochs):
    """
    Splits the training `images` and `labels` data into 
    training and testing subsets using of size (1-`test_size`) and `test_size`, respectively.

    Fit provided `model` on the training subset.
    Evaluate `model` performance on the train and test subset.

    Return fitted `model` and performance metrics.
    """
    # Categotical labels
    labels = tf.keras.utils.to_categorical(labels)
    
    # Split data into training and testing subsets
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), 
        np.array(labels), 
        test_size=test_size)

    # Fit and evaluate model performance on training subset
    tt_performance = model.fit(
        x_train, y_train, 
        epochs=epochs,
        validation_data=(x_test, y_test))

    return tt_performance