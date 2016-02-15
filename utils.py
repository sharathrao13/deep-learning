"""
Useful functions for the project are written in this file for easy management
and reuse.

Author: Angad Gill
"""

import sys
from keras.models import model_from_json


def save_model(model, name):
    """
    Saves a Keras model to disk as two files: a .json with description of the
    architecture and a .h5 with model weights

    Reference: http://keras.io/faq/#how-can-i-save-a-keras-model

    Parameteres:
    ------------
    model: Keras model that needs to be saved to disk
    name: Name of the model contained in the file names:
        <name>_architecture.json
        <name>_weights.h5

    Returns:
    --------
    True: Completed successfully
    False: Error while saving. The function will print the error.
    """
    try:
        # Uses 'with' to ensure that file is closed properly
        with open(name + '_architecture.json', 'w') as f:
            f.write(model.to_json())
        # Uses overwrite to avoid confirmation prompt
        model.save_weights(name + '_weights.h5', overwrite=True)
        return True  # Save successful
    except:
        print sys.exc_info()  # Prints exceptions
        return False  # Save failed


def load_model(name):
    """
    Loads a Keras model from disk. The model should be contained in two files:
    a .json with description of the architecture and a .h5 with model weights.

    See save_model() to save the model.

    Reference: http://keras.io/faq/#how-can-i-save-a-keras-model

    Parameters:
    -----------
    name: Name of the model contained in the file names:
        <name>_architecture.json
        <name>_weights.h5

    Returns:
    --------
    model: Keras model object.
    """
    # Uses 'with' to ensure that file is closed properly
    with open(name + '_architecture.json') as f:
        model = model_from_json(f.read())
    model.load_weights(name + '_weights.h5')
    return model


def output_at_layer(input_image, model, layer_num):
    """
    This function is used to visualize activations at any layer in a
    Convolutional Neural Network model in Keras. It returns the output image
    for a given input image at the layer_num layer of the model (layer numbers
    starting at 1).

    WARNING: This function will change the model. After using this function,
    you will need to reload/recreate the model if you want to use it for
    anything else.

    Reference: https://github.com/fchollet/keras/issues/431

    The idea is to keep the first layer_num layers of a trained model and
    remove the rest. Then compile the model and use the predict function to get
    the ouput.

    Parameters:
    -----------
    input_image: Numpy array that the model can accept
    model: Trained Keras model
    layer_num: Layer number between 1 and len(model.layers)

    Returns:
    --------
    output_image: Numpy array of the output at the layer_num layer
    """
    model.layers = model.layers[:layer_num]  # Truncates layers
    model.compile(loss=model.loss, optimizer=model.optimizer)  # Recompiles model
    output_image = model.predict(input_image)  # Uses predict to get ouput
    return output_image
