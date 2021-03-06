"""
Useful functions for the project are written in this file for easy management
and reuse.

Author: Angad Gill
"""

import sys
from keras.models import model_from_json
from keras.models import Sequential
from keras.datasets import cifar10
from keras.utils import np_utils

from matplotlib import pyplot as plt
import matplotlib.cm as cm

import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy import linalg

from multiprocessing import Pool
from multiprocessing import cpu_count


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


def output_at_layer(input_image, model, layer_num, verbose=False):
    """
    This function is used to visualize activations at any layer in a
    Convolutional Neural Network model in Keras. It returns the output image
    for a given input image at the layer_num layer of the model (layer numbers
    starting at 1). The model should be Sequential type. This function will
    not mutate the model.

    Reference: https://github.com/fchollet/keras/issues/431

    The idea is to keep the first layer_num layers of a trained model and
    remove the rest. Then compile the model and use the predict function to get
    the ouput.

    Parameters:
    -----------
        input_image: Image in Numpy format from the dataset
        model: Name to be used with the load_model() function
        layer_num: Layer number between 1 and len(model.layers)
        verbose: Prints layer info

    Returns:
    --------
        output_image: Numpy array of the output at the layer_num layer
    """
    model_temp = Sequential()
    model_temp.layers = model.layers[:layer_num]  # Truncates layers
    model_temp.compile(loss=model.loss, optimizer=model.optimizer)  # Recompiles model_temp
    output_image = model_temp.predict(np.array([input_image]))  # Uses predict to get ouput
    
    if verbose:
        # Print layer and image info
        layer = model_temp.layers[layer_num-1]
        print layer
        try:
            print layer.W_shape
        except:
            pass
        print "Image dimensions: " + str(output_image.shape)
        
    return output_image


def visualize_output(output_image, figsize=(15,15)):
    """
    This function will plot the output_image in a grid using Matplotlib.

    Parameters:
    ----------
        output_image: output produced by the output_at_layer function that
        needs to be visualized.

    Returns:
    -------
        No returns.

    """
    feature_maps = output_image.shape[1]
    rows_columns = int(np.ceil(np.sqrt(feature_maps)))  # To create a square grid

    plt.figure(figsize=figsize)
    for i, image in enumerate(output_image[0]):  # Iterate through all feature maps
        plt.subplot(rows_columns, rows_columns, i+1)
        plt.axis('off')  # Turn off axes to save space on the visualization
        plt.imshow(image.T, cmap=cm.Greys_r, interpolation='none')


# TODO: Combine this function with visualize_output()
def visualize_image_group(image_group, figsize=(15,15)):
    """
    Plot all images in an augmented group in a grid using Matplotlib.

    Parameters:
    ----------
        image_group: group of augmented images as Numpy arrays

    Returns:
    -------
        No returns.

    """
    plt.figure(figsize=(15, 6))
    for i, image in enumerate(image_group):
        plt.subplot(2, len(image_group), i+1)
        plt.axis('off')
        plt.imshow(image.T, interpolation='none')


def load_data():
    """
    Loads CIFAR-10 data using Keras.

    Returns:
    -------
        (X_train, Y_train), (X_test, Y_test) dataset in numpy format

    """
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize
    X_train /= 255
    X_test /= 255

    return (X_train, Y_train), (X_test, Y_test)


def move_image_channel(image):
    """
    Move the channel dimension from first to third

    Parameters:
        image: numpy array with 3 dimensions where channel is first

    Returns:
    -------
        image: numpy array with 3 dimensions where channel is last
    """
    image = np.swapaxes(image, 0, 1)
    image = np.swapaxes(image, 1, 2)
    return image


def augment_data(images, rotations):
    """
    Augment images with rotated versions of the image. This function returns a list of lists; the outer list is used to
    group the original image with this augmented versions.

    Parameters:
    ----------
        images: Numpy array images with channel dimension in the front
        rotations:

    Returns:
    -------
        augmented: list of lists containing the augmented images as Numpy arrays

    """
    pool = Pool(processes=cpu_count())
    data = zip(images, [rotations for _ in range(len(images))])
    augmented = pool.map(_augment_data, data)
    pool.close()
    pool.join()
    return augmented


def _augment_data(data):
    image, rotations = data
    augmented_images = [image]  # Add the original image to the list
    # Rotate and pad with 'nearest' pixels
    augmented_images += [rotate(image.T, r, reshape=False, mode='nearest').T for r in rotations]
    augmented_images = np.array(augmented_images)
    return augmented_images


def augment_image(image):
    """
    Returns a list with orginal image, its covariance matrix,
    and real part of FFT. This keep only the first channel.
    Image must have the first dimension as the channel.

    Parameters:
    -----------
        image: numpy array of (channel, x, y) shape.

    Returns:
    --------
        List containing 3 numpy arrays of (x, y) shape.
    """
    image = image[0]  # Keep the first channel
    images = []
    images += [image]
    images += [np.dot(image, image.T)]
    img_fft = np.fft.fftshift(np.fft.fftn(image))
    img_fft = np.log10(img_fft)
    images += [np.real(img_fft)]
    return images


def augmented_data_batches(data, labels, rotations, batch_size):
    """
    Generator that returns a new batch of augmented data each iteration.
    Example:
        for test_flat, labels_flat in util.augmented_data_batches(X_test, Y_test, [90, 180, 270], 100):
            loss, accuracy = model.evaluate(np.array(test_flat), np.array(labels_flat), show_accuracy=True)

    Parameters:
    -----------
        data: Images to be augmented.
        rotations: Array of angles to rotate the image in. 0-360.
        batch_size: Amount of images to draw from |data| each iteration.

    Returns:
    --------
        Flattened list of augmented and original images.
        Length of returned list is |batch_size| * len(|rotations|).
    """
    for i in xrange(0, data.shape[0], batch_size):
        X_test_aug = augment_data(data[i:i+batch_size], rotations)
        X_test_aug_flat = [x for X_test_group in X_test_aug for x in X_test_group]

        Y_test_aug_flat = []
        for y in labels[i:i+batch_size]:
            Y_test_aug_flat += [y for _ in range(len(rotations)+1)]

        yield X_test_aug_flat, Y_test_aug_flat


def visualize_augmented_images(images):
    """
    Plots input images in a row using Matplotlib. Input images
    must have only one channel.

    Parameters:
    -----------
        images: list containing images in numpy array of (x, y) shape.
    Returns:
    --------
        No return
    """
    plt.figure(figsize=(6,6))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.axis('off')
        plt.imshow(img, interpolation='none', cmap=cm.Greys_r)


def diff_at_layer(image_group, layer_num, model, verbose=True):
    """
    Calculates the difference in neural network layer outputs between the original image
    and its augmented versions.
    The "difference" is the norm of the absolute difference of the output matrices.

    Assumptions:
        First image is the original
        Model is sequential

    Args:
        image_group: list of numpy array is shape (channel, x, y)
        layer_num: int starting with 1, where 1 is the first layer
        model: Keras model
        verbose: Print layer details and diffs

    Returns:
        layer_diffs: list of tuples: (image #, diffs list)

    """
    if verbose:
        print model.layers[layer_num-1]

    # Get output images for image_group
    out = []
    for image in image_group:
        out += [output_at_layer(image, model, layer_num)]

    layer_diffs = []  # list of tuples: (image #, diffs list)

    maps = out[0].shape[1]  # number of feature maps

    for image_num in range(1, len(out)):
        diffs = []
        for m in range(maps):
            # diff_mat = np.abs(out[0][0][m] - out[image_num][0][m])
            diff_mat = out[0][0][m] - out[image_num][0][m]
            diffs += [np.linalg.norm(diff_mat)]
        diffs = np.array(diffs)
        if verbose:
            print "Diff with #%d augmented image at layer %d: min: %0.2f, mean: %0.2f, max: %0.2f" % \
            (image_num, layer_num, diffs.min(), diffs.mean(), diffs.max())
        layer_diffs += [(image_num, diffs)]

    return layer_diffs
