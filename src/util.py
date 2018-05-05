from keras.optimizers import SGD
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
import numpy as np
import cv2
import logging
from pathlib import Path

# define default values
POOLING_SIZE = 2
KERNEL_SIZE = 3
NUM_FILTERS = 32
NUM_LAYERS = 2
NUM_FOLDS = 5
NUM_EPOCHS = 10
IMG_DIMENSION = 256


def optimizer():
    """
    Get the optimizer to use for training
    :return: optimizer object
    """
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    return opt


def loss():
    """
    Get the loss function
    :return: loss function to use 
    """
    return "categorical_crossentropy"


def generate_folds(dirname, num_folds):
    """
    Generate folds
    :param dirname: the directory with images
    :param num_folds: the number of folds
    :return: dict of folds with 'A' -> [] for each fold
    """
    folds = {}
    files = Path(dirname).rglob("*")
    # for each file
    for i, f in enumerate(files):
        # get the fold num
        fold_num = i % num_folds
        # switch from int to char
        fold_letter = chr(65 + fold_num)
        # insert if not present
        if fold_letter not in folds:
            folds[fold_letter] = []
        # check that we have a valid extension
        if str(f).split(".")[-1] not in ["png", "tif", "jpg"
                                         ] or not Path(f).is_file():
            logging.info(
                "Found file {}, which is not an image. Skipping..".format(
                    str(f)))
            continue
        # append file to correct fold
        folds[fold_letter].append(str(f))

    return folds


def get_class_labels(array, invalid_images):
    """
    Generate array of class labels from an
    array of image paths
    :param array: the array of filenames
    :param invalid_images: the indexes in the array that are invalid images
    :return: class labels for files in array
    """

    # init
    data = []

    # foreach file
    for file in array:
        # get the class
        c = str(file).split("/")[-2]
        # save it
        data.append(c)

    # switch from str to int
    numerical_labels = []
    # map the string to the number
    label_map = {}
    # the next number used
    max_num = 0
    # for each label
    for label in data:
        # check if we have a label in the map
        if label not in label_map:
            label_map[label] = max_num
            max_num += 1
        # append the numerical label
        numerical_labels.append(label_map[label])
    # delete invalid ones
    numerical_labels = np.delete(numerical_labels, invalid_images)
    # one hot encode
    return to_categorical(numerical_labels)


def generate_train_data(filenames, img_size):
    """
    Convert an array of images to a
    numpy array for training
    :param filenames: the array of filenames
    :param img_size: the size of the image
    :return: data and classes for training
    """
    #create data
    data = np.empty((len(filenames), img_size, img_size, 3))
    # foreach image
    invalid_images = []
    for i, fname in enumerate(filenames):
        # read image
        img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
        if img.shape != (img_size, img_size, 3):

            logging.info("Found image that was not {}x{}: {}".format(
                img_size, img_size, fname))
            invalid_images.append(i)
        else:
            data[i] = preprocess_input(img.astype(np.float32), mode="tf")

    data = np.delete(data, invalid_images, axis=0)
    classes = get_class_labels(filenames, invalid_images)
    # return
    return data, classes
