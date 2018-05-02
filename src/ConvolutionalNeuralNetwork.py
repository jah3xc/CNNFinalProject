from pathlib import Path
import os
import logging
import numpy as np
import cv2

from keras.utils import to_categorical
from keras.models import Sequential as Model
from keras.layers import Convolution2D, MaxPooling2D as Pooling2D, Flatten, Dense

# define default values
POOLING_SIZE = 2
KERNEL_SIZE = 3
NUM_FILTERS = 32
NUM_LAYERS = 2
NUM_FOLDS = 5
NUM_EPOCHS = 10
IMG_DIMENSION = 256


class ConvolutionalNeuralNetwork:
    def __init__(self,
                 dirname,
                 pooling_size=POOLING_SIZE,
                 kernel_size=KERNEL_SIZE,
                 num_filters=NUM_FILTERS,
                 num_layers=NUM_LAYERS,
                 num_folds=NUM_FOLDS,
                 num_epochs=NUM_EPOCHS,
                 img_dimension=IMG_DIMENSION):
        """
        Constructor
        :param dirname: the directory with the images
        :param pooling_size: the pooling window size
        :param kernel_size: the kernel window size
        :param num_filters: number of convolutional filters
        :param num_layers: the number of convolutional and pooling filters
        :param num_folds: the number of folds
        """
        # save values
        self.dirname = dirname
        self.pooling_size = pooling_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_folds = num_folds
        self.num_epochs = num_epochs
        self.img_dimension = img_dimension
        self.num_classes = 0
        self.folds = {}
        self.cross_validation_accuracy = {}
        self.average_accuracy = 0.
        self.model = None

        # param chck
        if not Path(dirname).is_dir():
            logging.critical("{} does not exist!".format(dirname))

        # generate folds

        files = Path(self.dirname).rglob("*")
        # for each file
        for i, f in enumerate(files):
            # get the fold num
            fold_num = i % num_folds
            # switch from int to char
            fold_letter = chr(65 + fold_num)
            # insert if not present
            if fold_letter not in self.folds:
                self.folds[fold_letter] = []
            # check that we have a valid extension
            if str(f).split(".")[-1] not in ["png", "tif", "jpg"
                                             ] or not Path(f).is_file():
                logging.info(
                    "Found file {}, which is not an image. Skipping..".format(
                        str(f)))
                continue

            # append file to correct fold
            self.folds[fold_letter].append(str(f))

        # find the number of classes
        all_files = Path(self.dirname).glob("*")
        # check for directories
        for f in all_files:
            if Path(f).is_dir():
                self.num_classes += 1

    def compile(self):
        """
        Compile a model
        """

        # create model object
        self.model = Model()
        # is it the first layer
        first_layer = True
        # for each layer
        for i in range(self.num_layers):
            # create convolutional layer
            if first_layer:
                conv_layer = Convolution2D(
                    self.num_filters, (self.kernel_size, self.kernel_size),
                    activation="relu",
                    input_shape=(self.img_dimension, self.img_dimension, 3))
                first_layer = False
            else:
                conv_layer = Convolution2D(
                    self.num_filters, (self.kernel_size, self.kernel_size),
                    activation="relu")
            ## add convolutional layer
            self.model.add(conv_layer)

        # create pooling layer
        pool_layer = Pooling2D(
            pool_size=(self.pooling_size, self.pooling_size))
        # add the layers to the model
        self.model.add(pool_layer)
        # add flattening layer
        flat = Flatten()
        self.model.add(flat)
        # add the dense layer
        dense = Dense(128, activation="relu")
        self.model.add(dense)
        output = Dense(self.num_classes, activation="softmax")
        self.model.add(output)
        # compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=["accuracy"])

    def fit(self):
        """
        Fit the model with cross validation
        """
        if self.model is None:
            logging.critical(
                "Cannot fit without model! Please call the compile() function first!"
            )
            return

        # for each fold
        for fold in self.folds:
            logging.info("Training Fold {}".format(fold))
            # genearate filelist for training this fold
            training_files = []
            for f in self.folds:
                if f != fold:
                    training_files.append(self.folds[f])
            # flatten the array
            training_files = [
                item for sublist in training_files for item in sublist
            ]
            # generate the training data
            data, classes = self.generate_train_data(training_files)
            # sanity check
            if len(data) != len(classes):
                logging.critical(
                    "Training Data and Labels have different lengths ({} vs {})!".
                    format(len(data), len(classes)))
                return

            logging.debug("Found {} training images!".format(len(data)))

            # fit the model
            try:
                self.model.fit(data, classes, epochs=self.num_epochs)
            except Exception as err:
                logging.critical("Could not train! An error occurred")
                logging.debug(err)
                return

            # test the model
            test_data, test_classes = self.generate_train_data(
                self.folds[fold])
            # sanity check
            if len(test_data) != len(test_classes):
                logging.critical(
                    "Testing Data and Labels have different lengths ({} vs {})!".
                    format(len(test_data), len(test_classes)))
                return
            # eval
            try:
                loss, accuracy = self.model.evaluate(test_data, test_classes)
                self.cross_validation_accuracy[fold] = accuracy
            except Exception as err:
                logging.critical("Could not test! An error occurred")
                logging.debug(err)
                return

            print("Fold {} Accuracy: {}".format(
                fold, self.cross_validation_accuracy[fold]))

        total = 0.
        for score in self.cross_validation_accuracy:
            total += score
        self.average_accuracy = total / self.num_folds
        print("Average Accuracy: {}".format(self.average_accuracy))

    def get_class_labels(self, array, invalid_images):
        """
        Generate array of class labels from an
        array of image paths
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

    def generate_train_data(self, array):
        """
        Convert an array of images to a
        numpy array for training
        """
        #create data
        data = np.empty((len(array), self.img_dimension, self.img_dimension,
                         3))
        # foreach image
        invalid_images = []
        for i, fname in enumerate(array):
            # read image
            img = cv2.imread(str(fname), cv2.IMREAD_COLOR)
            if img.shape != (self.img_dimension, self.img_dimension, 3):

                logging.info("Found image that was not {}x{}: {}".format(
                    self.img_dimension, self.img_dimension, fname))
                invalid_images.append(i)
            else:
                data[i] = img

        data = np.delete(data, invalid_images, axis=0)
        classes = self.get_class_labels(array, invalid_images)
        # return
        return data, classes