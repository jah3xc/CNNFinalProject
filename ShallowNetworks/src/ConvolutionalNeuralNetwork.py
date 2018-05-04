from pathlib import Path
import os
import logging
import numpy as np
import cv2

from keras.applications.imagenet_utils import preprocess_input
from keras.utils import to_categorical
from keras.models import Sequential as Model
from keras.layers import Convolution2D, MaxPooling2D as Pooling2D, Flatten, Dense
from util.util import optimizer, loss, generate_train_data, generate_folds, POOLING_SIZE, KERNEL_SIZE, NUM_FILTERS, NUM_LAYERS, NUM_FOLDS, NUM_EPOCHS, IMG_DIMENSION




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
        self.folds = generate_folds(self.dirname, self.num_folds)

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
            optimizer=optimizer()
            loss=loss()
            metrics=["accuracy"])

    def fit(self):
        """
        Fit the model with cross validation
        """

        # for each fold
        for fold in [chr(i) for i in range(65, 65 + self.num_folds)]:

            logging.info("Training Fold {}".format(fold))
            # create fresh model
            del self.model
            self.compile()
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
            data, classes = generate_train_data(training_files, self.img_dimension)
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
            test_data, test_classes = generate_train_data(
                self.folds[fold], self.img_dimension)
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
            total += float(self.cross_validation_accuracy[score])
        self.average_accuracy = total / self.num_folds
        print("Average Accuracy: {}".format(self.average_accuracy))

    
