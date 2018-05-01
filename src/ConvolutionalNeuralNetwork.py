from pathlib import Path
import os
import logging
import numpy as np
import cv2

from keras.models import Sequential as Model
from keras.layers import Convolution2D, MaxPooling2D as Pooling2D, Flatten, Dense

# define default values
POOLING_SIZE = 2
KERNEL_SIZE = 3
NUM_FILTERS = 32
NUM_LAYERS = 3
NUM_FOLDS = 5
NUM_EPOCHS = 10
IMG_DIMENSION = 299


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
                logging.warning(
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
                    input_shape=(self.img_dimension, self.img_dimension))
                first_layer = False
            else:
                conv_layer = Convolution2D(
                    self.num_filters, (self.kernel_size, self.kernel_size),
                    activation="relu")
            # create pooling layer
            pool_layer = Pooling2D()
            # add the layers to the model
            self.model.add(conv_layer)
            self.model.add(pool_layer)
        # add flattening layer
        flat = Flatten()
        self.model.add(flat)
        # add the dense layer
        dense = Dense(self.num_classes, activation="softmax")
        self.model.add(dense)
        # compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

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
            training_files = []
            for f in self.folds:
                if f != fold:
                    training_files.append(self.folds[f])
            data = self.images_to_arrays(training_files)

        # fit the model
        self.model.fit(data, epochs=self.num_epochs)

    def images_to_arrays(self, array):
        """
        Convert an array of images to a
        numpy array for training
        """
        #create data
        data = np.empty((len(array), self.img_dimension, self.img_dimension))
        # foreach image
        for i, img in enumerate(array):
            # read image
            data[i] = cv2.imread(str(img))
        # return
        return data
