import pathlib
import os

# define default values
POOLING_SIZE = 2
KERNEL_SIZE = 3
NUM_FILTERS = 32
NUM_LAYERS = 3
NUM_FOLDS = 5


class ConvolutionalNeuralNetwork:
    def __init__(self,
                 dirname,
                 pooling_size=POOLING_SIZE,
                 kernel_size=KERNEL_SIZE,
                 num_filters=NUM_FILTERS,
                 num_layers=NUM_LAYERS,
                 num_folds=NUM_FOLDS):
        """
        Constructor
        :param dirname: the directory with the images
        :param pooling_size: the pooling window size
        :param kernel_size: the kernel window size
        :param num_filters: number of convolutional filters
        :param num_layers: the number of convolutional and pooling filters
        """
        # save values
        self.dirname = dirname
        self.pooling_size = pooling_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_folds = num_folds

        # param chck
        if not pathlib.Path(dirname).is_dir():
            logging.critical("{} does not exist!".format(dirname))

        # generate folds
        self.folds = {}
        files = pathlib.Path(self.dirname).rglob("*")
        i = 0
        for f in files:
            fold_num = i % num_folds
            i += 1
            fold_letter = chr(65 + fold_num)
            if fold_letter not in self.folds:
                self.folds[fold_letter] = []

            self.folds[fold_letter].append(str(f))
