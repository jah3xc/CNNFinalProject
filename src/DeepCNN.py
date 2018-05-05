import argparse
import logging
from pathlib import Path

from keras.models import Sequential, Model
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from src.util import optimizer, loss, generate_folds, generate_train_data, NUM_EPOCHS, NUM_FOLDS, IMG_DIMENSION


def run():
    args = setup()
    directory = args["dirname"]
    img = args["img_size"]
    residual = args["residual"]

    deep_cnn = DeepCNN(directory, img_size=img, residual=residual)
    deep_cnn.compile()
    deep_cnn.fit()


def setup():
    """
    Setup for running the DCNN
    """

    # create parser
    parser = argparse.ArgumentParser()

    parser.add_argument("dirname", type=str, help="Directory of images")
    parser.add_argument("--img_size", type=int, default=299, help="Image size")
    parser.add_argument("--residual", action="store_true", help="CNN or RNN")
    parser.add_argument(
        "--num_folds",
        type=int,
        help="Number of folds for cross validation",
        default=5)
    parser.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="WARNING",
        help="Logging level")
    args = vars(parser.parse_args())

    # extract logging
    logging.basicConfig(level=args["log"])

    # check that dirname exists
    if not Path(args["dirname"]).is_dir():
        logging.critical("Image Directory does not exist!")
        exit(1)

    if args["num_folds"] < 2:
        logging.critical("There must be 2 or more folds!")
        exit(1)

    return args


class DeepCNN:
    def __init__(self,
                 dirname,
                 img_size=IMG_DIMENSION,
                 residual=False,
                 num_epochs=NUM_EPOCHS,
                 num_folds=NUM_FOLDS):
        self.dirname = dirname
        self.residual = residual
        self.img_size = img_size
        self.num_classes = 0
        self.num_epochs = num_epochs
        self.num_folds = num_folds
        self.cross_validation_accuracy = {}
        self.average_accuracy = 0.
        self.model = None

        # get the folds
        self.folds = generate_folds(self.dirname, self.num_folds)

        # find the number of classes
        all_files = Path(self.dirname).glob("*")
        # check for directories
        for f in all_files:
            if Path(f).is_dir():
                self.num_classes += 1

    def compile(self):
        """
        Compile the model
        """
        if self.residual:
            stock_model = ResNet50(
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3),
                pooling='max')
        else:
            stock_model = Xception(
                include_top=False,
                input_shape=(self.img_size, self.img_size, 3),
                pooling='max')

        top_model = Sequential()
        top_model.add(
            Dense(
                self.num_classes,
                activation='softmax',
                input_shape=stock_model.output_shape[1:]))
        self.model = Model(
            inputs=stock_model.input, outputs=top_model(stock_model.output))

        self.model.compile(
            loss=loss(), optimizer=optimizer(), metrics=['accuracy'])

    def fit(self):
        """
        Train the network
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
            data, classes = generate_train_data(training_files, self.img_size)
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
                self.folds[fold], self.img_size)
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
