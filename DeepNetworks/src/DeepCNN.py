from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from util.util import optimizer, loss, generate_folds, NUM_EPOCHS, NUM_FOLDS, IMG_DIMENSION


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
            loss=loss()
            optimizer=optimizer(),
            metrics=['accuracy'])

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
