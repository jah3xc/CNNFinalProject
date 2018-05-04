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
        pass
