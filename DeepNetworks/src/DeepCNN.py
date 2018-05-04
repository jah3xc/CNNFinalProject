from keras.models import Sequential, Model
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.optimizers import SGD


class DeepCNN:
    def __init__(self,
                 dirname,
                 img_size=256,
                 residual=False,
                 num_epochs=20,
                 num_folds=5):
        self.dirname = dirname
        self.residual = residual
        self.img_size = img_size
        self.num_classes = 0
        self.num_epochs = num_epochs
        self.num_folds = num_folds

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

        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

    def fit(self):
        """
        Train the network
        """
        pass
