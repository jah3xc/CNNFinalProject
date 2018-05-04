from DeepNetworks.src.setup import setup
from DeepNetworks.src.DeepCNN import DeepCNN


def run():
    args = setup()
    directory = args["dirname"]
    img = args["img_size"]
    residual = args["residual"]

    deep_cnn = DeepCNN(directory, img_size=img, residual=residual)
