from src.setup import setup
from src.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork as CNN
import pathlib
import os
import logging


def final_project():
    """
    Run the final project
    """

    # get args
    args = setup()
    # parse args
    image_location = args["image_directory"]
    filters = args["num_filters"]
    kernel = args["kernel_size"]
    pooling = args["pooling_size"]
    layers = args["num_layers"]
    folds = args["num_folds"]
    epochs = args["num_epochs"]

    cnn = CNN(
        image_location,
        num_filters=filters,
        num_layers=layers,
        pooling_size=pooling,
        kernel_size=kernel,
        num_folds=folds,
        num_epochs=epochs)

    cnn.compile()
    cnn.fit()
