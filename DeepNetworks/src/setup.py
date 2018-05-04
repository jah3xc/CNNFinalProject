import argparse
from pathlib import Path
import logging


def setup():
    """
    Setup for running the DCNN
    """

    # create parser
    parser = argparse.ArgumentParser()

    parser.add_argument("dirname", type=str, help="Directory of images")
    parser.add_argument("--img_size", type=int, default=299, help="Image size")
    parser.add_argument("--residual", action=store_true, help="CNN or RNN")
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