from argparse import ArgumentParser
import logging
from pathlib import Path


def setup():
    """
    Setup for running the project
    :return: Parsed Arguments
    """
    parser = ArgumentParser()
    parser.add_argument("image_directory", type=str, help="Location of images")
    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="CRITICAL",
        help="Logging level")
    args = vars(parser.parse_args())

    # extract logging
    logging.basicConfig(level=args["log"])

    # make sure imgpath exists
    if not Path(args["image_directory"]).is_dir():
        logging.critical("Image Directory does not exist!")
        exit(1)

    return args
