import os
import sys
import logging


def get_logger(save_path, logger_name, file_name):
    """
    Initialize logger
    """
    with open((os.path.join(save_path, file_name)), 'w') as file:
        print("Create Experiment.log Finish!")
    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, file_name))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger
