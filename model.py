'''
Team yWaste's code submission to NUS Statistics DSC Hack 21.
Build using tensorflow, opencv, numpy and pandas.
Data paths are the unziped dataset in the root directory of the repo.
TODO:
1. Regular background masking and contour count for image detection,
2. use custom class to indentify where handwriting is, then run tesseract or which ever hand writing model
'''
import os
import cv2
import typing
import imutils
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from imutils.contours import sort_contours

# Tensorflow memory growth limit (gpu)
MEMORY_SIZE = 1024 * 4   # 4GB gpu memory
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_SIZE)])


# LOGGER
# create logger
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


# CONSTANTS
MAX_RGB = 255.0
CURR_DIR = os.getcwd()

TRAIN_DIR = os.path.join(CURR_DIR, 'train_data')
TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'train_images')
TRAIN_LABELS = os.path.join(TRAIN_DIR, 'train_labels.csv')

VAL_DIR = os.path.join(CURR_DIR, 'validation_data')
VAL_IMG_DIR = os.path.join(VAL_DIR, 'validation_images')
VAL_LABELS = os.path.join(VAL_DIR, 'validation_labels.csv')


def read_img_from_dir(image_dir: str) -> typing.List[np.ndarray]:
    '''
    Helper function to read in image files using cv2.
    TODO:
    1. resize image
    2. match with labels
    '''
    logger.info("Start reading images from directory: " + image_dir)
    imgs = []
    # Reading images in RGB numpy array
    for root, dirs, filenames in os.walk(image_dir):
        for filename in filenames:
            filepath = os.path.join(image_dir, filename)
            image = cv2.imread(filepath)
            image = image / MAX_RGB
            imgs.append(image)
            break
        logger.info("Finish reading images from directory: " + image_dir)
        break
    return imgs


if __name__ == '__main__':
    logger.info(gpus)
    logger.info("GPUs Available: %s" %
                tf.config.experimental.list_physical_devices('GPU'))
    images = read_img_from_dir(TRAIN_IMG_DIR)
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        memory = tf.config.experimental.get_memory_usage('GPU:0')
        logger.info(memory)
