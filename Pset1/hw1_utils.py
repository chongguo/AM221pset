"""
January 2019 Dor Verbin
"""

import gzip
import numpy as np


def load_images():
    """
    Loads the training images from train-images-idx3-ubyte.gz.
    """
    hw = 28 ** 2  # Number of pixels per image
    n = 60000     # Number of images

    with gzip.open('train-images-idx3-ubyte.gz', 'r') as f:
        f.read(16)

        buffer = f.read(hw * n)
        images = np.frombuffer(buffer, dtype=np.uint8)
        images = images.reshape(n, hw)

    return images


def load_labels():
    """
    Loads the training labels from train-labels-idx1-ubyte.gz.
    """
    n = 60000     # Number of images

    with gzip.open('train-labels-idx1-ubyte.gz', 'r') as f:
        f.read(8)

        buffer = f.read(n)
        labels = np.frombuffer(buffer, dtype=np.uint8)
        labels = labels.reshape(n)

    return labels
