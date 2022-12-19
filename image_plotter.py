"""
This script gets the index of an image in the Fashion MNIST dataset and plots the image.
"""

import tensorflow as tf
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
plt.imshow(train_images[0])
plt.show()








