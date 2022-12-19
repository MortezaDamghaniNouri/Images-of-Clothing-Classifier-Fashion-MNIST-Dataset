"""
This script gets the address of an CNN trained model and evaluates the input model using the
Fashion MNIST test dataset. Finally,  it prints the accuracy of the input model.
"""

import tensorflow as tf
from keras.models import load_model
from keras.utils import np_utils

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
test_labels = np_utils.to_categorical(test_labels)
model_address = input("Enter the full address of the model which you want to evaluate (example: models//model.h5): ")
model = load_model(model_address)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Accuracy: " + str(round(test_acc * 100, 2)) + " %")








