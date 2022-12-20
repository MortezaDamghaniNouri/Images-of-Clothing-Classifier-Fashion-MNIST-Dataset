"""
In this script, a classifier is implemented using Convolutional Neural Network with an innovative architecture.
The architecture consists of three convolutional layers, and each of them has sixty four 3 * 3 filters. After each
convolutional layer, there is a maxpooling layer. After the convolutional layers, a Fully Connected Neural Network
is used with one hidden layer which has 128 neurons. 5-fold cross-validation is used for evaluation.
"""

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import copy
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
all_images = []
for i in train_images:
    all_images.append(i)
for i in test_images:
    all_images.append(i)
all_labels = []
for i in train_labels:
    all_labels.append(i)
for i in test_labels:
    all_labels.append(i)

# k-fold cross-validation is implemented here
fold_number = 5
i = 1
accuracy_list = []
while i <= 5:
    print("Fold: " + str(i))
    test_images = []
    test_labels = []
    train_images = []
    train_labels = []
    all_images_copy = copy.deepcopy(all_images)
    all_labels_copy = copy.deepcopy(all_labels)
    j = int((i - 1) * (len(all_images) / fold_number))
    while j < (i * (len(all_images) / fold_number)):
        test_images.append(all_images_copy[j])
        test_labels.append(all_labels_copy[j])
        j += 1
    j = int((i - 1) * (len(all_images) / fold_number))
    k = 1
    while k <= (len(all_images) / fold_number):
        all_images_copy.pop(j)
        all_labels_copy.pop(j)
        k += 1
    train_images = np.array(all_images_copy)
    train_labels = np.array(all_labels_copy)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    # training a model
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    # evaluating the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    accuracy_list.append(test_acc)
    i += 1

# printing the results
i = 0
while i < len(accuracy_list):
    print("fold " + str(i + 1) + " accuracy: " + str(round(accuracy_list[i] * 100, 2)) + " %")
    i += 1
summation = 0
for i in accuracy_list:
    summation += round(i * 100, 2)
print("Mean accuracy: " + str(round(summation / len(accuracy_list), 2)) + " %")


