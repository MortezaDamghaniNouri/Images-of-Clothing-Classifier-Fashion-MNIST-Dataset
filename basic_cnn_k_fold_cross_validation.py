

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np
import copy
from keras.utils import np_utils

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
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
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(train_images, train_labels, epochs=3, batch_size=32, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    accuracy_list.append(test_acc)
    i += 1

i = 0
while i < len(accuracy_list):
    print("fold " + str(i + 1) + " accuracy: " + str(round(accuracy_list[i] * 100, 2)) + " %")
    i += 1
summation = 0
for i in accuracy_list:
    summation += round(i * 100, 2)
print("Mean accuracy: " + str(round(summation / len(accuracy_list), 2)) + " %")













































