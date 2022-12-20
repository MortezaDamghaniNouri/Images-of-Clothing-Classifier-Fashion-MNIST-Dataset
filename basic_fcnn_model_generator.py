"""
This script is written according to the TensorFlow website. In this script, a  model is trained based
on the Fashion MNIST dataset. Fully Connected Neural Network is used for training the model. The FCNN has one hidden layer with 128 neurons.
"""

import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
# evaluating the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# saving the trained model
model.save("model.h5")
print("Accuracy: " + str(round(test_acc * 100, 2)) + " %")






