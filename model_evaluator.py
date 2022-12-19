import tensorflow as tf
from keras.models import load_model


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
model_address = input("Enter the full address of the model which you want to evaluate (example: models//model.h5): ")
model = load_model(model_address)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Accuracy: " + str(round(test_acc * 100, 2)) + " %")















