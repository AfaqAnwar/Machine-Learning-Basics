import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

"""
Beginner / Simple MNIST hand written digit classification.
@Author Afaq Anwar
@Version 02/28/2019
"""

# Download and load the data.
digits_mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = digits_mnist.load_data()

print("Training Image Details: " + str(train_images.shape) + "\n60,000 images of 28 x 28 hand written digits. \n" +
      "Each pixel with a value of 0 - 255.")

print("Labels are integers 0-9. First digit of the training images is a 5. " + str(train_labels[0]))

print("Testing Image & Label Size: " + str(len(test_images)))

# Visual Representation.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Preprocess the training images to have a value between 0-1. Essentially scaling the data.
train_images = train_images / 255
test_images = test_images / 255

# Tries to load the existing model, otherwise creates and saves a new model.
try:
    model = tf.keras.models.load_model('digit_classifier_model.h5')
    print("Model loaded.")
except:
    print("File not found.")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trains the model.
    model.fit(train_images, train_labels, epochs=5)

    # Calculates the loss and accuracy. This model has about 98% accuracy.
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_accuracy)

    # Save the entire model along with all weights, biases and optimizations.
    model.save('digit_classifier_model.h5')
