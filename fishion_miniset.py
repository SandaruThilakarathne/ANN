import tensorflow as tf
import numpy as np
from tensorflow import keras


# defining the callbacks
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') < 0.9:
            print("\nreaching 90% accuracy. So cancelling")
            self.model.stop_training = True


callbacks = myCallback()
# loading data
fashion_miniset = keras.datasets.fashion_mnist

# splitting data in to training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_miniset.load_data()


# defining the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# starting training
model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])

# testing the results
model.evaluate(test_images, test_labels)

# cnn development

import tensorflow as tf
print(tf.__version__)
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images /255.0
testing_images = testing_images.reshape(10000, 28, 28, 1)
testing_images = testing_images/255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(testing_images, testing_labels, epochs=5, callbacks=[callbacks])
test_loss = model.evaluate(testing_images, testing_labels)

