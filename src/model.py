import tensorflow as tf
import pandas as pd
from utils import create_model
from keras import layers

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images, train_labels = x_train / 255, y_train
test_images, test_labels = x_test / 255, y_test

model = create_model()

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

model.fit(
    train_images,
    train_labels,
    batch_size=64,
    epochs=15,
    validation_split=0.2)

# What to predict
x_test = x_test[:1]

prob = model.predict(x_test)

print(f"Proba: {prob.round(2)}")
print(f"Answer: {y_test[0]}")

