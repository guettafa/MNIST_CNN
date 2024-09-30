import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers

def create_model():
    model = tf.keras.models.Sequential([
        # Input Layer
        layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=[28,28,1]),
        layers.MaxPooling2D(2),

        layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),

        layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),

        # Transform to 1D
        layers.Flatten(), 

        # 2 hidden layers
        layers.Dense(30, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(30, activation='relu'),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(10, activation="softmax")
    ])
    return model


def train_model():
    pass