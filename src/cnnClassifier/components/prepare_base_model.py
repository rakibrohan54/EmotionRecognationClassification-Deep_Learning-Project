import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Build the custom CNN model."""
        self.model = self.build_custom_cnn(
            input_shape=self.config.params_image_size,
            classes=self.config.params_classes
        )
        self.save_model(path=self.config.base_model_path, model=self.model)

    @staticmethod
    def build_custom_cnn(input_shape, classes):
        """Build a custom, efficient CNN model."""
        model = tf.keras.models.Sequential()

        # First Conv block
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Second Conv block
        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Third Conv block
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))

        # Flatten and Dense layers
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))  # Add some dropout to avoid overfitting

        # Output layer
        model.add(tf.keras.layers.Dense(classes, activation='softmax'))

    def update_base_model(self):
        """Update the base model to a full model (same as base in this case)."""
        # No need to prepare or update the full model separately; it's already ready.
        self.full_model = self.model
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the model to a specified path."""
        model.save(path)
