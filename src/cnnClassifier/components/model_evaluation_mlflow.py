import tensorflow as tf
from pathlib import Path
import mlflow
import os
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

        os.environ['MLFLOW_TRACKING_URI'] = self.config.mlflow_uri
        os.environ['MLFLOW_TRACKING_USERNAME'] = "rakibrohan54"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "68d2276306699e6c4c999c53f8494162ae6ca912"

    def _valid_generator(self):
        """Creates the validation data generator."""
        datagenerator_kwargs = dict(
            rescale=1. / 255,
            validation_split=0.20  # Ensure this matches your training split
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Size from your parameters
            batch_size=self.config.params_batch_size,
            color_mode="grayscale",  # Set to grayscale for 1 channel input
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=True,
            class_mode='sparse',  # Use 'sparse' as per your training configuration
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Loads the trained model from the specified path."""
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """Performs model evaluation and logs the results."""
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        """Saves the evaluation scores in a JSON file."""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """Logs parameters and metrics into MLflow."""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                # Register the model
                mlflow.keras.log_model(self.model, "model", registered_model_name="CustomCNNModel")
            else:
                mlflow.keras.log_model(self.model, "model")
