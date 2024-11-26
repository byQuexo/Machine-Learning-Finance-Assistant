from typing import Dict, List
import pandas as pd
from finacial_models.base_model import BaseModel
from finacial_models.lstm_model import LSTMModel
from finacial_models.prophet_model import ProphetModel


class ModelManager:
    def __init__(self):
        """Initialize the model manager with available models."""
        self.models: Dict[str, BaseModel] = {}
        self.available_models = {
            'lstm': LSTMModel,
            'prophet': ProphetModel
        }
        self.is_trained = False

    def add_model(self, model_name: str) -> None:
        """Add a new model instance to the manager.

        Args:
            model_name: The name of the model to add
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not supported")

        try:
            model_class = self.available_models[model_name]
            if model_name == 'lstm':
                self.models[model_name] = model_class(sequence_length=3)
            else:
                self.models[model_name] = model_class()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {model_name} model: {str(e)}")

    def train_model(self, model_name: str, data: pd.DataFrame, target_column: str) -> None:
        """Train a specific model with the provided data.

        Args:
            model_name: The name of the model to train
            data: The training data
            target_column: The target column for prediction
        """
        if model_name not in self.models:
            self.add_model(model_name)

        try:
            self.models[model_name].train(data, target_column)
            self.is_trained = True
        except Exception as e:
            raise RuntimeError(f"Failed to train {model_name} model: {str(e)}")

    def train_all_models(self, data: pd.DataFrame, target_column: str) -> None:
        """Train all available models with the provided data."""
        for model_name in self.available_models.keys():
            self.train_model(model_name, data, target_column)

    def get_prediction(self, model_name: str, periods: int) -> pd.DataFrame:
        """Get predictions from a specific model.

        Args:
            model_name: The name of the model to use for prediction
            periods: Number of periods to forecast

        Returns:
            DataFrame containing the predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} has not been trained")

        return self.models[model_name].predict(periods)

    def has_trained_models(self) -> bool:
        """Check if any models have been trained."""
        return self.is_trained

    def get_available_models(self) -> List[str]:
        """Get a list of available model names."""
        return list(self.available_models.keys())