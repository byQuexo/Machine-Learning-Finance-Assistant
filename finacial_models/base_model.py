from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all financial prediction models."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False

    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess the input data for the model."""
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame, target_column: str) -> None:
        """Train the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate predictions for the specified number of periods."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model."""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format and requirements."""
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")
        return True