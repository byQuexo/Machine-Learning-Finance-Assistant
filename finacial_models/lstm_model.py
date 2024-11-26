from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from finacial_models.base_model import BaseModel
import numpy as np
import pandas as pd


class LSTMModel(BaseModel):
    def __init__(self, sequence_length: int = 3):
        super().__init__()
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.last_values = None
        self.model = None
        self.is_trained = False

    def get_model_name(self) -> str:
        return "LSTM"

    def preprocess_data(self, data: pd.DataFrame) -> tuple:
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def build_model(self, input_shape: tuple) -> None:
        self.model = Sequential([
            LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(40, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')

    def train(self, data: pd.DataFrame, target_column: str) -> None:
        self.validate_data(data)

        # Store the original data values for later use in predictions
        self.last_values = data[target_column].values.copy()

        # Prepare and scale the data
        X, y = self.preprocess_data(data[[target_column]])

        # Build and train the model
        self.build_model((self.sequence_length, 1))
        self.model.fit(X, y, epochs=100, batch_size=32, verbose=0)

        self.is_trained = True

    def predict(self, periods: int) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        if self.last_values is None:
            raise RuntimeError("No training data available for prediction")

        # Prepare the last sequence for prediction
        last_sequence = self.scaler.transform(
            self.last_values[-self.sequence_length:].reshape(-1, 1)
        )

        # Generate predictions
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(periods):
            next_pred = self.model.predict(
                current_sequence.reshape(1, self.sequence_length, 1),
                verbose=0
            )
            predictions.append(next_pred[0, 0])
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred

        # Transform predictions back to original scale
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        return pd.DataFrame({'prediction': predictions.flatten()})