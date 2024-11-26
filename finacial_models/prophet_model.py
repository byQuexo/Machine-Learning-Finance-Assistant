from prophet import Prophet
import pandas as pd
from finacial_models.base_model import BaseModel


class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        self.last_values = None

    def get_model_name(self) -> str:
        """Return the name of the model.

        Returns:
            str: The model's name
        """
        return "Prophet"

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for Prophet model.

        Args:
            data: Input DataFrame containing financial data

        Returns:
            DataFrame: Processed data in Prophet's required format
        """
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data.values.flatten()
        })
        return prophet_data

    def train(self, data: pd.DataFrame, target_column: str) -> None:
        """Train the Prophet model on the provided data.

        Args:
            data: Training data DataFrame
            target_column: Column name containing target values
        """
        self.validate_data(data)
        self.last_values = data[target_column].values
        prophet_data = self.preprocess_data(data[[target_column]])
        self.model.fit(prophet_data)
        self.is_trained = True

    def predict(self, periods: int) -> pd.DataFrame:
        """Generate predictions for future periods.

        Args:
            periods: Number of periods to forecast

        Returns:
            DataFrame: Predictions with 'prediction' column
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        future = self.model.make_future_dataframe(periods=periods, freq='ME')
        forecast = self.model.predict(future)

        last_date = forecast['ds'].iloc[len(self.last_values) - 1]
        future_forecast = forecast[forecast['ds'] > last_date]

        return pd.DataFrame({'prediction': future_forecast['yhat'].values})