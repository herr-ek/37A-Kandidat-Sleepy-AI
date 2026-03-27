import numpy as np

from src.Cli.data_loader import DataLoader
from src.Cli.data_processor import DataProcessor
from src.Models import IModel


class TrainingModule:
    """Module responsible for training sleep stage classification models."""

    def __init__(
        self, model: IModel, data_loader: DataLoader, data_processor: DataProcessor
    ):
        self.model = model
        self.data_loader = data_loader
        self.data_processor = data_processor

    def train_model(self) -> IModel:
        """Train the model using the provided data loader."""
        X_train, y_train = self.data_loader.load_training_data()
        self.model.train(X_train, y_train)
        return self.model

    def select_training_data(self, records: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Select and preprocess training data based on user-selected records."""
        df = self.data_loader.load_data_for_records(records)
        X, y = self.data_processor.preprocess_for_training(df)
        return X, y
