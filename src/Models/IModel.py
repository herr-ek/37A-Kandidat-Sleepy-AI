from abc import ABC, abstractmethod

import numpy as np


class IModel(ABC):
    """Abstract base class for all sleep stage classification models."""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model and return performance metrics.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            Dictionary containing metric names and their values.
        """

    @abstractmethod
    def save(self, file_path: str) -> None:
        """Save the model to a file.

        Args:
            file_path: Path to the file where the model should be saved.
        """

    @abstractmethod
    def load(self, file_path: str) -> None:
        """Load the model from a file.

        Args:
            file_path: Path to the file from which the model should be loaded.
        """
