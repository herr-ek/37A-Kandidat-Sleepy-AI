from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)


class IModel(ABC):
    """Abstract base class for all sleep stage classification models."""

    @abstractmethod
    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        """Fit the model to training data.

        Args:
            X_tr: Feature matrix of shape (n_samples, n_features) for training data.
            y_tr: Target labels of shape (n_samples,) for training data.

            -- Optional validation data for per-epoch evaluation (used by deep learning models) --
            X_val (Optional): Feature matrix of shape (n_samples, n_features) for validation data.
            y_val (Optional): Target labels of shape (n_samples,) for validation data.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the given samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model and return performance metrics.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: True labels of shape (n_samples,).

        Returns:
            Dictionary containing metric names and their values.
                - "balanced_accuracy": Balanced accuracy score.
                - "accuracy": Accuracy score.
                - "recall": Macro-averaged recall score.
                - "f1_macro": Macro-averaged F1 score.
        """
        y_pred = self.predict(X)
        return {
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
            "confusion_matrix": confusion_matrix(
                y, y_pred, normalize="true"
            ),  # Placeholder for confusion matrix (can be added if needed)
        }

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
