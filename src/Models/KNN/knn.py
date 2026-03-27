import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.Models.IModel import IModel


class KNN(IModel):
    def __init__(self, n_neighbors: int = 5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro"),
        }

    def save(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)

    def load(self, file_path):
        self.model = joblib.load(file_path)
        return self
