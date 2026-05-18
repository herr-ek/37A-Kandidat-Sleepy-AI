import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_sample_weight

try:
    from ..IModel import IModel
except ImportError:
    from Models.IModel import IModel


class KNN(IModel):
    def __init__(self, n_neighbors: int = 5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
        self.weight = 0.5

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        self.weight = len(y_tr[y_tr == 1]) / len(y_tr)
        self.model.fit(X_tr, y_tr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(
            [
                1 if pred > self.weight else 0
                for pred in self.model.predict_proba(X)[:, 1]
            ]
        )

    def save(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "KNN":
        self.model = joblib.load(file_path)
        return self
