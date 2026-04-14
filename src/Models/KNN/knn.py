import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

try:
    from ..IModel import IModel
except ImportError:
    from Models.IModel import IModel


class KNN(IModel):
    def __init__(self, n_neighbors: int = 5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        self.model.fit(X_tr, y_tr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "KNN":
        self.model = joblib.load(file_path)
        return self
