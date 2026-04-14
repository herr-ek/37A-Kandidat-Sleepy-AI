import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    from ..IModel import IModel
except ImportError:
    from Models.IModel import IModel


class RandomForest(IModel):
    def __init__(
        self, n_estimators: int = 101, max_depth: int = 10, random_state: int = 42
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        sample_weight = compute_sample_weight("balanced", y)
        self.model.fit(X, y, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
        }

    def save(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "RandomForest":
        self.model = joblib.load(file_path)
        return self
