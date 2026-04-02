import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

try:
    from ..IModel import IModel
except ImportError:
    from Models.IModel import IModel


class SVM(IModel):
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LinearSVC(
            C=C,
            random_state=random_state,
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

    def load(self, file_path: str) -> "SVM":
        self.model = joblib.load(file_path)
        return self
