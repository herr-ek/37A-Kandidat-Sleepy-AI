import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import LinearSVC

try:
    from ..IModel import IModel
except ImportError:
    from Models.IModel import IModel


class SVM(IModel):
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.model = LinearSVC(
            C=C,
            random_state=random_state,
            class_weight="balanced",
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

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
