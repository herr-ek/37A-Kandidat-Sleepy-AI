import joblib
import numpy as np
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

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        sample_weight = compute_sample_weight("balanced", y_tr)
        self.model.fit(X_tr, y_tr, sample_weight=sample_weight)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def save(self, file_path: str) -> None:
        joblib.dump(self.model, file_path)

    def load(self, file_path: str) -> "SVM":
        self.model = joblib.load(file_path)
        return self
