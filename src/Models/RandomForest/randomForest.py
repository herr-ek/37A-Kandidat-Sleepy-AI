import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
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

    def load(self, file_path: str) -> "RandomForest":
        self.model = joblib.load(file_path)
        return self
