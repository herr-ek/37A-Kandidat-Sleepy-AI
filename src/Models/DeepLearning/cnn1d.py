import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..IModel import IModel
except ImportError:
    from Models.IModel import IModel


class _CNN1DNet(nn.Module):
    """Internal 1-D convolutional network for sequence classification."""

    def __init__(self, window_size: int, num_filters: int, hidden_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # Two MaxPool1d(2) layers halve the length twice: window_size → window_size // 4
        conv_out_len = window_size // 4
        self.fc = nn.Sequential(
            nn.Linear(conv_out_len * num_filters * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, W) → (B, 1, W)
        x = self.conv(x)  # (B, num_filters*2, W//4)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x).squeeze(1)  # (B,)


class CNN1D(IModel):
    """1-D CNN for apnea classification on raw SaO2 windows.

    Unlike the classical models, this model is trained on raw sliding-window
    signal data rather than hand-crafted features. Window length must match
    the ``window_size`` used when building the dataset.
    """

    FILE_EXTENSION = ".pt"

    def __init__(
        self,
        window_size: int = 60,
        num_filters: int = 16,
        hidden_size: int = 64,
        num_epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ):
        self.window_size = window_size
        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._build_net()

    def _build_net(self):
        self.net = _CNN1DNet(self.window_size, self.num_filters, self.hidden_size).to(
            self.device
        )

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size, shuffle=True
        )

        # Balanced weighting: pos_weight = n_negative / n_positive
        classes = np.unique(y)
        if len(classes) == 2:
            weights = compute_class_weight("balanced", classes=classes, y=y)
            pos_weight = torch.tensor(
                [weights[1] / weights[0]], dtype=torch.float32
            ).to(self.device)
        else:
            pos_weight = torch.ones(1).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.net.train()
        for _ in range(self.num_epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                criterion(self.net(Xb), yb).backward()
                optimizer.step()

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = torch.sigmoid(self.net(X_t))
        return (probs >= 0.5).cpu().numpy().astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
        }

    def save(self, file_path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "window_size": self.window_size,
                "num_filters": self.num_filters,
                "hidden_size": self.hidden_size,
            },
            file_path,
        )

    def load(self, file_path: str) -> "CNN1D":
        ckpt = torch.load(file_path, map_location=self.device)
        self.window_size = ckpt["window_size"]
        self.num_filters = ckpt["num_filters"]
        self.hidden_size = ckpt["hidden_size"]
        self._build_net()
        self.net.load_state_dict(ckpt["state_dict"])
        return self
