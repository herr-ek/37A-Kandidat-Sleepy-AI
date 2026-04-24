import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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
        max_pos_weight: float = 10.0,
        verbose: bool = True,
        show_progress: bool = False,
        device: str | None = None,
    ):
        self.window_size = window_size
        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_pos_weight = max_pos_weight
        self.verbose = verbose
        self.show_progress = show_progress
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._build_net()

    def _build_net(self):
        self.net = _CNN1DNet(self.window_size, self.num_filters, self.hidden_size).to(
            self.device
        )

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> None:
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr
            )

        X_t = torch.tensor(X_tr, dtype=torch.float32)
        y_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

        # Balanced weighting: pos_weight = n_negative / n_positive, capped to
        # avoid extreme over-correction on highly imbalanced datasets.
        classes = np.unique(y_tr)
        if len(classes) == 2:
            weights = compute_class_weight("balanced", classes=classes, y=y_tr)
            raw_pw = weights[1] / weights[0]
            capped_pw = min(raw_pw, self.max_pos_weight)
            if self.verbose and raw_pw != capped_pw:
                print(f"pos_weight capped: {raw_pw:.1f} -> {capped_pw:.1f}")
            pos_weight = torch.tensor([capped_pw], dtype=torch.float32).to(self.device)
        else:
            pos_weight = torch.ones(1).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        if self.verbose:
            print(f"Training CNN1D on device: {self.device}")

        self.net.train()
        epoch_bar = tqdm(
            range(self.num_epochs),
            desc="Training",
            unit="epoch",
            disable=not self.show_progress,
        )
        for epoch in epoch_bar:
            epoch_loss = 0.0
            batch_bar = tqdm(
                loader,
                desc=f"  Epoch {epoch + 1}/{self.num_epochs}",
                unit="batch",
                leave=False,
                disable=not self.show_progress,
            )
            for Xb, yb in batch_bar:
                Xb = Xb.to(self.device, non_blocking=self.device.type == "cuda")
                yb = yb.to(self.device, non_blocking=self.device.type == "cuda")
                optimizer.zero_grad()
                loss = criterion(self.net(Xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(Xb)
                if self.show_progress:
                    batch_bar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / max(len(X_t), 1)

            # Per-epoch evaluation on the held-out validation split
            self.net.eval()
            with torch.no_grad():
                val_Xb = X_val_t.to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                val_probs = torch.sigmoid(self.net(val_Xb))
                val_pred = (val_probs >= 0.5).cpu().numpy().astype(np.int64)
            val_ba = balanced_accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            self.net.train()

            if self.verbose:
                print(
                    f"  Epoch {epoch + 1:>{len(str(self.num_epochs))}}/{self.num_epochs}"
                    f"  loss={avg_loss:.4f}"
                    f"  bal_acc={val_ba:.3f}"
                    f"  f1={val_f1:.3f}"
                )

            if self.show_progress:
                epoch_bar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    bal_acc=f"{val_ba:.3f}",
                    f1=f"{val_f1:.3f}",
                )

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            probs = torch.sigmoid(self.net(X_t))
        return (probs >= 0.5).cpu().numpy().astype(np.int64)

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
        ckpt = torch.load(file_path, map_location=self.device, weights_only=False)
        self.window_size = ckpt["window_size"]
        self.num_filters = ckpt["num_filters"]
        self.hidden_size = ckpt["hidden_size"]
        self._build_net()
        self.net.load_state_dict(ckpt["state_dict"])
        return self
