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


class _RNNNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)  # (B, seq_len, hidden_size)
        return self.fc(out[:, -1, :]).squeeze(1)  # (B,)


class RNN(IModel):
    """Vanilla RNN for apnea classification on raw SaO2 windows.

    Each window of length ``window_size`` is treated as a sequence of
    scalar observations fed to the RNN one time-step at a time.
    Window length must match the ``window_size`` used when building the dataset.
    """

    FILE_EXTENSION = ".pt"

    def __init__(
        self,
        window_size: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_epochs: int = 20,
        batch_size: int = 1024,
        lr: float = 1e-3,
        max_pos_weight: float = 10.0,
        verbose: bool = True,
        show_progress: bool = False,
    ):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_pos_weight = max_pos_weight
        self.verbose = verbose
        self.show_progress = show_progress
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_mean: np.ndarray | None = None
        self._input_std: np.ndarray | None = None
        self._build_net()

    def _build_net(self):
        self.net = _RNNNet(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.model_selection import train_test_split

        # Hold out 10 % for per-epoch validation (stratified so both splits have apnea)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )

        # Fit input normalisation on training split only
        self._input_mean = X_tr.mean(axis=0)
        self._input_std = X_tr.std(axis=0) + 1e-8
        X_tr_n = (X_tr - self._input_mean) / self._input_std
        X_val_n = (X_val - self._input_mean) / self._input_std

        # Shape: (N, window_size) -> (N, window_size, 1)
        X_t = torch.tensor(X_tr_n, dtype=torch.float32).unsqueeze(-1)
        y_t = torch.tensor(y_tr, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_n, dtype=torch.float32).unsqueeze(-1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type == "cuda",
        )

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
            print(f"Training GRU on device: {self.device}")

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

            # Per-epoch evaluation on the held-out validation split (batched)
            val_pred = self._infer_batched(X_val_t)
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

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        if self._input_mean is not None:
            return (X - self._input_mean) / self._input_std
        return X

    def _infer_batched(self, X_t: torch.Tensor) -> np.ndarray:
        """Run inference in batches to avoid GPU OOM on large tensors."""
        self.net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                Xb = X_t[i : i + self.batch_size].to(
                    self.device, non_blocking=self.device.type == "cuda"
                )
                probs = torch.sigmoid(self.net(Xb))
                preds.append((probs >= 0.5).cpu().numpy().astype(np.int64))
        return np.concatenate(preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.tensor(self._normalise(X), dtype=torch.float32).unsqueeze(-1)
        return self._infer_batched(X_t)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "balanced_accuracy": balanced_accuracy_score(y, y_pred),
            "accuracy": accuracy_score(y, y_pred),
            "recall": recall_score(y, y_pred, average="macro"),
            "f1_macro": f1_score(y, y_pred, average="macro"),
        }

    def save(self, file_path: str) -> None:
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "window_size": self.window_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "input_mean": self._input_mean,
                "input_std": self._input_std,
            },
            file_path,
        )

    def load(self, file_path: str) -> "RNN":
        ckpt = torch.load(file_path, map_location=self.device)
        self.window_size = ckpt["window_size"]
        self.hidden_size = ckpt["hidden_size"]
        self.num_layers = ckpt["num_layers"]
        self.dropout = ckpt["dropout"]
        self._input_mean = ckpt.get("input_mean")
        self._input_std = ckpt.get("input_std")
        self._build_net()
        self.net.load_state_dict(ckpt["state_dict"])
        return self
