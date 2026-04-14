"""
Training module for the Sleep Data Analysis Pipeline CLI.

Handles dataset assembly from feature parquet files, train/test splitting,
model training, evaluation, and model persistence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.metrics import confusion_matrix

try:
    from ..Models import IModel
except ImportError:
    from Models import IModel

try:
    from .config import PROCESSED_DIR
except ImportError:
    from config import PROCESSED_DIR

LABEL_COLUMN = "apnea_event"
NON_FEATURE_COLUMNS = {"time_s", LABEL_COLUMN}


class TrainingSession:
    """Assembles a training dataset from feature parquet files and trains a model."""

    def __init__(self, console: Console):
        self.console = console

    # ------------------------------------------------------------------
    # Dataset assembly
    # ------------------------------------------------------------------

    def find_records_with_features(self) -> list[dict]:
        """Scan processed directory and return info about each record's feature files."""
        records = []
        for record_dir in sorted(PROCESSED_DIR.iterdir()):
            if not record_dir.is_dir():
                continue
            feature_file = record_dir / f"{record_dir.name}_features.parquet"
            if not feature_file.exists():
                continue
            try:
                df = pd.read_parquet(feature_file)
                has_labels = LABEL_COLUMN in df.columns
                has_normalized = self._check_normalized(record_dir.name)
                records.append(
                    {
                        "name": record_dir.name,
                        "path": feature_file,
                        "n_windows": len(df),
                        "has_normalized": has_normalized,
                        "has_labels": has_labels,
                    }
                )
            except Exception:
                pass
        return records

    def _check_normalized(self, record_name: str) -> bool:
        """Check if normalized feature file exists for the given record."""
        normalized_file = (
            PROCESSED_DIR / record_name / f"{record_name}_features_normalized.parquet"
        )
        return normalized_file.exists()

    def _get_processed_file(self, record_name: str) -> Optional[Path]:
        """Return the processed parquet path for a record.

        Historical exports used both `<record>.parquet` and
        `<record>_processed.parquet`, so training needs to accept either.
        """
        candidates = [
            PROCESSED_DIR / record_name / f"{record_name}_processed.parquet",
            PROCESSED_DIR / record_name / f"{record_name}.parquet",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def display_available_records(self, records: list[dict]):
        """Print a summary table of records with feature files."""
        table = Table(title="Records with Feature Files", box=box.ROUNDED)
        table.add_column("Record", style="green")
        table.add_column("Windows", justify="right", style="cyan")
        table.add_column("Labels", justify="center")
        table.add_column("Normalized", justify="center")

        for r in records:

            label_status = "[green]✓[/green]" if r["has_labels"] else "[red]✗[/red]"
            normalized_status = (
                "[green]✓[/green]" if r["has_normalized"] else "[red]✗[/red]"
            )
            table.add_row(
                r["name"], str(r["n_windows"]), label_status, normalized_status
            )

        self.console.print(table)

    def build_dataset(
        self, selected_records: list[str], use_normalized: bool = True
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load and concatenate feature files for the selected records.

        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,)
            feature_names: List of feature column names
        """
        frames = []

        for record_name in selected_records:
            if use_normalized and self._check_normalized(record_name):
                feature_file = (
                    PROCESSED_DIR
                    / record_name
                    / f"{record_name}_features_normalized.parquet"
                )
            else:
                feature_file = (
                    PROCESSED_DIR / record_name / f"{record_name}_features.parquet"
                )
            if not feature_file.exists():
                self.console.print(
                    f"[yellow]⚠ Skipping {record_name}: feature file not found[/yellow]"
                )
                continue
            df = pd.read_parquet(feature_file)
            if LABEL_COLUMN not in df.columns:
                self.console.print(
                    f"[yellow]⚠ Skipping {record_name}: no '{LABEL_COLUMN}' column[/yellow]"
                )
                continue
            frames.append(df)

        if not frames:
            raise ValueError(
                "No labelled feature files found for the selected records."
            )

        combined = pd.concat(frames, ignore_index=True).dropna()
        feature_cols = [c for c in combined.columns if c not in NON_FEATURE_COLUMNS]
        X = combined[feature_cols].to_numpy(dtype=np.float64)
        y = combined[LABEL_COLUMN].to_numpy(dtype=np.int64)

        self.console.print(
            f"[green]OK[/green] Dataset assembled: "
            f"{X.shape[0]} windows, {X.shape[1]} features, "
            f"{int(y.sum())} apnea / {int((y == 0).sum())} non-apnea"
        )
        return X, y, feature_cols

    def find_records_with_processed_data(self) -> list[dict]:
        """Scan processed directory and return records that have a processed parquet file."""
        records = []
        for record_dir in sorted(PROCESSED_DIR.iterdir()):
            if not record_dir.is_dir():
                continue
            processed_file = self._get_processed_file(record_dir.name)
            if processed_file is None:
                continue
            try:
                df = pd.read_parquet(processed_file)
                has_labels = "is_apnea" in df.columns and "is_hypopnea" in df.columns
                records.append(
                    {
                        "name": record_dir.name,
                        "path": processed_file,
                        "n_samples": len(df),
                        "has_labels": has_labels,
                    }
                )
            except Exception:
                pass
        return records

    def build_raw_dataset(
        self,
        selected_records: list[str],
        window_size: int = 60,
        step_size: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a dataset of raw SaO2 windows from processed parquet files.

        Args:
            selected_records: Record names to include.
            window_size: Number of samples per window (seconds at 1 Hz).
            step_size: Stride between windows; defaults to window_size // 2.

        Returns:
            X: (n_windows, window_size) float64
            y: (n_windows,) int64
        """
        if step_size is None:
            step_size = window_size // 2

        all_X: list[np.ndarray] = []
        all_y: list[int] = []

        for record_name in selected_records:
            processed_file = self._get_processed_file(record_name)
            if processed_file is None:
                self.console.print(
                    f"[yellow]⚠ Skipping {record_name}: processed file not found[/yellow]"
                )
                continue
            df = pd.read_parquet(processed_file).dropna()
            if "sao2_percent" not in df.columns or "is_apnea" not in df.columns:
                self.console.print(
                    f"[yellow]⚠ Skipping {record_name}: missing required columns[/yellow]"
                )
                continue

            sao2 = df["sao2_percent"].values
            is_apnea = df["is_apnea"].values
            is_hypopnea = df["is_hypopnea"].values
            n = len(sao2)

            for i in range(0, n - window_size + 1, step_size):
                w = sao2[i : i + window_size]
                a = is_apnea[i : i + window_size]
                h = is_hypopnea[i : i + window_size]
                label = (
                    1
                    if (a.sum() >= window_size / 2 or h.sum() >= window_size / 2)
                    else 0
                )
                all_X.append(w)
                all_y.append(label)

        if not all_X:
            raise ValueError("No valid processed files found for the selected records.")

        X = np.array(all_X, dtype=np.float64)
        y = np.array(all_y, dtype=np.int64)
        self.console.print(
            f"[green]OK[/green] Raw dataset assembled: "
            f"{X.shape[0]} windows x {window_size}s, "
            f"{int(y.sum())} apnea / {int((y == 0).sum())} non-apnea"
        )
        return X, y

    def split_dataset(
        self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train/test split."""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        self.console.print(
            f"[dim]Train: {len(X_train)} samples  |  Test: {len(X_test)} samples[/dim]"
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Training & evaluation
    # ------------------------------------------------------------------

    def train(
        self,
        model: IModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> IModel:
        """Fit the model and report."""
        self.console.print("\n[bold cyan]Training model...[/bold cyan]")
        model.train(X_train, y_train, X_val, y_val)
        self.console.print("[green]✓[/green] Training complete.")
        return model

    def evaluate(self, model: IModel, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model and display a results table.

        Returns:
            metrics: Dict of metric names and values.
                - "balanced_accuracy": Balanced accuracy score.
                - "accuracy": Accuracy score.
                - "recall": Macro-averaged recall score.
                - "f1_macro": Macro-averaged F1 score.
        """

        self.console.print("\n[bold cyan]Evaluating model...[/bold cyan]")
        metrics = model.evaluate(X_test, y_test)

        table = Table(title="Evaluation Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        for name, value in metrics.items():
            if name == "confusion_matrix":
                continue
            table.add_row(name, f"{value:.4f}")
        self.console.print(table)

        confusion_matrix = Table(title="Confusion Matrix", box=box.SIMPLE)
        confusion_matrix.add_column("", style="dim")
        confusion_matrix.add_column("Predicted 0", style="green")
        confusion_matrix.add_column("Predicted 1", style="red")
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            confusion_matrix.add_row("Actual 0", str(cm[0, 0]), str(cm[0, 1]))
            confusion_matrix.add_row("Actual 1", str(cm[1, 0]), str(cm[1, 1]))
            self.console.print(confusion_matrix)
        return metrics

    def save_model(
        self,
        model: IModel,
        model_name: str,
        output_dir: Path,
        hyperparams: dict = None,
        metrics: dict = None,
        records: list[str] = None,
        feature_names: list[str] = None,
    ):
        """Save the trained model and a JSON metadata sidecar to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = getattr(model, "FILE_EXTENSION", ".joblib")
        output_path = output_dir / f"{model_name}{ext}"
        model.save(str(output_path))

        meta = {
            "model": type(model).__name__,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "hyperparameters": hyperparams or {},
            "evaluation": {
                k: (v.tolist() if hasattr(v, "tolist") else round(float(v), 6))
                for k, v in (metrics or {}).items()
            },
            "records": records or [],
            "features": feature_names or [],
        }
        meta_path = output_dir / f"{model_name}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        self.console.print(f"[green]✓[/green] Model saved to {output_path}")
        self.console.print(f"[green]✓[/green] Metadata saved to {meta_path}")
        return output_path
