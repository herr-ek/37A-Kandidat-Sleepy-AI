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
            f"[green]✓[/green] Dataset assembled: "
            f"{X.shape[0]} windows, {X.shape[1]} features, "
            f"{int(y.sum())} apnea / {int((y == 0).sum())} non-apnea"
        )
        return X, y, feature_cols

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

    def train(self, model: IModel, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model and report."""
        self.console.print("\n[bold cyan]Training model...[/bold cyan]")
        model.train(X_train, y_train)
        self.console.print("[green]✓[/green] Training complete.")
        return model

    def evaluate(self, model: IModel, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate the model and display a results table.

        Returns:
            metrics: Dict of metric names and values."""
        self.console.print("\n[bold cyan]Evaluating model...[/bold cyan]")
        metrics = model.evaluate(X_test, y_test)

        table = Table(title="Evaluation Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        for name, value in metrics.items():
            table.add_row(name, f"{value:.4f}")
        self.console.print(table)
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
        output_path = output_dir / f"{model_name}.joblib"
        model.save(str(output_path))

        meta = {
            "model": type(model).__name__,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "hyperparameters": hyperparams or {},
            "evaluation": {k: round(float(v), 6) for k, v in (metrics or {}).items()},
            "records": records or [],
            "features": feature_names or [],
        }
        meta_path = output_dir / f"{model_name}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        self.console.print(f"[green]✓[/green] Model saved to {output_path}")
        self.console.print(f"[green]✓[/green] Metadata saved to {meta_path}")
        return output_path
