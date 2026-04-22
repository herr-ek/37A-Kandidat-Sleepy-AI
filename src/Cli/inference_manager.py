"""
Inference Manager for the Sleep Data Analysis Pipeline CLI.

Handles interactive selection and loading of a trained model,
then runs predictions on a single selected record.
"""

import inspect
import json

import numpy as np
import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.table import Table

try:
    from .config import MODELS_DIR, PROCESSED_DIR
    from .training import LABEL_COLUMN, NON_FEATURE_COLUMNS, TrainingSession
    from .training_manager import _build_model_registry
except ImportError:
    from config import MODELS_DIR, PROCESSED_DIR
    from training import LABEL_COLUMN, NON_FEATURE_COLUMNS, TrainingSession
    from training_manager import _build_model_registry


class InferenceManager:
    """Handles loading a trained model and running predictions on a single record."""

    def __init__(self, console: Console, style):
        self.console = console
        self.style = style
        self.session = TrainingSession(console)

    # ------------------------------------------------------------------
    # Top-level entry point called from cli.py
    # ------------------------------------------------------------------

    def run(self):
        """Interactive inference flow."""
        self.console.print(
            "\n[bold cyan]─── Model Inference ──────────────────────────────[/bold cyan]"
        )

        # 1. Select and load model
        result = self._select_and_load_model()
        if result is None:
            return
        model, meta = result
        is_deep = meta.get("model") in {"CNN1D", "FullyConnected", "RNN"}

        # 2. Select a single record and build its dataset
        try:
            X, y = self._select_and_build_record(is_deep, meta)
        except (ValueError, RuntimeError) as exc:
            self.console.print(f"[red]✗ {exc}[/red]")
            return

        # 3. Predict and display summary
        self.console.print("\n[bold cyan]Running predictions...[/bold cyan]")
        y_pred = model.predict(X)
        self._display_prediction_summary(y_pred)

        # 4. Evaluate against ground truth if labels are available
        if y is not None:
            self.session.evaluate(model, X, y)

    # ------------------------------------------------------------------
    # Model selection and loading
    # ------------------------------------------------------------------

    def _select_and_load_model(self):
        """Prompt user to pick a saved model, instantiate its class, and load weights."""
        model_files = sorted(MODELS_DIR.glob("*.json"))
        if not model_files:
            self.console.print(f"[red]✗ No saved models found in {MODELS_DIR}[/red]")
            return None

        meta_map = {}
        choices = []
        for jf in model_files:
            try:
                meta = json.loads(jf.read_text())
            except Exception:
                continue
            label = f"{jf.stem}  [{meta.get('model', '?')}]"
            choices.append(questionary.Choice(label, value=jf.stem))
            meta_map[jf.stem] = meta

        if not choices:
            self.console.print("[red]✗ No valid model metadata files found.[/red]")
            return None

        chosen_stem = questionary.select(
            "Select a trained model:", choices=choices, style=self.style
        ).ask()
        if chosen_stem is None:
            return None

        meta = meta_map[chosen_stem]
        model = self._instantiate_model(meta)
        if model is None:
            return None

        ext = getattr(model, "FILE_EXTENSION", ".joblib")
        model_path = MODELS_DIR / f"{chosen_stem}{ext}"
        if not model_path.exists():
            self.console.print(f"[red]✗ Model file not found: {model_path}[/red]")
            return None

        self.console.print(f"Loading [bold]{model_path.name}[/bold]...", style="cyan")
        try:
            model.load(str(model_path))
            self.console.print("[green]✓[/green] Model loaded.")
        except Exception as exc:
            self.console.print(f"[red]✗ Failed to load model: {exc}[/red]")
            return None

        return model, meta

    def _instantiate_model(self, meta: dict):
        """Instantiate the correct model class from JSON metadata."""
        model_class_name = meta.get("model")
        hyperparams = meta.get("hyperparameters", {})

        registry = _build_model_registry()
        entry = next(
            (v for v in registry.values() if v[0].__name__ == model_class_name), None
        )
        if entry is None:
            self.console.print(f"[red]✗ Unknown model type '{model_class_name}'.[/red]")
            return None

        model_cls, defaults = entry
        params = {**defaults, **hyperparams}
        valid_params = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
        try:
            model = model_cls(**{k: v for k, v in params.items() if k in valid_params})
        except Exception as exc:
            self.console.print(
                f"[red]✗ Could not instantiate {model_class_name}: {exc}[/red]"
            )
            return None
        return model

    # ------------------------------------------------------------------
    # Record selection and dataset assembly
    # ------------------------------------------------------------------

    def _select_and_build_record(
        self, is_deep: bool, meta: dict
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prompt user to pick one record and build its feature/window arrays."""
        if is_deep:
            return self._build_deep_record(meta)
        return self._build_classical_record(meta)

    def _build_deep_record(self, meta: dict) -> tuple[np.ndarray, np.ndarray | None]:
        records = self.session.find_records_with_processed_data()
        if not records:
            raise ValueError("No processed files found in data/processed/.")

        chosen = questionary.select(
            "Select a record for inference:",
            choices=[
                questionary.Choice(
                    f"{r['name']}  ({r['n_samples']} samples)", value=r["name"]
                )
                for r in records
            ],
            style=self.style,
        ).ask()
        if chosen is None:
            raise RuntimeError("No record selected.")

        window_size = meta.get("hyperparameters", {}).get("window_size", 60)
        X, y = self.session.build_raw_dataset([chosen], window_size=window_size)
        return X, y

    def _build_classical_record(
        self, meta: dict
    ) -> tuple[np.ndarray, np.ndarray | None]:
        records = self.session.find_records_with_features()
        if not records:
            raise ValueError("No feature files found. Run 'Extract features' first.")

        chosen = questionary.select(
            "Select a record for inference:",
            choices=[
                questionary.Choice(
                    f"{r['name']}  ({r['n_windows']} windows)", value=r["name"]
                )
                for r in records
            ],
            style=self.style,
        ).ask()
        if chosen is None:
            raise RuntimeError("No record selected.")

        record_meta = next(r for r in records if r["name"] == chosen)
        use_normalized = record_meta["has_normalized"]
        suffix = (
            "_features_normalized.parquet" if use_normalized else "_features.parquet"
        )
        feature_file = PROCESSED_DIR / chosen / f"{chosen}{suffix}"

        df = pd.read_parquet(feature_file).dropna()

        # Select and order features to match what the model was trained on
        model_features = meta.get("features", [])
        if model_features:
            missing = [f for f in model_features if f not in df.columns]
            if missing:
                self.console.print(
                    f"[yellow]⚠ Record is missing model features: {missing}[/yellow]"
                )
            available = [f for f in model_features if f in df.columns]
            X = df[available].to_numpy(dtype=np.float64)
        else:
            feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
            X = df[feature_cols].to_numpy(dtype=np.float64)

        y = (
            df[LABEL_COLUMN].to_numpy(dtype=np.int64)
            if LABEL_COLUMN in df.columns
            else None
        )
        return X, y

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def _display_prediction_summary(self, y_pred: np.ndarray):
        """Show a count/percentage breakdown of predicted classes."""
        total = len(y_pred)
        n_apnea = int(y_pred.sum())
        n_normal = total - n_apnea

        table = Table(title="Prediction Summary", box=box.ROUNDED)
        table.add_column("Class", style="cyan")
        table.add_column("Windows", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="dim")
        table.add_row("Normal (0)", str(n_normal), f"{100 * n_normal / total:.1f}%")
        table.add_row("Apnea  (1)", str(n_apnea), f"{100 * n_apnea / total:.1f}%")
        self.console.print(table)
