"""
Training manager for the Sleep Data Analysis Pipeline CLI.

Orchestrates the interactive flow: record selection → dataset assembly →
model configuration → train/test split → training → evaluation → save.
"""

import sys
from math import e
from pathlib import Path

import questionary

try:
    from .batch_processor import BatchProcessor
    from .config import MODELS_DIR, PROCESSED_DIR
    from .data_loader import DataLoader
    from .data_processor import DataProcessor
    from .data_saver import DataSaver
    from .training import TrainingSession
except ImportError:
    from batch_processor import BatchProcessor
    from config import MODELS_DIR, PROCESSED_DIR
    from data_loader import DataLoader
    from data_processor import DataProcessor
    from data_saver import DataSaver
    from training import TrainingSession


# Registry of available models: name → (class, default hyperparams)
def _build_model_registry() -> dict:
    registry = {}

    try:
        try:
            from ..Models.KNN import KNN
        except ImportError:
            from Models.KNN import KNN
        registry["KNN"] = (KNN, {"n_neighbors": 5})
    except ImportError:
        pass

    try:
        try:
            from ..Models.RandomForest import RandomForest
        except ImportError:
            from Models.RandomForest import RandomForest
        registry["Random Forest"] = (
            RandomForest,
            {"n_estimators": 100, "max_depth": 10},
        )
    except ImportError:
        pass

    try:
        try:
            from ..Models.SVM import SVM
        except ImportError:
            from Models.SVM import SVM
        registry["SVM"] = (SVM, {"C": 1.0})
    except ImportError:
        pass

    try:
        try:
            from ..Models.DeepLearning import CNN1D
        except ImportError:
            from Models.DeepLearning import CNN1D
        registry["Deep (1D CNN)"] = (
            CNN1D,
            {
                "window_size": 60,
                "num_filters": 16,
                "hidden_size": 64,
                "num_epochs": 20,
                "batch_size": 1024,
                "lr": 1e-3,
                "max_pos_weight": 10.0,
                "verbose": True,
            },
        )
    except ImportError:
        pass

    try:
        try:
            from ..Models.DeepLearning import FullyConnected
        except ImportError:
            from Models.DeepLearning import FullyConnected
        registry["Deep (FC)"] = (
            FullyConnected,
            {
                "window_size": 60,
                "hidden_sizes": [128, 64],
                "dropout": 0.3,
                "num_epochs": 20,
                "batch_size": 1024,
                "lr": 1e-3,
                "max_pos_weight": 10.0,
                "verbose": True,
            },
        )
    except ImportError:
        pass

    try:
        try:
            from ..Models.DeepLearning import RNN
        except ImportError:
            from Models.DeepLearning import RNN
        registry["Deep (RNN)"] = (
            RNN,
            {
                "window_size": 60,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.3,
                "num_epochs": 20,
                "batch_size": 1024,
                "lr": 1e-3,
                "max_pos_weight": 10.0,
                "verbose": True,
            },
        )
    except ImportError:
        pass

    return registry


class TrainingManager:
    """Handles all interactive training steps inside the CLI."""

    def __init__(self, console, style):
        self.console = console
        self.style = style
        self.session = TrainingSession(console)
        self.batch_processor = BatchProcessor(console)
        self.data_loader = DataLoader(console)
        self.data_processor = DataProcessor(console)
        self.data_saver = DataSaver(console)

    # ------------------------------------------------------------------
    # Top-level entry point called from cli.py
    # ------------------------------------------------------------------

    def run(self):
        """Interactive training flow."""
        self.console.print(
            "\n[bold cyan]─── Model Training ───────────────────────────────[/bold cyan]"
        )

        # 1. Model selection and hyperparameter configuration (determines data path)
        result = self._configure_model()
        if result is None:
            return
        model, hyperparams = result
        is_deep = getattr(model, "FILE_EXTENSION", ".joblib") == ".pt"

        if is_deep:
            # ── Deep model path: raw SpO2 windows ────────────────────────
            all_records = self.session.find_records_with_processed_data()
            if not all_records:
                self.console.print(
                    "[red]✗ No processed files found in data/processed/.[/red]"
                )
                return

            labelled = [r for r in all_records if r["has_labels"]]
            if not labelled:
                self.console.print(
                    "[red]✗ None of the processed files contain label columns.[/red]"
                )
                return

            self._display_raw_records(labelled)

            selected_names = questionary.checkbox(
                "Select records to include in the training set:",
                choices=[
                    questionary.Choice(
                        f"{r['name']}  ({r['n_samples']} samples)", value=r["name"]
                    )
                    for r in labelled
                ],
                style=self.style,
            ).ask()
            if not selected_names:
                self.console.print("[yellow]✗ No records selected, exiting.[/yellow]")
                return

            window_size = hyperparams.get("window_size", 60)
            try:
                X, y = self.session.build_raw_dataset(
                    selected_names, window_size=window_size
                )
            except ValueError as exc:
                self.console.print(f"[red]✗ {exc}[/red]")
                return
            feature_names = None

        else:
            # ── Classical model path: hand-crafted features ───────────────
            all_records = self.session.find_records_with_features()
            if not all_records:
                self.console.print(
                    "[red]✗ No feature files found in data/processed/.\n"
                    "  Run the batch pipeline with 'Extract features' first.[/red]"
                )
                return

            self.session.display_available_records(all_records)

            labelled = [r for r in all_records if r["has_labels"]]
            if not labelled:
                self.console.print(
                    "[red]✗ None of the feature files contain an 'apnea_event' label column.\n"
                    "  Re-extract features with training=True.[/red]"
                )
                return

            selected_names = questionary.checkbox(
                "Select records to include in the training set:",
                choices=[
                    questionary.Choice(
                        f"{r['name']}  ({r['n_windows']} windows)", value=r["name"]
                    )
                    for r in labelled
                ],
                style=self.style,
            ).ask()
            if not selected_names:
                self.console.print("[yellow]✗ No records selected, exiting.[/yellow]")
                return

            self._ensure_feature_coverage(selected_names)

            use_normalized = questionary.confirm(
                "Use normalized features (if available)?",
                default=True,
                style=self.style,
            ).ask()
            if use_normalized is None:
                return
            try:
                X, y, feature_names = self.session.build_dataset(
                    selected_names, use_normalized=use_normalized
                )
            except ValueError as exc:
                self.console.print(f"[red]✗ {exc}[/red]")
                return

        # 3. Train / test split
        test_pct = questionary.select(
            "Test set size:",
            choices=[
                questionary.Choice("10 %", value=0.10),
                questionary.Choice("20 %  (recommended)", value=0.20),
                questionary.Choice("30 %", value=0.30),
            ],
            style=self.style,
        ).ask()
        if test_pct is None:
            return

        X_train, X_test, y_train, y_test = self.session.split_dataset(
            X, y, test_size=test_pct
        )

        # 4. Train
        self.session.train(model, X_train, y_train)

        # 5. Evaluate
        metrics = self.session.evaluate(model, X_test, y_test)

        # 6. Save
        save = questionary.confirm("Save trained model to disk?", default=True).ask()
        if save:
            model_name = questionary.text(
                "Model filename (without extension):",
                default=f"{type(model).__name__.lower()}_apnea",
            ).ask()

            if model_name:
                if not is_deep:
                    hyperparams["use_normalized"] = use_normalized
                self.session.save_model(
                    model,
                    model_name,
                    MODELS_DIR,
                    hyperparams=hyperparams,
                    metrics=metrics,
                    records=selected_names,
                    feature_names=feature_names,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _display_raw_records(self, records: list[dict]):
        """Print a summary table of records with processed (raw) files."""
        from rich import box
        from rich.table import Table

        table = Table(title="Records with Processed Data", box=box.ROUNDED)
        table.add_column("Record", style="green")
        table.add_column("Samples", justify="right", style="cyan")
        table.add_column("Labels", justify="center")
        for r in records:
            label_status = "[green]✓[/green]" if r["has_labels"] else "[red]✗[/red]"
            table.add_row(r["name"], str(r["n_samples"]), label_status)
        self.console.print(table)

    def _ensure_feature_coverage(self, selected_names: list[str]):
        """Check that all selected records have feature files; offer to extract missing ones."""
        missing = [
            n
            for n in selected_names
            if not (PROCESSED_DIR / n / f"{n}_features.parquet").exists()
        ]
        if not missing:
            return

        self.console.print(
            f"\n[yellow]⚠ {len(missing)} record(s) are missing feature files:[/yellow]"
        )
        for n in missing:
            self.console.print(f"  [dim]• {n}[/dim]")

        proceed = questionary.confirm(
            "Extract features for the missing records now?", default=True
        ).ask()
        if not proceed:
            self.console.print(
                "[red]✗ Cannot build a complete training set without all features. Aborting.[/red]"
            )
            sys.exit(1)

        for record in missing:
            self.console.print(f"\n[bold]── Extracting features: {record} ──[/bold]")
            self.data_loader.clear()
            try:
                df = self.data_loader.load_data(
                    "processed", [record], load_features_mode="skip"
                )
                if df is None:
                    self.console.print(
                        f"[red]✗ Could not load {record}, skipping.[/red]"
                    )
                    continue
                processed_df, _ = self.data_processor.preprocess_signal(df)
                features_df, success = self.data_processor.extract_features(
                    processed_df
                )
                if success:
                    self.data_saver.save_features(features_df, [record])
            except Exception as exc:
                self.console.print(f"[red]✗ Failed for {record}: {exc}[/red]")

    def _configure_model(self):
        """Interactive model selection and hyperparameter configuration."""
        registry = _build_model_registry()
        if not registry:
            self.console.print("[red]✗ No model implementations found.[/red]")
            return None
        self.console.print("\n[bold]Available models:[/bold]")
        for name in registry:
            self.console.print(f"  [dim]• {name}[/dim]")
        model_name = questionary.select(
            "Select model:",
            choices=[questionary.Choice(name, value=name) for name in registry],
            style=self.style,
        ).ask()
        if model_name is None:
            return None

        model_cls, defaults = registry[model_name]
        params = dict(defaults)

        self.console.print(f"\n[bold]Configure {model_name} hyperparameters[/bold]")
        for param, default_val in defaults.items():
            raw = questionary.text(f"  {param}:", default=str(default_val)).ask()
            if raw is None:
                return None
            # Preserve type of the default value
            try:
                if isinstance(default_val, list):
                    import ast

                    params[param] = ast.literal_eval(raw)
                else:
                    params[param] = type(default_val)(raw)
            except (ValueError, TypeError, SyntaxError):
                self.console.print(
                    f"[yellow]⚠ Invalid value for {param}, using default {default_val}[/yellow]"
                )

        return model_cls(**params), params
