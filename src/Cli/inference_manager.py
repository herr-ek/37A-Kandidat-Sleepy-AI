"""
Inference Manager for the Sleep Data Analysis Pipeline CLI.

Handles interactive selection and loading of a trained model,
then runs predictions on a single selected record.
"""

import inspect
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
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
            X, y, record_ctx = self._select_and_build_record(is_deep, meta)
        except (ValueError, RuntimeError) as exc:
            self.console.print(f"[red]✗ {exc}[/red]")
            return

        # 3. Predict and display summary
        self.console.print("\n[bold cyan]Running predictions...[/bold cyan]")
        y_pred = model.predict(X)
        self._display_prediction_summary(y_pred)

        # 4. Reconstruct time series, compute AHI, offer plot
        try:
            ts_df = self._reconstruct_timeseries(y_pred, record_ctx)
            self._display_ahi(ts_df)
            plot = questionary.confirm(
                "Plot SaO2 signal with apnea annotations?", default=False
            ).ask()
            if plot:
                self._plot_signal(ts_df, record_ctx["name"])
        except Exception as exc:
            self.console.print(f"[yellow]⚠ Could not compute AHI: {exc}[/yellow]")

        # 5. Evaluate against ground truth if labels are available
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
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Prompt user to pick one record and build its feature/window arrays."""
        if is_deep:
            return self._build_deep_record(meta)
        return self._build_classical_record(meta)

    def _build_deep_record(
        self, meta: dict
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
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
        step_size = window_size // 2
        X, y = self.session.build_raw_dataset([chosen], window_size=window_size)
        n_windows = len(X)
        window_starts = np.arange(n_windows, dtype=np.float64) * step_size
        record_ctx = {
            "name": chosen,
            "window_size": window_size,
            "window_starts": window_starts,
        }
        return X, y, record_ctx

    def _build_classical_record(
        self, meta: dict
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
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

        # Derive window_size from time_s: time_s = time[i + window_size - 1], so
        # window_size = time_s[0] + 1 (since time starts at 0).
        time_s_vals = df["time_s"].values
        window_size = int(round(time_s_vals[0] + 1)) if len(time_s_vals) > 0 else 10
        # window_starts[i] = time_s[i] - (window_size - 1)  (start = end - size + 1)
        window_starts = time_s_vals - (window_size - 1)

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
        record_ctx = {
            "name": chosen,
            "window_size": window_size,
            "window_starts": window_starts,
        }
        return X, y, record_ctx

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

    # ------------------------------------------------------------------
    # Time-series reconstruction, AHI, and plotting
    # ------------------------------------------------------------------

    def _reconstruct_timeseries(
        self, y_pred: np.ndarray, record_ctx: dict
    ) -> pd.DataFrame:
        """Map window predictions back to a per-second time series via majority vote."""
        record_name = record_ctx["name"]
        processed_file = (
            PROCESSED_DIR / record_name / f"{record_name}_processed.parquet"
        )
        df = pd.read_parquet(processed_file)

        n = len(df)
        window_size = record_ctx["window_size"]
        window_starts = record_ctx["window_starts"]

        votes = np.zeros(n, dtype=np.int32)
        counts = np.zeros(n, dtype=np.int32)

        for i, pred in enumerate(y_pred):
            start = int(window_starts[i])
            end = min(start + window_size, n)
            if start >= n or start < 0:
                continue
            votes[start:end] += int(pred)
            counts[start:end] += 1

        covered = counts > 0
        predicted = np.zeros(n, dtype=np.int32)
        predicted[covered] = (votes[covered] / counts[covered] >= 0.5).astype(np.int32)
        df["predicted_apnea"] = predicted
        return df

    @staticmethod
    def _compute_ahi_stats(binary: np.ndarray, total_hours: float) -> dict:
        """Count contiguous apnea events and derive AHI statistics from a binary array."""
        n_events = 0
        in_event = False
        event_lengths: list[int] = []
        current_len = 0
        for v in binary:
            if v and current_len <= 40:
                current_len += 1
                if not in_event:
                    n_events += 1
                    in_event = True
            else:
                if in_event:
                    event_lengths.append(current_len)
                    current_len = 0
                in_event = False
        if in_event:
            event_lengths.append(current_len)

        return {
            "n_events": n_events,
            "ahi": n_events / total_hours if total_hours > 0 else 0.0,
            "avg_duration": float(np.mean(event_lengths)) if event_lengths else 0.0,
            "apnea_minutes": binary.sum() / 60.0,
        }

    @staticmethod
    def _ahi_severity(ahi: float) -> tuple[str, str]:
        if ahi < 5:
            return "Normal", "green"
        elif ahi < 15:
            return "Mild", "yellow"
        elif ahi < 30:
            return "Moderate", "orange3"
        return "Severe", "red"

    def _display_ahi(self, df: pd.DataFrame) -> None:
        """Compute and display AHI statistics, with ground-truth comparison if available."""
        total_seconds = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) + 1.0
        total_hours = total_seconds / 3600.0

        pred_stats = self._compute_ahi_stats(df["predicted_apnea"].values, total_hours)

        has_gt = "is_apnea" in df.columns and "is_hypopnea" in df.columns
        gt_stats: dict | None = None
        if has_gt:
            gt_binary = ((df["is_apnea"].values + df["is_hypopnea"].values) > 0).astype(
                np.int32
            )
            gt_stats = self._compute_ahi_stats(gt_binary, total_hours)

        table = Table(title="Sleep Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Predicted", justify="right", style="green")
        if gt_stats is not None:
            table.add_column("Ground Truth", justify="right", style="blue")

        def row(label: str, pred_val: str, gt_val: str | None = None) -> None:
            if gt_stats is not None:
                table.add_row(label, pred_val, gt_val or "—")
            else:
                table.add_row(label, pred_val)

        row("Recording duration", f"{total_hours:.2f} h", f"{total_hours:.2f} h")
        row(
            "Apnea events",
            str(pred_stats["n_events"]),
            str(gt_stats["n_events"]) if gt_stats else None,
        )
        row(
            "AHI  (events / hour)",
            f"{pred_stats['ahi']:.1f}",
            f"{gt_stats['ahi']:.1f}" if gt_stats else None,
        )
        row(
            "Mean event duration",
            f"{pred_stats['avg_duration']:.1f} s",
            f"{gt_stats['avg_duration']:.1f} s" if gt_stats else None,
        )
        row(
            "Total apnea time",
            f"{pred_stats['apnea_minutes']:.1f} min",
            f"{gt_stats['apnea_minutes']:.1f} min" if gt_stats else None,
        )
        self.console.print(table)

        sev, colour = self._ahi_severity(pred_stats["ahi"])
        self.console.print(
            f"Predicted AHI severity: [{colour}]{sev}[/{colour}] "
            f"(AHI = {pred_stats['ahi']:.1f} events/h)"
        )
        if gt_stats is not None:
            sev_gt, colour_gt = self._ahi_severity(gt_stats["ahi"])
            self.console.print(
                f"Ground truth AHI severity: [{colour_gt}]{sev_gt}[/{colour_gt}] "
                f"(AHI = {gt_stats['ahi']:.1f} events/h)"
            )

    def _plot_signal(self, df: pd.DataFrame, record_name: str) -> None:
        """Plot SaO2 over time with shaded predicted (and ground-truth) apnea regions."""
        time_h = df["time_s"].values / 3600.0
        sao2 = df["sao2_percent"].values
        predicted = df["predicted_apnea"].values
        has_gt = "is_apnea" in df.columns

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(
            time_h, sao2, color="#4a9eda", linewidth=0.7, zorder=3, label="SaO₂ (%)"
        )

        legend_patches = [mpatches.Patch(color="#4a9eda", label="SaO₂ (%)")]

        if has_gt:
            gt = (
                df["is_apnea"].values
                + df.get("is_hypopnea", pd.Series(np.zeros(len(df)))).values
            ) > 0
            self._shade_regions(ax, time_h, gt.astype(int), color="#5cb85c", alpha=0.30)
            legend_patches.append(
                mpatches.Patch(color="#5cb85c", alpha=0.6, label="Ground truth")
            )

        self._shade_regions(ax, time_h, predicted, color="#d9534f", alpha=0.40)
        legend_patches.append(
            mpatches.Patch(color="#d9534f", alpha=0.7, label="Predicted apnea")
        )

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("SaO₂ (%)")
        ax.set_title(f"SaO₂ Signal — {record_name}")
        ax.legend(handles=legend_patches, loc="lower right")
        ax.set_xlim(time_h[0], time_h[-1])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _shade_regions(
        ax: plt.Axes,
        time_h: np.ndarray,
        binary: np.ndarray,
        color: str,
        alpha: float,
    ) -> None:
        """Shade contiguous runs of 1s in *binary* on *ax*."""
        in_region = False
        start = 0.0
        for i, v in enumerate(binary):
            if v and not in_region:
                start = time_h[i]
                in_region = True
            elif not v and in_region:
                ax.axvspan(start, time_h[i - 1], color=color, alpha=alpha, linewidth=0)
                in_region = False
        if in_region:
            ax.axvspan(start, time_h[-1], color=color, alpha=alpha, linewidth=0)
