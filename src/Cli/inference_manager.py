"""
Inference Manager for the Sleep Data Analysis Pipeline CLI.

Handles interactive selection and loading of a trained model,
then runs predictions on a single selected record.
"""

import inspect
import json
import math

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
        while True:
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
                pred_stats, gt_stats = self._display_ahi(ts_df)
                plot = questionary.confirm(
                    "Plot SaO2 signal with apnea annotations?", default=False
                ).ask()
                if plot:
                    self._plot_signal(ts_df, record_ctx["name"], pred_stats, gt_stats)
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
    def _compute_ahi_stats(
        binary: np.ndarray,
        time_s: np.ndarray,
        total_hours: float,
        max_duration: int = 30,
        min_gap: int = 10,
    ) -> dict:
        """Count apnea events using merge-then-split logic.

        1. Find contiguous apnea runs.
        2. Merge runs separated by fewer than *min_gap* non-apnea seconds.
        3. Walk each merged run: take up to *max_duration* seconds as one event,
           then skip *min_gap* seconds (mandatory gap), then start the next event —
           even if still inside a predicted apnea run.

        Returns a dict with n_events, ahi, avg_duration, apnea_minutes, and
        ``events`` — a list of (start_s, end_s) float tuples for plotting.
        """
        # Step 1: find contiguous 1-runs as index ranges
        runs: list[tuple[int, int]] = []
        in_run = False
        run_start = 0
        for i, v in enumerate(binary):
            if v and not in_run:
                run_start = i
                in_run = True
            elif not v and in_run:
                runs.append((run_start, i - 1))
                in_run = False
        if in_run:
            runs.append((run_start, len(binary) - 1))

        # Step 2: merge runs where the gap is shorter than min_gap
        merged: list[tuple[int, int]] = []
        for run in runs:
            if merged and (run[0] - merged[-1][1] - 1) < min_gap:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)

        # Step 3: split runs longer than max_duration.
        # Each event lasts at most max_duration seconds; after it ends we wait
        # min_gap seconds (still inside the run) before the next event starts.
        events: list[tuple[float, float]] = []
        for start, end in merged:
            cursor = start
            while cursor <= end:
                event_end = min(cursor + max_duration - 1, end)
                events.append((float(time_s[cursor]), float(time_s[event_end])))
                cursor = event_end + 1 + min_gap  # mandatory gap before next event

        durations = [e - s + 1.0 for s, e in events]
        return {
            "n_events": len(events),
            "ahi": len(events) / total_hours if total_hours > 0 else 0.0,
            "avg_duration": float(np.mean(durations)) if durations else 0.0,
            "apnea_minutes": float(binary.sum()) / 60.0,
            "events": events,
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

    def _display_ahi(self, df: pd.DataFrame) -> tuple[dict, dict | None]:
        """Compute and display AHI statistics, with ground-truth comparison if available."""
        total_seconds = float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) + 1.0
        total_hours = total_seconds / 3600.0
        time_s = df["time_s"].values

        pred_stats = self._compute_ahi_stats(
            df["predicted_apnea"].values, time_s, total_hours
        )

        has_gt = "is_apnea" in df.columns and "is_hypopnea" in df.columns
        gt_stats: dict | None = None
        if has_gt:
            gt_binary = ((df["is_apnea"].values + df["is_hypopnea"].values) > 0).astype(
                np.int32
            )
            gt_stats = self._compute_ahi_stats(gt_binary, time_s, total_hours)

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

        return pred_stats, gt_stats

    def _plot_signal(
        self,
        df: pd.DataFrame,
        record_name: str,
        pred_stats: dict,
        gt_stats: dict | None,
    ) -> None:
        """Plot SaO2 with split predicted events and (optionally) split ground-truth events."""
        time_h = df["time_s"].values / 3600.0
        sao2 = df["sao2_percent"].values

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(
            time_h, sao2, color="#4a9eda", linewidth=0.7, zorder=3, label="SaO₂ (%)"
        )

        legend_patches = [mpatches.Patch(color="#4a9eda", label="SaO₂ (%)")]

        if gt_stats is not None:
            self._shade_event_list(ax, gt_stats["events"], color="#5cb85c", alpha=0.35)
            legend_patches.append(
                mpatches.Patch(
                    color="#5cb85c",
                    alpha=0.6,
                    label=f"Ground truth ({gt_stats['n_events']} events, AHI {gt_stats['ahi']:.1f})",
                )
            )

        self._shade_event_list(ax, pred_stats["events"], color="#d9534f", alpha=0.45)
        legend_patches.append(
            mpatches.Patch(
                color="#d9534f",
                alpha=0.7,
                label=f"Predicted ({pred_stats['n_events']} events, AHI {pred_stats['ahi']:.1f})",
            )
        )

        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("SaO₂ (%)")
        ax.set_title(f"SaO₂ Signal — {record_name}")
        ax.legend(handles=legend_patches, loc="lower right")
        ax.set_xlim(time_h[0], time_h[-1])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _shade_event_list(
        ax: plt.Axes,
        events: list[tuple[float, float]],
        color: str,
        alpha: float,
    ) -> None:
        """Shade a list of (start_s, end_s) event intervals on *ax* (x-axis in hours)."""
        for start_s, end_s in events:
            ax.axvspan(
                start_s / 3600.0, end_s / 3600.0, color=color, alpha=alpha, linewidth=0
            )
