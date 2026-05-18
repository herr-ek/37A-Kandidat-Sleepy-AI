from __future__ import annotations

"""
Run an overfitting/underfitting fit check for a saved classical model.

Load a trained model from a JSON + joblib checkpoint, build the predefined
train / validation / test splits, evaluate on each, and print a comparison
table.

Usage:
    python jobs/train_err.py --model-json data/models/batch/<job>/knn_apnea_101.json

params:
    --model-json: Path to the .json metadata sidecar of a saved model (required).
    --set-dist-dir: Directory containing training_set.txt etc. Defaults to
                    <root>/set_distribution/.
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = Path(__file__).resolve().parent
SET_DIST_DIR = ROOT_DIR / "set_distribution"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(JOBS_DIR) not in sys.path:
    sys.path.insert(0, str(JOBS_DIR))

from train_utils import load_predefined_split

from src.Cli.training import TrainingSession
from src.Cli.training_manager import _build_model_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit check (overfitting/underfitting) for a saved classical model."
    )
    parser.add_argument(
        "--model-json",
        type=Path,
        required=True,
        help="Path to the .json metadata sidecar of the saved model.",
    )
    parser.add_argument(
        "--set-dist-dir",
        type=Path,
        default=None,
        help="Directory containing split txt files. Defaults to <root>/set_distribution/.",
    )
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    console = Console()
    session = TrainingSession(console)

    try:
        meta_path = args.model_json.resolve()
        if not meta_path.exists():
            console.print(f"[red]✗ Metadata file not found: {meta_path}[/red]")
            return 1

        meta = json.loads(meta_path.read_text())
        model_class_name = meta.get("model")
        hyperparams = meta.get("hyperparameters", {})

        console.print(f"[cyan]Model:[/cyan] {model_class_name}")
        console.print(f"[cyan]Hyperparameters:[/cyan] {hyperparams}")

        # --- Instantiate model ---
        registry = _build_model_registry()
        entry = next(
            (v for v in registry.values() if v[0].__name__ == model_class_name), None
        )
        if entry is None:
            console.print(f"[red]✗ Unknown model type '{model_class_name}'.[/red]")
            return 1

        model_cls, defaults = entry
        params = {**defaults, **hyperparams}
        params.pop("device", None)
        valid_params = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
        model = model_cls(**{k: v for k, v in params.items() if k in valid_params})

        ext = getattr(model, "FILE_EXTENSION", ".joblib")
        model_path = meta_path.with_suffix(ext)
        if not model_path.exists():
            console.print(f"[red]✗ Model file not found: {model_path}[/red]")
            return 1

        console.print(f"Loading [bold]{model_path.name}[/bold]...")
        model.load(str(model_path))
        console.print("[green]✓[/green] Model loaded.")

        # --- Build splits ---
        set_dist_dir = args.set_dist_dir if args.set_dist_dir else SET_DIST_DIR
        all_records = session.find_records_with_features()
        available = {r["name"] for r in all_records if r["has_labels"]}

        train_records, test_records, val_records = load_predefined_split(
            available, set_dist_dir, record_filter=None
        )
        console.print(
            f"[cyan]Train:[/cyan] {len(train_records)} record(s)  "
            f"[cyan]Val:[/cyan] {len(val_records)} record(s)  "
            f"[cyan]Test:[/cyan] {len(test_records)} record(s)"
        )

        use_normalized = hyperparams.get("use_normalized", True)
        console.print(
            f"[dim]Normalised features: {'yes' if use_normalized else 'no'}[/dim]"
        )

        X_train, y_train, _ = session.build_dataset(
            train_records, use_normalized=use_normalized
        )
        X_val, y_val, _ = session.build_dataset(
            val_records, use_normalized=use_normalized
        )
        X_test, y_test, _ = session.build_dataset(
            test_records, use_normalized=use_normalized
        )

        # --- Evaluate ---
        console.print("[cyan]Evaluating splits...[/cyan]")
        train_m = model.evaluate(X_train, y_train)
        val_m = model.evaluate(X_val, y_val)
        test_m = model.evaluate(X_test, y_test)

        fit_table = Table(title="Overfitting / Underfitting Check", box=box.ROUNDED)
        fit_table.add_column("Metric", style="cyan")
        fit_table.add_column("Train", justify="right", style="green")
        fit_table.add_column("Validation", justify="right", style="yellow")
        fit_table.add_column("Test", justify="right", style="blue")
        fit_table.add_column("Train - Test", justify="right", style="dim")
        for key in ("balanced_accuracy", "f1_macro", "recall", "precision"):
            tr = train_m.get(key, float("nan"))
            va = val_m.get(key, float("nan"))
            te = test_m.get(key, float("nan"))
            fit_table.add_row(
                key,
                f"{tr:.4f}",
                f"{va:.4f}",
                f"{te:.4f}",
                f"{tr - te:+.4f}",
            )
        console.print(fit_table)

        gap = train_m.get("balanced_accuracy", 0) - test_m.get("balanced_accuracy", 0)
        train_bal = train_m.get("balanced_accuracy", 0)
        if gap > 0.10:
            console.print(
                f"[red]⚠ Possible overfitting: train-test gap = {gap:.4f}[/red]"
            )
        elif train_bal < 0.65:
            console.print(
                f"[yellow]⚠ Possible underfitting: train balanced accuracy = {train_bal:.4f}[/yellow]"
            )
        else:
            console.print(
                f"[green]✓ No strong overfitting/underfitting signal (gap = {gap:.4f})[/green]"
            )

        return 0

    except Exception as exc:
        console.print(f"[red]Fit check failed:[/red] {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
