from __future__ import annotations

import argparse
import sys
from pathlib import Path
from re import X

import torch
from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = Path(__file__).resolve().parent
SET_DIST_DIR = ROOT_DIR / "set_distribution"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(JOBS_DIR) not in sys.path:
    sys.path.insert(0, str(JOBS_DIR))

from train_utils import build_model_name, get_job_id, load_predefined_split

from src.Cli.config import MODELS_DIR
from src.Cli.training import TrainingSession

"""
Train a single deep learning model on the SaO2 feature files. Uses the same predefined train/test/validate split as train_single_classical.py.
params:
    --model: Which deep model to train (default: Fully Connected).
    --records: Optional whitelist of record IDs. When given, only these records are used from training_set.txt.
    --set-dist-dir: Directory containing training_set.txt and test_set.txt. Defaults to <root>/set_distribution/.
    --window-size: Window size in samples/seconds for raw SaO2 windows (default: 60).
    --step-size: Stride between windows. Defaults to window_size // 2.
    --num-filters: [CNN1D] Filters in the first convolution layer (default: 16).
    --hidden-size: [CNN1D] FC layer size after conv blocks; [RNN] GRU hidden size (default: 64).
    --hidden-sizes: [FullyConnected] Comma-separated list of hidden layer sizes (default: 128,64).
    --dropout: [FullyConnected / RNN] Dropout probability (default: 0.3).
    --num-layers: [RNN] Number of GRU layers (default: 2).
    --epochs: Number of training epochs (default: 20).
    --batch-size: Training batch size (default: 1024).
    --lr: Adam learning rate (default: 1e-3).
    --max-pos-weight: Maximum positive class weight for BCEWithLogitsLoss (default: 10.0).
    --model-name: Output model name without extension. Defaults to a descriptive timestamped name.
    --no-save: Skip writing the model checkpoint and metadata sidecar.
    --verbose: Enable verbose output including progress bars (default: False).
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deep learning apnea classifier from processed parquet files."
    )

    # --- Model selection ---
    parser.add_argument(
        "--model",
        default="FullyConnected",
        choices=["CNN1D", "FullyConnected", "RNN"],
        help="Which deep model to train (default: FullyConnected).",
    )

    # --- Data ---
    parser.add_argument(
        "--records",
        nargs="+",
        help="Optional whitelist of record IDs. When given, only these records are used from training_set.txt.",
    )
    parser.add_argument(
        "--set-dist-dir",
        type=Path,
        default=None,
        help="Directory containing training_set.txt and test_set.txt. Defaults to <root>/set_distribution/.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Window size in samples/seconds for raw SaO2 windows (default: 60).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Stride between windows. Defaults to window_size // 2.",
    )

    # --- CNN1D-specific ---
    parser.add_argument(
        "--num-filters",
        type=int,
        default=16,
        help="[CNN1D] Filters in the first convolution layer (default: 16).",
    )

    # --- CNN1D / RNN shared ---
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="[CNN1D] FC layer size after conv blocks; [RNN] GRU hidden size (default: 64).",
    )

    # --- FullyConnected-specific ---
    parser.add_argument(
        "--hidden-sizes",
        default="128,64",
        help="[FullyConnected] Comma-separated list of hidden layer sizes (default: 128,64).",
    )

    # --- RNN-specific ---
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="[RNN] Number of GRU layers (default: 2).",
    )

    # --- Shared regularisation ---
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="[FullyConnected / RNN] Dropout probability (default: 0.3).",
    )

    # --- Training ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size (default: 1024).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=10.0,
        help="Maximum positive class weight for BCEWithLogitsLoss (default: 10.0).",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Output model name without extension. Defaults to a descriptive timestamped name.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the model checkpoint and metadata sidecar.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output including progress bars (default: False).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Request CUDA training. Fails if no GPU is available.",
    )
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: str | None = None):
    """Instantiate the requested model and return (model, hyperparams_dict)."""
    common = {
        "window_size": args.window_size,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_pos_weight": args.max_pos_weight,
        "verbose": True,
        "show_progress": args.verbose,
        "device": device,
    }

    if args.model == "CNN1D":
        from src.Models.DeepLearning import CNN1D

        extra = {"num_filters": args.num_filters, "hidden_size": args.hidden_size}
        return CNN1D(**common, **extra), {**common, **extra}

    if args.model == "FullyConnected":
        from src.Models.DeepLearning import FullyConnected

        hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
        extra = {"hidden_sizes": hidden_sizes, "dropout": args.dropout}
        return FullyConnected(**common, **extra), {
            **common,
            "hidden_sizes": hidden_sizes,
            "dropout": args.dropout,
        }

    if args.model == "RNN":
        from src.Models.DeepLearning import RNN

        extra = {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        }
        return RNN(**common, **extra), {**common, **extra}

    raise ValueError(f"Unknown model: {args.model}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    console = Console()
    session = TrainingSession(console)

    device: str | None = None
    if args.use_gpu:
        if not torch.cuda.is_available():
            console.print(
                "[red]--use-gpu requested but no CUDA device is available.[/red]"
            )
            return 1
        device = "cuda"

    else:
        device = "cpu"

    try:
        set_dist_dir = args.set_dist_dir if args.set_dist_dir else SET_DIST_DIR
        all_records = session.find_records_with_processed_data()
        available = {r["name"] for r in all_records if r["has_labels"]}

        train_records, test_records, validate_records = load_predefined_split(
            available, set_dist_dir, args.records
        )
        console.print(
            f"[cyan]Train:[/cyan] {len(train_records)} record(s)  "
            f"[cyan]Test:[/cyan] {len(test_records)} record(s)  "
            f"[cyan]Validate:[/cyan] {len(validate_records)} record(s)"
        )

        X_train, y_train = session.build_raw_dataset(
            train_records,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        X_val, y_val = session.build_raw_dataset(
            validate_records,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        X_test, y_test = session.build_raw_dataset(
            test_records,
            window_size=args.window_size,
            step_size=args.step_size,
        )

        model, hyperparams = build_model(args, device=device)
        hyperparams["step_size"] = args.step_size

        console.print(
            f"[cyan]Model:[/cyan] {args.model}  |  [cyan]Device:[/cyan] {model.device}"
        )
        session.train(model, X_train, y_train, X_val, y_val)
        metrics = session.evaluate(model, X_test, y_test)

        if not args.no_save:
            model_name = build_model_name(args)
            job_id = get_job_id()
            output_dir = MODELS_DIR / "batch" / job_id
            output_path = session.save_model(
                model,
                model_name,
                output_dir,
                hyperparams=hyperparams,
                metrics=metrics,
                records=train_records,
            )
            console.print(f"[green]Saved model:[/green] {output_path}")
            console.print(f"[dim]Job ID:[/dim] {job_id}")
        else:
            console.print("[yellow]Skipping model save (--no-save).[/yellow]")

        console.print("[green]Training run finished successfully.[/green]")
        return 0
    except Exception as exc:
        console.print(f"[red]Training failed:[/red] {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
