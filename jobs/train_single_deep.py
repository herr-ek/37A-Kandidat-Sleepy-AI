from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.Cli.config import MODELS_DIR
from src.Cli.training import TrainingSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a deep learning apnea classifier from processed parquet files."
    )

    # --- Model selection ---
    parser.add_argument(
        "--model",
        default="CNN1D",
        choices=["CNN1D", "FullyConnected", "RNN"],
        help="Which deep model to train (default: CNN1D).",
    )

    # --- Data ---
    parser.add_argument(
        "--records",
        nargs="+",
        help="Record IDs to include. Defaults to all processed records with labels.",
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
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
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
    return parser.parse_args()


def select_records(
    session: TrainingSession, requested_records: list[str] | None
) -> list[str]:
    available = session.find_records_with_processed_data()
    labelled = [record["name"] for record in available if record["has_labels"]]

    if not labelled:
        raise ValueError(
            "No processed records with labels were found in data/processed."
        )

    if requested_records is None:
        return labelled

    available_set = set(labelled)
    missing = [record for record in requested_records if record not in available_set]
    if missing:
        raise ValueError(
            "Requested record(s) are missing processed label data: "
            + ", ".join(missing)
        )
    return requested_records


def build_model(args: argparse.Namespace):
    """Instantiate the requested model and return (model, hyperparams_dict)."""
    common = {
        "window_size": args.window_size,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_pos_weight": args.max_pos_weight,
        "verbose": True,
        "show_progress": args.verbose,
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


def build_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model == "CNN1D":
        return f"cnn1d_w{args.window_size}_f{args.num_filters}_h{args.hidden_size}_{timestamp}"
    if args.model == "FullyConnected":
        sizes = args.hidden_sizes.replace(",", "x")
        return f"fc_w{args.window_size}_{sizes}_{timestamp}"
    if args.model == "RNN":
        return f"rnn_w{args.window_size}_h{args.hidden_size}_l{args.num_layers}_{timestamp}"
    return f"{args.model.lower()}_{timestamp}"


def get_job_id() -> str:
    """Get the SLURM job ID from environment, or generate a timestamp-based ID."""
    # Try SLURM_ARRAY_JOB_ID first (for array jobs)
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    if job_id:
        return job_id

    # Fall back to SLURM_JOB_ID (for single jobs)
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        return job_id

    # If not running in SLURM, use timestamp
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    console = Console()
    session = TrainingSession(console)

    try:
        selected_records = select_records(session, args.records)
        console.print(
            f"[cyan]Using {len(selected_records)} record(s):[/cyan] {', '.join(selected_records[:10])}"
            + (" ..." if len(selected_records) > 10 else "")
        )

        X, y = session.build_raw_dataset(
            selected_records,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        X_train, X_test, y_train, y_test = session.split_dataset(
            X,
            y,
            test_size=args.test_size,
            seed=args.seed,
        )

        model, hyperparams = build_model(args)
        hyperparams["step_size"] = args.step_size
        hyperparams["test_size"] = args.test_size
        hyperparams["seed"] = args.seed

        console.print(
            f"[cyan]Model:[/cyan] {args.model}  |  [cyan]Device:[/cyan] {model.device}"
        )
        session.train(model, X_train, y_train)
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
                records=selected_records,
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
