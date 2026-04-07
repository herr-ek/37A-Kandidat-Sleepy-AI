from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Cli.config import MODELS_DIR
from Cli.training import TrainingSession
from Models.DeepLearning import CNN1D


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CNN1D apnea classifier from processed parquet files."
    )
    parser.add_argument(
        "--records",
        nargs="+",
        help="Record IDs to include. Defaults to all processed records with labels.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        help="Window size in samples/seconds for raw SaO2 windows.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Stride between windows. Defaults to window_size // 2.",
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        default=16,
        help="Number of filters in the first convolution layer.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden layer size after convolution blocks.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Output model name without extension. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the model checkpoint and metadata sidecar.",
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


def build_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"cnn1d_{timestamp}"


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

        model = CNN1D(
            window_size=args.window_size,
            num_filters=args.num_filters,
            hidden_size=args.hidden_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=True,
        )

        console.print(f"[cyan]Torch device selected:[/cyan] {model.device}")
        session.train(model, X_train, y_train)
        metrics = session.evaluate(model, X_test, y_test)

        if not args.no_save:
            model_name = build_model_name(args)
            output_path = session.save_model(
                model,
                model_name,
                MODELS_DIR,
                hyperparams={
                    "window_size": args.window_size,
                    "step_size": args.step_size,
                    "num_filters": args.num_filters,
                    "hidden_size": args.hidden_size,
                    "num_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "test_size": args.test_size,
                    "seed": args.seed,
                },
                metrics=metrics,
                records=selected_records,
            )
            console.print(f"[green]Saved model:[/green] {output_path}")
        else:
            console.print(
                "[yellow]Skipping model save because --no-save was set.[/yellow]"
            )

        console.print("[green]Training run finished successfully.[/green]")
        return 0
    except Exception as exc:
        console.print(f"[red]Training failed:[/red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
