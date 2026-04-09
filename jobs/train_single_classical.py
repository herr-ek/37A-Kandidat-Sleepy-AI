from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

ROOT_DIR = Path(__file__).resolve().parents[1]
SET_DIST_DIR = ROOT_DIR / "set_distribution"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.Cli.config import MODELS_DIR
from src.Cli.training import TrainingSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classical ML model (KNN / RandomForest / SVM) on SaO2 feature files."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["KNN", "RandomForest", "SVM"],
        help="Which classifier to train.",
    )
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
        "--normalized",
        action="store_true",
        default=True,
        help="Use normalised feature files when available (default: true).",
    )
    parser.add_argument(
        "--no-normalized",
        dest="normalized",
        action="store_false",
        help="Use raw (non-normalised) feature files.",
    )

    # --- KNN ---
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="[KNN] Number of neighbours (default: 5).",
    )

    # --- RandomForest ---
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="[RandomForest] Number of trees (default: 100).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=10,
        help="[RandomForest] Maximum tree depth (default: 10).",
    )

    # --- SVM ---
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="[SVM] Regularisation parameter C for LinearSVC (default: 1.0).",
    )

    # --- Common ---
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
    return parser.parse_args()


def build_model(args: argparse.Namespace):
    """Instantiate the requested model and return (model, hyperparams_dict)."""
    if args.model == "KNN":
        from src.Models.KNN import KNN

        hyperparams = {"n_neighbors": args.n_neighbors}
        return KNN(**hyperparams), hyperparams

    if args.model == "RandomForest":
        from src.Models.RandomForest import RandomForest

        hyperparams = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        }
        return RandomForest(**hyperparams), hyperparams

    if args.model == "SVM":
        from src.Models.SVM import SVM

        hyperparams = {"C": args.C}
        return SVM(**hyperparams), hyperparams

    raise ValueError(f"Unknown model: {args.model}")


def load_predefined_split(
    available_records: set[str],
    set_dist_dir: Path,
    record_filter: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Read training_set.txt and test_set.txt and return (train_records, test_records)
    restricted to records that have labelled feature files."""

    def _read(path: Path) -> list[str]:
        return [l.strip() for l in path.read_text().splitlines() if l.strip()]

    train_all = _read(set_dist_dir / "training_set.txt")
    test_all = _read(set_dist_dir / "test_set.txt")

    train_records = [r for r in train_all if r in available_records]
    test_records = [r for r in test_all if r in available_records]

    if record_filter is not None:
        filter_set = set(record_filter)
        train_records = [r for r in train_records if r in filter_set]

    if not train_records:
        raise ValueError(
            "No training records with labelled feature files found in training_set.txt."
        )
    if not test_records:
        raise ValueError(
            "No test records with labelled feature files found in test_set.txt."
        )

    return train_records, test_records


def build_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "norm" if args.normalized else "raw"
    if args.model == "KNN":
        return f"knn_k{args.n_neighbors}_{suffix}_{timestamp}"
    if args.model == "RandomForest":
        return f"rf_n{args.n_estimators}_d{args.max_depth}_{suffix}_{timestamp}"
    if args.model == "SVM":
        c_str = str(args.C).replace(".", "p")
        return f"svm_C{c_str}_{suffix}_{timestamp}"
    return f"{args.model.lower()}_{timestamp}"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    console = Console()
    session = TrainingSession(console)

    try:
        set_dist_dir = args.set_dist_dir if args.set_dist_dir else SET_DIST_DIR
        all_records = session.find_records_with_features()
        available = {r["name"] for r in all_records if r["has_labels"]}

        train_records, test_records = load_predefined_split(
            available, set_dist_dir, args.records
        )
        console.print(
            f"[cyan]Train:[/cyan] {len(train_records)} record(s)  "
            f"[cyan]Test:[/cyan] {len(test_records)} record(s)"
        )
        console.print(
            f"[dim]Normalised features: {'yes' if args.normalized else 'no'}[/dim]"
        )

        X_train, y_train, feature_names = session.build_dataset(
            train_records, use_normalized=args.normalized
        )
        X_test, y_test, _ = session.build_dataset(
            test_records, use_normalized=args.normalized
        )

        model, hyperparams = build_model(args)
        hyperparams["use_normalized"] = args.normalized

        session.train(model, X_train, y_train)
        metrics = session.evaluate(model, X_test, y_test)

        if not args.no_save:
            model_name = build_model_name(args)
            output_path = session.save_model(
                model,
                model_name,
                MODELS_DIR,
                hyperparams=hyperparams,
                metrics=metrics,
                records=train_records,
                feature_names=feature_names,
            )
            console.print(f"[green]Saved model:[/green] {output_path}")
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
    sys.exit(main())
