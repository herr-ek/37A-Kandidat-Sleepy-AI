import argparse
import os
from datetime import datetime
from pathlib import Path

from src.Cli.training import TrainingSession


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


def load_predefined_split(
    available_records: set[str],
    set_dist_dir: Path,
    record_filter: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Read training_set.txt and test_set.txt and return (train_records, test_records)
    restricted to records that have processed data files."""

    def _read(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    train_all = _read(set_dist_dir / "training_set.txt")
    test_all = _read(set_dist_dir / "test_set.txt")

    train_records = [r for r in train_all if r in available_records]
    test_records = [r for r in test_all if r in available_records]

    if record_filter is not None:
        filter_set = set(record_filter)
        train_records = [r for r in train_records if r in filter_set]

    if not train_records:
        raise ValueError(
            "No training records with processed data found in training_set.txt."
        )
    if not test_records:
        raise ValueError("No test records with processed data found in test_set.txt.")

    return train_records, test_records


def build_model_name(args: argparse.Namespace) -> str:
    if args.model_name:
        return args.model_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized = getattr(args, "normalized", None)
    norm_suffix = (
        f"_{'norm' if normalized else 'raw'}" if normalized is not None else ""
    )
    if args.model == "CNN1D":
        return f"cnn1d_w{args.window_size}_f{args.num_filters}_h{args.hidden_size}_{timestamp}"
    if args.model == "FullyConnected":
        sizes = args.hidden_sizes.replace(",", "x")
        return f"fc_w{args.window_size}_{sizes}_{timestamp}"
    if args.model == "RNN":
        return f"rnn_w{args.window_size}_h{args.hidden_size}_l{args.num_layers}_{timestamp}"
    if args.model == "KNN":
        return f"knn_k{args.n_neighbors}{norm_suffix}_{timestamp}"
    if args.model == "RandomForest":
        return f"rf_n{args.n_estimators}_d{args.max_depth}{norm_suffix}_{timestamp}"
    if args.model == "SVM":
        c_str = str(args.C).replace(".", "p")
        return f"svm_C{c_str}{norm_suffix}_{timestamp}"
    return f"{args.model.lower()}_{timestamp}"
