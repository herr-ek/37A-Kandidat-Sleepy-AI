"""
Data loading and management module for the Sleep Data Analysis Pipeline CLI.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from .. import Data_management as dm
    from .config import PROCESSED_DIR, RAW_DIR
except ImportError:
    from config import PROCESSED_DIR, RAW_DIR

    import Data_management as dm


class DataLoader:
    """Handles data loading from various sources."""

    def __init__(self, console):
        self.console = console
        self.current_dataframe = None
        self.current_features = None
        self.current_features_normalized = None
        self.applied_operations = []

    def load_data(
        self,
        data_source: str,
        selected_records: List[str],
        custom_data_dir: Optional[str] = None,
        alt_record: Optional[str] = None,
        load_features_mode: str = "prompt",
    ) -> pd.DataFrame:
        """Load data based on current data source and selected record.

        Args:
            data_source: "raw" or "processed"
            selected_records: List of selected record names
            custom_data_dir: Optional custom directory path
            alt_record: Optional specific record to load (used for batch mode)
            load_features_mode: "prompt" (ask user), "auto" (load silently),
                                 or "skip" (never load feature files)

        Returns:
            Loaded DataFrame
        """
        if self.current_dataframe is not None:
            return self.current_dataframe

        record = selected_records[0] if alt_record is None else alt_record

        if data_source == "raw":
            record_path = (
                Path(custom_data_dir) / record if custom_data_dir else RAW_DIR / record
            )
            df = dm.load_from_mat_and_arousal_to_pandas(str(record_path / record))
        else:
            # Handle both "record_name" and "record_name/file_name" formats
            if "/" in record:
                # Full path to specific parquet file
                parquet_file = PROCESSED_DIR / f"{record}.parquet"
            else:
                # Just record name, use default parquet file
                parquet_file = PROCESSED_DIR / record / f"{record}.parquet"

            df = dm.load_from_parquet_to_pandas(str(parquet_file))

            # Try to load metadata if it exists
            metadata_file = parquet_file.with_suffix(".metadata.txt")
            if metadata_file.exists():
                self._load_metadata(metadata_file)

            # Optionally load features from the same processed record directory.
            feature_candidates = sorted(
                [
                    p
                    for p in parquet_file.parent.glob("*feature*.parquet")
                    if p != parquet_file
                ]
            )
            if feature_candidates and load_features_mode != "skip":
                if load_features_mode == "prompt":
                    import questionary

                    should_load = questionary.confirm(
                        "Feature parquet file(s) found. Load features as well?",
                        default=True,
                    ).ask()
                else:  # "auto"
                    should_load = True

                if should_load:
                    normal_feature_files = [
                        p
                        for p in feature_candidates
                        if "normalized" not in p.stem.lower()
                    ]
                    normalized_feature_files = [
                        p for p in feature_candidates if "normalized" in p.stem.lower()
                    ]

                    if normal_feature_files:
                        self.current_features = dm.load_from_parquet_to_pandas(
                            str(normal_feature_files[0])
                        )
                        self.console.print(
                            f"[dim]🧩 Loaded features: {normal_feature_files[0].name}[/dim]"
                        )

                    if normalized_feature_files:
                        self.current_features_normalized = (
                            dm.load_from_parquet_to_pandas(
                                str(normalized_feature_files[0])
                            )
                        )
                        self.console.print(
                            f"[dim]📐 Loaded normalized features: {normalized_feature_files[0].name}[/dim]"
                        )

        self.current_dataframe = df
        return df

    def get_features(self) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Get the currently extracted features, if available."""
        return self.current_features, self.current_features_normalized

    def _load_metadata(self, metadata_file: Path):
        """Load and parse metadata file to restore operations history."""
        try:
            with open(metadata_file, "r") as f:
                content = f.read()

            # Parse operations from metadata
            operations = []
            in_operations_section = False

            for line in content.split("\n"):
                line = line.strip()

                if line == "Operations applied:":
                    in_operations_section = True
                    continue

                if in_operations_section:
                    # Check if line is an operation (starts with number and dot)
                    if line and line[0].isdigit() and "." in line:
                        # Extract operation name (after the "1. " part)
                        operation = line.split(".", 1)[1].strip()
                        operations.append(operation)
                    elif line.startswith("DataFrame") or line.startswith("Columns"):
                        # End of operations section
                        break

            if operations:
                self.applied_operations = operations
                self.console.print(
                    f"[dim]📋 Loaded metadata: {len(operations)} operation(s) found[/dim]"
                )

        except Exception as e:
            self.console.print(
                f"[dim yellow]⚠ Could not load metadata: {str(e)}[/dim yellow]"
            )

    def get_available_records(
        self, data_source: str, custom_data_dir: Optional[str] = None
    ) -> List[str]:
        """Get list of available record directories from the selected data source."""
        # Use custom directory if set, otherwise use default
        if custom_data_dir:
            data_dir = Path(custom_data_dir)
        elif data_source == "raw":
            data_dir = RAW_DIR
        else:
            data_dir = PROCESSED_DIR

        if not data_dir.exists():
            return []

        # Get subdirectories (each represents a record)
        records = [
            d.name
            for d in data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        return sorted(records)

    def select_file_from_record(self, record: str) -> Optional[str]:
        """Select a specific file from within a record directory.

        Args:
            record: The record directory name

        Returns:
            The filename (without .parquet extension) or None if cancelled
        """
        import questionary
        from rich import box
        from rich.table import Table

        try:
            from .utils import format_file_size, get_questionary_style
        except ImportError:
            from utils import format_file_size, get_questionary_style

        record_dir = PROCESSED_DIR / record

        # Get all parquet files in the directory
        parquet_files = [
            f for f in record_dir.glob("*.parquet") if not f.name.startswith(".")
        ]

        if not parquet_files:
            self.console.print(f"[red]✗ No parquet files found in {record}[/red]")
            return None

        # Display available files
        table = Table(title=f"Files in {record}", box=box.ROUNDED)
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("File Name", style="green")
        table.add_column("Size", style="dim")
        table.add_column("Metadata", style="dim")

        file_info = []
        for idx, file_path in enumerate(sorted(parquet_files), 1):
            file_size = file_path.stat().st_size
            size_str = format_file_size(file_size)

            metadata_file = file_path.with_suffix(".metadata.txt")
            has_metadata = "✓" if metadata_file.exists() else "-"

            file_info.append(
                {
                    "path": file_path,
                    "name": file_path.stem,
                    "size": size_str,
                    "metadata": has_metadata,
                }
            )

            table.add_row(str(idx), file_path.name, size_str, has_metadata)

        self.console.print(table)

        # Select file with autocomplete
        file_choices = [info["name"] for info in file_info]
        selected = questionary.autocomplete(
            "Select a file (start typing for autocomplete):",
            choices=file_choices,
            style=get_questionary_style(),
        ).ask()

        return selected

    def clear(self):
        """Clear the cached dataframe."""
        self.current_dataframe = None
        self.current_features = None
        self.current_features_normalized = None
        self.applied_operations = []
