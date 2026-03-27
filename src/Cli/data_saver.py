"""
Data saving module for the Sleep Data Analysis Pipeline CLI.
"""

import shutil
from pathlib import Path

import pandas as pd

try:
    from .. import Data_management as dm
    from .config import PROCESSED_DIR, RAW_DIR
except ImportError:
    from config import PROCESSED_DIR, RAW_DIR

    import Data_management as dm


class DataSaver:
    """Handles saving processed data and features."""

    def __init__(self, console):
        self.console = console

    def save_dataframe(
        self,
        df: pd.DataFrame,
        selected_records: list,
        applied_operations: list,
        data_source: str,
    ) -> bool:
        """Save the current processed dataframe, overwriting any previous saved version."""
        if df is None:
            self.console.print("[yellow]⚠ No dataframe loaded yet[/yellow]")
            return False

        record = selected_records[0]
        record_dir = record.split("/")[0] if "/" in record else record
        output_dir = PROCESSED_DIR / record_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{record_dir}_processed.parquet"

        try:
            df.to_parquet(output_file, index=False)

            metadata_file = output_file.with_suffix(".metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Record: {record}\n")
                f.write(f"Source: {data_source}\n")
                f.write(f"Original file: {selected_records[0]}\n")
                f.write("Operations applied:\n")
                for i, op in enumerate(applied_operations, 1):
                    f.write(f"  {i}. {op}\n")
                f.write(f"\nDataFrame shape: {df.shape}\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")

            self.console.print(f"[green]✓[/green] Saved to {output_file}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to save: {str(e)}[/red]")
            return False

    def save_features(
        self,
        features_df: pd.DataFrame,
        selected_records: list,
        normalized_features_df: pd.DataFrame = None,
    ) -> bool:
        """Save extracted features, overwriting any previous saved version."""
        if features_df is None or features_df.empty:
            self.console.print("[yellow]⚠ No features extracted yet[/yellow]")
            return False

        record = selected_records[0]
        record_dir = record.split("/")[0] if "/" in record else record
        output_dir = PROCESSED_DIR / record_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{record_dir}_features.parquet"

        try:
            features_df.to_parquet(output_file, index=False)

            normalized_output_file = output_file.with_name(
                f"{output_file.stem}_normalized{output_file.suffix}"
            )

            if normalized_features_df is not None:
                normalized_features_df.to_parquet(normalized_output_file, index=False)

            metadata_file = output_file.with_suffix(".metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Record: {record}\n")
                f.write("Type: features\n")
                f.write("Normalized: no\n")
                f.write(f"DataFrame shape: {features_df.shape}\n")
                f.write(f"Columns: {', '.join(features_df.columns)}\n")

            if normalized_features_df is not None:
                normalized_metadata_file = normalized_output_file.with_suffix(
                    ".metadata.txt"
                )
                with open(normalized_metadata_file, "w") as f:
                    f.write(f"Record: {record}\n")
                    f.write("Type: features\n")
                    f.write("Normalized: yes\n")
                    f.write(f"DataFrame shape: {normalized_features_df.shape}\n")
                    f.write(f"Columns: {', '.join(normalized_features_df.columns)}\n")

            self.console.print(f"[green]✓[/green] Features saved to {output_file}")
            if normalized_features_df is not None:
                self.console.print(
                    f"[green]✓[/green] Normalized features saved to {normalized_output_file}"
                )
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to save features: {str(e)}[/red]")
            return False

    def export_to_parquet(self, selected_records: list, data_source: str) -> bool:
        """Export raw data to parquet format."""
        if data_source != "raw":
            self.console.print(
                "[yellow]⚠ This action only works with raw data[/yellow]"
            )
            return False

        self.console.print("\n[bold cyan]Exporting to parquet...[/bold cyan]")

        record = selected_records[0]
        record_path = RAW_DIR / record

        # Create output directory
        output_dir = PROCESSED_DIR / record
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{record}.parquet"

        # Extract and save
        try:
            dm.extract_and_save_to_parquet(str(record_path / record))

            # Move to processed directory
            source = Path(str(record_path / record) + ".parquet")
            if source.exists():
                shutil.move(str(source), str(output_file))
                self.console.print(f"[green]✓[/green] Exported to {output_file}")
                return True
            else:
                self.console.print("[red]✗ Export failed[/red]")
                return False
        except Exception as e:
            self.console.print(f"[red]✗ Export failed: {str(e)}[/red]")
            return False
