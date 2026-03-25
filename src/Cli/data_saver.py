"""
Data saving module for the Sleep Data Analysis Pipeline CLI.
"""

import shutil
from pathlib import Path

import pandas as pd
import questionary

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
        style,
    ) -> bool:
        """Save the current dataframe to a parquet file."""
        if df is None:
            self.console.print("[yellow]⚠ No dataframe loaded yet[/yellow]")
            return False

        if not applied_operations:
            self.console.print(
                "[yellow]⚠ No modifications have been made to the dataframe[/yellow]"
            )
            should_continue = questionary.confirm(
                "Save original data anyway?", default=False
            ).ask()
            if not should_continue:
                return False

        self.console.print("\n[bold cyan]Saving dataframe...[/bold cyan]")

        # Get filename
        record = selected_records[0]
        base_record = record.split("/")[-1] if "/" in record else record
        operations_suffix = (
            "_".join(applied_operations)
            .lower()
            .replace("(", "_")
            .replace(")", "")
            .replace(".", "_")
        )
        default_filename = (
            f"{base_record}_{operations_suffix}.parquet"
            if operations_suffix
            else f"{base_record}_modified.parquet"
        )

        filename = questionary.text(
            "Enter filename (without path):", default=default_filename
        ).ask()

        if not filename:
            self.console.print("[yellow]✗ Save cancelled[/yellow]")
            return False

        # Ensure .parquet extension
        if not filename.endswith(".parquet"):
            filename += ".parquet"

        # Choose where to save
        save_location = questionary.select(
            "Where do you want to save?",
            choices=[
                questionary.Choice(
                    f"📂 Processed folder (data/processed/{record}/)", value="processed"
                ),
                questionary.Choice("📁 Custom path", value="custom"),
            ],
            style=style,
        ).ask()

        if save_location == "processed":
            base_record = record.split("/")[0] if "/" in record else record
            output_dir = PROCESSED_DIR / base_record
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / filename
        else:
            custom_path = questionary.path(
                "Enter full path to save location:",
                default=str(PROCESSED_DIR.parent),
            ).ask()

            if not custom_path:
                self.console.print("[yellow]✗ Save cancelled[/yellow]")
                return False

            output_file = Path(custom_path) / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        try:
            df.to_parquet(output_file, index=False)

            # Create metadata file with operations log
            metadata_file = output_file.with_suffix(".metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Record: {record}\n")
                f.write(f"Source: {data_source}\n")
                f.write(f"Original file: {selected_records[0]}\n")
                f.write(f"Operations applied:\n")
                for i, op in enumerate(applied_operations, 1):
                    f.write(f"  {i}. {op}\n")
                f.write(f"\nDataFrame shape: {df.shape}\n")
                f.write(f"Columns: {', '.join(df.columns)}\n")

            self.console.print(f"[green]✓[/green] Saved to {output_file}")
            self.console.print(f"[green]✓[/green] Metadata saved to {metadata_file}")
            return True

        except Exception as e:
            self.console.print(f"[red]✗ Failed to save: {str(e)}[/red]")
            return False

    def save_features(
        self, features_df: pd.DataFrame, selected_records: list, style
    ) -> bool:
        """Save the extracted features to a parquet file."""
        if features_df is None or features_df.empty:
            self.console.print("[yellow]⚠ No features extracted yet[/yellow]")
            return False

        self.console.print("\n[bold cyan]Saving features...[/bold cyan]")

        # Get filename
        record = selected_records[0]
        base_record = record.split("/")[-1] if "/" in record else record
        default_filename = f"{base_record}_features.parquet"

        filename = questionary.text(
            "Enter filename (without path):", default=default_filename
        ).ask()

        if not filename:
            self.console.print("[yellow]✗ Save cancelled[/yellow]")
            return False

        # Ensure .parquet extension
        if not filename.endswith(".parquet"):
            filename += ".parquet"

        # Choose where to save
        save_location = questionary.select(
            "Where do you want to save?",
            choices=[
                questionary.Choice(
                    f"📂 Processed folder (data/processed/{record}/)", value="processed"
                ),
                questionary.Choice("📁 Custom path", value="custom"),
            ],
            style=style,
        ).ask()

        if save_location == "processed":
            base_record = record.split("/")[0] if "/" in record else record
            output_dir = PROCESSED_DIR / base_record
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / filename
        else:
            custom_path = questionary.path(
                "Enter full path to save location:",
                default=str(PROCESSED_DIR.parent),
            ).ask()

            if not custom_path:
                self.console.print("[yellow]✗ Save cancelled[/yellow]")
                return False

            output_file = Path(custom_path) / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the features
        try:
            features_df.to_parquet(output_file, index=False)
            self.console.print(f"[green]✓[/green] Features saved to {output_file}")
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
