#!/usr/bin/env python3
"""
Sleep Data Analysis Pipeline CLI

Interactive command-line interface for processing and analyzing sleep apnea data.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import questionary
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import Data_management as dm
import Feature_extraction as fe
import Plotting as pl
import Preprocessing as pp
import Resampling as rs

console = Console()

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class SleepDataPipeline:
    """Main pipeline controller for sleep data analysis."""

    def __init__(self):
        self.data_source = None
        self.mode = None
        self.selected_records = []
        self.current_dataframe = None
        self.current_features = None
        self.applied_operations = []  # Track what's been done to the dataframe
        self.custom_data_dir = None  # For custom directory in batch mode

    def run(self):
        """Main entry point for the CLI."""
        console.print(
            Panel.fit(
                "[bold cyan]Sleep Data Analysis Pipeline[/bold cyan]\n"
                "[dim]Process and analyze sleep apnea recordings[/dim]",
                border_style="cyan",
                box=box.DOUBLE,
            )
        )

        try:
            # Step 1: Choose data source
            self.choose_data_source()

            # Step 2: Select record(s)
            self.select_records()

            # Step 3: Main action loop
            while True:
                action = self.choose_action()
                if action == "exit":
                    break
                self.execute_action(action)

        except KeyboardInterrupt:
            console.print("\n[yellow]✗ Operation cancelled by user[/yellow]")
            sys.exit(0)

        console.print("[green]✓ Pipeline completed successfully![/green]")

    def choose_data_source(self):
        """Step 1: Choose between raw or processed data."""
        console.print("\n[bold]Step 1:[/bold] Choose data source", style="cyan")

        choice = questionary.select(
            "What type of data do you want to work with?",
            choices=[
                questionary.Choice(
                    "📁 Raw data (load from .mat + .arousal files)", value="raw"
                ),
                questionary.Choice(
                    "📂 Multiple raw records (batch process)", value="raw_batch"
                ),
                questionary.Choice(
                    "📊 Processed data (load from .parquet files)", value="processed"
                ),
                questionary.Choice(
                    "📊 Multiple processed records (batch process)",
                    value="processed_batch",
                ),
                questionary.Choice(
                    "⬇️  Download data from PhysioNet (OBS. download speed capped to ~1.5 MB/s)",
                    value="download",
                ),
                questionary.Choice("❌ Exit", value="exit"),
            ],
            style=self._get_style(),
        ).ask()

        if choice == "exit":
            console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)
        elif choice == "download":
            self._download_physionet_data()
            # After download, let user choose data source again
            self.choose_data_source()
            return

        self.data_source = choice
        self.mode = "single"
        if choice == "raw_batch" or choice == "processed_batch":
            self.data_source = "raw" if choice == "raw_batch" else "processed"
            self.mode = "batch"
        console.print(f"[green]✓[/green] Using {choice} data")

    def select_records(self):
        """Step 2: Select record(s) to process."""
        console.print("\n[bold]Step 2:[/bold] Select recording(s)", style="cyan")

        # For batch raw data, ask if user wants to load from custom directory
        if self.mode == "batch" and self.data_source == "raw":
            use_custom = questionary.confirm(
                "Load from custom directory?", default=False
            ).ask()
            if use_custom:
                self._set_custom_directory()

        # Get available record directories
        records = self._get_available_records()

        if not records:
            default_location = (
                self.custom_data_dir
                if self.custom_data_dir
                else (RAW_DIR if self.data_source == "raw" else PROCESSED_DIR)
            )
            console.print(f"[red]✗ No records found in {default_location}[/red]")
            sys.exit(1)

        if self.mode == "batch":
            # For batch mode, select multiple records with checkbox
            selected = questionary.checkbox(
                "Select records to process (use space to select):",
                choices=records,
                style=self._get_style(),
            ).ask()

            if not selected:
                console.print("[red]✗ No records selected[/red]")
                sys.exit(1)

            self.selected_records = selected
            console.print(f"[green]✓[/green] Selected {len(selected)} record(s)")
            return

        # Display available records (directories only)
        table = Table(title="Available Records", box=box.ROUNDED)
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Record Name", style="green")
        table.add_column("Files", style="dim")

        for idx, record in enumerate(records, 1):
            # Count files in this record
            if self.data_source == "processed":
                record_dir = PROCESSED_DIR / record
                file_count = len(list(record_dir.glob("*.parquet")))
                table.add_row(str(idx), record, f"{file_count} file(s)")
            else:
                table.add_row(str(idx), record, "-")

        console.print(table)

        # Select record directory with autocomplete
        selected_record = questionary.autocomplete(
            "Select a record (start typing for autocomplete):",
            choices=records,
            style=self._get_style(),
        ).ask()

        if not selected_record:
            console.print("[red]✗ No record selected[/red]")
            sys.exit(1)

        console.print(f"[green]✓[/green] Selected record: {selected_record}")

        # For processed data, show files within the selected directory
        if self.data_source == "processed":
            selected_file = self._select_file_from_record(selected_record)
            if selected_file:
                self.selected_records = [f"{selected_record}/{selected_file}"]
                console.print(f"[green]✓[/green] Selected file: {selected_file}")
            else:
                console.print("[red]✗ No file selected[/red]")
                sys.exit(1)
        else:
            self.selected_records = [selected_record]

    def choose_action(self) -> str:
        """Step 3: Choose what action to perform with the data."""
        console.print("\n[bold]Step 3:[/bold] Choose an action", style="cyan")

        # Show current dataframe status if loaded
        if self.current_dataframe is not None:
            self._display_dataframe_status()

        if self.mode == "batch":
            # For batch mode, only allow certain actions
            batch_actions = [
                questionary.Choice(
                    "🔄 Process all selected records with default pipeline",
                    value="batch_process",
                ),
                questionary.Choice(
                    "⏲️ Resample all records to target resolution",
                    value="batch_resample",
                ),
                questionary.Choice("🔙 Back to data source selection", value="restart"),
                questionary.Choice("❌ Exit", value="exit"),
            ]

            choice = questionary.select(
                "What would you like to do?",
                choices=batch_actions,
                style=self._get_style(),
            ).ask()

            return choice
        else:

            actions = [
                questionary.Choice("📋 Display data as DataFrame", value="display_df"),
                questionary.Choice("📊 Show data statistics", value="show_stats"),
                questionary.Choice(
                    "📈 Plot signal with annotations", value="plot_signal"
                ),
                questionary.Choice(
                    "🧹 Preprocess signal (clean artifacts)", value="preprocess"
                ),
                questionary.Choice(
                    "🔍 Analyze signal for resampling", value="resample_analysis"
                ),
                questionary.Choice(
                    "⏲️ Resample signal to different time resolution",
                    value="resample_signal",
                ),
                questionary.Choice(
                    "✨ Extract features and save to parquet", value="extract_features"
                ),
                # Data management actions
                questionary.Choice(
                    "💾 Save current dataframe to parquet", value="save_dataframe"
                ),
                questionary.Choice(
                    "📤 Export to parquet (if raw data)", value="export_parquet"
                ),
                questionary.Choice("🔄 Select different record", value="change_record"),
                questionary.Choice("🔙 Back to data source selection", value="restart"),
                questionary.Choice("❌ Exit", value="exit"),
            ]

            choice = questionary.select(
                "What would you like to do?", choices=actions, style=self._get_style()
            ).ask()

            return choice

    def execute_action(self, action: str):
        """Execute the selected action."""
        try:
            if action == "display_df":
                self._display_dataframe()
            elif action == "show_stats":
                self._show_statistics()
            elif action == "plot_signal":
                self._plot_signal()
            elif action == "preprocess":
                self._preprocess_signal()
            elif action == "resample_signal":
                self._resample_signal()
            elif action == "resample_analysis":
                self._resample_analysis()

            elif action == "extract_features":
                self._extract_and_save_features()
            elif action == "save_dataframe":
                self._save_dataframe()
            elif action == "export_parquet":
                self._export_to_parquet()
            elif action == "change_record":
                self.selected_records = []
                self.current_dataframe = None
                self.applied_operations = []
                self.select_records()
            elif action == "restart":
                self.current_dataframe = None
                self.applied_operations = []
                self.choose_data_source()
                self.select_records()
            elif action == "batch_process":
                self._batch_process_records()
            elif action == "batch_resample":
                self._batch_resample_records()
        except Exception as e:
            console.print(f"[red]✗ Error: {str(e)}[/red]")

    def _display_dataframe(self):
        """Display the data as a pandas DataFrame."""
        console.print("\n[bold cyan]Loading data...[/bold cyan]")

        df = self._load_data()

        console.print(f"\n[bold]DataFrame Preview[/bold] ({len(df)} rows)")
        console.print(df.head(20).to_string())

        # Show column info
        console.print("\n[bold]Columns:[/bold]")
        for col in df.columns:
            console.print(f"  • {col} ({df[col].dtype})")

    def _display_dataframe_status(self):
        """Display status of current dataframe and applied operations."""
        status_table = Table(
            title="Current DataFrame Status",
            box=box.ROUNDED,
            show_header=False,
            border_style="dim",
        )
        status_table.add_column("Info", style="dim")

        if self.current_dataframe is not None:
            rows, cols = self.current_dataframe.shape
            status_table.add_row(f"📊 Loaded: {rows:,} rows × {cols} columns")

            if self.applied_operations:
                ops_str = " → ".join(self.applied_operations)
                status_table.add_row(f"⚙️  Pipeline: {ops_str}")
            else:
                status_table.add_row(
                    "⚙️  Pipeline: [yellow]No modifications yet[/yellow]"
                )
        else:
            status_table.add_row("📊 No dataframe loaded yet")

        console.print(status_table)

    def _show_statistics(self):
        """Show statistical summary of the data."""
        console.print("\n[bold cyan]Computing statistics...[/bold cyan]")

        df = self._load_data()

        console.print("\n[bold]Data Statistics[/bold]")
        console.print(df.describe().to_string())

        # Additional sleep-specific stats
        if "is_apnea" in df.columns and "is_hypopnea" in df.columns:
            apnea_count = df["is_apnea"].sum()
            hypopnea_count = df["is_hypopnea"].sum()
            total_samples = len(df)

            stats_table = Table(title="Sleep Event Statistics", box=box.ROUNDED)
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")

            stats_table.add_row("Total samples", f"{total_samples:,}")
            stats_table.add_row("Apnea events", f"{apnea_count:,}")
            stats_table.add_row("Hypopnea events", f"{hypopnea_count:,}")
            stats_table.add_row("Apnea %", f"{(apnea_count/total_samples)*100:.2f}%")
            stats_table.add_row(
                "Hypopnea %", f"{(hypopnea_count/total_samples)*100:.2f}%"
            )

            console.print(stats_table)

    def _plot_signal(self):
        """Plot the signal with annotations."""
        console.print("\n[bold cyan]Plotting signal...[/bold cyan]")

        record = self.selected_records[0]

        df = self._load_data()
        pl.plot_with_annotations(df, title=f"SaO2 with Annotations - {record}")

        console.print("[green]✓[/green] Plot displayed (close window to continue)")

    def _preprocess_signal(self):
        """Preprocess the signal to remove artifacts."""
        console.print("\n[bold cyan]Preprocessing signal...[/bold cyan]")

        df = self._load_data()

        if "sao2_percent" not in df.columns:
            console.print("[red]✗ No SaO2 signal found in data[/red]")
            return

        original_signal = df["sao2_percent"].to_numpy()
        cleaned_signal = pp.preproccess_signal(original_signal)

        df["sao2_percent"] = cleaned_signal

        # Show comparison
        console.print("\n[bold]Preprocessing Results[/bold]")
        console.print(
            f"Original signal range: {original_signal.min():.2f} - {original_signal.max():.2f}"
        )
        console.print(
            f"Cleaned signal range: {cleaned_signal.min():.2f} - {cleaned_signal.max():.2f}"
        )

        # Calculate differences
        diff = np.abs(original_signal - cleaned_signal)
        console.print(f"Mean difference: {diff.mean():.4f}")
        console.print(f"Max difference: {diff.max():.4f}")
        console.print(
            f"Samples modified: {(diff > 0.01).sum()} ({(diff > 0.01).sum()/len(diff)*100:.2f}%)"
        )

        # Store cleaned data
        self.current_dataframe = df
        self.applied_operations.append("Preprocessed")
        console.print("[green]✓[/green] Signal preprocessed")

        # Ask if user wants to save
        save = questionary.confirm(
            "Save preprocessed data to parquet?", default=False
        ).ask()

        if save:
            output_dir = PROCESSED_DIR / self.selected_records[0]
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{self.selected_records[0]}_cleaned.parquet"
            df.to_parquet(output_file, index=False)
            console.print(f"[green]✓[/green] Saved to {output_file}")

    def _resample_analysis(self):
        """Analyze signal to estimate original sampling rate before resampling."""
        console.print("\n[bold cyan]Analyzing signal for resampling...[/bold cyan]")

        df = self._load_data()

        if "sao2_percent" not in df.columns:
            console.print("[red]✗ No SaO2 signal found in data[/red]")
            return

        signal = df["sao2_percent"].to_numpy(dtype=float, na_value=np.nan)
        report = rs.find_pre_resampled_rate(signal, current_fs=200)
        rs.print_analysis_results(self.selected_records[0], report)

    def _resample_signal(self):
        """Resample the signal to a different time resolution."""
        console.print("\n[bold cyan]Resampling signal...[/bold cyan]")

        df = self._load_data()

        if "time_s" not in df.columns or "sao2_percent" not in df.columns:
            console.print(
                "[red]✗ Required columns 'time_s' and 'sao2_percent' not found[/red]"
            )
            return

        target_resolution = questionary.text(
            "Enter target time resolution in seconds (e.g., 0.5 for 500ms):",
            validate=lambda text: text.replace(".", "", 1).isdigit(),
        ).ask()

        target_resolution = float(target_resolution)
        resampled_df = rs.resample_to_time_resolution(df, target_resolution)
        self.current_dataframe = resampled_df
        self.applied_operations.append(f"Resampled({target_resolution}s)")

        console.print(
            f"[green]✓[/green] Signal resampled to {target_resolution} seconds"
        )
        console.print(resampled_df.head().to_string())

    def _extract_and_save_features(self):
        """Extract features from the signal and save to a new parquet file."""
        console.print("\n[bold cyan]Extracting features...[/bold cyan]")

        df = self._load_data()

        if "time_s" not in df.columns or "sao2_percent" not in df.columns:
            console.print(
                "[red]✗ Required columns 'time_s' and 'sao2_percent' not found[/red]"
            )
            return

        features_df = fe.extract_features(df)
        self.current_features = features_df

        console.print(f"[green]✓[/green] Features extracted")
        console.print(features_df.head().to_string())

        # Ask if user wants to save
        if self.current_features is not None:
            save = questionary.confirm(
                "Save extracted features to parquet?", default=True
            ).ask()

            if save:
                self._save_features()

    def _export_to_parquet(self):
        """Export raw data to parquet format."""
        if self.data_source != "raw":
            console.print("[yellow]⚠ This action only works with raw data[/yellow]")
            return

        console.print("\n[bold cyan]Exporting to parquet...[/bold cyan]")

        record = self.selected_records[0]
        record_path = RAW_DIR / record

        # Create output directory
        output_dir = PROCESSED_DIR / record
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{record}.parquet"

        # Extract and save
        dm.extract_and_save_to_parquet(str(record_path / record))

        # Move to processed directory
        import shutil

        source = Path(str(record_path / record) + ".parquet")
        if source.exists():
            shutil.move(str(source), str(output_file))
            console.print(f"[green]✓[/green] Exported to {output_file}")
        else:
            console.print("[red]✗ Export failed[/red]")

    def _save_dataframe(self):
        """Save the current dataframe to a parquet file."""
        if self.current_dataframe is None:
            console.print("[yellow]⚠ No dataframe loaded yet[/yellow]")
            return

        if not self.applied_operations:
            console.print(
                "[yellow]⚠ No modifications have been made to the dataframe[/yellow]"
            )
            should_continue = questionary.confirm(
                "Save original data anyway?", default=False
            ).ask()
            if not should_continue:
                return

        console.print("\n[bold cyan]Saving dataframe...[/bold cyan]")

        # Show what will be saved
        self._display_dataframe_status()

        # Get filename
        record = self.selected_records[0]
        # Get base record name (without subdirectory path)
        base_record = record.split("/")[-1] if "/" in record else record
        operations_suffix = (
            "_".join(self.applied_operations)
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
            console.print("[yellow]✗ Save cancelled[/yellow]")
            return

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
            style=self._get_style(),
        ).ask()

        if save_location == "processed":
            # Use base record name for output directory
            base_record = record.split("/")[0] if "/" in record else record
            output_dir = PROCESSED_DIR / base_record
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / filename
        else:
            custom_path = questionary.path(
                "Enter full path to save location:", default=str(DATA_DIR)
            ).ask()

            if not custom_path:
                console.print("[yellow]✗ Save cancelled[/yellow]")
                return

            output_file = Path(custom_path) / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the file
        try:
            self.current_dataframe.to_parquet(output_file, index=False)

            # Create metadata file with operations log
            metadata_file = output_file.with_suffix(".metadata.txt")
            with open(metadata_file, "w") as f:
                f.write(f"Record: {record}\n")
                f.write(f"Source: {self.data_source}\n")
                f.write(f"Original file: {self.selected_records[0]}\n")
                f.write(f"Operations applied:\n")
                for i, op in enumerate(self.applied_operations, 1):
                    f.write(f"  {i}. {op}\n")
                f.write(f"\nDataFrame shape: {self.current_dataframe.shape}\n")
                f.write(f"Columns: {', '.join(self.current_dataframe.columns)}\n")

            console.print(f"[green]✓[/green] Saved to {output_file}")
            console.print(f"[green]✓[/green] Metadata saved to {metadata_file}")

        except Exception as e:
            console.print(f"[red]✗ Failed to save: {str(e)}[/red]")

    def _save_features(self):
        """Save the extracted features to a parquet file."""
        if self.current_features is None:
            console.print("[yellow]⚠ No features extracted yet[/yellow]")
            return

        console.print("\n[bold cyan]Saving features...[/bold cyan]")

        # Get filename
        record = self.selected_records[0]
        base_record = record.split("/")[-1] if "/" in record else record
        default_filename = f"{base_record}_features.parquet"

        filename = questionary.text(
            "Enter filename (without path):", default=default_filename
        ).ask()

        if not filename:
            console.print("[yellow]✗ Save cancelled[/yellow]")
            return

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
            style=self._get_style(),
        ).ask()

        if save_location == "processed":
            base_record = record.split("/")[0] if "/" in record else record
            output_dir = PROCESSED_DIR / base_record
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / filename
        else:
            custom_path = questionary.path(
                "Enter full path to save location:", default=str(DATA_DIR)
            ).ask()

            if not custom_path:
                console.print("[yellow]✗ Save cancelled[/yellow]")
                return

            output_file = Path(custom_path) / filename
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save the features
        try:
            self.current_features.to_parquet(output_file, index=False)

            # Create metadata file with operations log
            # metadata_file = output_file.with_suffix(".metadata.txt")
            # with open(metadata_file, "w") as f:
            #     f.write(f"Record: {record}\n")
            #     f.write(f"Source: {self.data_source}\n")
            #     f.write(f"Original file: {self.selected_records[0]}\n")
            #     f.write(f"Operations applied:\n")
            #     for i, op in enumerate(self.applied_operations, 1):
            #         f.write(f"  {i}. {op}\n")
            #     f.write(f"\nFeatures shape: {self.current_features.shape}\n")
            #     f.write(f"Columns: {', '.join(self.current_features.columns)}\n")

            console.print(f"[green]✓[/green] Features saved to {output_file}")
            # console.print(f"[green]✓[/green] Metadata saved to {metadata_file}")

        except Exception as e:
            console.print(f"[red]✗ Failed to save features: {str(e)}[/red]")

    def _load_data(self, alt_record: Optional[str] = None) -> pd.DataFrame:
        """Load data based on current data source and selected record.

        Args:
            alt_record: Optional specific record to load (used for batch mode)

        Returns:
            Loaded DataFrame
        """
        if self.current_dataframe is not None:
            return self.current_dataframe

        record = self.selected_records[0] if alt_record is None else alt_record

        if self.data_source == "raw":

            record_path = (
                Path(self.custom_data_dir) / record
                if self.custom_data_dir
                else RAW_DIR / record
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

        self.current_dataframe = df
        return df

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
                console.print(
                    f"[dim]📋 Loaded metadata: {len(operations)} operation(s) found[/dim]"
                )

        except Exception as e:
            console.print(
                f"[dim yellow]⚠ Could not load metadata: {str(e)}[/dim yellow]"
            )

    def _get_available_records(self) -> List[str]:
        """Get list of available record directories from the selected data source."""
        # Use custom directory if set, otherwise use default
        if self.custom_data_dir:
            data_dir = Path(self.custom_data_dir)
        elif self.data_source == "raw":
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

    def _select_file_from_record(self, record: str) -> Optional[str]:
        """Select a specific file from within a record directory.

        Args:
            record: The record directory name

        Returns:
            The filename (without .parquet extension) or None if cancelled
        """
        record_dir = PROCESSED_DIR / record

        # Get all parquet files in the directory
        parquet_files = [
            f for f in record_dir.glob("*.parquet") if not f.name.startswith(".")
        ]

        if not parquet_files:
            console.print(f"[red]✗ No parquet files found in {record}[/red]")
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
            size_str = self._format_file_size(file_size)

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

        console.print(table)

        # Select file with autocomplete
        file_choices = [info["name"] for info in file_info]
        selected = questionary.autocomplete(
            "Select a file (start typing for autocomplete):",
            choices=file_choices,
            style=self._get_style(),
        ).ask()

        return selected

    def _download_physionet_data(self):
        """Download data from PhysioNet Challenge 2018."""
        console.print("\n[bold cyan]PhysioNet Data Downloader[/bold cyan]")
        console.print(
            "[dim]Source: https://physionet.org/files/challenge-2018/1.0.0/[/dim]"
        )
        console.print(
            "[dim yellow]Note: PhysioNet may limit download speeds to ~1-2 MB/s[/dim yellow]\n"
        )

        # Check if wget is available
        try:
            subprocess.run(
                ["which", "wget"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            console.print(
                "[red]✗ wget is not installed. Please install it first:[/red]"
            )
            console.print("  macOS: brew install wget")
            console.print("  Linux: sudo apt-get install wget")
            return

        # Download options
        download_type = questionary.select(
            "What would you like to download?",
            choices=[
                questionary.Choice("📦 Specific training records", value="specific"),
                questionary.Choice(
                    "📦 Range of training records",
                    value="range",
                ),
                questionary.Choice(
                    "📦 All training data (~1.5GB)", value="all_training"
                ),
                questionary.Choice("🔙 Back", value="back"),
            ],
            style=self._get_style(),
        ).ask()

        if download_type == "back":
            return
        elif download_type == "specific":
            self._download_specific_records()
        elif download_type == "range":
            self._download_range_records()
        elif download_type == "all_training":
            self._download_all_training()

    def _download_specific_records(self):
        """Download specific records by entering record IDs."""
        console.print("\n[bold]Download Specific Records[/bold]")
        console.print(
            "[dim]Enter record IDs (e.g., tr03-0146, tr03-0147) separated by commas[/dim]"
        )

        record_ids = questionary.text(
            "Record IDs:",
            default="tr03-0146",
        ).ask()

        if not record_ids:
            return

        # Parse record IDs
        records = [r.strip() for r in record_ids.split(",")]

        # Download each record
        for record in records:
            self._download_single_record(record)

        console.print(
            f"\n[green]✓[/green] Downloaded {len(records)} record(s) to {RAW_DIR}"
        )

    def _download_range_records(self):
        """Download a range of records."""
        console.print("\n[bold]Download Range of Records[/bold]")
        console.print("[dim]Enter start and end record numbers[/dim]")

        start = questionary.text(
            "Start record (e.g., tr03-0100):", default="tr03-0100"
        ).ask()
        end = questionary.text(
            "End record (e.g., tr03-0110):", default="tr03-0110"
        ).ask()

        if not start or not end:
            return

        # Extract numbers from record IDs
        start_match = re.match(r"(tr\d+-?)(\d+)", start)
        end_match = re.match(r"(tr\d+-?)(\d+)", end)

        if not start_match or not end_match:
            console.print("[red]✗ Invalid record format[/red]")
            return

        prefix = start_match.group(1)
        start_num = int(start_match.group(2))
        end_num = int(end_match.group(2))

        if start_num > end_num:
            console.print("[red]✗ Start record must be less than end record[/red]")
            return

        # Download records in range
        records = []
        for num in range(start_num, end_num + 1):
            record = f"{prefix}{num:04d}"
            records.append(record)

        console.print(f"\n[bold]Downloading {len(records)} records...[/bold]")

        success_count = 0
        for record in records:
            if self._download_single_record(record, show_progress=False):
                success_count += 1

        console.print(
            f"\n[green]✓[/green] Successfully downloaded {success_count}/{len(records)} records to {RAW_DIR}"
        )

    def _download_all_training(self):
        """Download all training data using wget recursive."""
        console.print("\n[yellow]⚠ This will download ~1.5GB of data[/yellow]")

        confirm = questionary.confirm("Continue with download?", default=False).ask()

        if not confirm:
            return

        console.print("\n[bold cyan]Downloading all training data...[/bold cyan]")

        # Create data directory
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        # Download using wget
        base_url = "https://physionet.org/files/challenge-2018/1.0.0/training/"

        cmd = [
            "wget",
            "-r",  # recursive
            "-N",  # timestamping
            "-c",  # continue
            "-np",  # no parent
            "-nH",  # no host directories
            "--cut-dirs=4",  # cut directory depth
            "-P",
            str(RAW_DIR),  # output directory
            "-A",
            "*.mat,*.arousal,*.hea",  # accept only these file types
            base_url,
        ]

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(description="Downloading...", total=None)
                result = subprocess.run(
                    cmd,
                    cwd=str(RAW_DIR.parent),
                    capture_output=False,
                )

            if result.returncode == 0:
                console.print(f"\n[green]✓[/green] Download completed to {RAW_DIR}")
            else:
                console.print(f"\n[red]✗[/red] Download failed")

        except KeyboardInterrupt:
            console.print("\n[yellow]✗ Download cancelled[/yellow]")
        except Exception as e:
            console.print(f"\n[red]✗ Error during download: {str(e)}[/red]")

    def _download_single_record(self, record: str, show_progress: bool = True) -> bool:
        """
        Download a single record from PhysioNet.

        Args:
            record: Record ID (e.g., "tr03-0146")
            show_progress: Whether to show progress messages

        Returns:
            True if download successful, False otherwise
        """
        base_url = "https://physionet.org/files/challenge-2018/1.0.0/training/"

        # Create record directory
        record_dir = RAW_DIR / record
        record_dir.mkdir(parents=True, exist_ok=True)

        # Files to download for each record
        extensions = [".mat", ".arousal", ".hea"]

        if show_progress:
            console.print(f"\n[bold]Downloading {record}...[/bold]")

        success = True
        for ext in extensions:
            url = f"{base_url}{record}/{record}{ext}"
            output_file = record_dir / f"{record}{ext}"

            # Skip if file already exists
            if output_file.exists():
                if show_progress:
                    console.print(f"  [dim]✓ {record}{ext} (already exists)[/dim]")
                continue

            try:
                # Use wget with progress bar for better feedback during slow downloads
                cmd = ["wget", "--progress=bar:force", "-O", str(output_file), url]
                result = subprocess.run(
                    cmd, capture_output=False, timeout=900
                )  # 15 minutes timeout

                if result.returncode == 0 and output_file.exists():
                    if show_progress:
                        file_size = output_file.stat().st_size
                        size_str = self._format_file_size(file_size)
                        console.print(f"  [green]✓[/green] {record}{ext} ({size_str})")
                else:
                    if show_progress:
                        console.print(f"  [red]✗[/red] {record}{ext} (failed)")
                    success = False
            except subprocess.TimeoutExpired:
                if show_progress:
                    console.print(
                        f"  [red]✗[/red] {record}{ext} (timeout after 5 minutes)"
                    )
                success = False
            except Exception as e:
                if show_progress:
                    console.print(f"  [red]✗[/red] {record}{ext} ({str(e)})")
                success = False

        return success

    def _batch_process_records(self):
        """Batch process all selected records with default pipeline."""
        console.print("\n[bold cyan]Batch Processing Records[/bold cyan]")
        console.print(
            f"[dim]Processing {len(self.selected_records)} record(s) with default pipeline...[/dim]"
        )

        for record in self.selected_records:
            console.print(f"\n[bold]Processing {record}...[/bold]")
            try:
                # Load data
                self.current_dataframe = None  # Clear current dataframe to force reload
                df = self._load_data(record)

                # Apply default processing pipeline
                # df = pp.preproccess_signal(df["sao2_percent"].to_numpy())
                # df = rs.resample_to_time_resolution(df, 0.5)

                # Save processed data
                output_dir = PROCESSED_DIR / record
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{record}.parquet"
                df.to_parquet(output_file, index=False)

                console.print(f"[green]✓[/green] Processed and saved to {output_file}")
            except Exception as e:
                console.print(f"[red]✗ Failed to process {record}: {str(e)}[/red]")

    def _batch_resample_records(self):
        """Batch resample all selected records to target resolution."""
        console.print("\n[bold cyan]Batch Resampling Records[/bold cyan]")
        console.print(
            f"[dim]Resampling {len(self.selected_records)} record(s) to target resolution...[/dim]"
        )

        target_resolution = questionary.text(
            "Enter target time resolution in seconds (e.g., 0.5 for 500ms):",
            validate=lambda text: text.replace(".", "", 1).isdigit(),
        ).ask()

        target_resolution = float(target_resolution)

        for record in self.selected_records:
            console.print(f"\n[bold]Resampling {record}...[/bold]")
            try:
                # Load data
                self.current_dataframe = None  # Clear current dataframe to force reload
                df = self._load_data(record)

                # Resample signal
                resampled_df = rs.resample_to_time_resolution(df, target_resolution)

                # Create metadata file with operations log
                metadata_file = (
                    PROCESSED_DIR / record / f"{record}_resampled.metadata.txt"
                )
                with open(metadata_file, "w") as f:
                    f.write(f"Record: {record}\n")
                    f.write(f"Source: {self.data_source}\n")
                    f.write(f"Original file: {record}\n")
                    f.write(f"Operations applied:\n")
                    f.write(f"  1. Resampled to {target_resolution} seconds\n")
                    f.write(f"\nResampled DataFrame shape: {resampled_df.shape}\n")
                    f.write(f"Columns: {', '.join(resampled_df.columns)}\n")

                # Save resampled data
                output_dir = PROCESSED_DIR / record
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{record}.parquet"
                resampled_df.to_parquet(output_file, index=False)

                console.print(f"[green]✓[/green] Resampled and saved to {output_file}")
            except Exception as e:
                console.print(f"[red]✗ Failed to resample {record}: {str(e)}[/red]")

    @staticmethod
    def _format_file_size(size_bytes: float) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _get_record_path(self, record: str) -> Path:
        """Get the full path to a record directory."""
        if self.data_source == "raw":
            return RAW_DIR / record
        else:
            return PROCESSED_DIR / record

    def _set_custom_directory(self):
        """Prompt user for a custom data directory path."""
        console.print(
            "[bold cyan]📁 Enter custom directory path[/bold cyan]",
        )
        console.print(
            "[dim]Examples: /path/to/data/raw, C:\\Users\\Data, ./my_records[/dim]"
        )

        path_input = questionary.text(
            "Directory path:",
            validate=lambda text: bool(text.strip()),
        ).ask()

        if not path_input:
            console.print("[yellow]↩️ Using default directory[/yellow]")
            self.custom_data_dir = None
            return

        custom_path = Path(path_input.strip())

        if not custom_path.exists():
            console.print(f"[red]✗ Directory does not exist: {custom_path}[/red]")
            retry = questionary.confirm("🔄 Try another path?", default=True).ask()
            if retry:
                self._set_custom_directory()
            else:
                console.print("[yellow]↩️ Using default directory[/yellow]")
                self.custom_data_dir = None
            return

        if not custom_path.is_dir():
            console.print(f"[red]✗ Path is not a directory: {custom_path}[/red]")
            retry = questionary.confirm("🔄 Try another path?", default=True).ask()
            if retry:
                self._set_custom_directory()
            else:
                console.print("[yellow]↩️ Using default directory[/yellow]")
                self.custom_data_dir = None
            return

        self.custom_data_dir = str(custom_path)
        console.print(f"[green]✓[/green] Using custom directory: {custom_path}")
        console.print(f"[dim]📁 Loading records from: {custom_path}[/dim]")

    @staticmethod
    def _get_style():
        """Get questionary style configuration."""
        from questionary import Style

        return Style(
            [
                ("qmark", "fg:#e91e63 bold"),
                ("question", "fg:#673ab7 bold"),
                ("answer", "fg:#2196f3 bold"),
                ("pointer", "fg:#e91e63 bold"),
                ("highlighted", "fg:#e91e63 bold"),
                ("selected", "fg:#4caf50"),
                ("separator", "fg:#999999"),
                ("instruction", "fg:#999999"),
            ]
        )


def main():
    """Main entry point."""
    pipeline = SleepDataPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
