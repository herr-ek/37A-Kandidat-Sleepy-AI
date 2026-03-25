"""
Batch processing module for the Sleep Data Analysis Pipeline CLI.
"""

try:
    from .. import Resampling as rs
    from .config import PROCESSED_DIR
except ImportError:
    from config import PROCESSED_DIR

    import Resampling as rs


class BatchProcessor:
    """Handles batch processing of multiple records."""

    def __init__(self, console):
        self.console = console

    def batch_process_records(
        self, selected_records: list, data_loader, data_processor
    ):
        """Batch process all selected records with default pipeline."""
        self.console.print("\n[bold cyan]Batch Processing Records[/bold cyan]")
        self.console.print(
            f"[dim]Processing {len(selected_records)} record(s) with default pipeline...[/dim]"
        )

        for record in selected_records:
            self.console.print(f"\n[bold]Processing {record}...[/bold]")
            try:
                # Load data
                data_loader.clear()
                df = data_loader.load_data("raw", [record])

                # Apply default processing pipeline
                # (In this case, just export to parquet)

                # Save processed data
                output_dir = PROCESSED_DIR / record
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{record}.parquet"
                df.to_parquet(output_file, index=False)

                self.console.print(
                    f"[green]✓[/green] Processed and saved to {output_file}"
                )
            except Exception as e:
                self.console.print(f"[red]✗ Failed to process {record}: {str(e)}[/red]")

    def batch_resample_records(
        self, selected_records: list, target_resolution: float, data_loader
    ):
        """Batch resample all selected records to target resolution."""
        self.console.print("\n[bold cyan]Batch Resampling Records[/bold cyan]")
        self.console.print(
            f"[dim]Resampling {len(selected_records)} record(s) to target resolution...[/dim]"
        )

        for record in selected_records:
            self.console.print(f"\n[bold]Resampling {record}...[/bold]")
            try:
                # Load data
                data_loader.clear()
                df = data_loader.load_data("raw", [record])

                # Resample signal
                resampled_df = rs.resample_to_time_resolution(df, target_resolution)

                # Create metadata file with operations log
                output_dir = PROCESSED_DIR / record
                output_dir.mkdir(parents=True, exist_ok=True)

                metadata_file = output_dir / f"{record}_resampled.metadata.txt"
                with open(metadata_file, "w") as f:
                    f.write(f"Record: {record}\n")
                    f.write(f"Source: raw\n")
                    f.write(f"Original file: {record}\n")
                    f.write(f"Operations applied:\n")
                    f.write(f"  1. Resampled to {target_resolution} seconds\n")
                    f.write(f"\nResampled DataFrame shape: {resampled_df.shape}\n")
                    f.write(f"Columns: {', '.join(resampled_df.columns)}\n")

                # Save resampled data
                output_file = output_dir / f"{record}.parquet"
                resampled_df.to_parquet(output_file, index=False)

                self.console.print(
                    f"[green]✓[/green] Resampled and saved to {output_file}"
                )
            except Exception as e:
                self.console.print(
                    f"[red]✗ Failed to resample {record}: {str(e)}[/red]"
                )
