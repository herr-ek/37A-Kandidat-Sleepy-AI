"""
Data display and visualization module for the Sleep Data Analysis Pipeline CLI.
"""

import pandas as pd
from rich import box
from rich.table import Table

try:
    from .. import Plotting as pl
except ImportError:
    import Plotting as pl


class DataDisplay:
    """Handles data display and visualization."""

    def __init__(self, console):
        self.console = console

    def display_dataframe(self, df: pd.DataFrame):
        """Display the data as a pandas DataFrame."""
        self.console.print("\n[bold cyan]Loading data...[/bold cyan]")

        self.console.print(f"\n[bold]DataFrame Preview[/bold] ({len(df)} rows)")
        self.console.print(df.head(20).to_string())

        # Show column info
        self.console.print("\n[bold]Columns:[/bold]")
        for col in df.columns:
            self.console.print(f"  • {col} ({df[col].dtype})")

    def display_dataframe_status(
        self,
        current_dataframe: pd.DataFrame,
        applied_operations: list,
        current_features: pd.DataFrame = None,
        current_features_normalized: pd.DataFrame = None,
        record_name: str = None,
    ):
        """Display status of current dataframe and applied operations."""
        status_table = Table(
            title="Current DataFrame Status",
            box=box.ROUNDED,
            show_header=False,
            border_style="dim",
        )
        status_table.add_column("Info", style="dim")
        status_table.add_row(f"📁 Record: {record_name if record_name else 'Unknown'}")

        if current_dataframe is not None:
            rows, cols = current_dataframe.shape
            status_table.add_row(f"📊 Loaded: {rows:,} rows × {cols} columns")

            if applied_operations:
                ops_str = " → ".join(applied_operations)
                status_table.add_row(f"⚙️  Pipeline: {ops_str}")
            else:
                status_table.add_row(
                    "⚙️  Pipeline: [yellow]No modifications yet[/yellow]"
                )
        else:
            status_table.add_row("📊 No dataframe loaded yet")

        if current_features is not None:
            f_rows, f_cols = current_features.shape
            status_table.add_row(
                f"🧩 Features: Extracted ({f_rows:,} rows × {f_cols} columns)"
            )
            normalization_status = (
                "Yes" if current_features_normalized is not None else "No"
            )
            status_table.add_row(f"📐 Features normalized: {normalization_status}")
        else:
            status_table.add_row("🧩 Features: Not extracted")

        self.console.print(status_table)

    def show_statistics(self, df: pd.DataFrame):
        """Show statistical summary of the data."""
        self.console.print("\n[bold cyan]Computing statistics...[/bold cyan]")

        self.console.print("\n[bold]Data Statistics[/bold]")
        self.console.print(df.describe().to_string())

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

            self.console.print(stats_table)

    def plot_signal(self, df: pd.DataFrame, record_name: str):
        """Plot the signal with annotations."""
        self.console.print("\n[bold cyan]Plotting signal...[/bold cyan]")

        pl.plot_with_annotations(df, title=f"SpO2 with Annotations - {record_name}")

        self.console.print("[green]✓[/green] Plot displayed (close window to continue)")
