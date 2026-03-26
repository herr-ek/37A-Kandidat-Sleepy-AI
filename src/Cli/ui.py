"""
User interface module for the Sleep Data Analysis Pipeline CLI.
"""

from typing import List

import questionary
from rich import box
from rich.table import Table

try:
    from .config import PROCESSED_DIR
except ImportError:
    from config import PROCESSED_DIR


class CLI_UI:
    """Handles all user interface interactions."""

    def __init__(self, console, style):
        self.console = console
        self.style = style

    def show_data_source_menu(self) -> str:
        """Step 1: Choose between raw or processed data."""
        self.console.print("\n[bold]Step 1:[/bold] Choose data source", style="cyan")

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
            style=self.style,
        ).ask()

        return choice

    def show_single_record_menu(self) -> str:
        """Display available records and prompt selection."""
        self.console.print("\n[bold]Step 2:[/bold] Select recording(s)", style="cyan")
        return "continue"

    def display_available_records(self, records: List[str], data_source: str):
        """Display available records in a table."""
        table = Table(title="Available Records", box=box.ROUNDED)
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Record Name", style="green")
        table.add_column("Files", style="dim")

        for idx, record in enumerate(records, 1):
            if data_source == "processed":
                record_dir = PROCESSED_DIR / record
                file_count = len(list(record_dir.glob("*.parquet")))
                table.add_row(str(idx), record, f"{file_count} file(s)")
            else:
                table.add_row(str(idx), record, "-")

        self.console.print(table)

    def prompt_record_selection(self, records: List[str]) -> str:
        """Prompt user to select a record."""
        selected = questionary.autocomplete(
            "Select a record (start typing for autocomplete):",
            choices=records,
            style=self.style,
        ).ask()

        return selected

    def prompt_checkbox_selection(
        self, records: List[str], title: str = "Select records"
    ) -> List[str]:
        """Prompt user to select multiple records using checkbox."""
        selected = questionary.checkbox(
            title + " (use space to select):",
            choices=records,
            style=self.style,
        ).ask()

        return selected if selected else []

    def show_single_mode_menu(self, state: dict = None) -> str:
        """Show action menu for single mode."""
        self.console.print("\n[bold]Step 3:[/bold] Choose an action", style="cyan")

        if state is None:
            state = {}

        df_loaded = state.get("dataframe_loaded", False)
        preprocessed = state.get("preprocessed", False)
        features_extracted = state.get("features_extracted", False)
        features_normalized = state.get("features_normalized", False)

        def _disabled(condition: bool, reason: str):
            return None if condition else reason

        actions = [
            questionary.Separator("── View ──────────────────────────────"),
            questionary.Choice(
                "📋 Display data as DataFrame",
                value="display_df",
                disabled=_disabled(df_loaded, "Load data first"),
            ),
            questionary.Choice(
                "📊 Show data statistics",
                value="show_stats",
                disabled=_disabled(df_loaded, "Load data first"),
            ),
            questionary.Choice(
                "📈 Plot signal with annotations",
                value="plot_signal",
                disabled=_disabled(df_loaded, "Load data first"),
            ),
            questionary.Separator("── Pipeline ──────────────────────────"),
            questionary.Choice(
                "🧹 Preprocess signal",
                value="preprocess",
                disabled=_disabled(df_loaded, "Load data first"),
            ),
            questionary.Choice(
                "✨ Extract features",
                value="extract_features",
                disabled=_disabled(preprocessed, "Preprocess signal first"),
            ),
            questionary.Choice(
                "🥞 Normalize features",
                value="normalize_features",
                disabled=_disabled(features_extracted, "Extract features first"),
            ),
            questionary.Choice(
                "💾 Save features",
                value="save_features",
                disabled=_disabled(features_extracted, "Extract features first"),
            ),
            questionary.Separator("── Resampling ────────────────────────"),
            questionary.Choice(
                "🔍 Analyze signal for resampling",
                value="resample_analysis",
                disabled=_disabled(df_loaded, "Load data first"),
            ),
            questionary.Choice(
                "⌚ Resample signal",
                value="resample_signal",
                disabled=_disabled(df_loaded, "Load data first"),
            ),
            questionary.Separator("── File ──────────────────────────────"),
            questionary.Choice(
                "📤 Export to parquet (raw data only)", value="export_parquet"
            ),
            questionary.Separator("── Navigation ────────────────────────"),
            questionary.Choice("🔄 Select different record", value="change_record"),
            questionary.Choice("🔙 Back to data source selection", value="restart"),
            questionary.Choice("❌ Exit", value="exit"),
        ]

        choice = questionary.select(
            "What would you like to do?", choices=actions, style=self.style
        ).ask()

        return choice

    def show_batch_mode_menu(self) -> str:
        """Show action menu for batch mode."""
        self.console.print("\n[bold]Step 3:[/bold] Choose an action", style="cyan")

        batch_actions = [
            questionary.Choice(
                "🔄 Process all selected records with default pipeline",
                value="batch_process",
            ),
            questionary.Choice(
                "⌚ Resample all records to target resolution",
                value="batch_resample",
            ),
            questionary.Choice("🔙 Back to data source selection", value="restart"),
            questionary.Choice("❌ Exit", value="exit"),
        ]

        choice = questionary.select(
            "What would you like to do?", choices=batch_actions, style=self.style
        ).ask()

        return choice

    def prompt_custom_directory(self) -> bool:
        """Prompt user if they want to use custom directory."""
        use_custom = questionary.confirm(
            "Load from custom directory?", default=False
        ).ask()
        return use_custom

    def confirm_selection(
        self, records: List[str], text: str = "No records selected"
    ) -> bool:
        """Show confirmation if no records selected."""
        if not records:
            self.console.print(f"[red]✗ {text}[/red]")
            return False
        return True
