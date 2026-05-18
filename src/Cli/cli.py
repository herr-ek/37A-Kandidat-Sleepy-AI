#!/usr/bin/env python3
"""
Sleep Data Analysis Pipeline CLI

Interactive command-line interface for processing and analyzing sleep apnea data.
"""

import sys
from pathlib import Path

# Add Cli and src directories to path for script execution
_cli_dir = str(Path(__file__).parent)
_src_dir = str(Path(__file__).parent.parent)
if _cli_dir not in sys.path:
    sys.path.insert(0, _cli_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from rich import box
from rich.console import Console
from rich.panel import Panel

# Import application modules
try:
    # Try relative imports first (when run as module)
    from .batch_processor import BatchProcessor
    from .config import PROCESSED_DIR, RAW_DIR
    from .csv_loader import CsvLoader
    from .data_display import DataDisplay
    from .data_loader import DataLoader
    from .data_processor import DataProcessor
    from .data_saver import DataSaver
    from .downloader import PhysioNetDownloader
    from .inference_manager import InferenceManager
    from .resampling_manager import ResamplingManager
    from .results_browser import ResultsBrowser
    from .training_manager import TrainingManager
    from .ui import CLI_UI
    from .utils import get_questionary_style, set_custom_directory
except ImportError:
    # Fall back to absolute imports (when run as script)
    from batch_processor import BatchProcessor
    from config import PROCESSED_DIR, RAW_DIR
    from csv_loader import CsvLoader
    from data_display import DataDisplay
    from data_loader import DataLoader
    from data_processor import DataProcessor
    from data_saver import DataSaver
    from downloader import PhysioNetDownloader
    from inference_manager import InferenceManager
    from resampling_manager import ResamplingManager
    from results_browser import ResultsBrowser
    from training_manager import TrainingManager
    from ui import CLI_UI
    from utils import get_questionary_style, set_custom_directory

try:
    from ..Data_management.generate_sets import generate_sets
except ImportError:
    import os as _os
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Data_management.generate_sets import generate_sets


class SleepDataPipeline:
    """Main pipeline controller for sleep data analysis."""

    def __init__(self, console: Console):
        self.console = console
        self.data_source = None
        self.mode = None
        self.selected_records = []
        self.custom_data_dir = None
        self.batch_steps = []

        # Initialize component modules
        self.style = get_questionary_style()
        self.ui = CLI_UI(self.console, self.style)
        self.data_loader = DataLoader(self.console)
        self.data_display = DataDisplay(self.console)
        self.data_processor = DataProcessor(self.console)
        self.data_saver = DataSaver(self.console)
        self.downloader = PhysioNetDownloader(self.console)
        self.batch_processor = BatchProcessor(self.console)
        self.training_manager = TrainingManager(self.console, self.style)
        self.inferene_manager = InferenceManager(self.console, self.style)
        self.results_browser = ResultsBrowser(self.console, self.style)
        self.resampling_manager = ResamplingManager(self.console, self.style)
        self.csv_loader = CsvLoader(self.console, self.style)

    def run(self):
        """Main entry point for the CLI."""
        self.console.print(
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
            self.console.print("\n[yellow]✗ Operation cancelled by user[/yellow]")
            sys.exit(0)

        self.console.print("[green]✓ Pipeline completed successfully![/green]")

    def choose_data_source(self):
        """Step 1: Choose between raw or processed data."""
        choice = self.ui.show_data_source_menu()

        if choice is None or choice == "exit":
            self.console.print("[yellow]Goodbye![/yellow]")
            sys.exit(0)
        elif choice == "train":
            self.training_manager.run()
            # Return to main menu after training completes
            self.choose_data_source()
            return
        elif choice == "generate_sets":
            self._handle_generate_sets()
            self.choose_data_source()
            return
        elif choice == "inference":
            self.inferene_manager.run()
            self.choose_data_source()
            return
        elif choice == "results":
            self.results_browser.run()
            self.choose_data_source()
            return
        elif choice == "resample":
            self.resampling_manager.run()
            self.choose_data_source()
            return
        elif choice == "csv":
            result = self.csv_loader.run()
            if result is None:
                self.choose_data_source()
                return
            df, record_name = result
            self.data_source = "csv"
            self.mode = "single"
            self.selected_records = [record_name]
            self.data_loader.current_dataframe = df
            return
        elif choice == "download":
            self._handle_download()
            # After download, let user choose data source again
            self.choose_data_source()
            return

        self.data_source = choice
        self.mode = "single"
        if choice == "raw_batch" or choice == "processed_batch":
            self.data_source = "raw" if choice == "raw_batch" else "processed"
            self.mode = "batch"
        self.console.print(f"[green]✓[/green] Using {choice} data")

    def select_records(self):
        """Step 2: Select record(s) to process."""
        # CSV source: data is already loaded in choose_data_source — skip.
        if self.data_source == "csv":
            return

        self.ui.show_single_record_menu()

        # For batch raw data, ask if user wants to load from custom directory
        if self.mode == "batch" and self.data_source == "raw":
            if self.ui.prompt_custom_directory():
                self.custom_data_dir = set_custom_directory(self.console)

        # Get available record directories
        records = self.data_loader.get_available_records(
            self.data_source, self.custom_data_dir
        )

        if not records:
            default_location = (
                self.custom_data_dir
                if self.custom_data_dir
                else (RAW_DIR if self.data_source == "raw" else PROCESSED_DIR)
            )
            self.console.print(f"[red]✗ No records found in {default_location}[/red]")
            sys.exit(1)

        if self.mode == "batch":
            # For batch mode, select multiple records with checkbox
            selected = self.ui.prompt_checkbox_selection(
                records, "Select records to process"
            )

            if not self.ui.confirm_selection(selected, "No records selected"):
                sys.exit(1)

            self.selected_records = selected
            self.console.print(f"[green]✓[/green] Selected {len(selected)} record(s)")
            return

        # Display available records (directories only)
        self.ui.display_available_records(records, self.data_source)

        # Select record directory with autocomplete
        selected_record = self.ui.prompt_record_selection(records)

        if not selected_record:
            self.console.print("[red]✗ No record selected[/red]")
            sys.exit(1)

        self.console.print(f"[green]✓[/green] Selected record: {selected_record}")

        # For processed data, show files within the selected directory
        if self.data_source == "processed":
            selected_file = self.data_loader.select_file_from_record(selected_record)
            if selected_file:
                self.selected_records = [f"{selected_record}/{selected_file}"]
                self.console.print(f"[green]✓[/green] Selected file: {selected_file}")
            else:
                self.console.print("[red]✗ No file selected[/red]")
                sys.exit(1)
        else:
            self.selected_records = [selected_record]

        # Load data immediately so pipeline state reflects a loaded record
        self.data_loader.load_data(
            self.data_source, self.selected_records, self.custom_data_dir
        )

    def choose_action(self) -> str:
        """Step 3: Choose what action to perform with the data."""
        # Show current dataframe status if loaded
        if (
            self.data_loader.current_dataframe is not None
            or self.data_loader.current_features is not None
        ):
            self.data_display.display_dataframe_status(
                self.data_loader.current_dataframe,
                self.data_loader.applied_operations,
                self.data_loader.current_features,
                self.data_loader.current_features_normalized,
                self.selected_records[0] if self.selected_records else None,
            )

        if self.mode == "batch":
            action = self.ui.show_batch_mode_menu()
            if action == "run_pipeline":
                self.batch_steps = self.ui.prompt_batch_pipeline_steps(self.data_source)
                return "batch_pipeline"
            return action
        else:
            state = {
                "dataframe_loaded": self.data_loader.current_dataframe is not None,
                "preprocessed": "Preprocessed" in self.data_loader.applied_operations,
                "features_extracted": self.data_loader.current_features is not None,
                "features_normalized": self.data_loader.current_features_normalized
                is not None,
            }
            return self.ui.show_single_mode_menu(state)

    def execute_action(self, action: str):
        """Execute the selected action."""
        try:
            if action == "display_df":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                self.data_display.display_dataframe(df)

            elif action == "show_stats":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                self.data_display.show_statistics(df)

            elif action == "plot_signal":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                self.data_display.plot_signal(df, self.selected_records[0])

            elif action == "preprocess":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                processed_df, success = self.data_processor.preprocess_signal(df)
                if success:
                    self.data_loader.current_dataframe = processed_df
                    self.data_loader.applied_operations.append("Preprocessed")
                    self.data_saver.save_dataframe(
                        processed_df,
                        self.selected_records,
                        self.data_loader.applied_operations,
                        self.data_source,
                    )
            elif action == "normalize_features":
                features, _ = self.data_loader.get_features()
                if features is None:
                    self.console.print(
                        "[red]✗ No features available. Please extract features first.[/red]"
                    )
                    return
                normalized_features, success = self.data_processor.normalize_features(
                    features
                )
                if success:
                    self.data_loader.current_features_normalized = normalized_features
                    self.data_loader.applied_operations.append("Postprocessed Features")
            elif action == "save_features":
                features, normalized_features = self.data_loader.get_features()
                if features is None:
                    self.console.print(
                        "[red]✗ No features available. Please extract features first.[/red]"
                    )
                    return
                self.data_saver.save_features(
                    features,
                    self.selected_records,
                    normalized_features_df=normalized_features,
                )
            elif action == "resample_analysis":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                self.data_processor.analyze_for_resampling(df)

            elif action == "resample_signal":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                target_resolution = self.data_processor.prompt_target_resolution()
                resampled_df, success = self.data_processor.resample_signal(
                    df, target_resolution
                )
                if success:
                    self.data_loader.current_dataframe = resampled_df
                    self.data_loader.applied_operations.append(
                        f"Resampled({target_resolution}s)"
                    )
                    self.data_saver.save_dataframe(
                        resampled_df,
                        self.selected_records,
                        self.data_loader.applied_operations,
                        self.data_source,
                    )

            elif action == "trim_signal":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                trimmed_df, success = self.data_processor.trim_signal(df)
                if success:
                    self.data_loader.current_dataframe = trimmed_df
                    self.data_loader.applied_operations.append("Trimmed")
                    self.data_saver.save_dataframe(
                        trimmed_df,
                        self.selected_records,
                        self.data_loader.applied_operations,
                        self.data_source,
                    )

            elif action == "extract_features":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                features_df, success = self.data_processor.extract_features(df)
                self.data_loader.current_features = features_df
                self.data_loader.current_features_normalized = None
                if success:
                    # Ask if user wants to save
                    import questionary

                    save = questionary.confirm(
                        "Save extracted features to parquet?", default=True
                    ).ask()
                    if save:
                        self.data_saver.save_features(
                            features_df,
                            self.selected_records,
                            normalized_features_df=self.data_loader.current_features_normalized,
                        )

            elif action == "save_dataframe":
                df = self.data_loader.load_data(
                    self.data_source, self.selected_records, self.custom_data_dir
                )
                if df is not None:
                    self.data_saver.save_dataframe(
                        df,
                        self.selected_records,
                        self.data_loader.applied_operations,
                        self.data_source,
                    )

            elif action == "export_parquet":
                self.data_saver.export_to_parquet(
                    self.selected_records, self.data_source
                )

            elif action == "change_record":
                self.data_loader.clear()
                self.selected_records = []
                self.select_records()

            elif action == "restart":
                self.data_loader.clear()
                self.choose_data_source()
                self.select_records()

            elif action == "batch_pipeline":
                if not self.batch_steps:
                    self.console.print("[yellow]⚠ No pipeline steps selected.[/yellow]")
                    return
                self.batch_processor.run_pipeline(
                    self.selected_records,
                    self.batch_steps,
                    self.data_source,
                    self.data_loader,
                    self.data_processor,
                    self.data_saver,
                )

        except Exception as e:
            self.console.print(f"[red]✗ Error: {str(e)}[/red]")

    def _handle_download(self):
        """Handle PhysioNet data download."""
        download_type = self.downloader.download_menu()

        if download_type == "back":
            return
        elif download_type == "specific":
            self.downloader.download_specific_records()
        elif download_type == "range":
            self.downloader.download_range_records()
        elif download_type == "all_training":
            self.downloader.download_all_training()

    def _handle_generate_sets(self):
        """Generate train/test/validate set distribution from processed records."""
        import questionary

        train_ratio, test_ratio, validate_ratio = 0.8, 0.1, 0.1

        self.console.print("\n[bold cyan]Generate Set Distribution[/bold cyan]")

        use_custom = questionary.confirm(
            f"Use default split (train={int(train_ratio*100)}% / test={int(test_ratio*100)}% / validate={int(validate_ratio*100)}%)? "
            "Select 'No' to customise.",
            default=True,
        ).ask()
        if use_custom is None:
            return

        if not use_custom:
            while True:
                raw = questionary.text(
                    "Enter train/test/validate split as percentages (e.g. 70 20 10):",
                    validate=lambda v: (
                        True
                        if len(v.split()) == 3
                        and all(p.replace(".", "", 1).isdigit() for p in v.split())
                        and abs(sum(float(p) for p in v.split()) - 100) < 0.01
                        else "Enter three numbers that sum to 100"
                    ),
                ).ask()
                if raw is None:
                    return
                parts = [float(p) / 100 for p in raw.split()]
                train_ratio, test_ratio, validate_ratio = parts[0], parts[1], parts[2]
                break

        self.console.print(
            f"[dim]Split: train={int(train_ratio*100)}% / test={int(test_ratio*100)}% / validate={int(validate_ratio*100)}%[/dim]"
        )

        save = questionary.confirm(
            "Save the generated sets to set_distribution/?", default=True
        ).ask()
        if save is None:
            return

        try:
            train, test, validate = generate_sets(
                save=save,
                train=train_ratio,
                test=test_ratio,
                validate=validate_ratio,
            )
            self.console.print(
                f"[green]✓[/green] Generated sets: "
                f"[cyan]train[/cyan]={len(train)}, "
                f"[cyan]test[/cyan]={len(test)}, "
                f"[cyan]validate[/cyan]={len(validate)}"
            )
            if save:
                self.console.print(
                    "[green]✓[/green] Saved to [dim]set_distribution/[/dim]"
                )
        except Exception as e:
            self.console.print(f"[red]✗ Failed to generate sets: {e}[/red]")


def main():
    """Main entry point."""
    console = Console()
    pipeline = SleepDataPipeline(console)
    pipeline.run()


if __name__ == "__main__":
    main()
