"""
Batch processing module for the Sleep Data Analysis Pipeline CLI.
"""

import questionary

try:
    from .config import PROCESSED_DIR
except ImportError:
    from config import PROCESSED_DIR


class BatchProcessor:
    """Handles batch processing of multiple records."""

    def __init__(self, console):
        self.console = console

    # Steps are executed in this fixed order regardless of selection order
    _STEP_ORDER = [
        "export_parquet",
        "preprocess",
        "resample",
        "extract_features",
        "normalize_features",
    ]

    def _feature_file_exists(self, record: str) -> bool:
        """Check whether a features parquet file exists for a record."""
        record_dir = record.split("/")[0] if "/" in record else record
        return (PROCESSED_DIR / record_dir / f"{record_dir}_features.parquet").exists()

    def _check_feature_coverage(
        self, selected_records: list, ordered_steps: list
    ) -> list:
        """
        When extract_features is NOT in the pipeline, check that every record
        already has a feature file. Returns the list of records that are missing
        one, or an empty list if coverage is complete (or extraction is planned).
        """
        if "extract_features" in ordered_steps:
            return []
        return [r for r in selected_records if not self._feature_file_exists(r)]

    def run_pipeline(
        self,
        selected_records: list,
        steps: list,
        data_source: str,
        data_loader,
        data_processor,
        data_saver,
        resample_resolution: float = None,
    ):
        """Run the selected pipeline steps on each record in order."""
        ordered_steps = [s for s in self._STEP_ORDER if s in steps]

        self.console.print("\n[bold cyan]Batch Pipeline[/bold cyan]")
        self.console.print(
            f"[dim]Steps: {', '.join(ordered_steps)} "
            f"— {len(selected_records)} record(s)[/dim]"
        )

        # --- Pre-run feature coverage check ---
        missing = self._check_feature_coverage(selected_records, ordered_steps)
        if missing:
            self.console.print(
                f"\n[yellow]⚠ {len(missing)} record(s) are missing feature files:[/yellow]"
            )
            for r in missing:
                self.console.print(f"  [dim]• {r}[/dim]")

            choice = questionary.select(
                "How do you want to proceed?",
                choices=[
                    questionary.Choice(
                        "✨ Extract features for missing records, then continue",
                        value="extract_missing",
                    ),
                    questionary.Choice(
                        "✨ Extract features for ALL records (ensures consistency)",
                        value="extract_all",
                    ),
                    questionary.Choice(
                        "❌ Abort — a complete training set requires all features",
                        value="abort",
                    ),
                ],
            ).ask()

            if choice == "abort" or choice is None:
                self.console.print("[yellow]✗ Batch run cancelled.[/yellow]")
                return

            if choice == "extract_all":
                ordered_steps = [
                    s
                    for s in self._STEP_ORDER
                    if s in ordered_steps or s == "extract_features"
                ]
                records_needing_extraction = selected_records
            else:  # extract_missing
                ordered_steps = [
                    s
                    for s in self._STEP_ORDER
                    if s in ordered_steps or s == "extract_features"
                ]
                records_needing_extraction = set(missing)
        else:
            records_needing_extraction = set()

        # --- Per-record execution ---
        succeeded = 0
        failed = 0

        for record in selected_records:
            self.console.print(f"\n[bold]── {record} ──[/bold]")
            data_loader.clear()
            try:
                will_extract = "extract_features" in ordered_steps and (
                    records_needing_extraction is selected_records
                    or record in records_needing_extraction
                )

                if "export_parquet" in ordered_steps:
                    data_saver.export_to_parquet([record], data_source)

                df = data_loader.load_data(
                    data_source,
                    [record],
                    load_features_mode="skip" if will_extract else "auto",
                )
                if df is None:
                    self.console.print("[red]✗ Could not load data, skipping.[/red]")
                    failed += 1
                    continue

                if "preprocess" in ordered_steps:
                    processed_df, success = data_processor.preprocess_signal(df)
                    if success:
                        df = processed_df
                        data_loader.current_dataframe = df
                        data_loader.applied_operations.append("Preprocessed")

                if "resample" in ordered_steps and resample_resolution is not None:
                    resampled_df, success = data_processor.resample_signal(
                        df, resample_resolution
                    )
                    if success:
                        df = resampled_df
                        data_loader.current_dataframe = df
                        data_loader.applied_operations.append(
                            f"Resampled({resample_resolution}s)"
                        )

                if data_loader.applied_operations:
                    data_saver.save_dataframe(
                        df, [record], data_loader.applied_operations, data_source
                    )

                if will_extract:
                    features_df, success = data_processor.extract_features(df)
                    if success:
                        data_loader.current_features = features_df
                        data_loader.current_features_normalized = None

                        if "normalize_features" in ordered_steps:
                            norm_df, norm_success = data_processor.normalize_features(
                                features_df
                            )
                            if norm_success:
                                data_loader.current_features_normalized = norm_df

                        data_saver.save_features(
                            features_df,
                            [record],
                            normalized_features_df=data_loader.current_features_normalized,
                        )

                succeeded += 1

            except Exception as e:
                self.console.print(f"[red]✗ Failed: {str(e)}[/red]")
                failed += 1

        self.console.print(
            f"\n[green]✓ Done — {succeeded} succeeded"
            + (f", [red]{failed} failed[/red]" if failed else "")
            + "[/green]"
        )
