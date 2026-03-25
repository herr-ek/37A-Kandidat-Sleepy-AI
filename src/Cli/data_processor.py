"""
Data processing module for the Sleep Data Analysis Pipeline CLI.
"""

import numpy as np
import pandas as pd
import questionary

try:
    from .. import Feature_extraction as fe
    from .. import Preprocessing as pp
    from .. import Resampling as rs
except ImportError:
    import Feature_extraction as fe
    import Preprocessing as pp
    import Resampling as rs


class DataProcessor:
    """Handles data processing operations."""

    def __init__(self, console):
        self.console = console

    def preprocess_signal(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Preprocess the signal to remove artifacts.

        Returns:
            Tuple of (processed_dataframe, was_modified)
        """
        self.console.print("\n[bold cyan]Preprocessing signal...[/bold cyan]")

        if "sao2_percent" not in df.columns:
            self.console.print("[red]✗ No SaO2 signal found in data[/red]")
            return df, False

        original_signal = df["sao2_percent"].to_numpy()
        cleaned_signal = pp.preproccess_signal(original_signal)

        df["sao2_percent"] = cleaned_signal

        # Show comparison
        self.console.print("\n[bold]Preprocessing Results[/bold]")
        self.console.print(
            f"Original signal range: {original_signal.min():.2f} - {original_signal.max():.2f}"
        )
        self.console.print(
            f"Cleaned signal range: {cleaned_signal.min():.2f} - {cleaned_signal.max():.2f}"
        )

        # Calculate differences
        diff = np.abs(original_signal - cleaned_signal)
        self.console.print(f"Mean difference: {diff.mean():.4f}")
        self.console.print(f"Max difference: {diff.max():.4f}")
        self.console.print(
            f"Samples modified: {(diff > 0.01).sum()} ({(diff > 0.01).sum()/len(diff)*100:.2f}%)"
        )

        self.console.print("[green]✓[/green] Signal preprocessed")
        return df, True

    def analyze_for_resampling(self, df: pd.DataFrame):
        """Analyze signal to estimate original sampling rate before resampling."""
        self.console.print(
            "\n[bold cyan]Analyzing signal for resampling...[/bold cyan]"
        )

        if "sao2_percent" not in df.columns:
            self.console.print("[red]✗ No SaO2 signal found in data[/red]")
            return

        signal = df["sao2_percent"].to_numpy(dtype=float, na_value=np.nan)
        report = rs.find_pre_resampled_rate(signal, current_fs=200)
        rs.print_analysis_results("", report)

    def resample_signal(
        self, df: pd.DataFrame, target_resolution: float
    ) -> tuple[pd.DataFrame, bool]:
        """Resample the signal to a different time resolution.

        Args:
            df: Input dataframe
            target_resolution: Target time resolution in seconds

        Returns:
            Tuple of (resampled_dataframe, was_successful)
        """
        self.console.print("\n[bold cyan]Resampling signal...[/bold cyan]")

        if "time_s" not in df.columns or "sao2_percent" not in df.columns:
            self.console.print(
                "[red]✗ Required columns 'time_s' and 'sao2_percent' not found[/red]"
            )
            return df, False

        resampled_df = rs.resample_to_time_resolution(df, target_resolution)

        self.console.print(
            f"[green]✓[/green] Signal resampled to {target_resolution} seconds"
        )
        self.console.print(resampled_df.head().to_string())

        return resampled_df, True

    def extract_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Extract features from the signal.

        Returns:
            Tuple of (features_dataframe, was_successful)
        """
        self.console.print("\n[bold cyan]Extracting features...[/bold cyan]")

        if "time_s" not in df.columns or "sao2_percent" not in df.columns:
            self.console.print(
                "[red]✗ Required columns 'time_s' and 'sao2_percent' not found[/red]"
            )
            return pd.DataFrame(), False

        features_df = fe.extract_features(df)

        self.console.print(f"[green]✓[/green] Features extracted")
        self.console.print(features_df.head().to_string())

        return features_df, True

    def prompt_target_resolution(self) -> float:
        """Prompt user for target resampling resolution."""
        target_str = questionary.text(
            "Enter target time resolution in seconds (e.g., 0.5 for 500ms):",
            validate=lambda text: text.replace(".", "", 1).isdigit(),
        ).ask()

        return float(target_str)
