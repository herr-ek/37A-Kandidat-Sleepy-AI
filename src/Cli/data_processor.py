"""
Data processing module for the Sleep Data Analysis Pipeline CLI.
"""

import numpy as np
import pandas as pd
import questionary

try:
    from .. import Feature_extraction as fe
    from .. import Resampling as rs
    from .. import Signal_processing as sp
except ImportError:
    import Feature_extraction as fe
    import Resampling as rs
    import Signal_processing as sp


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
            self.console.print("[red]✗ No SpO2 signal found in data[/red]")
            return df, False

        original_signal = df["sao2_percent"].to_numpy()
        cleaned_signal = sp.preproccess_signal(original_signal)

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

    def normalize_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Normalize extracted features.

        Returns:
            Tuple of (normalized_dataframe, was_modified)
        """
        self.console.print("\n[bold cyan]Normalizing features...[/bold cyan]")

        if df.empty:
            self.console.print("[red]✗ No features to normalize[/red]")
            return df, False

        normalized_df = sp.normalize_feature_df(df)

        self.console.print("[green]✓[/green] Features normalized")
        self.console.print(normalized_df.head().to_string())

        return normalized_df, True

    def _detect_source_resolution(self, df: pd.DataFrame) -> tuple[float, float]:
        """Estimate (resolution_s, freq_hz) from the median time step in time_s."""
        dts = np.diff(df.sort_values("time_s")["time_s"].values)
        res_s = float(np.median(dts)) if len(dts) > 0 else 1.0
        freq_hz = 1.0 / res_s if res_s > 0 else 1.0
        return res_s, freq_hz

    def analyze_for_resampling(self, df: pd.DataFrame):
        """Analyze signal to estimate original sampling rate before resampling."""
        self.console.print(
            "\n[bold cyan]Analyzing signal for resampling...[/bold cyan]"
        )

        if "sao2_percent" not in df.columns:
            self.console.print("[red]✗ No SpO2 signal found in data[/red]")
            return

        res_s, freq_hz = self._detect_source_resolution(df)
        self.console.print(
            f"Auto-detected source resolution: [cyan]{res_s:.4f} s/sample "
            f"({freq_hz:.4g} Hz)[/cyan]"
        )

        fs_str = questionary.text(
            "Confirm current sampling rate (Hz):",
            default=f"{freq_hz:.4g}",
            validate=lambda t: t.replace(".", "", 1).isdigit() and float(t) > 0,
        ).ask()
        if fs_str is None:
            return
        current_fs = float(fs_str)

        signal = df["sao2_percent"].to_numpy(dtype=float, na_value=np.nan)
        report = rs.find_pre_resampled_rate(signal, current_fs=int(round(current_fs)))
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

        src_res_s, src_freq_hz = self._detect_source_resolution(df)

        src_str = questionary.text(
            "Source resolution (s/sample):",
            default=f"{src_res_s:.6g}",
            validate=lambda t: t.replace(".", "", 1).isdigit() and float(t) > 0,
        ).ask()
        if src_str is None:
            return df, False
        src_res_s = float(src_str)
        src_freq_hz = 1.0 / src_res_s

        self.console.print(
            f"Source : [cyan]{src_res_s:.4f} s/sample ({src_freq_hz:.4g} Hz)[/cyan]  →  "
            f"Target : [cyan]{target_resolution:.4f} s/sample "
            f"({1.0/target_resolution:.4g} Hz)[/cyan]"
        )

        method = questionary.select(
            "Resampling method:",
            choices=[
                questionary.Choice(
                    "Average  — mean of all samples in each bin (no data fabricated)",
                    value="average",
                ),
                questionary.Choice(
                    "Interpolate  — linear interpolation onto a new time grid",
                    value="interpolate",
                ),
            ],
        ).ask()
        if method is None:
            return df, False

        if method == "average":
            reindex = questionary.confirm(
                "Reindex time to a clean evenly-spaced grid? "
                "(No = keep mean timestamp of each bin)",
                default=True,
            ).ask()
            if reindex is None:
                return df, False
            resampled_df = rs.resample_average(
                df, target_resolution, reindex_time=reindex
            )
        else:
            resampled_df = rs.resample_to_time_resolution(df, target_resolution)

        self.console.print(
            f"[green]✓[/green] Signal resampled to {target_resolution} s/sample "
            f"using '{method}'"
        )
        self.console.print(resampled_df.head().to_string())

        return resampled_df, True

    def trim_signal(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
        """Trim seconds from the start and/or end of the signal.

        Returns:
            Tuple of (trimmed_dataframe, was_modified)
        """
        self.console.print("\n[bold cyan]Trimming signal...[/bold cyan]")

        if "time_s" not in df.columns:
            self.console.print("[red]✗ No 'time_s' column found in data[/red]")
            return df, False

        trim_low_str = questionary.text(
            "Seconds to trim from the start (e.g. 30):",
            default="0",
            validate=lambda t: t.replace(".", "", 1).isdigit(),
        ).ask()
        if trim_low_str is None:
            return df, False

        trim_high_str = questionary.text(
            "Seconds to trim from the end (e.g. 30):",
            default="0",
            validate=lambda t: t.replace(".", "", 1).isdigit(),
        ).ask()
        if trim_high_str is None:
            return df, False

        trim_low = float(trim_low_str)
        trim_high = float(trim_high_str)

        df = df.sort_values("time_s").reset_index(drop=True)

        # Estimate sampling rate from median time step
        dts = np.diff(df["time_s"].values)
        fs_est = int(round(1.0 / np.median(dts))) if len(dts) > 0 else 1

        original_len = len(df)
        result: dict[str, np.ndarray] = {}
        for col in df.columns:
            result[col] = rs.trim_signal(df[col].values, trim_low, trim_high, fs=fs_est)

        trimmed_df = pd.DataFrame(result)
        trimmed_len = len(trimmed_df)

        self.console.print(
            f"[green]✓[/green] Trimmed {original_len - trimmed_len} rows "
            f"(start: {trim_low}s, end: {trim_high}s, fs≈{fs_est} Hz)"
        )
        self.console.print(
            f"New duration: {trimmed_df['time_s'].iloc[-1] - trimmed_df['time_s'].iloc[0]:.1f}s "
            f"({trimmed_len} rows)"
        )

        return trimmed_df, True

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

        features_df = fe.extract_features(df, training=True)

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
