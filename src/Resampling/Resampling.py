import os

import numpy as np
import pandas as pd
import scipy.io
from numpy.typing import ArrayLike


def find_pre_resampled_rate(signal: ArrayLike, current_fs: int = 200) -> dict:
    """
    Estimates the original sampling rate of a resampled signal by finding
    the smallest run of consecutive identical values.

    Args:
        signal: 1D array of signal values
        current_fs: Current sampling rate in Hz (default 200)

    Returns:
        dict with estimated original sampling rate and run statistics
    """
    signal = np.asarray(signal).flatten()

    # Find where signal values change (consecutive samples differ)
    changes = np.where(np.diff(signal) != 0)[0]

    if len(changes) < 2:
        return {
            "estimated_original_fs": None,
            "error": "Not enough value changes detected",
        }

    # Calculate run lengths (number of consecutive identical values)
    # Add boundaries at start and end for complete run calculation
    change_positions = np.concatenate([[0], changes + 1, [len(signal)]])
    run_lengths = np.diff(change_positions)

    # Find the smallest run length (excluding outliers)
    # Remove runs of length 1 (single-sample glitches) and very small runs
    filtered_runs = run_lengths[run_lengths > 1]

    if len(filtered_runs) == 0:
        return {
            "estimated_original_fs": None,
            "error": "No valid runs found",
        }

    # Calculate statistics
    min_run = np.min(filtered_runs)
    mode_run = np.bincount(filtered_runs.astype(int)).argmax()
    mean_run = np.mean(filtered_runs)
    median_run = np.median(filtered_runs)

    # Estimate original sampling rate based on smallest run
    # If values repeat for N samples at 200Hz, original rate was 200/N Hz
    estimated_fs_min = current_fs / min_run
    estimated_fs_mode = current_fs / mode_run
    estimated_fs_median = current_fs / median_run

    return {
        "estimated_original_fs_from_min": estimated_fs_min,
        "estimated_original_fs_from_mode": estimated_fs_mode,
        "estimated_original_fs_from_median": estimated_fs_median,
        "smallest_run_length": int(min_run),
        "mode_run_length": int(mode_run),
        "median_run_length": median_run,
        "mean_run_length": mean_run,
        "total_runs": len(run_lengths),
        "signal_length": len(signal),
        "signal_duration_seconds": len(signal) / current_fs,
    }


def find_intervals_shorter_than(
    signal: ArrayLike, threshold_samples: int
) -> list[tuple[int, int]]:
    """
    Finds intervals in the signal where values are constant for fewer than threshold_samples.

    Args:
        signal: 1D array of signal values
        threshold_samples: Minimum number of consecutive identical samples to be considered a valid run

    Returns:
        List of tuples representing the start and end indices of intervals shorter than the threshold.
    """
    signal = np.asarray(signal).flatten()
    intervals = []
    start = 0
    for i in range(1, len(signal)):
        if signal[i] != signal[i - 1]:
            if i - start < threshold_samples:
                intervals.append((start, i))
            start = i
    return intervals


def print_analysis_results(record, result):
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Analysis Results for {record}, Channel 11 (SpO2):")
        print(f"  Signal duration: {result['signal_duration_seconds']:.1f} seconds")
        print(f"  Total runs detected: {result['total_runs']}")
        print(f"  Smallest run length: {result['smallest_run_length']} samples")
        print(f"  Mode run length: {result['mode_run_length']} samples")
        print(f"  Median run length: {result['median_run_length']:.2f} samples")
        print(f"  Mean run length: {result['mean_run_length']:.2f} samples")
        print(
            f"\n  Estimated original sampling rate (from smallest): {result['estimated_original_fs_from_min']:.2f} Hz"
        )
        print(
            f"  Estimated original sampling rate (from mode): {result['estimated_original_fs_from_mode']:.2f} Hz"
        )
        print(
            f"  Estimated original sampling rate (from median): {result['estimated_original_fs_from_median']:.2f} Hz"
        )


def trim_signal(
    signal: ArrayLike, trim_low: float, trim_high: float, fs: int = 200
) -> np.ndarray:
    """
    Trims the start and end of a signal by a specified number of seconds.

    Args:
        signal: 1D array of signal values
        trim_low: Number of seconds to trim from the start
        trim_high: Number of seconds to trim from the end
        fs: Sampling rate in Hz (default 200)

    Returns:
        Trimmed signal array.
    """
    trim_samples_low = int(trim_low * fs)
    trim_samples_high = int(trim_high * fs)
    return signal[trim_samples_low:-trim_samples_high]


def resample_to_time_resolution(df: pd.DataFrame, target_resolution: float):
    """
    Resamples a DataFrame to a specified time resolution.

    Args:
        df: Input DataFrame with a seconds column.
        target_resolution: Desired time resolution in seconds (e.g., 0.5 for 500ms = 2Hz).
            Conversion: target_resolution = 1 / target frequency (Hz)
    Returns:
        Resampled DataFrame with the specified time resolution.
    """
    # Ensure the seconds column is sorted and has a consistent time step
    df = df.sort_values(by="time_s").reset_index(drop=True)

    # Create a new time index based on the target resolution
    start_time = df["time_s"].min()
    end_time = df["time_s"].max()
    new_time_index = np.arange(start_time, end_time, target_resolution)

    # Resample the DataFrame using interpolation
    resampled_df = pd.DataFrame({"time_s": new_time_index})

    for column in df.columns:
        if column != "time_s":
            resampled_df[column] = np.interp(new_time_index, df["time_s"], df[column])

    return resampled_df


def resample_average(
    df: pd.DataFrame,
    target_resolution: float,
    numeric_cols: list[str] | None = None,
    binary_cols: list[str] | None = None,
    reindex_time: bool = True,
) -> pd.DataFrame:
    """
    Downsample a DataFrame by grouping rows into non-overlapping time bins and
    averaging within each bin.

    Unlike ``resample_to_time_resolution`` (which uses interpolation), this
    function takes the **mean** of all samples that fall inside each bin, so
    no information is fabricated between samples.  Binary/label columns are
    majority-voted: a bin is labelled 1 if more than half its rows are 1.

    Args:
        df: Input DataFrame with a ``time_s`` column (seconds, 1-based float).
        target_resolution: Desired bin width in seconds (e.g., 1.0 for 1 Hz).
        numeric_cols: Columns to average.  Defaults to all non-time numeric
            columns not in *binary_cols*.
        binary_cols: Columns to majority-vote.  Defaults to columns whose name
            starts with ``is_`` or ``apnea``.
        reindex_time: If ``True`` (default), replace the aggregated ``time_s``
            values with a clean, evenly-spaced sequence
            ``t0, t0 + res, t0 + 2*res, …``.  Set to ``False`` to keep the
            mean timestamp of each bin (useful when the input timestamps are
            already irregular and you want to preserve the original offsets).

    Returns:
        Resampled DataFrame with one row per bin and a ``time_s`` column set to
        the centre of each bin.
    """
    df = df.sort_values("time_s").reset_index(drop=True)

    t0 = df["time_s"].iloc[0]
    bin_indices = ((df["time_s"] - t0) / target_resolution).astype(int)
    df = df.copy()
    df["_bin"] = bin_indices

    all_cols = [c for c in df.columns if c not in ("time_s", "_bin")]
    if binary_cols is None:
        binary_cols = [
            c for c in all_cols if c.startswith("is_") or c.startswith("apnea")
        ]
    if numeric_cols is None:
        numeric_cols = [c for c in all_cols if c not in binary_cols]

    agg: dict[str, object] = {"time_s": "mean"}
    for c in numeric_cols:
        agg[c] = "mean"
    for c in binary_cols:
        agg[c] = lambda s: int(s.mean() >= 0.5)

    result = df.groupby("_bin", sort=True).agg(agg).reset_index(drop=True)

    if reindex_time:
        result["time_s"] = 0 + np.arange(len(result)) * target_resolution

    return result


if __name__ == "__main__":
    # Load data
    record = "tr03-0146"
    channel_index = 11

    # Construct path to data folder
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )
    mat_file = os.path.join(data_dir, f"{record}.mat")
    mat = scipy.io.loadmat(mat_file)
    vals = mat["val"]
    signal = vals[channel_index, :]

    # Analyze the signal
    result = find_pre_resampled_rate(signal, current_fs=200)

    print_analysis_results(record, result)
