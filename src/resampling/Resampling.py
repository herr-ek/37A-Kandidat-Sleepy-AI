import scipy.io
import numpy as np
import os


def find_pre_resampled_rate(record, channel_index=11, current_fs=200):
    """
    Estimates the original sampling rate of a resampled signal by finding
    the smallest run of consecutive identical values.

    Args:
        record: Record identifier (e.g., "tr03-0146")
        channel_index: Channel to analyze (default 11 for SaO2)
        current_fs: Current sampling rate in Hz (default 200)

    Returns:
        dict with estimated original sampling rate and run statistics
    """
    # Construct path to data folder
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data"
    )
    mat_file = os.path.join(data_dir, f"{record}.mat")
    mat = scipy.io.loadmat(mat_file)
    vals = mat["val"]
    signal = vals[channel_index, :]

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


if __name__ == "__main__":
    # Test with a record
    record = "tr03-0146"
    result = find_pre_resampled_rate(record, channel_index=11)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Analysis Results for {record}, Channel 11 (SaO2):")
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
