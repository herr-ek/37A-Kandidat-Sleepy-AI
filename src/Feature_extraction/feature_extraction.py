import warnings

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew


def _safe_skew(w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(_skew(w))


def _safe_kurtosis(w):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(_kurtosis(w))


def extract_features(
    data: pd.DataFrame,
    windowSize: int = 10,
    overlap: float = 0.5,
    training: bool = False,
) -> pd.DataFrame:
    """
    Extracts features from the SpO2 signal in the given DataFrame.
    Utilized a sliding window approach to compute features for each segment of the signal.

    Args:
        data: DataFrame containing 'time_s', 'sao2_percent', and 'is_apnea'/'is_hypopnea' columns.
        windowSize: The window size for feature extraction. Default is 10.
        overlap: The proportion of overlap between consecutive windows. Default is 0.5 (50% overlap).
        training: Boolean indicating whether to include training labels. Default is False.

    Returns:
        DataFrame containing the extracted features.
            - mean_sao2: Mean of SpO2 values in the window.
            - std_sao2: Standard deviation of SpO2 values in the window.
            - skew_sao2: Skewness of SpO2 values in the window.
            - min_sao2: Minimum SpO2 value in the window.
            - kurtosis_sao2: Kurtosis of SpO2 values in the window.
            - o2_sat_>96: Proportion of SpO2 values > 96% in the extended window.
            - o2_sat_90_96: Proportion of SpO2 values between 90% and 96% in the extended window.
            - o2_sat_80_90: Proportion of SpO2 values between 80% and 90% in the extended window.
            - o2_sat_<80: Proportion of SpO2 values < 80% in the extended window.
            - apnea_event (if training=True): Binary label indicating presence of apnea/hypopnea event in the window.
    """
    stepSize = int(windowSize * (1 - overlap))
    extended_window_size = windowSize * 3
    n = len(data)

    # Extract numpy arrays once — avoids repeated DataFrame slicing in the loop
    sao2 = data["sao2_percent"].values
    time = data["time_s"].values
    if training and "is_apnea" in data.columns and "is_hypopnea" in data.columns:
        is_apnea = data["is_apnea"].values
        is_hypopnea = data["is_hypopnea"].values

    rows = []
    for i in range(0, n - windowSize + 1, stepSize):
        w = sao2[i : i + windowSize]
        ext_end = min(i + extended_window_size, n)
        w_ext = sao2[i:ext_end]

        row = {
            "time_s": time[i + windowSize - 1],
            "mean_sao2": w.mean(),
            "std_sao2": w.std(),
            "skew_sao2": float(v) if np.isfinite(v := _safe_skew(w)) else 0.0,
            "min_sao2": w.min(),
            "kurtosis_sao2": float(v) if np.isfinite(v := _safe_kurtosis(w)) else 0.0,
            "o2_sat_>96": (w_ext > 96).sum() / extended_window_size,
            "o2_sat_90_96": ((w_ext > 90) & (w_ext <= 96)).sum() / extended_window_size,
            "o2_sat_80_90": ((w_ext < 90) & (w_ext >= 80)).sum() / extended_window_size,
            "o2_sat_<80": (w_ext < 80).sum() / extended_window_size,
        }
        if training and "is_apnea" in data.columns and "is_hypopnea" in data.columns:
            a = is_apnea[i : i + windowSize]
            h = is_hypopnea[i : i + windowSize]
            row["apnea_event"] = (
                1 if (a.sum() >= windowSize / 2 or h.sum() >= windowSize / 2) else 0
            )

        rows.append(row)

    return pd.DataFrame(rows)
