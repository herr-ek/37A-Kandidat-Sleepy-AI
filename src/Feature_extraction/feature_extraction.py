import pandas as pd
from antropy import lziv_complexity


def extract_features(
    data: pd.DataFrame,
    windowSize: int = 10,
    overlap: float = 0.5,
    training: bool = False,
) -> pd.DataFrame:
    """
    Extracts features from the SaO2 signal in the given DataFrame.
    Utilized a sliding window approach to compute features for each segment of the signal.

    Args:
        data: DataFrame containing 'time_s', 'sao2_percent', and 'is_apnea'/'is_hypopnea' columns.
        windowSize: The window size for feature extraction. Default is 10.
        overlap: The proportion of overlap between consecutive windows. Default is 0.5 (50% overlap).
        training: Boolean indicating whether to include training labels. Default is False.

    Returns:
        DataFrame containing the extracted features.
            - mean_sao2: Mean of SaO2 values in the window.
            - std_sao2: Standard deviation of SaO2 values in the window.
            - skew_sao2: Skewness of SaO2 values in the window.
            - min_sao2: Minimum SaO2 value in the window.
            - lempel_ziv: Lempel-Ziv complexity of the SaO2 values in the window.
            - o2_sat_>96: Proportion of SaO2 values > 96% in the extended window.
            - o2_sat_90_96: Proportion of SaO2 values between 90% and 96% in the extended window.
            - o2_sat_80_90: Proportion of SaO2 values between 80% and 90% in the extended window.
            - o2_sat_<80: Proportion of SaO2 values < 80% in the extended window.
            - apnea_event (if training=True): Binary label indicating presence of apnea/hypopnea event in the window.
    """
    # Example feature extraction (replace with actual implementation)
    features = pd.DataFrame()

    stepSize = int(windowSize * (1 - overlap))

    extended_window_size = (
        windowSize * 3
    )  # Extend window to capture future values for certain features
    for i in range(0, len(data) - windowSize + 1, stepSize):
        window_data = data.iloc[i : i + windowSize]
        if i + extended_window_size <= len(data):
            window_data_extended = data.iloc[
                i : i + extended_window_size
            ]  # Extend window for future features
        else:
            window_data_extended = data.iloc[i:]

        if len(window_data) < windowSize:
            break  # Skip incomplete windows

        # Extract features
        features.loc[i, "time_s"] = window_data["time_s"].iloc[-1]
        features.loc[i, "mean_sao2"] = window_data["sao2_percent"].mean()
        features.loc[i, "std_sao2"] = window_data["sao2_percent"].std()
        features.loc[i, "skew_sao2"] = window_data["sao2_percent"].skew()
        features.loc[i, "min_sao2"] = window_data["sao2_percent"].min()
        features.loc[i, "lempel_ziv"] = lziv_complexity(
            window_data["sao2_percent"].values
        )
        # Calculate proportions of SaO2 values in different ranges for the extended window
        features.loc[i, "o2_sat_>96"] = (
            (window_data_extended["sao2_percent"]) > 96
        ).sum() / extended_window_size
        features.loc[i, "o2_sat_90_96"] = (
            (window_data_extended["sao2_percent"] > 90)
            & (window_data_extended["sao2_percent"] <= 96)
        ).sum() / extended_window_size
        features.loc[i, "o2_sat_80_90"] = (
            (window_data_extended["sao2_percent"] < 90)
            & (window_data_extended["sao2_percent"] >= 80)
        ).sum() / extended_window_size
        features.loc[i, "o2_sat_<80"] = (
            (window_data_extended["sao2_percent"]) < 80
        ).sum() / extended_window_size
        if training:
            apnea_event = ((window_data["is_apnea"].sum()) >= windowSize / 2) or (
                (window_data["is_hypopnea"].sum()) >= windowSize / 2
            )
            features.loc[i, "apnea_event"] = 1 if apnea_event else 0

    return features
