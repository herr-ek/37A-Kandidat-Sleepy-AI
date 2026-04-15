import numpy as np
import pandas as pd


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    """
    Normalize a feature to the range [0, 1].

    Parameters:
    feature (numpy.ndarray): The input feature to be normalized.

    Returns:
    numpy.ndarray: The normalized feature.
    """
    min_val = feature.min()
    max_val = feature.max()

    if max_val - min_val == 0:
        return feature  # Avoid division by zero, return original feature if all values are the same

    normalized_feature = (feature - min_val) / (max_val - min_val)

    return normalized_feature


def normalize_with_context(feature: np.ndarray, down: float, up: float) -> np.ndarray:
    """Normalize a feature to the range [down, up] using provided upper and lower bounds.

    Parameters:
    feature (numpy.ndarray): The input feature to be normalized.
    up (float): The upper bound of the normalization range.
    down (float): The lower bound of the normalization range.

    Returns:
    numpy.ndarray: The normalized feature.
    """

    normalized_feature = (feature - down) / (up - down)

    return normalized_feature


def normalize_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all numeric features in a DataFrame to the range [0, 1].

    Parameters:
    df (pandas.DataFrame): The input DataFrame with features to be normalized.

    Returns:
    pandas.DataFrame: A new DataFrame with normalized features.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    normalized_df = df.copy()

    for col in numeric_cols:
        if (
            col != "time_s"
        ):  # Assuming 'time_s' is a time feature that should not be normalized
            if col == "mean_sao2" or col == "min_sao2":
                # Normalize mean_sao2 and min_sao2 to [0, 1] based on expected SaO2 range (e.g., 70-100%)
                normalized_df[col] = normalize_with_context(
                    df[col].to_numpy(), down=70, up=100
                )
            else:
                normalized_df[col] = normalize_feature(df[col].to_numpy())

    return normalized_df
