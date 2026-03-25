import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _find_event_intervals(series: pd.Series) -> list[tuple[float, float]]:
    """
    Find start and end times of consecutive intervals where series == 1.

    Args:
        series: Series with binary values (0 or 1) and time_s as index

    Returns:
        List of (start_time, end_time) tuples for each interval
    """
    intervals = []
    in_event = False
    start_time = None

    for idx, value in series.items():
        if value == 1 and not in_event:
            # Start of new event
            start_time = idx
            in_event = True
        elif value == 0 and in_event:
            # End of event
            intervals.append((start_time, idx))
            in_event = False

    # Handle case where event extends to end of data
    if in_event:
        intervals.append((start_time, series.index[-1]))

    return intervals


def plot_with_annotations(df: pd.DataFrame, title: str = "SaO2 with Annotations"):
    """
    Plots the SaO2 signal with apnea and hypopnea annotations as vertical bars.

    Args:
        df: DataFrame containing 'time_s', 'sao2_percent', 'is_apnea', and 'is_hypopnea' columns.
        title: Title of the plot.
    """
    plt.figure(figsize=(15, 5))
    plt.plot(
        df["time_s"],
        df["sao2_percent"],
        label="SaO2 (%)",
        color="green",
        zorder=3,
        linewidth=1,
    )

    # Set time_s as index for easier interval finding
    df_indexed = df.set_index("time_s")

    # Draw apnea event intervals as vertical bars
    apnea_intervals = _find_event_intervals(df_indexed["is_apnea"])
    for start, end in apnea_intervals:
        plt.axvspan(
            start,
            end,
            color="red",
            alpha=0.3,
            label="Apnea" if start == apnea_intervals[0][0] else "",
        )

    # Draw hypopnea event intervals as vertical bars
    hypopnea_intervals = _find_event_intervals(df_indexed["is_hypopnea"])
    for start, end in hypopnea_intervals:
        plt.axvspan(
            start,
            end,
            color="blue",
            alpha=0.3,
            label="Hypopnea" if start == hypopnea_intervals[0][0] else "",
        )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("SaO2 (%)")
    plt.legend()
    plt.grid()
    plt.show()
