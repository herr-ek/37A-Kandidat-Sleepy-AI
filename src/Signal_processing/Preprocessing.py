import os

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from numpy.typing import ArrayLike

# SpO2 ARTIFACT REMOVAL (4 steps: remove impossible values, remove spikes, interpolate, median filter)


def preproccess_signal(signal: ArrayLike) -> np.ndarray:
    """
    Cleans the signal by removing artifacts and unrealistic values.
    Uses a combination of physiological plausibility checks, spike detection, interpolation, and median filtering.
    """
    clean = signal.copy()

    # Remove impossible physiological values (<70% or >100%)
    clean[(clean < 70) | (clean > 100)] = np.nan

    # Remove unrealistic spikes using derivative (sao2 should move slowly)
    # Unsure if this is unnecessary, but it removes very sharp jumps that are likely artifacts
    diff = np.abs(np.diff(clean))
    spike_threshold = 4

    spikes = np.where(diff > spike_threshold)[0]

    for idx in spikes:
        clean[idx] = np.nan
        if idx + 1 < len(clean):
            clean[idx + 1] = np.nan

    # Interpolate missing values
    nans = np.isnan(clean)
    if np.any(nans):
        clean[nans] = np.interp(
            np.flatnonzero(nans), np.flatnonzero(~nans), clean[~nans]
        )

    return clean


# EVENT EXTRACTION (RESPIRATORY EVENTS) (resp_obstructiveapnea, resp_centralapnea, resp_hypopnea)
def get_respiratory_events(ann):
    notes = np.array(ann.aux_note)
    mask = [(n.startswith("(resp_") or n.startswith("resp_")) for n in notes]
    samples = np.array(ann.sample)[mask]
    return samples, notes[mask]


def visualize_preprocessed_signal(
    signal: np.ndarray, ann: wfdb.Annotation, fs: int = 200
):
    """Visualizes the cleaned SpO2 signal with respiratory events marked."""

    clean_signal = preproccess_signal(signal)

    resp_samples, resp_notes = get_respiratory_events(ann)

    if len(resp_samples) == 0:
        raise RuntimeError("no respiratory events found")

    # Pick first x events
    num_events = 10
    if len(resp_samples) < num_events:
        num_events = len(resp_samples)

    selected_samples = resp_samples[:num_events]
    selected_notes = resp_notes[:num_events]

    print(f"Selected {num_events} events:")
    for s, n in zip(selected_samples, selected_notes):
        print(f"  sample {s} ({n}) -> {s/fs:.2f}s")

    # WINDOW AROUND EVENTS

    window_s = 30.0
    nbefore = int(window_s * fs)
    after = int(window_s * fs)

    start = max(0, selected_samples[0] - nbefore)
    end = min(signal.size, selected_samples[-1] + after)

    print(f"window samples: {start}..{end} ({(end-start)/fs:.1f}s total)")

    t = np.arange(start, end) / fs

    data_raw = signal[start:end]
    data_clean = clean_signal[start:end]

    event_mask = (resp_samples >= start) & (resp_samples < end)
    events_in_window = resp_samples[event_mask]

    # PLOT

    plt.figure(figsize=(12, 5))

    plt.plot(t, data_raw, label="Raw SpO₂ (Removed Artifacts)", alpha=0.4)
    plt.plot(t, data_clean, label="Cleaned SpO₂", linewidth=2)

    for ev in events_in_window:
        plt.axvline(ev / fs, color="red", linestyle="--", alpha=0.6)

    plt.xlabel("Time (s)")
    plt.ylabel("SpO₂ (%)")
    plt.title("Apnea-related events (Artifact Removal)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Optional span
    data_span = clean_signal[start:end]
    print("Span array shape:", data_span.shape)


if __name__ == "__main__":
    record = "tr03-0146"
    channel_index = 11

    # Construct absolute path to data folder
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    record_path = os.path.join(data_dir, record)

    # Load signal and annotations
    signal, fields = wfdb.rdsamp(record_path, channels=[channel_index])
    ann = wfdb.rdann(record_path, extension="arousal")

    visualize_preprocessed_signal(signal.flatten(), ann)
