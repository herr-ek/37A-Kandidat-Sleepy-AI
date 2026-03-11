import scipy.io
import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# Pick a record to analyze
record = "tr03-0005"
fs = 200
mat_file = f"{record}.mat"

# Load annotation and data
ann = wfdb.rdann(record, 'arousal')
mat = scipy.io.loadmat(mat_file)
vals = mat['val']

# Choose channel (SaO2)
channel_index = 11
signal = vals[channel_index, :]

# Apply scaling
hdr = wfdb.rdheader(record)
gain = hdr.adc_gain[channel_index]
baseline = hdr.baseline[channel_index]

print(f"applying gain={gain}, baseline={baseline} to channel {channel_index}")

signal = signal.astype(np.float64)
signal = (signal - baseline) / gain


# SaO2 ARTIFACT REMOVAL (4 steps: remove impossible values, remove spikes, interpolate, median filter)

def clean_sao2(sig, fs):
    clean = sig.copy()

    # Remove impossible physiological values (<70% or >100%)
    clean[(clean < 70) | (clean > 100)] = np.nan

    # Remove unrealistic spikes using derivative (sao2 should move slowly)
    # Unsure if this is unnecessary, but it removes very sharp jumps that are likely artifacts
    diff = np.abs(np.diff(clean))
    spike_threshold = 8

    spikes = np.where(diff > spike_threshold)[0]

    for idx in spikes:
        clean[idx] = np.nan
        if idx + 1 < len(clean):
            clean[idx + 1] = np.nan

    # Interpolate missing values
    nans = np.isnan(clean)
    if np.any(nans):
        clean[nans] = np.interp(
            np.flatnonzero(nans),
            np.flatnonzero(~nans),
            clean[~nans]
        )

    # Median filter to remove small spikes
    clean = medfilt(clean, kernel_size=9)

    return clean


# Apply preprocessing
clean_signal = clean_sao2(signal, fs)


# EVENT EXTRACTION (RESPIRATORY EVENTS) (resp_obstructiveapnea, resp_centralapnea, resp_hypopnea)

def get_respiratory_events():
    notes = np.array(ann.aux_note)
    mask = [(n.startswith('(resp_') or n.startswith('resp_')) for n in notes]
    samples = np.array(ann.sample)[mask]
    return samples, notes[mask]


resp_samples, resp_notes = get_respiratory_events()

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

plt.figure(figsize=(12,5))

plt.plot(t, data_raw, label="Raw SaO₂ (Removed Artifacts)", alpha=0.4)
plt.plot(t, data_clean, label="Cleaned SaO₂", linewidth=2)

for ev in events_in_window:
    plt.axvline(ev/fs, color='red', linestyle='--', alpha=0.6)

plt.xlabel('Time (s)')
plt.ylabel('SaO₂ (%)')
plt.title('Apnea-related events (Artifact Removal)')
plt.legend()
plt.tight_layout()
plt.show()


# Optional span
data_span = clean_signal[start:end]
print("Span array shape:", data_span.shape)