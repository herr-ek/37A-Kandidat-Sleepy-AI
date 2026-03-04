import scipy.io as sio
import numpy as np
import wfdb
import pandas as pd
import os 

GAIN = 655.35
BASELINE = -32768
FREQ = 200

def load_from_mat_and_arousal_to_pandas(file_record) -> pd.DataFrame:
  """
  Returns a pandas.DataFrame containing relevant data, extracted from .mat and .arousal
  """
  if '.' in file_record:
    file_record = os.path.splitext(file_record)[0]


  mat = sio.loadmat(f"{file_record}.mat")
  signal = mat["val"]
  sao2_raw = signal[11, :].astype(np.float64)

  sao2 = (sao2_raw - BASELINE) / GAIN


  nans = np.isnan(sao2)

  if np.any(~nans): # This cleans up any NaN and "draws" a line between the last and the next known points.
    sao2[nans] = np.interp(
        np.flatnonzero(nans),
        np.flatnonzero(~nans),
        sao2[~nans]
    )


  ann = wfdb.rdann(file_record, "arousal")
  samples = np.array(ann.sample)
  labels  = np.array(ann.aux_note, dtype=object)
  is_event = np.array([("apnea" in s) or ("hypopnea" in s) for s in labels])

  event_samples = samples[is_event]
  event_labels  = labels[is_event]

  is_apnea_array = np.zeros(sao2.shape[0], dtype=int)
  is_hypopnea_array = np.zeros(sao2.shape[0], dtype=int)

  for i in range(0, event_labels.shape[0], 2): # Every 2 rows are start and end of an event
    start, end = event_samples[i], event_samples[i+1]
    if "hypopnea" in event_labels[i]:
      is_hypopnea_array[start:end] = 1
    else:
      is_apnea_array[start:end] = 1
  

  df = pd.DataFrame({
      "time_s": np.arange(len(sao2)) / FREQ, # Just an array with the time, given the sampling rate
      "sao2_percent": sao2,
      "is_apnea": is_apnea_array,
      "is_hypopnea": is_hypopnea_array
  })


def extract_and_save_to_parquet(file_record):
  """
  Extracts relevant data from .mat and .arousal and converts it into .parquet 
  """
  df = load_from_mat_and_arousal_to_pandas(file_record)
  
  df.to_parquet(f"{file_record}.parquet", index=False)
  
  
def load_from_parquet_to_pandas(file_record) -> pd.DataFrame:
  """
  Loads pandas.DataFrame from existing .parquet 
  """

  if '.' in file_record and not file_record.endswith('.parquet'):
    file_record = file_record.split(".")[0]

  filename = file_record if file_record.endswith('.parquet') else f"{file_record}.parquet"

  df = pd.read_parquet(filename)

  return df
  

if __name__ == "__main__":


  # extract_and_save_to_parquet("tr03-0029")
  df = load_from_parquet_to_pandas("tr03-0029")


  diff_sao2 = df["sao2_percent"].diff()

  s = set()

  unique_levels = np.unique(df["sao2_percent"])
  print(len(unique_levels))
  print(unique_levels)

  changed_rows = df[diff_sao2 != 0]

  # Print them
  print(changed_rows)

  # Parquet istället för csv?

  # kolumner: tid, nivå (i procent?), is_apnea (true/false eller 1/0)

  # Ladda in .mat och .arousal (med wfdb) -> convert sao2-raw till sao2-%, gör en ny array med 1 där innanför ett apnea intervall, 0 annars -> storea i egen dataformat 

