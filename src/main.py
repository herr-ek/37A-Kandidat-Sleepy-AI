import Preprocessing as pp
import Data_management as dm
import Resampling as rs

import os
import wfdb
import pandas as pd
import numpy as np

if __name__ == "__main__":
    record = "tr03-0146"
    channel_index = 11  # SaO2 channel

    record_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data/raw", record
    )
    print(record_path)

    df = dm.load_from_mat_and_arousal_to_pandas(record_path)
    report = rs.find_pre_resampled_rate(df["sao2_percent"].values, current_fs=200.0)
    rs.print_analysis_results(record, report)
