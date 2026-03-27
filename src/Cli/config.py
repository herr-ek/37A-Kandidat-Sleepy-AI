"""
Configuration and constants for the Sleep Data Analysis Pipeline CLI.
"""

from pathlib import Path

# Directory Configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Constants
DEFAULT_BATCH_PREPROCESS_RESOLUTION = 0.5  # seconds
DEFAULT_RECORD_PREFIX = "tr03"
DOWNLOAD_SPEED_CAP = "1.5 MB/s"

# Export settings
PARQUET_COMPRESSION = None  # Use default compression
METADATA_SUFFIX = ".metadata.txt"
