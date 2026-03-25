# Sleep Data Analysis Pipeline - CLI Guide

## Overview

This CLI provides an interactive interface for processing and analyzing sleep apnea recordings. It features:

- 🎯 **Navigable menus** with arrow key selection
- 🔍 **Autocomplete** for record selection
- 🎨 **Color-coded output** for better readability
- 🔧 **Extensible architecture** for easy feature additions
- 📦 **Batch processing** for multiple records
- ⬇️  **PhysioNet downloader** to fetch open-source sleep data
- ✨ **Feature extraction** for machine learning pipelines
- ⏲️  **Resampling utilities** for signal time-resolution adjustment

## Installation

First, ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

### Running the CLI

**As a standalone script** (from project root):

```bash
python src/Cli/cli.py
```

Or make it executable and run directly:

```bash
chmod +x src/Cli/cli.py
./src/Cli/cli.py
```

**As an imported module** (e.g., from `main.py` or another script):

```python
from rich.console import Console
from src.Cli.cli import SleepDataPipeline

console = Console()
pipeline = SleepDataPipeline(console)
pipeline.run()
```

This allows you to integrate the CLI into your own application or notebook.

### Workflow

#### Step 1: Choose Data Source

Select between:

- **Raw data**: Load a single recording from `.mat` and `.arousal` files (4 files per recording)
- **Multiple raw records (batch process)**: Process multiple raw recordings in sequence
- **Processed data**: Load from pre-processed `.parquet` files (faster, single file per record)
- **Multiple processed records (batch process)**: Batch process multiple parquet files
- **Download from PhysioNet**: Download open-source sleep apnea data (~1.5 MB/s speed cap)

Use arrow keys ↑/↓ and press Enter to select.

#### Step 2: Select Recording(s)

**Single Mode:**
The CLI will scan the chosen directory and display all available recordings in a table.
Type the recording name (autocomplete will help you) or paste it, then press Enter.
For processed data, you can also select a specific file within a record.
Example recordings: `tr03-0146`, `tr03-0029`, etc.

**Batch Mode:**
Select multiple recordings using the checkbox interface. Press Space to select/deselect, then Enter to confirm.
Optionally load from a custom directory instead of the default data folder.

**Custom Directory:**
For batch raw mode, you can specify a custom data directory path to load recordings from an alternative location.

#### Step 3: Choose Action

**Single Mode Actions:**

- **📋 Display data as DataFrame**: View the data in tabular format with column info
- **📊 Show data statistics**: Get statistical summary and sleep event counts
- **📈 Plot signal with annotations**: Visualize the SaO2 signal with respiratory events
- **🧹 Preprocess signal**: Clean artifacts from the signal (removes outliers, spikes, smooths)
- **🔍 Analyze signal for resampling**: Analyze the signal to determine optimal resampling parameters
- **⏲️ Resample signal**: Resample to a different time resolution (e.g., 0.5s instead of 0.005s)
- **✨ Extract features and save to parquet**: Extract ML-ready features from the signal
- **💾 Save current dataframe to parquet**: Save modified data with applied operations
- **📤 Export to parquet**: Convert raw data to parquet format (raw data only)
- **🔄 Select different record**: Choose another recording without changing data source
- **🔙 Back to data source selection**: Start over with a different data source
- **❌ Exit**: Quit the program

**Batch Mode Actions:**

- **🔄 Process all selected records with default pipeline**: Apply default preprocessing to all selected records
- **⏲️ Resample all records to target resolution**: Batch resample multiple recordings
- **🔙 Back to data source selection**: Start over
- **❌ Exit**: Quit the program

## Directory Structure

```
src/
├── Cli/                          # Modular CLI package
│   ├── __init__.py               # Package initialization
│   ├── cli.py                    # Main entry point (orchestrator)
│   ├── config.py                 # Configuration and constants
│   ├── utils.py                  # Utility functions
│   ├── ui.py                     # User interface (menus/prompts)
│   ├── data_loader.py            # Data loading and file I/O
│   ├── data_display.py           # Display and visualization
│   ├── data_processor.py         # Signal processing operations
│   ├── data_saver.py             # Saving and export functionality
│   ├── downloader.py             # PhysioNet data downloading
│   └── batch_processor.py        # Batch processing operations
├── Data_management/              # Data loading utilities
├── Feature_extraction/           # ML feature extraction
├── Plotting/                     # Visualization functions
├── Preprocessing/                # Signal preprocessing
├── Resampling/                   # Signal resampling
└── ...

data/
├── raw/                          # Raw recordings
│   └── tr03-0146/                # One folder per recording
│       ├── tr03-0146.mat
│       ├── tr03-0146.arousal
│       ├── tr03-0146.hea
│       └── tr03-0146-arousal.mat
└── processed/                    # Processed data
    └── tr03-0146/                # One folder per recording
        └── tr03-0146.parquet
```

## Architecture

The CLI has been refactored into a modular, component-based architecture for better maintainability, testability, and reusability:

### Core Components

- **`cli.py`**: Main orchestrator class `SleepDataPipeline` that coordinates all components and manages the workflow
- **`config.py`**: Centralized configuration (paths, constants, defaults) for easy customization
- **`utils.py`**: Reusable utility functions (file formatting, styling, directory prompts)
- **`ui.py`**: `CLI_UI` class handling all user interface menus and prompts with rich styling
- **`data_loader.py`**: `DataLoader` class managing data loading from raw/processed sources with metadata tracking
- **`data_display.py`**: `DataDisplay` class for visualization and statistical analysis
- **`data_processor.py`**: `DataProcessor` class for signal processing operations (preprocess, resample, extract features)
- **`data_saver.py`**: `DataSaver` class for saving and exporting data with metadata generation
- **`downloader.py`**: `PhysioNetDownloader` class for fetching data from PhysioNet Challenge 2018
- **`batch_processor.py`**: `BatchProcessor` class for batch operations on multiple records

### Design Patterns

- **Dependency Injection**: Components receive `console` object for consistent output
- **Single Responsibility**: Each module handles one specific aspect of the pipeline
- **Composable**: Modules can be used independently or combined in custom workflows
- **Dual-Mode Imports**: Supports both relative imports (when run as module) and absolute imports (when run as script)
- **State Management**: `DataLoader` maintains current dataframe and operation history

## Features

### Current Actions

1. **Display DataFrame**: Shows first 20 rows and column information with data types
2. **Statistics**: Provides statistical summary including apnea/hypopnea event counts and percentages
3. **Plot Signal**: Visualizes the SaO2 signal with respiratory events marked
4. **Preprocess**: Applies artifact removal and noise reduction (removes outliers, spikes, and smooths)
5. **Resample Analysis**: Analyzes signal to estimate optimal resampling parameters
6. **Resample Signal**: Resamples data to a custom time resolution while preserving event annotations
7. **Feature Extraction**: Extracts ML-ready features from the signal for model training
8. **Save DataFrame**: Saves current dataframe with applied operations to parquet format
9. **Export to Parquet**: Converts raw .mat/.arousal files to efficient parquet format
10. **Batch Processing**: Apply actions to multiple records automatically
11. **Download Data**: Fetch sleep apnea recordings directly from PhysioNet Challenge 2018

### Data Columns

When loading data, you'll see these columns:

- `time_s`: Time in seconds
- `sao2_percent`: SpO2 (oxygen saturation) percentage
- `is_apnea`: Binary flag for apnea events
- `is_hypopnea`: Binary flag for hypopnea events
- `sao2_cleaned`: (after preprocessing) Cleaned SpO2 signal

## Batch Processing

### Batch Process All Records
Automatically apply the default preprocessing pipeline to multiple selected records:
1. Loads each record
2. Applies preprocessing
3. Exports to parquet in the processed folder

### Batch Resample
Resample multiple recordings to the same target time resolution:
- Specify target resolution once (e.g., 0.5 seconds)
- CLI applies to all selected records
- Useful for standardizing datasets before ML training

## Resampling

### Analyze for Resampling
Before resampling, analyze the signal to understand:
- Current sampling rate characteristics
- Recommended resampling parameters
- Impact on event preservation

### Resample to Target Resolution
- Specify desired time interval (e.g., 0.5s, 1.0s)
- Preserves event annotations (is_apnea, is_hypopnea)
- Reduces file size and computation time

## Feature Extraction

Extract machine-learning-ready features from sleep signals:
- Calculates statistical features from signal segments
- Generates features for apnea/hypopnea prediction
- Saves features to parquet for ML pipeline integration

## Downloading Data from PhysioNet

The CLI can download open-source sleep apnea data from PhysioNet Challenge 2018:

### Download Options
1. **Specific records**: Enter record IDs (e.g., tr03-0146, tr03-0147)
2. **Range of records**: Specify start and end record numbers
3. **All training data**: Download entire training set (~1.5GB)

### Download Notes
- Requires `wget` to be installed and in PATH
- Download speed is capped by PhysioNet at ~1-2 MB/s
- Records are saved to `data/raw/`
- After download, you can immediately process the data

## Extending the CLI

### Adding New Actions

To add a new action to the CLI:

1. **Add the action to the menu** in `ui.py` (in the appropriate menu method):

```python
questionary.Choice("🆕 Your new action", value="your_action"),
```

2. **Handle the action** in `cli.py` `execute_action()` method:

```python
elif action == "your_action":
    df = self.data_loader.load_data(
        self.data_source, self.selected_records, self.custom_data_dir
    )
    # Your implementation here
    self.data_loader.current_dataframe = modified_df
    self.data_loader.applied_operations.append("YourOperation")
```

### Adding New Processing Steps

To add a new signal processing operation:

1. **Add method to `data_processor.py`** in the `DataProcessor` class:

```python
def your_operation(self, df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Your operation description."""
    self.console.print("\n[bold cyan]Processing...[/bold cyan]")
    
    modified_df = df.copy()
    # Your implementation here
    
    self.console.print("[green]✓[/green] Processing complete")
    return modified_df, True  # (dataframe, success)
```

2. **Use it in `cli.py`**:

```python
elif action == "your_operation":
    df = self.data_loader.load_data(...)
    result_df, success = self.data_processor.your_operation(df)
    if success:
        self.data_loader.current_dataframe = result_df
        self.data_loader.applied_operations.append("YourOperation")
```

### Using Components Independently

Each component can be used standalone:

```python
from rich.console import Console
from src.Cli.data_loader import DataLoader
from src.Cli.config import RAW_DIR

console = Console()
loader = DataLoader(console)

df = loader.load_data("raw", ["tr03-0146"])
print(df.head())
```

### Adding New Data Sources

Extend `data_loader.py` to support new formats:

```python
def load_data(
    self,
    data_source: str,
    selected_records: List[str],
    custom_data_dir: Optional[str] = None,
    alt_record: Optional[str] = None,
) -> pd.DataFrame:
    # Add new source handling
    if data_source == "your_new_source":
        df = your_custom_loader(record_path)
    # ...
```

### Creating a Custom Pipeline

You can create custom workflows by combining components:

```python
from rich.console import Console
from src.Cli.cli import SleepDataPipeline
from src.Cli.data_processor import DataProcessor

console = Console()
pipeline = SleepDataPipeline(console)

# Custom workflow
pipeline.data_loader.load_data("raw", ["tr03-0146"])
pipeline.data_processor.preprocess_signal(pipeline.data_loader.current_dataframe)
pipeline.data_processor.resample_signal(pipeline.data_loader.current_dataframe, 0.5)
pipeline.data_saver.save_dataframe(...)
```

## Data Management

### Saving Data

The CLI provides multiple save options:

**Save Dataframe**: Save modified data (after preprocessing, resampling, etc.)
- Automatically tracks operations in filename
- Choose between processed folder or custom location
- Retains all event annotations

**Save Features**: Save extracted ML features
- Generates feature matrix for model training
- Default location: processed folder
- Can be saved with custom naming

**Export to Parquet**: Convert raw .mat/.arousal files to parquet
- Initial conversion from raw format
- Preserves all signal data and annotations
- Enables fast loading in future sessions

### DataFrame Status

The CLI shows current dataframe status including:
- Number of rows and columns
- Applied operations (preprocessing, resampling, etc.)
- Ready for visualization or export

## Keyboard Shortcuts

- **↑/↓ Arrow keys**: Navigate menu options
- **Enter**: Select option
- **Ctrl+C**: Cancel/Exit at any time
- **Start typing**: Activate autocomplete in record selection

## Tips

1. **Raw vs Processed**: Use raw data when you need to visualize, export, or preprocess. Use processed data for faster analysis of pre-converted parquet files.

2. **Preprocessing**: The preprocessing step cleans the signal but keeps it in memory. Save it to parquet if you want to reuse it later.

3. **Batch Processing**: Ideal for preparing datasets. Select multiple records and batch process them to convert/preprocess them all at once.

4. **Resampling Before Features**: Resample your data to a consistent time resolution before extracting features for better ML model training.

5. **Feature Extraction**: Extract features once and save the parquet file for use in machine learning pipelines.

6. **Custom Directories**: Use custom directories to process data from external sources without moving files.

7. **Download Strategy**: Download data in ranges (e.g., tr03-0100 to tr03-0200) rather than all at once to avoid long waits.

8. **Error Handling**: If something goes wrong, the CLI will display an error message and let you continue working or try again.

9. **Integration**: Import `SleepDataPipeline` into your own scripts or notebooks for programmatic access to all CLI features.

10. **Modular Components**: Use individual components (`DataLoader`, `DataProcessor`, etc.) in your own workflows for maximum flexibility.

## Troubleshooting

### No records found

- Check that your data directory structure matches the expected format
- Ensure recordings are in subdirectories under `data/raw/` or `data/processed/`
- For batch mode with raw data, verify you have the correct custom directory set

### Import errors

- Make sure you're running from the correct directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Plotting doesn't work

- Ensure matplotlib backend is properly configured
- Try running from a terminal that supports GUI windows

### Download fails

- Verify `wget` is installed: `wget --version`
- Check internet connection and PhysioNet server availability
- Disk space: ~1.5GB needed for full training dataset
- PhysioNet may rate-limit downloads; try downloading smaller ranges

### Resampling produces unexpected results

- Use "Analyze signal for resampling" first to understand optimal parameters
- Ensure time_s and sao2_percent columns exist in data
- Check that target resolution is reasonable (typically 0.1-1.0 seconds)

### Custom directory errors

- Verify path exists and is readable
- Use forward slashes (/) or backslashes (\\) depending on OS
- Ensure directory contains proper subdirectories with record data
