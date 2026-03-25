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

From the project root directory:

```bash
python src/cli.py
```

Or make it executable and run directly:

```bash
chmod +x src/cli.py
./src/cli.py
```

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
data/
├── raw/                   # Raw recordings
│   └── tr03-0146/         # One folder per recording
│       ├── tr03-0146.mat
│       ├── tr03-0146.arousal
│       ├── tr03-0146.hea
│       └── tr03-0146-arousal.mat
└── processed/             # Processed data
    └── tr03-0146/         # One folder per recording
        └── tr03-0146.parquet
```

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

1. **Add the action to the menu** in `choose_action()`:

```python
questionary.Choice("🆕 Your new action", value="your_action"),
```

2. **Handle the action** in `execute_action()`:

```python
elif action == "your_action":
    self._your_action_function()
```

3. **Implement the function** in the `SleepDataPipeline` class:

```python
def _your_action_function(self):
    """Your action description."""
    console.print("\n[bold cyan]Doing something...[/bold cyan]")

    df = self._load_data()

    # Your implementation here

    console.print("[green]✓[/green] Action completed")
```

### Adding Multi-Record Support

The `selected_records` attribute is a list, and batch processing is already supported. To add batch-compatible actions:

1. Add action to `choose_action()` only in the non-batch section
2. Or add to batch_actions list for batch-only functionality
3. Update action functions to handle single records in single mode

Example for single-mode actions:

```python
def _your_action(self):
    """Process the selected record."""
    record = self.selected_records[0]
    df = self._load_data()
    # Process single record...
```

Example for batch-compatible actions:

```python
def _your_batch_action(self):
    """Process multiple records."""
    # For batch mode
    if self.mode == "batch":
        for record in self.selected_records:
            df = self._load_data(record)
            # Process each record...
    else:
        df = self._load_data()
        # Process single record...
```

### Working with DataFrames

Track applied operations using the `self.applied_operations` list:

```python
# After modifying data
self.current_dataframe = modified_df
self.applied_operations.append("YourOperation")

# Operations show in status when saved
# Example filename: tr03-0146_preprocessed_resampled_0.5s.parquet
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
