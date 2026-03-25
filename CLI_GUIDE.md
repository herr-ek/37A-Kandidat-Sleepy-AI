# Sleep Data Analysis Pipeline - CLI Guide

## Overview

This CLI provides an interactive interface for processing and analyzing sleep apnea recordings. It features:

- 🎯 **Navigable menus** with arrow key selection
- 🔍 **Autocomplete** for record selection
- 🎨 **Color-coded output** for better readability
- 🔧 **Extensible architecture** for easy feature additions

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

- **Raw data**: Load directly from `.mat` and `.arousal` files (4 files per recording)
- **Processed data**: Load from pre-processed `.parquet` files (faster, single file)

Use arrow keys ↑/↓ and press Enter to select.

#### Step 2: Select Recording

The CLI will scan the chosen directory and display all available recordings in a table.

Type the recording name (autocomplete will help you) or paste it, then press Enter.

Example recordings: `tr03-0146`, `tr03-0029`, etc.

#### Step 3: Choose Action

Select from available actions:

- **📋 Display data as DataFrame**: View the data in tabular format with column info
- **📊 Show data statistics**: Get statistical summary and sleep event counts
- **📈 Plot signal with annotations**: Visualize the SaO2 signal with respiratory events
- **🧹 Preprocess signal**: Clean artifacts from the signal (removes outliers, spikes, smooths)
- **💾 Export to parquet**: Convert raw data to parquet format (raw data only)
- **🔄 Select different record**: Choose another recording
- **🔙 Back to data source selection**: Start over with different data source
- **❌ Exit**: Quit the program

## Directory Structure

```
data/
├── raw/                    # Raw recordings
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

1. **Display DataFrame**: Shows first 20 rows and column information
2. **Statistics**: Provides statistical summary including apnea/hypopnea event counts
3. **Plot Signal**: Visualizes the SaO2 signal with respiratory events marked
4. **Preprocess**: Applies artifact removal and noise reduction (4-step process)
5. **Export**: Converts raw data to efficient parquet format

### Data Columns

When loading data, you'll see these columns:

- `time_s`: Time in seconds
- `sao2_percent`: SpO2 (oxygen saturation) percentage
- `is_apnea`: Binary flag for apnea events
- `is_hypopnea`: Binary flag for hypopnea events
- `sao2_cleaned`: (after preprocessing) Cleaned SpO2 signal

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

The `selected_records` attribute is a list, so when you implement multi-record support:

1. Modify `select_records()` to allow multiple selections
2. Update action functions to loop through `self.selected_records`
3. Consider aggregation strategies for combined results

Example:

```python
def _your_batch_action(self):
    """Process multiple records."""
    results = []
    for record in self.selected_records:
        # Process each record
        results.append(result)

    # Aggregate or display combined results
```

## Keyboard Shortcuts

- **↑/↓ Arrow keys**: Navigate menu options
- **Enter**: Select option
- **Ctrl+C**: Cancel/Exit at any time
- **Start typing**: Activate autocomplete in record selection

## Tips

1. **Raw vs Processed**: Use raw data when you need to visualize or export. Use processed data for faster analysis.

2. **Preprocessing**: The preprocessing step cleans the signal but keeps it in memory. Save it to parquet if you want to reuse it.

3. **Batch Processing**: Currently supports single records, but the architecture is ready for batch processing - just extend the selection logic.

4. **Error Handling**: If something goes wrong, the CLI will display an error message and let you continue working.

## Troubleshooting

### No records found

- Check that your data directory structure matches the expected format
- Ensure recordings are in subdirectories under `data/raw/` or `data/processed/`

### Import errors

- Make sure you're running from the correct directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Plotting doesn't work

- Ensure matplotlib backend is properly configured
- Try running from a terminal that supports GUI windows

## Future Enhancements

Planned features:

- [ ] Multi-record selection and batch processing
- [ ] Feature extraction for ML models
- [ ] Export preprocessed data in multiple formats
- [ ] Progress bars for long operations
- [ ] Configuration file for custom settings
- [ ] Report generation (PDF/HTML)
