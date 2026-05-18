# 37A-Kandidat-Sleepy-AI

Bachelor thesis project VT26 for group 37A in Data-IT division at Chalmers.

## Setup

This project uses Python with a virtual environment to manage dependencies.

**Recommended**: Open the project in VS Code using the workspace file `sleepy.code-workspace` for the best development experience (automatic environment detection, kernel selection, and terminal activation).

### Prerequisites

- Python 3.12

### Installation

#### macOS/Linux

1. Clone the repository
2. Run the setup script:
   ```bash
   ./setup.sh
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

#### Windows

1. Clone the repository
2. Run the setup script in PowerShell:
   ```powershell
   .\setup.ps1
   ```
   _Note: If you encounter an execution policy error, run PowerShell as Administrator and execute:_
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Activate the virtual environment:
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - CMD: `.venv\Scripts\activate.bat`

### Deactivating the Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

### Setup Options

The project provides multiple setup scripts to optimize installation time:

#### Standard Setup (First Time Setup)

```bash
./setup.sh
```

- Creates fresh virtual environment
- Installs all dependencies including Jupyter (slowest, ~5-10 minutes)
- **Now optimized**: Only recreates venv if you confirm, otherwise just updates packages

#### Fast Setup (Quick Updates)

```bash
./setup-fast.sh
```

- Updates existing environment only
- Only installs missing/outdated packages
- Takes seconds instead of minutes
- **Use this** for daily development after initial setup

#### Lightweight Setup (No Jupyter)

```bash
# For running scripts/CLI only (not notebooks)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-light.txt
```

- Skips Jupyter and its many dependencies
- Much faster (~2-3 minutes)
- Use if you only need to run Python scripts or the CLI

**Why is setup slow?** Jupyter has 100+ dependencies which take time to download and install. If you don't need notebooks, use the lightweight setup.

## Dependencies

The project includes the following Python libraries (see `requirements.txt`):

- **Data Processing**: numpy, pandas, scipy
- **Data Management**: wfdb, pyarrow
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **CLI Interface**: questionary, rich
- **Development**: jupyter, ipykernel, tqdm

Additional deep learning frameworks (PyTorch, TensorFlow) can be uncommented in `requirements.txt` if needed.

## GPU Training On Windows

If you want to train the 1-D CNN on an NVIDIA GPU such as an RTX 3070, the default setup is not enough on its own. The current virtual environment may contain a CPU-only PyTorch build.

### 1. Install CUDA-enabled PyTorch in the project venv

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Then verify that PyTorch can see the GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If this prints `True` and shows your NVIDIA GPU name, training will run on CUDA automatically.

### 2. Run the direct CNN training script

The repository includes a non-interactive training entrypoint for the deep model:

```powershell
python jobs\train_single.py --epochs 20 --batch-size 1024
```

Useful options:

- `--records tr03-0005 tr03-0146` to train on specific processed records only
- `--window-size 60` to control the raw SpO2 window size
- `--test-size 0.2` to change the evaluation split
- `--model-name cnn1d_3070` to control the saved checkpoint name
- `--no-save` to do a quick smoke test without writing model files

Models are saved to `data/models/`.

### 3. Use the interactive CLI instead

If you prefer the menu flow, this also works once CUDA-enabled PyTorch is installed:

```powershell
python src\Cli\cli.py
```

Choose `Train a model`, then select `Deep (1D CNN)`. The model code uses CUDA automatically when available.
