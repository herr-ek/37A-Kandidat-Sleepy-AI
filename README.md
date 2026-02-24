# 37A-Kandidat-Sleepy-AI

Bachelor thesis project VT26 for group 37A in Data-IT division at Chalmers.

## Setup

This project uses Python with a virtual environment to manage dependencies.

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

## Dependencies

The project includes the following Python libraries (see `requirements.txt`):

- **Data Processing**: numpy, pandas, scipy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Development**: jupyter, ipykernel, tqdm

Additional deep learning frameworks (PyTorch, TensorFlow) can be uncommented in `requirements.txt` if needed.
