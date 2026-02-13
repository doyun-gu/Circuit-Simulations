# Dynamic Phasor Framework - Setup Guide

## Quick Start (5 minutes)

### Step 1: Create Project Directory
```bash
mkdir dynamic-phasor-sim
cd dynamic-phasor-sim
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install numpy scipy matplotlib pandas numba jupyter jupyterlab ipywidgets
```

### Step 4: Verify Installation
```bash
python -c "import numpy; import scipy; import matplotlib; print('All packages installed successfully!')"
```

### Step 5: Launch Jupyter
```bash
jupyter lab
```

---

## Project Structure

After setup, your directory should look like:
```
dynamic-phasor-sim/
├── venv/                      # Virtual environment (don't edit)
├── src/
│   └── dynamic_phasor/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── components.py  # R, L, C components
│       │   ├── phasor.py      # Phasor transformations
│       │   └── circuit.py     # Circuit classes
│       ├── solvers/
│       │   ├── __init__.py
│       │   ├── time_domain.py
│       │   ├── phasor_domain.py
│       │   └── hybrid.py
│       ├── benchmarks/
│       │   ├── __init__.py
│       │   └── rim2025.py     # Rim et al. benchmark
│       └── utils/
│           ├── __init__.py
│           └── plotting.py
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_rim2025_validation.ipynb
│   └── 03_method_comparison.ipynb
├── tests/
│   └── test_phasor.py
├── examples/
│   └── basic_rlc.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Detailed Installation Options

### Option A: Minimal (for quick testing)
```bash
pip install numpy scipy matplotlib
```

### Option B: Full Development
```bash
pip install numpy scipy matplotlib pandas numba jupyter jupyterlab ipywidgets pytest black
```

### Option C: Using requirements.txt
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Issue: `python3` not found
```bash
# Try using 'python' instead
python -m venv venv
```

### Issue: Permission denied on Windows
```bash
# Run PowerShell as Administrator, then:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Jupyter not launching
```bash
# Install jupyter separately
pip install jupyter
# Or use classic notebook
jupyter notebook
```

### Issue: Import errors
```bash
# Make sure you're in the project root and venv is activated
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or on Windows:
set PYTHONPATH=%PYTHONPATH%;%cd%\src
```

---

## Next Steps

1. Copy all source files from the provided framework
2. Run the quickstart notebook
3. Validate against Rim et al. (2025) benchmark
4. Generate comparison figures for your paper
