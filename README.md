# AE117 Parkinson Disease Detection (local run)

This project is a Flask web app that demonstrates image- and voice-based predictions for Parkinson's detection.

The repository includes two modes:

- **Minimal mode** (recommended to start): runs the web UI with stubs for ML predictions so you can explore the site without installing heavy libs. Required packages are in `requirements.txt`.

- **Full ML mode**: installs TensorFlow and `praat-parselmouth` for real predictions. See `requirements_full.txt` for the full list. These packages are large and may require additional system dependencies on Windows.

## Quick Start (Windows)

The easiest way to run the app is using the provided helper scripts. These scripts will automatically:
1. Detect your Python installation.
2. Create a virtual environment (`.venv`).
3. Install necessary requirements.
4. Start the Flask application.

### Using Batch Script
Double-click `run_app.bat` or run it from the Command Prompt:
```cmd
run_app.bat
```

### Using PowerShell Script
Right-click `run_app.ps1` and select "Run with PowerShell" or run it from a PowerShell terminal:
```powershell
.\run_app.ps1
```

Once started, open your browser and go to:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## Manual Setup

If you prefer to set up the environment manually:

1. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   ```

2. **Activate and install requirements:**
   ```powershell
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. **Run the app:**
   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

## Notes and Troubleshooting

- The repository contains guarded fallbacks so the UI runs without TensorFlow or Praat. Image predictions will return a default response when `keras_model.h5` is missing; voice predictions will return a default when `praat-parselmouth` is not installed.
- **Python Version**: Ensure you are using Python 3.8-3.10 for best compatibility with dependencies.
