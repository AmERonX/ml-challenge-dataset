param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param([string]$Message, [scriptblock]$Action)
    Write-Host "==> $Message"
    & $Action
}

$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonCmd = "py -3"
} else {
    throw "Python was not found. Install Python 3.9+ and re-run."
}

Invoke-Step "Creating virtual environment at '$VenvPath'" {
    Invoke-Expression "$pythonCmd -m venv `"$VenvPath`""
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Invoke-Step "Upgrading pip inside virtual environment" {
    & $venvPython -m pip install --upgrade pip
}

Invoke-Step "Installing dependencies from requirements.txt" {
    & $venvPython -m pip install -r requirements.txt
}

Invoke-Step "Registering Jupyter kernel for this virtual environment" {
    & $venvPython -m ipykernel install --user --name "ml-challenge-venv" --display-name "Python (.venv ML Challenge)"
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Use this command to generate predictions:"
Write-Host ".\.venv\Scripts\python.exe .\predict.py --model trained_model.json --test TEST.csv --output FINAL.csv"
