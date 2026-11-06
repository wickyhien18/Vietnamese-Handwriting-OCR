# Local setup for Windows (PowerShell): virtual environment, dependencies, local folders, checks.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$script:UsePyLauncher = $false
if (Get-Command py -ErrorAction SilentlyContinue) {
    $script:UsePyLauncher = $true
} elseif (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python not found. Install Python 3.10 or newer (see README section 2)."
    exit 1
}

function Invoke-Python {
    if ($script:UsePyLauncher) { & py -3 @args } else { & python @args }
}

Invoke-Python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python 3.10 or newer is required."
    exit 1
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: git not found. Install it before cloning or pushing (see README section 2)."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    Invoke-Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: could not create the virtual environment."; exit 1 }
}

$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: dependency installation failed."; exit 1 }

New-Item -ItemType Directory -Force -Path "data\uit_hwdb", "data\corpus", "checkpoints\ocr", "checkpoints\corrector", "outputs" | Out-Null

Write-Host "Running unit tests ..."
& $venvPython -m unittest discover -s tests -t .
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: unit tests failed."; exit 1 }

Write-Host "Validating the sample contract file ..."
& $venvPython scripts\validate_ocr_out.py contracts\examples\ocr_out.sample.jsonl
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: sample file validation failed."; exit 1 }

Write-Host ""
Write-Host "Setup finished. To activate the environment in a new terminal:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
