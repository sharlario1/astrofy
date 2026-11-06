# Native Windows 11 setup (PowerShell). Run from this folder:  powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# Optional GPU provider for the (optional) face-parsing ONNX:  -Gpu dml   (DirectML, any GPU)   or   -Gpu cuda   (NVIDIA)
param([ValidateSet("none","dml","cuda")][string]$Gpu = "none")
$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -e ".[dev]"
if ($Gpu -ne "none") {
    & $py -m pip uninstall -y onnxruntime
    & $py -m pip install $(if ($Gpu -eq "dml") { "onnxruntime-directml" } else { "onnxruntime-gpu" })
}
& $py -m photoflow fetch-models
& $py -m pytest -q
Write-Host "`nReady. Activate with  .\.venv\Scripts\Activate.ps1   then run  photoflow --help"
