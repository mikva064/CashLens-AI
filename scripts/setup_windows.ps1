param(
    [string]$PythonExe = 'python',
    [switch]$WithModel
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot '.venv'
& $PythonExe -m venv $environmentPath
if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment' }
$runtimePython = Join-Path $environmentPath 'Scripts/python.exe'
& $runtimePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip' }
if ($WithModel) {
    & $runtimePython -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
    if ($LASTEXITCODE -ne 0) { throw 'Could not install CUDA-enabled PyTorch' }
    & $runtimePython -m pip install -e "${projectRoot}[data,model]"
} else {
    & $runtimePython -m pip install -e "${projectRoot}[data]"
}
if ($LASTEXITCODE -ne 0) { throw 'Could not install CashLens dependencies' }
& $runtimePython -m cashlens_ai.cli doctor
exit $LASTEXITCODE
