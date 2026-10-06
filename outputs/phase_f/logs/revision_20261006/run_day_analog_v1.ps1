param([ValidateSet('smoke', 'search')][string]$Stage = 'smoke')

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONWARNINGS = 'ignore'
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Remove-Item Env:CUDA_HOME -ErrorAction SilentlyContinue

$analogRoot = (Get-Location).Path
$analogPython = Join-Path $analogRoot 'outputs/phase_f/env/Scripts/python.exe'
$analogSmokeLog = Join-Path $analogRoot 'outputs/phase_f/logs/day_analog_smoke_v1_20261006.log'
$analogSearchLog = Join-Path $analogRoot 'outputs/phase_f/logs/day_analog_search_v1_20261006.log'
if (($Stage -eq 'smoke' -and (Test-Path -LiteralPath $analogSmokeLog)) -or
    ($Stage -eq 'search' -and (Test-Path -LiteralPath $analogSearchLog))) {
    throw 'Preserve prior analog logs; this first-launch script cannot replace them.'
}
if (-not (Test-Path -LiteralPath $analogPython)) {
    throw 'Phase F experiment Python is unavailable.'
}

if ($Stage -eq 'smoke') {
    & $analogPython -u -m phase_f.goal_day_analog --stage smoke 2>&1 | Tee-Object -FilePath $analogSmokeLog
    $analogSmokeExit = $LASTEXITCODE
    exit $analogSmokeExit
}

if (-not (Test-Path -LiteralPath $analogSmokeLog)) {
    throw 'First complete and verify the dedicated technical smoke.'
}

& $analogPython -u -m phase_f.goal_day_analog --stage search 2>&1 | Tee-Object -FilePath $analogSearchLog
$analogSearchExit = $LASTEXITCODE
exit $analogSearchExit
