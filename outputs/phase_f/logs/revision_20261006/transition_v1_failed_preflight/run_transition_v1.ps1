$phaseRoot = (Get-Location).Path
$phasePython = Join-Path $phaseRoot 'outputs/phase_f/env/Scripts/python.exe'
$phaseSmokeLog = Join-Path $phaseRoot 'outputs/phase_f/logs/transition_smoke_v1_20261006.log'
$phaseSearchLog = Join-Path $phaseRoot 'outputs/phase_f/logs/transition_search_v1_20261006.log'
if (-not (Test-Path -LiteralPath $phasePython)) { throw 'Phase F runtime is missing' }
if ((Test-Path -LiteralPath $phaseSmokeLog) -or (Test-Path -LiteralPath $phaseSearchLog)) {
    throw 'Existing transition logs must be preserved; do not start a duplicate driver'
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONWARNINGS = 'ignore'
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
Remove-Item Env:CUDA_HOME -ErrorAction SilentlyContinue
& $phasePython -u -m phase_f.goal_transition --stage smoke *> $phaseSmokeLog
$phaseSmokeExit = $LASTEXITCODE
if ($phaseSmokeExit -ne 0) {
    Write-Output "TRANSITION_SMOKE_FAILED exit=$phaseSmokeExit log=$phaseSmokeLog"
    exit $phaseSmokeExit
}
Write-Output "TRANSITION_SMOKE_COMPLETED log=$phaseSmokeLog"
& $phasePython -u -m phase_f.goal_transition --stage search *> $phaseSearchLog
$phaseSearchExit = $LASTEXITCODE
Write-Output "TRANSITION_SEARCH_TERMINAL exit=$phaseSearchExit log=$phaseSearchLog"
exit $phaseSearchExit
