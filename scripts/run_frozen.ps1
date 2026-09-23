$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $projectRoot
$pipelinePython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pipelinePython)) { throw 'Install the project .venv first.' }
$logPath = Join-Path $projectRoot 'outputs\logs\scheduled_freeze.log'
# Keep native stderr warnings from becoming terminating PowerShell errors.
# Python enforces the KST deadline and one-time completed/interrupted locks.
$errorLogPath = Join-Path $projectRoot 'outputs\logs\scheduled_freeze.stderr.log'
$process = Start-Process -FilePath $pipelinePython -ArgumentList @('-u', '-X', 'utf8', 'run_all.py', '--from', 'final') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $logPath -RedirectStandardError $errorLogPath -Wait -PassThru
exit $process.ExitCode
