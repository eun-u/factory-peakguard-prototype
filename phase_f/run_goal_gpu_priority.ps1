param([Parameter(Mandatory=$true)][int]$ExpectedBaselinePid, [switch]$RecoverVerifiedExit)
$ErrorActionPreference = 'Stop'
$phaseRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $phaseRoot
$phasePython = Join-Path $phaseRoot 'outputs/phase_f/env/Scripts/python.exe'
$phaseLogRoot = Join-Path $phaseRoot 'outputs/phase_f/logs'
$phaseAllocation = Join-Path $phaseLogRoot 'revision_20261006/goal_lora_gpu_allocation.json'
$phasePreviousAllocation = $phaseAllocation
if ($RecoverVerifiedExit) {
    $phaseAllocation = Join-Path $phaseLogRoot 'revision_20261006/goal_lora_gpu_allocation_recovery.json'
}
$phaseOriginalStatus = Join-Path $phaseRoot 'outputs/phase_f/walkforward_v2/logs/driver_status.json'
if (Test-Path -LiteralPath $phaseAllocation) {
    throw 'An allocation record already exists; preserve it and use a new transaction.'
}
$phaseRecord = [ordered]@{
    status = 'ownership_check'; started_at = [DateTime]::UtcNow.ToString('o')
    baseline_pid = $ExpectedBaselinePid; goal_achieved = $false; holdout_read = $false
    full_phase_budgets = 'unchanged'; resume_original_in_finally = $true
}
function Save-Allocation {
    $phaseRecord | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $phaseAllocation -Encoding utf8
}
$phaseOwner = Get-CimInstance Win32_Process -Filter "ProcessId = $ExpectedBaselinePid"
if ($RecoverVerifiedExit) {
    if ($phaseOwner -or (Get-Process -Id $ExpectedBaselinePid -ErrorAction SilentlyContinue)) {
        throw 'Recovery requires the original process to be absent; no new job launched.'
    }
    $phasePrevious = Get-Content -LiteralPath $phasePreviousAllocation -Raw | ConvertFrom-Json
    if ($phasePrevious.baseline_pid -ne $ExpectedBaselinePid -or $phasePrevious.status -ne 'lora_failed' -or
        $phasePrevious.error -ne 'The original process has not exited; no model files moved or new jobs launched.' -or
        -not $phasePrevious.baseline_command.StartsWith('"' + $phasePython + '"') -or
        $phasePrevious.baseline_command -notmatch '\s-m\s+phase_f\.run\s+--stage\s+all\s+--retry\s*$') {
        throw 'Recovery ownership evidence does not match the preserved failed transaction.'
    }
    $phaseRecord['baseline_command'] = $phasePrevious.baseline_command
    $phaseRecord['recovery_of'] = $phasePreviousAllocation
    $phaseRecord['recovery_record_sha256'] = (Get-FileHash -LiteralPath $phasePreviousAllocation -Algorithm SHA256).Hash.ToLowerInvariant()
} else {
    if (-not $phaseOwner -or -not $phaseOwner.CommandLine.StartsWith('"' + $phasePython + '"') -or
        $phaseOwner.CommandLine -notmatch '\s-m\s+phase_f\.run\s+--stage\s+all\s+--retry\s*$') {
        throw 'Expected PID is not the exact owned original Phase F driver; no process stopped.'
    }
    $phaseRecord['baseline_command'] = $phaseOwner.CommandLine
}
$phaseRecord['original_driver_snapshot'] = Get-Content -LiteralPath $phaseOriginalStatus -Raw | ConvertFrom-Json
Save-Allocation
$phaseCheckpointed = $false
$phaseLoraFailed = $false
try {
    if (-not $RecoverVerifiedExit) {
        Stop-Process -Id $ExpectedBaselinePid -ErrorAction Stop
        Wait-Process -Id $ExpectedBaselinePid -Timeout 10 -ErrorAction SilentlyContinue
        for ($phaseExitPoll=0; $phaseExitPoll -lt 20; $phaseExitPoll++) {
            if (-not (Get-CimInstance Win32_Process -Filter "ProcessId = $ExpectedBaselinePid" -ErrorAction Stop)) { break }
            Start-Sleep -Milliseconds 250
        }
    }
    if (Get-CimInstance Win32_Process -Filter "ProcessId = $ExpectedBaselinePid" -ErrorAction Stop) {
        throw 'The original process has not exited; no model files moved or new jobs launched.'
    }
    $phaseCheckpointed = $true
    $phaseModelRoot = Join-Path $phaseRoot 'outputs/phase_f/walkforward_v2/models'
    $phaseJunction = Get-Item -LiteralPath $phaseModelRoot
    if ($phaseJunction.LinkType -ne 'Junction' -or
        $phaseJunction.Target -ne 'D:\PeakGuard_PhaseF_20261003\models\walkforward_v2') {
        throw 'Owned model junction changed; no checkpoint files moved.'
    }
    $phaseFits = @()
    $phaseOrphans = @()
    $phaseStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffff')
    foreach ($phaseDir in Get-ChildItem -LiteralPath $phaseModelRoot -Directory -Filter 'F0-1-M2__wf_explore__seed*') {
        foreach ($phaseFile in Get-ChildItem -LiteralPath $phaseDir.FullName -File) {
            if ($phaseFile.Name -notmatch '^h\d{2}_f\d+\.(pt|json|pt\.tmp)$') { continue }
            if ($phaseFile.Extension -eq '.tmp') {
                $phaseOrphan = $phaseFile.FullName + '.interrupted-' + $phaseStamp
                Move-Item -LiteralPath $phaseFile.FullName -Destination $phaseOrphan
                $phaseOrphans += $phaseOrphan
                continue
            }
            $phasePeer = [System.IO.Path]::ChangeExtension($phaseFile.FullName,
                $(if ($phaseFile.Extension -eq '.pt') { '.json' } else { '.pt' }))
            if (-not (Test-Path -LiteralPath $phasePeer)) {
                $phaseOrphan = $phaseFile.FullName + '.interrupted-' + $phaseStamp
                Move-Item -LiteralPath $phaseFile.FullName -Destination $phaseOrphan
                $phaseOrphans += $phaseOrphan
                continue
            }
            if ($phaseFile.Extension -eq '.pt') {
                $phaseMetadata = Get-Content -LiteralPath $phasePeer -Raw | ConvertFrom-Json
                $phaseDigest = (Get-FileHash -LiteralPath $phaseFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($phaseDigest -ne $phaseMetadata.sha256) { throw 'Completed baseline fit hash mismatch.' }
                $phaseFits += [ordered]@{path=$phaseFile.FullName; sha256=$phaseDigest; seed=$phaseMetadata.seed}
            }
        }
    }
    $phaseRecord['completed_fits'] = $phaseFits
    $phaseRecord['preserved_incomplete_files'] = $phaseOrphans
    $phaseRecord['status'] = 'lora_smoke'
    Save-Allocation
    [ordered]@{status='checkpointed_for_goal_lora'; previous_pid=$ExpectedBaselinePid;
        at=[DateTime]::UtcNow.ToString('o'); completed_fit_count=$phaseFits.Count;
        automatic_resume='after LoRA wave, including failure'; full_phase_complete=$false;
        holdout_read=$false} | ConvertTo-Json | Set-Content -LiteralPath $phaseOriginalStatus -Encoding utf8
    $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; $env:PYTHONWARNINGS='ignore'
    $env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
    $env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
    Remove-Item Env:CUDA_HOME -ErrorAction SilentlyContinue
    & $phasePython -u -m phase_f.goal_foundation --stage smoke 2>&1 |
        Tee-Object -FilePath (Join-Path $phaseLogRoot 'goal_lora_smoke_v1_20261006.log')
    if ($LASTEXITCODE -ne 0) { throw "LoRA smoke exited $LASTEXITCODE; full candidate search not started." }
    $phaseRecord['status'] = 'lora_search'; $phaseRecord['smoke_passed_at'] = [DateTime]::UtcNow.ToString('o')
    Save-Allocation
    & $phasePython -u -m phase_f.goal_foundation --stage search 2>&1 |
        Tee-Object -FilePath (Join-Path $phaseLogRoot 'goal_lora_search_v1_20261006.log')
    if ($LASTEXITCODE -ne 0) { throw "LoRA search exited $LASTEXITCODE." }
    $phaseRecord['status'] = 'lora_explore_complete'; $phaseRecord['lora_finished_at'] = [DateTime]::UtcNow.ToString('o')
    Save-Allocation
} catch {
    $phaseLoraFailed = $true
    $phaseRecord['status'] = 'lora_failed'; $phaseRecord['error'] = $_.Exception.Message
    Save-Allocation
    Write-Output ('GOAL_LORA_FAILED ' + $_.Exception.Message)
} finally {
    if ($phaseCheckpointed) {
        $phaseRecord['original_resume_started_at'] = [DateTime]::UtcNow.ToString('o')
        Save-Allocation
        Write-Output 'RESUMING_ORIGINAL_FULL_PHASE_F'
        & $phasePython -u -m phase_f.run --stage all --retry 2>&1 |
            Tee-Object -FilePath (Join-Path $phaseLogRoot 'revised_full_resume4_20261006.log')
        $phaseRecord['original_resume_exit_code'] = $LASTEXITCODE
        $phaseRecord['original_resume_finished_at'] = [DateTime]::UtcNow.ToString('o')
        Save-Allocation
    }
}
if ($phaseLoraFailed) { exit 1 }
if ($phaseCheckpointed -and $phaseRecord['original_resume_exit_code'] -ne 0) { exit 1 }
