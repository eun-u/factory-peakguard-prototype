$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$freezeScript = Join-Path $PSScriptRoot 'run_frozen.ps1'
$freezeInstant = [DateTimeOffset]::Parse('2026-10-01T18:00:00+09:00')
$taskName = 'FactoryPeakguard-Task05-Freeze-20261001'
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument ('-NoProfile -NonInteractive -WindowStyle Hidden -File "{0}"' -f $freezeScript) -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Once -At $freezeInstant.LocalDateTime
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$taskIdentity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.Actions.Arguments -notlike ('*' + $freezeScript + '*')) {
    throw 'A task with the same name belongs to another checkout; preserve it and inspect before replacement.'
}
$needsInteractive = $false
try {
    $principal = New-ScheduledTaskPrincipal -UserId $taskIdentity -LogonType S4U -RunLevel Limited
    $task = Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Task05 local-only frozen evaluation and report generation after the authorized deadline.' -Force
} catch {
    $needsInteractive = $true
    $principal = New-ScheduledTaskPrincipal -UserId $taskIdentity -LogonType Interactive -RunLevel Limited
    $task = Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Task05 frozen evaluation after deadline; interactive-session fallback.' -Force
}
$result = [ordered]@{task_name=$taskName;state=[string]$task.State;scheduled_kst=$freezeInstant.ToString('o');interactive_logon_required=$needsInteractive;start_when_available=$true;execution_completed=$false;note='PC must be available. If interactive_logon_required is true, this Windows user must be logged on. Python date and one-time lock remain authoritative.'}
$result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'outputs\logs\scheduled_freeze.json') -Encoding UTF8
$result | ConvertTo-Json
