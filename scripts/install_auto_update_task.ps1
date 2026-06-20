param(
    [string]$TaskName = "WorldCupPredictorAutoUpdate",
    [int]$IntervalMinutes = 60
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = (Get-Command python).Source
$Script = Join-Path $ProjectRoot "src\auto_update.py"

if (-not (Test-Path $Script)) {
    throw "Could not find $Script"
}

$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$Script`" --once" `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
$Trigger.Repetition = New-ScheduledTaskRepetitionSettings `
    -Interval (New-TimeSpan -Minutes $IntervalMinutes) `
    -Duration ([TimeSpan]::MaxValue)

$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Automatically refresh World Cup predictor fixtures, FotMob data, predictions, and analysis reports." `
    -Force | Out-Null

Write-Host "Installed scheduled task: $TaskName"
Write-Host "Project root: $ProjectRoot"
Write-Host "Runs every $IntervalMinutes minutes."
Write-Host "Check status: reports\update_status.json"
