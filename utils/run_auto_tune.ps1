param(
    [string]$StudyName = "dma_v15_auto_tune",
    [string]$TaskName = "iTentformerAutoTune",
    [int]$ScreenEpochs = 24,
    [int]$Finalists = 3
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = "D:\SoftWare\Environment_M\envs\lucky\python.exe"
$studyDir = Join-Path $root ("tuning_results\" + $StudyName)
$launcherLog = Join-Path $studyDir "scheduled_launcher.log"

New-Item -ItemType Directory -Force -Path (Join-Path $root ".tmp") | Out-Null
New-Item -ItemType Directory -Force -Path $studyDir | Out-Null
$env:TEMP = Join-Path $root ".tmp"
$env:TMP = Join-Path $root ".tmp"
Set-Location $root

# Task Scheduler starts long-running jobs below normal priority by default.
# Restore the same priority used by an interactive PowerShell training run.
(Get-Process -Id $PID).PriorityClass = "Normal"

try {
    & $python "utils\auto_tune_itentformer.py" `
        --study-name $StudyName `
        --screen-epochs $ScreenEpochs `
        --finalists $Finalists *>> $launcherLog
    exit $LASTEXITCODE
}
catch {
    $_ | Out-String | Add-Content -LiteralPath $launcherLog -Encoding UTF8
    exit 1
}
finally {
    schtasks.exe /Delete /TN $TaskName /F 2>$null | Out-Null
}
