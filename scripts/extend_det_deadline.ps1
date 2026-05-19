# scripts/extend_det_deadline.ps1
# Watch for iter_epoch_4.pdparams (current epoch save), then gracefully restart
# the watchdog with extended deadline and disabled early-stop patience.

param(
    [string]$WaitForFile = "E:\OCR_Project\PaddleOCR\output\det_v0\iter_epoch_4.pdparams",
    [int]$CurrentWatchdogPid = 11468,
    [string]$NewDeadline = "2026-05-18 20:00",
    [double]$HmeanTarget = 0.95,
    [int]$PatienceEvals = 999,
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Continue"
$logFile = "E:\OCR_Project\artifacts\det_training\extend_deadline.log"

function W($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $logFile -Value $line -Encoding utf8
    Write-Host $line
}

W "extend_det_deadline started. Waiting for $WaitForFile"
W "  current watchdog PID=$CurrentWatchdogPid"
W "  new deadline=$NewDeadline, patience=$PatienceEvals"

# ----- 1) wait until target checkpoint appears -----
while ($true) {
    if (Test-Path $WaitForFile) {
        $age = (Get-Date) - (Get-Item $WaitForFile).LastWriteTime
        if ($age.TotalSeconds -gt 20) {
            W "checkpoint detected (age=$([int]$age.TotalSeconds)s): $WaitForFile"
            break
        }
    }
    Start-Sleep -Seconds $PollSeconds
}

# ----- 2) verify epoch 4 eval ran (look for best_epoch: 4 or higher in log) -----
$trainLog = (Get-ChildItem 'E:\OCR_Project\artifacts\det_training\logs\det_*.log' |
             Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
W "tail log: $trainLog"

# wait up to 5 more minutes for eval line for epoch 4 to be written
$waitedExtra = 0
while ($waitedExtra -lt 300) {
    $content = Get-Content $trainLog -Raw -ErrorAction SilentlyContinue
    if ($content -match 'best_epoch:\s*[4-9]|best_epoch:\s*\d{2,}') {
        W "epoch 4 eval line confirmed"
        break
    }
    Start-Sleep -Seconds 15
    $waitedExtra += 15
}

# ----- 3) kill watchdog + child python -----
W "killing watchdog PID=$CurrentWatchdogPid"
try { Stop-Process -Id $CurrentWatchdogPid -Force -ErrorAction Stop } catch { W "  stop watchdog failed: $_" }
Start-Sleep -Seconds 3

# kill any python.exe running tools/train.py with our config
$pythons = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
           Where-Object { $_.CommandLine -like '*tools/train.py*' -or $_.CommandLine -like '*tools\train.py*' }
foreach ($p in $pythons) {
    W "killing python.exe PID=$($p.ProcessId)"
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch { W "  failed: $_" }
}
Start-Sleep -Seconds 8

# ----- 4) confirm latest.pdparams is up to date -----
$latest = "E:\OCR_Project\PaddleOCR\output\det_v0\latest.pdparams"
if (Test-Path $latest) {
    $lts = (Get-Item $latest).LastWriteTime
    W "latest.pdparams timestamp: $lts (will resume from here)"
} else {
    W "WARNING: latest.pdparams missing. fresh start will happen."
}

# clear old sentinel since new session is starting
Remove-Item "E:\OCR_Project\artifacts\det_training\SAFE_TO_POWER_OFF.txt" -ErrorAction SilentlyContinue

# ----- 5) launch new watchdog -----
$args = @(
    '-NoProfile','-ExecutionPolicy','Bypass','-Command',
    "& 'E:\OCR_Project\scripts\run_det_session.ps1' -DeadlineTime '$NewDeadline' -HmeanTarget $HmeanTarget -PatienceEvals $PatienceEvals"
)
$proc = Start-Process powershell.exe -ArgumentList $args -WindowStyle Minimized -PassThru
W "new watchdog launched: PID=$($proc.Id)"
W "extend_det_deadline DONE"
