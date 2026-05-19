# chain_next_session.ps1
# -----------------------------------------------------------
# 현재 세션이 SAFE_TO_POWER_OFF.txt 를 만들면 즉시 다음 세션 자동 시작.
# -----------------------------------------------------------

param(
    [int]$NextSession,
    [Parameter(Mandatory=$true)][string]$NextDeadline,
    [string]$ProjectRoot = "E:\OCR_Project"
)

Set-Location $ProjectRoot
$safeFile = Join-Path $ProjectRoot "artifacts/training_full/SAFE_TO_POWER_OFF.txt"
$startWatch = Get-Date
$startMtime = (Get-Item $safeFile -ErrorAction SilentlyContinue).LastWriteTime

Write-Host "[chain] watching $safeFile for update..."
Write-Host "[chain] initial mtime = $startMtime"
Write-Host "[chain] next session = $NextSession, deadline = $NextDeadline"

while ($true) {
    Start-Sleep -Seconds 60
    $cur = Get-Item $safeFile -ErrorAction SilentlyContinue
    if ($cur -and $cur.LastWriteTime -ne $startMtime) {
        Write-Host "[chain] previous session finished at $($cur.LastWriteTime). Launching session $NextSession..."
        break
    }
    $elapsed = [math]::Round(((Get-Date) - $startWatch).TotalMinutes, 1)
    Write-Host "[chain] still waiting... elapsed=$elapsed min"
}

Start-Sleep -Seconds 30   # 체크포인트 flush 여유

$trainScript = Join-Path $ProjectRoot "scripts/run_training_session.ps1"
& powershell.exe -ExecutionPolicy Bypass -NoProfile -File $trainScript `
    -Session $NextSession -DeadlineTime $NextDeadline
