# run_pipeline_session1.ps1
# -------------------------------------------------------------------
# 세션 1: 크롭 생성 -> 학습. 전체 wall time을 -TotalHours 안으로 제한.
# -------------------------------------------------------------------

param(
    [double]$TotalHours = 6.0,
    [double]$SafetyMarginMin = 30,
    [string]$DeadlineTime = "",   # absolute deadline, e.g. "2026-05-12 05:00". overrides TotalHours.
    [string]$ProjectRoot = "E:\OCR_Project"
)

Set-Location $ProjectRoot
$startTime = Get-Date
if ($DeadlineTime -ne "") {
    $deadline = [DateTime]::Parse($DeadlineTime)
} else {
    $deadline = $startTime.AddHours($TotalHours).AddMinutes(-$SafetyMarginMin)
}
Write-Host "[pipeline] start    = $startTime"
Write-Host "[pipeline] deadline = $deadline"

$recTrain = Join-Path $ProjectRoot "data/rec_dataset_full/rec_train.txt"
$recVal   = Join-Path $ProjectRoot "data/rec_dataset_full/rec_val.txt"
$prepLog  = Join-Path $ProjectRoot "artifacts/training_full/prep_crops.log"
$safeFile = Join-Path $ProjectRoot "artifacts/training_full/SAFE_TO_POWER_OFF.txt"

Write-Host "[pipeline] waiting for crop generation..."
while ($true) {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[pipeline] deadline reached during crop gen - aborting" -ForegroundColor Red
        "TIMEOUT_DURING_CROP_GEN at $(Get-Date)" | Out-File -FilePath $safeFile -Encoding utf8
        return
    }
    if ((Test-Path $recTrain) -and (Test-Path $recVal) -and (Test-Path $prepLog)) {
        $tail = Get-Content $prepLog -Tail 5 -ErrorAction SilentlyContinue
        if ($tail -match "output root") {
            Write-Host "[pipeline] crop generation complete."
            break
        }
    }
    $elapsedMin = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
    $remMin     = [math]::Round(($deadline - (Get-Date)).TotalMinutes, 1)
    Write-Host "[pipeline] crops in progress... elapsed=$elapsedMin min, deadline in $remMin min"
    Start-Sleep -Seconds 120
}

$nTrain = (Get-Content $recTrain | Measure-Object -Line).Lines
$nVal   = (Get-Content $recVal   | Measure-Object -Line).Lines
Write-Host "[pipeline] rec_train=$nTrain  rec_val=$nVal"

$remainingHours = ($deadline - (Get-Date)).TotalHours
Write-Host "[pipeline] remaining for training = $([math]::Round($remainingHours,2)) h"
if ($remainingHours -lt 0.5) {
    Write-Host "[pipeline] not enough time for training; will let next session do training" -ForegroundColor Yellow
    "SAFE_TO_POWER_OFF (crops only, no training this session) at $(Get-Date)" |
        Out-File -FilePath $safeFile -Encoding utf8
    return
}

$trainScript = Join-Path $ProjectRoot "scripts/run_training_session.ps1"
$deadlineStr = $deadline.ToString("yyyy-MM-dd HH:mm:ss")
& powershell.exe -ExecutionPolicy Bypass -NoProfile -File $trainScript -Session 1 -DeadlineTime $deadlineStr
