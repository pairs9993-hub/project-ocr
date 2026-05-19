# run_training_session.ps1
# -------------------------------------------------------------------
# 6시간 단위 PaddleOCR 학습 세션 러너 (resume-aware, auto-stop).
#
# 동작:
#   1. 슬립/모니터 절전 비활성화 (학습 중 자동 절전 차단)
#   2. output/v0_full/latest.* 가 있으면 checkpoint 자동 resume
#   3. PaddleOCR train.py 를 백그라운드로 띄우고 stdout/stderr 를 로그파일에 기록
#   4. 지정한 -DurationHours 만큼 경과하면 그 시점에 graceful kill
#      (PaddleOCR 은 매 epoch 마다 latest.pdparams 를 저장하므로 안전)
#   5. 종료 후 절전 설정 복원, 상태 요약을 status 파일에 기록
#
# 사용:
#   pwsh -File scripts/run_training_session.ps1 -Session 1 -DurationHours 5.5
#   pwsh -File scripts/run_training_session.ps1 -Session 2 -DurationHours 5.5
#   ...
#
# 세션 사이에 노트북을 꺼도 OK. 다음 세션은 같은 명령으로 -Session 번호만 올려서 재실행.
# -------------------------------------------------------------------

param(
    [int]$Session = 1,
    [double]$DurationHours = 5.5,
    [string]$DeadlineTime = "",   # absolute deadline string; overrides DurationHours
    [string]$Config = "configs/rec/finetune_v0_full.yml",
    [string]$OutputDir = "output/v0_full",
    [string]$ProjectRoot = "E:\OCR_Project"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$paddleDir = Join-Path $ProjectRoot "PaddleOCR"
$logsDir = Join-Path $ProjectRoot "artifacts/training_full/logs"
$statusFile = Join-Path $ProjectRoot "artifacts/training_full/SESSION_STATUS.txt"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionTag = "session{0:D2}_{1}" -f $Session, $stamp
$logFile = Join-Path $logsDir "$sessionTag.log"

# Resolve absolute checkpoint path (PaddleOCR output dir is relative to PaddleOCR/)
$ckptDir = Join-Path $paddleDir $OutputDir
$latestPdparams = Join-Path $ckptDir "latest.pdparams"
$resumeFlag = @()
if (Test-Path $latestPdparams) {
    $ckptStem = (Join-Path $ckptDir "latest").Replace("\","/")
    $resumeFlag = @("-o", "Global.checkpoints=$ckptStem")
    Write-Host "[session $Session] RESUMING from $ckptStem" -ForegroundColor Cyan
} else {
    Write-Host "[session $Session] FRESH START (no latest checkpoint found)" -ForegroundColor Yellow
}

# ----- 1. 절전 비활성화 -----
Write-Host "[session $Session] disabling sleep / hibernate for AC power"
powercfg /change standby-timeout-ac 0      | Out-Null
powercfg /change hibernate-timeout-ac 0    | Out-Null
powercfg /change monitor-timeout-ac 0      | Out-Null
# 노트북 덮개 닫기에도 동작 유지 (선택)
powercfg -setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 | Out-Null
powercfg -setactive SCHEME_CURRENT | Out-Null

# ----- 2. 학습 프로세스 시작 -----
$python = Join-Path $ProjectRoot ".venv_train/Scripts/python.exe"
$trainArgs = @(
    "tools/train.py",
    "-c", $Config
) + $resumeFlag

# CUDA DLL 경로를 PATH 에 prepend (venv activation 우회 대응)
$venvScripts = Join-Path $ProjectRoot ".venv_train/Scripts"
$venvLibBin  = Join-Path $ProjectRoot ".venv_train/Library/bin"
$nvidiaRoot  = Join-Path $ProjectRoot ".venv_train/Lib/site-packages/nvidia"
$cudaDllDirs = @()
if (Test-Path $nvidiaRoot) {
    $cudaDllDirs = Get-ChildItem -Path $nvidiaRoot -Directory -Recurse |
                   Where-Object { $_.Name -eq "bin" } |
                   ForEach-Object { $_.FullName }
}
$pathPrepend = @($venvScripts, $venvLibBin) + $cudaDllDirs
$env:PATH = ($pathPrepend -join ";") + ";" + $env:PATH
Write-Host "[session $Session] PATH prepended with $($pathPrepend.Count) CUDA dirs"

$startTime = Get-Date
if ($DeadlineTime -ne "") {
    $deadline = [DateTime]::Parse($DeadlineTime)
} else {
    $deadline = $startTime.AddHours($DurationHours)
}

"[session $Session] start=$startTime deadline=$deadline" | Tee-Object -FilePath $statusFile -Append
Write-Host "[session $Session] launching: $python $($trainArgs -join ' ')"
Write-Host "[session $Session] log file : $logFile"
Write-Host "[session $Session] deadline : $deadline"

# Start training as background process from PaddleOCR/ directory
Push-Location $paddleDir
$proc = Start-Process -FilePath $python `
    -ArgumentList $trainArgs `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$logFile.err"
Pop-Location

Write-Host "[session $Session] PID = $($proc.Id)"

# ----- 3. Deadline 도달 또는 자연 종료까지 대기 -----
$stopReason = "natural_exit"
while (-not $proc.HasExited) {
    if ((Get-Date) -gt $deadline) {
        Write-Host "[session $Session] deadline reached - stopping training gracefully"
        $stopReason = "deadline_reached"
        # graceful: 자식 트리까지 종료
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } catch {
            Write-Host "Stop-Process failed: $_"
        }
        break
    }
    Start-Sleep -Seconds 60
}

# 한번 더 정리
if (-not $proc.HasExited) { try { $proc.Kill() } catch {} }

$endTime = Get-Date
$durationMin = [math]::Round(($endTime - $startTime).TotalMinutes, 1)

# ----- 4. 마무리: 절전 복원 (균형으로 되돌림) -----
powercfg /change standby-timeout-ac 30   | Out-Null
powercfg /change hibernate-timeout-ac 0  | Out-Null
powercfg /change monitor-timeout-ac 15   | Out-Null

# ----- 5. 상태 요약 -----
$summary = @"
=== SESSION $Session SUMMARY ===
start       : $startTime
end         : $endTime
duration_min: $durationMin
stop_reason : $stopReason
exit_code   : $($proc.ExitCode)
log         : $logFile
checkpoint  : $latestPdparams (exists=$(Test-Path $latestPdparams))

NEXT STEPS:
  - 안전하게 전원 꺼도 됨. 다음 세션은 아래 명령 그대로 실행:
      pwsh -File scripts/run_training_session.ps1 -Session $($Session + 1) -DurationHours 5.5
  - 학습 로그 마지막 100줄 확인:
      Get-Content $logFile -Tail 100
"@

$summary | Tee-Object -FilePath $statusFile -Append
Write-Host $summary -ForegroundColor Green

# Write a "safe to shutdown" sentinel file
"SAFE_TO_POWER_OFF since $endTime (session $Session)" |
    Out-File -FilePath (Join-Path $ProjectRoot "artifacts/training_full/SAFE_TO_POWER_OFF.txt") -Encoding utf8
Write-Host "`n[session $Session] DONE - safe to power off." -ForegroundColor Green
