# run_det_session.ps1
# -------------------------------------------------------------------
# PaddleOCR det fine-tune 무인 학습 러너.
#
# 동작:
#   1. 절전/모니터/덮개 비활성화 (학습 도중 자동 절전 차단)
#   2. CUDA DLL 경로를 PATH 에 prepend
#   3. output/det_v0/latest.* 가 있으면 자동 resume
#   4. tools/train.py 백그라운드 실행, stdout/stderr 를 로그 파일에 기록
#   5. 워치독 루프:
#        a) -DeadlineTime 도달하면 graceful stop
#        b) 매 평가 후 로그에서 hmean 파싱 →
#             - hmean >= -HmeanTarget 이면 즉시 stop ("target hit")
#             - 연속 -PatienceEvals 회 동안 best 갱신 없으면 stop ("no improvement")
#   6. 종료 후 절전 복원, SAFE_TO_POWER_OFF.txt 작성, SESSION_STATUS 갱신
#
# 사용 예:
#   pwsh -File scripts/run_det_session.ps1 -DeadlineTime "2026-05-17 12:00"
#   pwsh -File scripts/run_det_session.ps1 -DeadlineTime "2026-05-17 12:00" `
#        -HmeanTarget 0.95 -PatienceEvals 2
# -------------------------------------------------------------------

param(
    [string]$DeadlineTime = "",          # 절대 종료 시각 (예: "2026-05-17 12:00")
    [double]$DurationHours = 13.0,       # DeadlineTime 미지정 시 사용
    [double]$HmeanTarget = 0.95,         # 이 hmean 이상이면 즉시 중단
    [int]$PatienceEvals = 2,             # 연속 N회 best 갱신 없으면 중단
    [double]$ImprovementEps = 0.001,     # best 갱신 최소폭
    [string]$Config = "configs/det/finetune_det_v0.yml",
    [string]$OutputDir = "output/det_v0",
    [string]$ProjectRoot = "E:\OCR_Project",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$paddleDir  = Join-Path $ProjectRoot "PaddleOCR"
$logsDir    = Join-Path $ProjectRoot "artifacts/det_training/logs"
$statusFile = Join-Path $ProjectRoot "artifacts/det_training/SESSION_STATUS.txt"
$safeFile   = Join-Path $ProjectRoot "artifacts/det_training/SAFE_TO_POWER_OFF.txt"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
Remove-Item $safeFile -ErrorAction SilentlyContinue

$stamp      = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionTag = "det_$stamp"
$logFile    = Join-Path $logsDir "$sessionTag.log"
$errFile    = "$logFile.err"

# ----- resume? -----
$ckptDir = Join-Path $paddleDir $OutputDir
$latestPdparams = Join-Path $ckptDir "latest.pdparams"
$resumeFlag = @()
if (Test-Path $latestPdparams) {
    $ckptStem = (Join-Path $ckptDir "latest").Replace("\","/")
    $resumeFlag = @("-o", "Global.checkpoints=$ckptStem")
    Write-Host "[det] RESUMING from $ckptStem" -ForegroundColor Cyan
} else {
    Write-Host "[det] FRESH START (no latest checkpoint)" -ForegroundColor Yellow
}

# ----- 절전 비활성화 -----
Write-Host "[det] disabling sleep / hibernate / monitor / lid"
powercfg /change standby-timeout-ac 0      | Out-Null
powercfg /change hibernate-timeout-ac 0    | Out-Null
powercfg /change monitor-timeout-ac 0      | Out-Null
powercfg -setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0 | Out-Null
powercfg -setactive SCHEME_CURRENT | Out-Null

# ----- CUDA PATH prepend -----
$python      = Join-Path $ProjectRoot ".venv_train/Scripts/python.exe"
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
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
Write-Host "[det] PATH prepended with $($pathPrepend.Count) dirs"

# ----- deadline 결정 -----
$startTime = Get-Date
if ($DeadlineTime -ne "") {
    $deadline = [DateTime]::Parse($DeadlineTime)
} else {
    $deadline = $startTime.AddHours($DurationHours)
}

$header = @"
=== DET SESSION ===
start          : $startTime
deadline       : $deadline
config         : $Config
output_dir     : $OutputDir
hmean_target   : $HmeanTarget
patience_evals : $PatienceEvals
improvement_eps: $ImprovementEps
log            : $logFile
"@
$header | Tee-Object -FilePath $statusFile -Append | Out-Null
Write-Host $header

# ----- 학습 프로세스 시작 -----
$trainArgs = @("tools/train.py", "-c", $Config) + $resumeFlag
Push-Location $paddleDir
$proc = Start-Process -FilePath $python `
    -ArgumentList $trainArgs `
    -PassThru -NoNewWindow `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile
Pop-Location
Write-Host "[det] PID=$($proc.Id) launched."

# ----- 워치독 -----
$bestHmean      = -1.0
$lastEvalSeen   = -1
$evalsSinceBest = 0
$stopReason     = "natural_exit"
$evalRegex      = [regex]'cur metric,[^\n]*?hmean:\s*([0-9.]+)'
# fallback pattern (older PaddleOCR):
$evalRegex2     = [regex]'hmean\s*:\s*([0-9.]+)'

function Get-LatestHmeans {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return @() }
    try {
        $content = Get-Content $Path -Raw -ErrorAction Stop
    } catch { return @() }
    $hits = @()
    foreach ($m in $evalRegex.Matches($content)) {
        $hits += [double]$m.Groups[1].Value
    }
    if ($hits.Count -eq 0) {
        foreach ($m in $evalRegex2.Matches($content)) {
            $hits += [double]$m.Groups[1].Value
        }
    }
    return ,$hits
}

while (-not $proc.HasExited) {
    Start-Sleep -Seconds $PollSeconds

    # 1) deadline
    if ((Get-Date) -gt $deadline) {
        $stopReason = "deadline_reached"
        Write-Host "[det] deadline reached. stopping." -ForegroundColor Yellow
        break
    }

    # 2) hmean parse
    $hmeans = Get-LatestHmeans -Path $logFile
    if ($hmeans.Count -gt $lastEvalSeen + 1) {
        # new evals appeared
        for ($i = $lastEvalSeen + 1; $i -lt $hmeans.Count; $i++) {
            $h = $hmeans[$i]
            $isBest = $false
            if ($h -ge $bestHmean + $ImprovementEps) {
                $bestHmean = $h
                $evalsSinceBest = 0
                $isBest = $true
            } else {
                $evalsSinceBest += 1
            }
            $msg = "[det] eval #{0}: hmean={1:N4} best={2:N4} since_best={3} {4}" -f `
                   ($i + 1), $h, $bestHmean, $evalsSinceBest, ($(if($isBest){"<-- NEW BEST"}else{""}))
            Write-Host $msg -ForegroundColor Cyan
            $msg | Out-File -FilePath $statusFile -Append -Encoding utf8
        }
        $lastEvalSeen = $hmeans.Count - 1

        # target?
        if ($bestHmean -ge $HmeanTarget) {
            $stopReason = "hmean_target_hit (best=$bestHmean >= $HmeanTarget)"
            Write-Host "[det] $stopReason" -ForegroundColor Green
            break
        }
        # patience?
        if ($evalsSinceBest -ge $PatienceEvals) {
            $stopReason = "no_improvement ($evalsSinceBest evals without >= $ImprovementEps)"
            Write-Host "[det] $stopReason" -ForegroundColor Yellow
            break
        }
    }
}

# ----- 종료 처리 -----
if (-not $proc.HasExited) {
    Write-Host "[det] stopping train.py (reason=$stopReason)"
    try { Stop-Process -Id $proc.Id -Force -ErrorAction Stop } catch { Write-Host "Stop-Process failed: $_" }
    Start-Sleep -Seconds 5
    if (-not $proc.HasExited) { try { $proc.Kill() } catch {} }
}

$endTime = Get-Date
$durationMin = [math]::Round(($endTime - $startTime).TotalMinutes, 1)

# 절전 복원
powercfg /change standby-timeout-ac 30   | Out-Null
powercfg /change hibernate-timeout-ac 0  | Out-Null
powercfg /change monitor-timeout-ac 15   | Out-Null

$summary = @"

=== DET SESSION SUMMARY ===
start        : $startTime
end          : $endTime
duration_min : $durationMin
stop_reason  : $stopReason
exit_code    : $($proc.ExitCode)
best_hmean   : $bestHmean
evals_seen   : $($lastEvalSeen + 1)
log          : $logFile
checkpoint   : $latestPdparams (exists=$(Test-Path $latestPdparams))

NEXT:
  - 로그 마지막 200줄: Get-Content "$logFile" -Tail 200
  - 재개하려면 동일 명령 다시 실행 (latest.pdparams 에서 resume).
"@
$summary | Tee-Object -FilePath $statusFile -Append
Write-Host $summary -ForegroundColor Green

"SAFE_TO_POWER_OFF since $endTime (best_hmean=$bestHmean reason=$stopReason)" |
    Out-File -FilePath $safeFile -Encoding utf8
Write-Host "[det] DONE - safe to power off." -ForegroundColor Green
