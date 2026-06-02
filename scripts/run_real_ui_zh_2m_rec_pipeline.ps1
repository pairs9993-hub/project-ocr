# Generate zh-CN/zh-TW/zh-HK synthetic data, prepare rec crops, and start no-deadline training.

param(
    [string]$ProjectRoot = "E:\OCR_Project",
    [int]$TotalCount = 2000000,
    [int]$ChunkSize = 125000,
    [int]$ValCount = 20000,
    [int]$Seed = 20260601,
    [string[]]$Languages = @("zh_cn", "zh_tw", "zh_hk"),
    [string]$PythonExe = "",
    [int]$ShardSize = 10000,
    [int]$TrainingSession = 1
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

if ($PythonExe -eq "") {
    $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}
if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$generatedRoot = "generated_2000000_real_ui_zh"
$generatedPath = Join-Path $ProjectRoot $generatedRoot
$chunksRoot = Join-Path $generatedPath "chunks"
$trainManifest = "dataset\train_manifest_real_ui_zh_2m.jsonl"
$valRoot = "artifacts\real_ui_zh_val_20000"
$valManifest = "artifacts\real_ui_zh_val_20000_manifest.jsonl"
$charDict = "data\dict\ppocr_keys_real_ui_zh_2m.txt"
$coverage = "artifacts\charset\coverage_real_ui_zh_2m.json"
$recDir = "data\rec_dataset_real_ui_zh_2m"
$readyMarker = Join-Path $ProjectRoot (Join-Path $recDir "REC_DATA_READY.txt")
$statusDir = Join-Path $ProjectRoot "artifacts\real_ui_zh_2m_training"
$statusFile = Join-Path $statusDir "PIPELINE_STATUS.txt"
New-Item -ItemType Directory -Force -Path $chunksRoot, $statusDir | Out-Null

function Write-Status {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    $line | Tee-Object -FilePath $statusFile -Append
}

function Count-Lines {
    param([string]$Path)
    if (!(Test-Path $Path)) { return 0 }
    return (Get-Content $Path | Measure-Object -Line).Lines
}

function Assert-LastExit {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Test-TrainingProcessActive {
    $needle = "finetune_real_ui_zh_2m.yml"
    $matches = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like "*$needle*" })
    return $matches.Count -gt 0
}

Write-Status "pipeline start: languages=$($Languages -join ',') total=$TotalCount val=$ValCount"

for ($start = 0; $start -lt $TotalCount; $start += $ChunkSize) {
    $end = [Math]::Min($start + $ChunkSize - 1, $TotalCount - 1)
    $thisCount = $end - $start + 1
    $chunkName = "chunk_{0:D7}_{1:D7}" -f $start, $end
    $outDir = Join-Path $chunksRoot $chunkName
    $labelsPath = Join-Path $outDir "labels.jsonl"
    $existingRows = Count-Lines $labelsPath

    if ($existingRows -eq $thisCount) {
        Write-Status "skip chunk ${chunkName}: $existingRows rows"
        continue
    }
    if ($existingRows -gt 0 -or (Test-Path $outDir)) {
        Write-Status "regenerate chunk ${chunkName}: existing rows=$existingRows expected=$thisCount"
        Remove-Item -Recurse -Force $outDir -ErrorAction SilentlyContinue
    }

    Write-Status "generate chunk $chunkName start=$start count=$thisCount"
    $generatorArgs = @(
        "scripts\generate_real_ui_synth.py",
        "--output-dir", $outDir,
        "--count", $TotalCount,
        "--start-index", $start,
        "--chunk-count", $thisCount,
        "--seed", $Seed,
        "--languages"
    ) + $Languages
    & $PythonExe @generatorArgs
    Assert-LastExit "generate chunk $chunkName"
}

if ((Count-Lines (Join-Path $ProjectRoot $trainManifest)) -ne $TotalCount) {
    Write-Status "build train manifest: $trainManifest"
    & $PythonExe "scripts\build_manifest_from_labels.py" --project-root $ProjectRoot --source "$generatedRoot/chunks/*/labels.jsonl" --output $trainManifest
    Assert-LastExit "build train manifest"
} else {
    Write-Status "train manifest already ready: $TotalCount rows"
}

$valLabels = Join-Path $ProjectRoot (Join-Path $valRoot "labels.jsonl")
if ((Count-Lines $valLabels) -ne $ValCount) {
    Write-Status "generate validation set: $ValCount"
    Remove-Item -Recurse -Force (Join-Path $ProjectRoot $valRoot) -ErrorAction SilentlyContinue
    $valGeneratorArgs = @(
        "scripts\generate_real_ui_synth.py",
        "--output-dir", $valRoot,
        "--count", $ValCount,
        "--seed", ($Seed + 101),
        "--languages"
    ) + $Languages
    & $PythonExe @valGeneratorArgs
    Assert-LastExit "generate validation set"
} else {
    Write-Status "validation labels already ready: $ValCount rows"
}

if ((Count-Lines (Join-Path $ProjectRoot $valManifest)) -ne $ValCount) {
    Write-Status "build validation manifest: $valManifest"
    & $PythonExe "scripts\build_manifest_from_labels.py" --project-root $ProjectRoot --source "$valRoot\labels.jsonl" --output $valManifest
    Assert-LastExit "build validation manifest"
} else {
    Write-Status "validation manifest already ready: $ValCount rows"
}

Write-Status "build character dictionary: $charDict"
& $PythonExe "scripts\build_char_dict.py" --project-root $ProjectRoot --no-defaults --label-file $trainManifest --label-file $valManifest --output-dict $charDict --output-coverage $coverage
Assert-LastExit "build character dictionary"

if (!(Test-Path $readyMarker)) {
    Write-Status "prepare recognition crops: $recDir"
    Remove-Item -Recurse -Force (Join-Path $ProjectRoot $recDir) -ErrorAction SilentlyContinue
    & $PythonExe "scripts\prepare_rec_data.py" --project-root $ProjectRoot --train-source $trainManifest --val-source $valManifest --output-dir $recDir --char-dict $charDict --margin 4 --shard-size $ShardSize
    Assert-LastExit "prepare recognition crops"
    "ready $(Get-Date -Format o)" | Out-File -FilePath $readyMarker -Encoding utf8
} else {
    Write-Status "recognition crops already marked ready"
}

if (Test-TrainingProcessActive) {
    Write-Status "training process already active for finetune_real_ui_zh_2m.yml; not launching duplicate"
} else {
    Write-Status "launch detached no-deadline recognizer training"
    & powershell.exe -ExecutionPolicy Bypass -NoProfile -File "scripts\run_training_session.ps1" -ProjectRoot $ProjectRoot -Config "../configs/rec/finetune_real_ui_zh_2m.yml" -OutputDir "output/real_ui_zh_2m_rec" -Session $TrainingSession -NoDeadline -Detach
    Assert-LastExit "launch recognizer training"
}

Write-Status "pipeline finished setup; training output=PaddleOCR\output\real_ui_zh_2m_rec"