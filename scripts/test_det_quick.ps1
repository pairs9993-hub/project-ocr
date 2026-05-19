# scripts/test_det_quick.ps1
# 학습된 DB det 모델 빠른 시각 테스트
#   1) best_accuracy.pdparams -> inference dir export
#   2) predict_det.py 로 sample 이미지에 박스 그려서 inference_results/ 에 저장
#
# 사용:
#   powershell -ExecutionPolicy Bypass -File scripts\test_det_quick.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\test_det_quick.ps1 -SampleCount 20

param(
    [string]$ProjectRoot = "E:\OCR_Project",
    [string]$Config      = "configs/det/finetune_det_v0.yml",
    [string]$Checkpoint  = "./output/det_v0/best_accuracy",
    [string]$InferenceDir = "./output/det_v0/inference",
    [string]$ImageSrcDir = "E:\OCR_Project\dataset\test\images",
    [int]$SampleCount    = 10,
    [string]$VizOutDir   = "E:\OCR_Project\artifacts\det_v0_viz"
)

$ErrorActionPreference = "Stop"

# ----- CUDA PATH prepend (run_det_session.ps1 와 동일 패턴) -----
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
Write-Host "[test_det] CUDA PATH prepend: $($pathPrepend.Count) dirs" -ForegroundColor Cyan

Push-Location (Join-Path $ProjectRoot "PaddleOCR")
try {
    # ----- 1) export inference dir -----
    Write-Host "`n[test_det] (1/3) export_model -> $InferenceDir" -ForegroundColor Green
    & $python tools/export_model.py `
        -c $Config `
        -o "Global.pretrained_model=$Checkpoint" "Global.save_inference_dir=$InferenceDir"
    if ($LASTEXITCODE -ne 0) { throw "export_model failed (exit=$LASTEXITCODE)" }

    # ----- 2) 샘플 이미지 임시 폴더 구성 -----
    $tmpSampleDir = Join-Path $env:TEMP ("det_test_sample_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
    New-Item -ItemType Directory -Path $tmpSampleDir -Force | Out-Null

    $samples = Get-ChildItem -Path $ImageSrcDir -File -Filter "*.png" |
               Get-Random -Count $SampleCount
    foreach ($f in $samples) {
        Copy-Item $f.FullName -Destination $tmpSampleDir
    }
    Write-Host "[test_det] (2/3) sampled $($samples.Count) images -> $tmpSampleDir" -ForegroundColor Green

    # ----- 3) predict_det.py 실행 -----
    Write-Host "`n[test_det] (3/3) predict_det" -ForegroundColor Green
    if (Test-Path $VizOutDir) { Remove-Item $VizOutDir -Recurse -Force }
    New-Item -ItemType Directory -Path $VizOutDir -Force | Out-Null

    # predict_det.py 는 결과를 ./inference_results 에 저장
    & $python tools/infer/predict_det.py `
        --image_dir=$tmpSampleDir `
        --det_model_dir=$InferenceDir `
        --det_limit_type=min `
        --det_limit_side_len=640 `
        --det_db_box_thresh=0.5 `
        --det_db_unclip_ratio=1.5 `
        --use_gpu=True
    if ($LASTEXITCODE -ne 0) { throw "predict_det failed (exit=$LASTEXITCODE)" }

    # 결과 복사
    $defaultOut = Join-Path (Get-Location) "inference_results"
    if (Test-Path $defaultOut) {
        Copy-Item (Join-Path $defaultOut "*") -Destination $VizOutDir -Recurse -Force
        Write-Host "`n[test_det] DONE. visualizations -> $VizOutDir" -ForegroundColor Yellow
        Get-ChildItem $VizOutDir | Select-Object -First 5 | Format-Table Name, Length
    } else {
        Write-Host "[test_det] WARNING: inference_results not found" -ForegroundColor Red
    }
}
finally {
    Pop-Location
}
