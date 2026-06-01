# Wait for 1M synthetic generation, prepare data, then start training.

param(
    [string]$ProjectRoot = "E:\OCR_Project",
    [int]$ExpectedTrainRows = 1000000,
    [int]$SyntheticValCount = 5000,
    [int]$CompanyTestCount = 1045,
    [double]$DetDurationHours = 12.0,
    [double]$RecDurationHours = 12.0,
    [int]$PollSeconds = 300
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$manifest = Join-Path $ProjectRoot "dataset\train_manifest_real_ui_1m.jsonl"
$valManifest = Join-Path $ProjectRoot "artifacts\real_ui_synth_val_5000_manifest.jsonl"
$companyAll = Join-Path $ProjectRoot "artifacts\company_real_screens\llm_labels\gpt_all_labels.jsonl"
$companyTest = Join-Path $ProjectRoot ("artifacts\company_real_screens\llm_labels\gpt_test_{0}.jsonl" -f $CompanyTestCount)
$statusDir = Join-Path $ProjectRoot "artifacts\real_ui_1m_training"
$statusFile = Join-Path $statusDir "PIPELINE_STATUS.txt"
New-Item -ItemType Directory -Path $statusDir -Force | Out-Null

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

Write-Status "waiting for train manifest: $manifest"
while ($true) {
    $rows = Count-Lines $manifest
    if ($rows -eq $ExpectedTrainRows) {
        Write-Status "train manifest ready: $rows rows"
        break
    }
    Write-Status "train manifest rows=$rows/$ExpectedTrainRows; sleeping ${PollSeconds}s"
    Start-Sleep -Seconds $PollSeconds
}

Write-Status "creating company test set: $CompanyTestCount rows"
& $python -c "from pathlib import Path; src=Path(r'$companyAll'); out=Path(r'$companyTest'); rows=[line for line in src.read_text(encoding='utf-8').splitlines() if line.strip()]; selected=rows[:$CompanyTestCount]; out.write_text('\n'.join(selected)+'\n', encoding='utf-8', newline='\n'); print(len(selected))"

if ((Count-Lines $valManifest) -ne $SyntheticValCount) {
    Write-Status "generating synthetic validation set: $SyntheticValCount"
    & $python "scripts\generate_real_ui_synth.py" --output-dir "artifacts\real_ui_synth_val_5000" --count $SyntheticValCount --languages en fr es --seed 20260523
    & $python "scripts\build_manifest_from_labels.py" --project-root $ProjectRoot --source "artifacts\real_ui_synth_val_5000\labels.jsonl" --output "artifacts\real_ui_synth_val_5000_manifest.jsonl"
} else {
    Write-Status "synthetic validation manifest already ready"
}

Write-Status "building character dictionary"
& $python "scripts\build_char_dict.py" --project-root $ProjectRoot --no-defaults --label-file "dataset\train_manifest_real_ui_1m.jsonl" --label-file "artifacts\real_ui_synth_val_5000_manifest.jsonl" --label-file "artifacts\company_real_screens\llm_labels\gpt_all_labels.jsonl" --output-dict "data\dict\ppocr_keys_real_ui_en_fr_es.txt" --output-coverage "artifacts\charset\coverage_real_ui_en_fr_es.json"

$detTrain = Join-Path $ProjectRoot "data\det_dataset_real_ui_1m\det_train.txt"
$detVal = Join-Path $ProjectRoot "data\det_dataset_real_ui_1m\det_val.txt"
if ((Count-Lines $detTrain) -eq $ExpectedTrainRows -and (Count-Lines $detVal) -eq $SyntheticValCount) {
    Write-Status "detection labels already ready"
} else {
    Write-Status "preparing detection labels"
    & $python "scripts\prepare_det_data.py" --project-root $ProjectRoot --train-source "dataset\train_manifest_real_ui_1m.jsonl" --val-source "artifacts\real_ui_synth_val_5000_manifest.jsonl" --val-prefix "." --out-dir "data\det_dataset_real_ui_1m"
}

Write-Status "starting detector training"
& powershell.exe -ExecutionPolicy Bypass -NoProfile -File "scripts\run_det_session.ps1" -ProjectRoot $ProjectRoot -Config "configs/det/finetune_det_real_ui_1m.yml" -OutputDir "output/det_real_ui_1m" -DurationHours $DetDurationHours -HmeanTarget 0.995 -PatienceEvals 4

Write-Status "preparing recognition crops"
& $python "scripts\prepare_rec_data.py" --project-root $ProjectRoot --train-source "dataset\train_manifest_real_ui_1m.jsonl" --val-source "artifacts\real_ui_synth_val_5000_manifest.jsonl" --output-dir "data\rec_dataset_real_ui_1m" --char-dict "data\dict\ppocr_keys_real_ui_en_fr_es.txt"

Write-Status "starting recognizer training"
& powershell.exe -ExecutionPolicy Bypass -NoProfile -File "scripts\run_training_session.ps1" -ProjectRoot $ProjectRoot -Config "configs/rec/finetune_real_ui_1m.yml" -OutputDir "output/real_ui_1m_rec" -Session 1 -DurationHours $RecDurationHours

Write-Status "pipeline finished initial det+rec sessions; company test labels: $companyTest"