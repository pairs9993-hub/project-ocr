# Generate real-style washer/dryer UI synthetic data in resumable chunks.

param(
    [int]$TotalCount = 1000000,
    [int]$ChunkSize = 125000,
    [string]$OutputRoot = "generated_1000000_real_ui_en_fr_es",
    [string]$ProjectRoot = "E:\OCR_Project",
    [int]$Seed = 20260520,
    [string[]]$Languages = @("en", "fr", "es"),
    [string]$PythonExe = "",
    [string]$ManifestOut = "dataset\train_manifest_real_ui_1m.jsonl",
    [string]$CharDictOut = "data\dict\ppocr_keys_real_ui_en_fr_es.txt",
    [string]$CoverageOut = "artifacts\charset\coverage_real_ui_en_fr_es.json",
    [switch]$BuildManifest
)

Set-Location $ProjectRoot

if ($PythonExe -eq "") {
    $PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
}

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$chunksRoot = Join-Path $ProjectRoot (Join-Path $OutputRoot "chunks")
New-Item -ItemType Directory -Force -Path $chunksRoot | Out-Null

for ($start = 0; $start -lt $TotalCount; $start += $ChunkSize) {
    $end = [Math]::Min($start + $ChunkSize - 1, $TotalCount - 1)
    $thisCount = $end - $start + 1
    $chunkName = "chunk_{0:D6}_{1:D6}" -f $start, $end
    $outDir = Join-Path $chunksRoot $chunkName
    $labelsPath = Join-Path $outDir "labels.jsonl"

    if (Test-Path $labelsPath) {
        $existingRows = (Get-Content $labelsPath | Measure-Object -Line).Lines
        if ($existingRows -eq $thisCount) {
            Write-Host "[skip] $chunkName already has $existingRows rows"
            continue
        }
        Write-Host "[resume-overwrite] $chunkName has $existingRows rows; regenerating chunk"
    }

    Write-Host "[generate] $chunkName start=$start count=$thisCount"
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
    if ($LASTEXITCODE -ne 0) {
        throw "generation failed for $chunkName"
    }
}

if ($BuildManifest) {
    $sourcePattern = "$OutputRoot/chunks/*/labels.jsonl"

    Write-Host "[manifest] $ManifestOut"
    & $PythonExe "scripts\build_manifest_from_labels.py" --project-root $ProjectRoot --source $sourcePattern --output $ManifestOut
    if ($LASTEXITCODE -ne 0) {
        throw "manifest build failed"
    }

    Write-Host "[char-dict] $CharDictOut"
    & $PythonExe "scripts\build_char_dict.py" --project-root $ProjectRoot --no-defaults --label-file $ManifestOut --label-file "artifacts\company_real_screens\llm_labels\gpt_all_labels.jsonl" --output-dict $CharDictOut --output-coverage $CoverageOut
    if ($LASTEXITCODE -ne 0) {
        throw "char dictionary build failed"
    }
}

Write-Host "[done] generated synthetic data under $OutputRoot"