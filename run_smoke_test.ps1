param(
    [int]$Epochs = 1,
    [int]$WindowStride = 20,
    [string]$RunName = "smoke",
    [string]$CondaEnv = "lucky"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TEMP = Join-Path $repoRoot ".tmp"
$env:TMP = Join-Path $repoRoot ".tmp"

$smokeResults = Join-Path $repoRoot ".tmp\smoke_results"
$smokeModels = Join-Path $repoRoot ".tmp\smoke_models"
New-Item -ItemType Directory -Force -Path $smokeResults, $smokeModels | Out-Null

conda run --no-capture-output -n $CondaEnv python (Join-Path $repoRoot "iTentformer.py") `
    --run_folds 1 `
    --epochs $Epochs `
    --window_stride $WindowStride `
    --plot_count 0 `
    --results_dir $smokeResults `
    --model_dir $smokeModels `
    --model_prefix "smoke_tmp" `
    --run_name $RunName
