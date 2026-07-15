param(
    [int]$Epochs = 1,
    [int]$WindowStride = 20,
    [string]$RunName = "smoke",
    [string]$CondaEnv = "lucky",
    [switch]$WithQwen
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TEMP = Join-Path $repoRoot ".tmp"
$env:TMP = Join-Path $repoRoot ".tmp"

$smokeResults = Join-Path $repoRoot ".tmp\smoke_results"
$smokeModels = Join-Path $repoRoot ".tmp\smoke_models"
New-Item -ItemType Directory -Force -Path $smokeResults, $smokeModels | Out-Null

$arguments = @(
    "run", "--no-capture-output", "-n", $CondaEnv,
    "python", (Join-Path $repoRoot "iTentformer.py"),
    "--run_folds", "1",
    "--epochs", "$Epochs",
    "--window_stride", "$WindowStride",
    "--plot_count", "0",
    "--results_dir", $smokeResults,
    "--model_dir", $smokeModels,
    "--model_prefix", "smoke_tmp",
    "--run_name", $RunName
)

if ($WithQwen) {
    $arguments += @(
        "--qwen_reranker_epochs", "1",
        "--qwen_reranker_batch_size", "1",
        "--qwen_train_max_windows", "8",
        "--qwen_valid_max_windows", "4"
    )
} else {
    $arguments += "--no-use_qwen_reranker"
}

& conda @arguments
