param(
    [int]$Epochs = 1,
    [int]$WindowStride = 20,
    [int]$CandidateWarmupEpochs = 0,
    [int]$PlotCount = 1,
    [string]$RunName = "smoke",
    [string]$CondaEnv = "lucky",
    [switch]$SplitOnly,
    [switch]$NoQwenSemantic
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:TEMP = Join-Path $repoRoot ".tmp"
$env:TMP = Join-Path $repoRoot ".tmp"

$smokeResults = Join-Path $repoRoot ".tmp\smoke_results"
$smokeModels = Join-Path $repoRoot ".tmp\smoke_models"
$smokeManifest = Join-Path $repoRoot ".tmp\smoke_fixed_split.json"
New-Item -ItemType Directory -Force -Path $env:TEMP, $smokeResults, $smokeModels | Out-Null

$arguments = @(
    "run", "--no-capture-output", "-n", $CondaEnv,
    "python", (Join-Path $repoRoot "iTentformer.py"),
    "--epochs", "$Epochs",
    "--window_stride", "$WindowStride",
    "--candidate_selector_warmup_epochs", "$CandidateWarmupEpochs",
    "--split_manifest_path", $smokeManifest,
    "--plot_count", "$PlotCount",
    "--results_dir", $smokeResults,
    "--model_dir", $smokeModels,
    "--model_prefix", "smoke_tmp",
    "--run_name", $RunName
)

if ($SplitOnly) {
    $arguments += "--split_only"
}

if ($NoQwenSemantic) {
    $arguments += "--no-use_qwen_semantic_teacher"
}

& conda @arguments
