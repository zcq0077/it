param(
    [int]$WaitForProcessId = 0,
    [string]$PythonPath = 'D:\SoftWare\Environment_M\envs\lucky\python.exe',
    [int]$TargetTotal = 350
)

$ErrorActionPreference = 'Stop'
try {
    (Get-Process -Id $PID).PriorityClass = 'BelowNormal'
} catch {
    Write-Warning "Could not lower pipeline process priority: $($_.Exception.Message)"
}
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TempRoot = Join-Path $ProjectRoot '.tmp'
$OutputRoot = Join-Path $ProjectRoot 'dataset\dma_raw_2023_06_07_08_plus_09_10_oa_s00'
$LogPath = Join-Path $OutputRoot 'pipeline.log'
$Prefix = "dma_2023_06_07_08_plus_09_10_oa_s00_target$TargetTotal"

New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:TEMP = $TempRoot
$env:TMP = $TempRoot

function Write-PipelineLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    $line | Tee-Object -FilePath $LogPath -Append
}

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    Write-PipelineLog "START $Name"
    # Python/Transformers writes harmless progress and load reports to stderr.
    # Keep them in the log without letting PowerShell's strict mode turn them
    # into terminating NativeCommandError records.
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $PythonPath @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $PythonExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($PythonExitCode -ne 0) {
        throw "$Name failed with exit code $PythonExitCode."
    }
    Write-PipelineLog "DONE $Name"
}

if ($WaitForProcessId -gt 0) {
    Write-PipelineLog "Waiting for training process $WaitForProcessId to finish."
    while (Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-PipelineLog "Training process $WaitForProcessId finished."
}

foreach ($Month in @('09', '10')) {
    $MonthRoot = Join-Path $ProjectRoot "dataset\dma_raw_2023_$Month"
    $ZipPath = "D:\AIS\2023_06_09\aisdk-2023-$Month.zip"
    $AllPath = Join-Path $MonthRoot 'dma_itentformer_all.pkl'
    $DatabasePath = Join-Path $MonthRoot 'dma_filtered.sqlite'
    $RoutePath = Join-Path $MonthRoot 'dma_itentformer_ti_4class_revnorm_lasthit_fromall.pkl'
    $RouteLabelsPath = Join-Path $MonthRoot 'dma_route_labels_ti_4class_revnorm_lasthit_fromall.json'

    New-Item -ItemType Directory -Force -Path $MonthRoot | Out-Null
    if (-not (Test-Path -LiteralPath $AllPath)) {
        $Stage = if (Test-Path -LiteralPath $DatabasePath) { 'build' } else { 'all' }
        Invoke-PythonStep "preprocess 2023-$Month" @(
            'utils\preprocess_dma_zip.py',
            '--input_zip', $ZipPath,
            '--output_dir', $MonthRoot,
            '--stage', $Stage,
            '--output_name', 'dma_itentformer_all.pkl',
            '--report_name', 'dma_preprocess_all_report.json'
        )
    } else {
        Write-PipelineLog "SKIP preprocess 2023-$Month; output already exists."
    }

    if (-not (Test-Path -LiteralPath $RoutePath)) {
        Invoke-PythonStep "classify routes 2023-$Month" @(
            'utils\classify_itentformer_routes.py',
            '--data_path', $AllPath,
            '--output_path', $RoutePath,
            '--output_labels_path', $RouteLabelsPath,
            '--report_path', (Join-Path $MonthRoot 'dma_classify_ti_4class_revnorm_lasthit_fromall_report.json'),
            '--source_name', "2023-$Month",
            '--endpoint_policy', 'last_hit',
            '--include_direct_c_route',
            '--reverse_mode', 'normalize'
        )
    } else {
        Write-PipelineLog "SKIP classify 2023-$Month; output already exists."
    }

    Invoke-PythonStep "quality 2023-$Month" @(
        'utils\check_dma_quality.py',
        '--data_path', $RoutePath,
        '--report_path', (Join-Path $MonthRoot 'dma_quality_ti_4class_revnorm_lasthit_fromall_report.json')
    )
}

$AugmentedData = Join-Path $OutputRoot "$Prefix.pkl"
$AugmentedRouteLabels = Join-Path $OutputRoot "${Prefix}_route_labels.json"
$AugmentedSubrouteLabels = Join-Path $OutputRoot "${Prefix}_subroute_labels.json"
$SupplementData = Join-Path $OutputRoot "${Prefix}_supplement.pkl"
$SupplementRouteLabels = Join-Path $OutputRoot "${Prefix}_supplement_route_labels.json"
$SupplementContext = Join-Path $OutputRoot "${Prefix}_supplement_context.pkl"
$MergedContext = Join-Path $OutputRoot "${Prefix}_voyage_context.pkl"
$SemanticPath = Join-Path $OutputRoot "${Prefix}_qwen_semantic.pkl"

Invoke-PythonStep 'assign OA_S00 supplement' @(
    'utils\augment_oa_s00.py',
    '--base_data_path', 'dataset\dma_raw_2023_06_07_08\dma_itentformer_ti_4class_revnorm_lasthit.pkl',
    '--base_route_labels_path', 'dataset\dma_raw_2023_06_07_08\dma_route_labels_ti_4class_revnorm_lasthit.json',
    '--base_subroute_labels_path', 'dataset\dma_raw_2023_06_07_08\dma_subroutes_ti_4class_compact6_v1_labels.json',
    '--base_split_manifest', 'dataset\dma_raw_2023_06_07_08\dma_fixed_split_70_10_20_seed42.json',
    '--candidate_data_paths', 'dataset\dma_raw_2023_09\dma_itentformer_ti_4class_revnorm_lasthit_fromall.pkl,dataset\dma_raw_2023_10\dma_itentformer_ti_4class_revnorm_lasthit_fromall.pkl',
    '--candidate_route_labels_paths', 'dataset\dma_raw_2023_09\dma_route_labels_ti_4class_revnorm_lasthit_fromall.json,dataset\dma_raw_2023_10\dma_route_labels_ti_4class_revnorm_lasthit_fromall.json',
    '--output_dir', $OutputRoot,
    '--prefix', $Prefix,
    '--target_total', "$TargetTotal",
    '--max_per_mmsi', '2',
    '--force'
)

if (-not (Test-Path -LiteralPath $SupplementContext)) {
    Invoke-PythonStep 'build supplement voyage context' @(
        'utils\build_dma_voyage_context.py',
        '--input_zip', 'D:\AIS\2023_06_09\aisdk-2023-09.zip',
        '--input_zip', 'D:\AIS\2023_06_09\aisdk-2023-10.zip',
        '--data_path', $SupplementData,
        '--labels_path', $SupplementRouteLabels,
        '--output_path', $SupplementContext,
        '--force'
    )
} else {
    Write-PipelineLog 'SKIP supplement voyage context; output already exists.'
}

if (-not (Test-Path -LiteralPath $MergedContext)) {
    Invoke-PythonStep 'merge voyage contexts' @(
        'utils\merge_voyage_context.py',
        '--input_paths', 'dataset\dma_raw_2023_06_07_08\dma_voyage_context_2023_06_07_08.pkl', $SupplementContext,
        '--data_path', $AugmentedData,
        '--labels_path', $AugmentedRouteLabels,
        '--output_path', $MergedContext,
        '--force'
    )
} else {
    Write-PipelineLog 'SKIP merge voyage contexts; output already exists.'
}

if (-not (Test-Path -LiteralPath $SemanticPath)) {
    Invoke-PythonStep 'build merged Qwen semantic teacher' @(
        'utils\build_qwen_semantic_teacher.py',
        '--context_path', $MergedContext,
        '--model_path', 'D:\Jason1982\wsl\Models\Qwen3-1.7B',
        '--output_path', $SemanticPath,
        '--batch_size', '16',
        '--max_length', '128',
        '--device', 'auto',
        '--force'
    )
} else {
    Write-PipelineLog 'SKIP merged Qwen semantic teacher; output already exists.'
}

Invoke-PythonStep 'quality augmented dataset' @(
    'utils\check_dma_quality.py',
    '--data_path', $AugmentedData,
    '--report_path', (Join-Path $OutputRoot "${Prefix}_quality_report.json")
)

Write-PipelineLog "PIPELINE COMPLETE: $AugmentedData"
