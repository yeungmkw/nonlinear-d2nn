$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

Set-Location "C:\Users\Jiangqianxian\source\repos\d2nn"

$root = "C:\Users\Jiangqianxian\source\repos\d2nn\reports\post_rs_compressed\2026-04-10"
$logs = Join-Path $root "logs"
$queueLog = Join-Path $root "queue.log"
$saveDir = "checkpoints_proxy"

New-Item -ItemType Directory -Force -Path $root, $logs, $saveDir | Out-Null

$queue = @(
    [pscustomobject]@{
        Name = "fashion_mnist_phase_only_5ep_size100_seed42_post_rs_fft_proxy"
        Args = @(
            "--task", "classification",
            "--dataset", "fashion-mnist",
            "--epochs", "5",
            "--size", "100",
            "--layers", "5",
            "--batch-size", "64",
            "--seed", "42",
            "--experiment-stage", "post-rs-compressed-proxy",
            "--run-name", "fashion_mnist_phase_only_5ep_size100_seed42_post_rs_fft_proxy",
            "--rs-backend", "fft",
            "--allow-tf32",
            "--num-workers", "2",
            "--pin-memory",
            "--save-dir", $saveDir
        )
    },
    [pscustomobject]@{
        Name = "fashion_mnist_incoherent_back_5ep_size100_seed42_post_rs_fft_proxy"
        Args = @(
            "--task", "classification",
            "--dataset", "fashion-mnist",
            "--epochs", "5",
            "--size", "100",
            "--layers", "5",
            "--batch-size", "64",
            "--seed", "42",
            "--experiment-stage", "post-rs-compressed-proxy",
            "--run-name", "fashion_mnist_incoherent_back_5ep_size100_seed42_post_rs_fft_proxy",
            "--activation-type", "incoherent_intensity",
            "--activation-placement", "back",
            "--activation-preset", "balanced",
            "--rs-backend", "fft",
            "--allow-tf32",
            "--num-workers", "2",
            "--pin-memory",
            "--save-dir", $saveDir
        )
    }
)

function Write-QueueLog {
    param([string]$Line)
    $Line | Out-File -FilePath $queueLog -Encoding utf8 -Append
}

Write-QueueLog ("[QUEUE START] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

foreach ($job in $queue) {
    $logPath = Join-Path $logs ($job.Name + ".log")
    $commandPreview = "uv run python train.py " + ($job.Args -join " ")
    Write-QueueLog ("[START] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $job.Name)
    Write-QueueLog ("[COMMAND] " + $commandPreview)

    try {
        & uv run python train.py @($job.Args) *> $logPath
        $exitCode = $LASTEXITCODE
    } catch {
        $_ | Out-File -FilePath $logPath -Encoding utf8 -Append
        $exitCode = 1
    }

    Write-QueueLog ("[EXIT] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " " + $job.Name + " exit=" + $exitCode)
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

Write-QueueLog ("[QUEUE DONE] " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
