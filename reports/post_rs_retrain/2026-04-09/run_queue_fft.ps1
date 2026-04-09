$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
Set-Location 'C:\Users\Jiangqianxian\source\repos\d2nn'
$logs = 'C:\Users\Jiangqianxian\source\repos\d2nn\reports\post_rs_retrain\2026-04-09\logs_fft'
$queueLog = 'C:\Users\Jiangqianxian\source\repos\d2nn\reports\post_rs_retrain\2026-04-09\queue_fft.log'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$queue = @(
    [pscustomobject]@{ Name = 'fashion_mnist_phase_only_20ep_seed42_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_phase_only_20ep_seed42_post_rs_fft --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'fashion_mnist_incoherent_back_20ep_seed42_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_incoherent_back_20ep_seed42_post_rs_fft --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'fashion_mnist_incoherent_back_20ep_seed7_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 7 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_incoherent_back_20ep_seed7_post_rs_fft --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'fashion_mnist_incoherent_back_20ep_seed123_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 123 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_incoherent_back_20ep_seed123_post_rs_fft --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_phase_only_20ep_seed42_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_phase_only_20ep_seed42_post_rs_fft --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_incoherent_back_20ep_seed42_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_incoherent_back_20ep_seed42_post_rs_fft --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_phase_only_20ep_seed7_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 7 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_phase_only_20ep_seed7_post_rs_fft --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_incoherent_back_20ep_seed7_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 7 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_incoherent_back_20ep_seed7_post_rs_fft --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_phase_only_20ep_seed123_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 123 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_phase_only_20ep_seed123_post_rs_fft --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_incoherent_back_20ep_seed123_post_rs_fft'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 123 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_incoherent_back_20ep_seed123_post_rs_fft --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --rs-backend fft --allow-tf32 --num-workers 2 --pin-memory' }
)
'[QUEUE START] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $queueLog -Encoding utf8 -Append
foreach ($job in $queue) {
    $logPath = Join-Path $logs ("$($job.Name).log")
    "[START] 2026-04-10 00:05:51 $($job.Name)" | Out-File -FilePath $queueLog -Encoding utf8 -Append
    "[COMMAND] $($job.Command)" | Out-File -FilePath $queueLog -Encoding utf8 -Append
    & cmd /c $job.Command *> $logPath
    $exitCode = $LASTEXITCODE
    "[EXIT] 2026-04-10 00:05:51 $($job.Name) exit=$exitCode" | Out-File -FilePath $queueLog -Encoding utf8 -Append
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}
'[QUEUE DONE] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $queueLog -Encoding utf8 -Append
