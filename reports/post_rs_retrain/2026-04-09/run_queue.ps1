$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
Set-Location 'C:\Users\Jiangqianxian\source\repos\d2nn'
$root = 'C:\Users\Jiangqianxian\source\repos\d2nn\reports\post_rs_retrain\2026-04-09'
$logs = 'C:\Users\Jiangqianxian\source\repos\d2nn\reports\post_rs_retrain\2026-04-09\logs'
$queueLog = 'C:\Users\Jiangqianxian\source\repos\d2nn\reports\post_rs_retrain\2026-04-09\queue.log'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$queue = @(
    [pscustomobject]@{ Name = 'fashion_mnist_phase_only_20ep_seed42_post_rs'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_phase_only_20ep_seed42_post_rs --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'fashion_mnist_incoherent_back_20ep_seed42_post_rs'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_incoherent_back_20ep_seed42_post_rs --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'fashion_mnist_incoherent_back_20ep_seed7_post_rs'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 7 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_incoherent_back_20ep_seed7_post_rs --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'fashion_mnist_incoherent_back_20ep_seed123_post_rs'; Command = 'uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 123 --experiment-stage post-rs-full-retrain --run-name fashion_mnist_incoherent_back_20ep_seed123_post_rs --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_phase_only_20ep_seed42_post_rs'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_phase_only_20ep_seed42_post_rs --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_incoherent_back_20ep_seed42_post_rs'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 42 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_incoherent_back_20ep_seed42_post_rs --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_phase_only_20ep_seed7_post_rs'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 7 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_phase_only_20ep_seed7_post_rs --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_incoherent_back_20ep_seed7_post_rs'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 7 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_incoherent_back_20ep_seed7_post_rs --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_phase_only_20ep_seed123_post_rs'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 123 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_phase_only_20ep_seed123_post_rs --allow-tf32 --num-workers 2 --pin-memory' },
    [pscustomobject]@{ Name = 'cifar10_rgb_incoherent_back_20ep_seed123_post_rs'; Command = 'uv run python train.py --task classification --dataset cifar10-rgb --epochs 20 --size 200 --layers 5 --batch-size 64 --seed 123 --experiment-stage post-rs-full-retrain --run-name cifar10_rgb_incoherent_back_20ep_seed123_post_rs --activation-type incoherent_intensity --activation-placement back --activation-preset balanced --allow-tf32 --num-workers 2 --pin-memory' }
)
'[QUEUE START] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $queueLog -Encoding utf8 -Append
foreach ($job in $queue) {
    $logPath = Join-Path $logs ("$($job.Name).log")
    '[START] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $job.Name | Tee-Object -FilePath $queueLog -Append | Out-Host
    '[COMMAND] ' + $job.Command | Tee-Object -FilePath $queueLog -Append | Out-Host
    & cmd /c $job.Command 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        '[FAIL] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $job.Name + ' exit=' + $LASTEXITCODE | Tee-Object -FilePath $queueLog -Append | Out-Host
        exit $LASTEXITCODE
    }
    '[DONE] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + ' ' + $job.Name | Tee-Object -FilePath $queueLog -Append | Out-Host
}
'[QUEUE DONE] ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $queueLog -Encoding utf8 -Append
