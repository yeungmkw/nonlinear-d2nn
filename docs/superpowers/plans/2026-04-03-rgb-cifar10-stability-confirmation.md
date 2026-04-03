# RGB CIFAR-10 Stability Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current RGB CIFAR-10 nonlinear result from 2 seeds to 5 paired seeds under the same frozen 10-epoch setup, then review and log the stability conclusion.

**Architecture:** Keep the current RGB CIFAR-10 experiment surface fixed and treat this as an evidence-building pass, not a new feature pass. Reuse the existing `train.py` entrypoint, existing artifact naming scheme, and existing Obsidian logs; only add three new paired seed runs, compute the five-seed summary, perform an independent review, and append the result to the two reproduction notes.

**Tech Stack:** Python 3.13, uv, PyTorch, PowerShell, Git, GitHub CLI release artifacts already published, Obsidian Markdown notes

---

### Task 1: Reconfirm the frozen RGB baseline and create the five-seed result frame

**Files:**
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json`
- Modify later: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018.md`
- Modify later: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018 · 非线性层方案.md`

- [ ] **Step 1: Read the four existing manifests and extract the frozen reference values**

Run:

```powershell
$paths = @(
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json'
)
$paths | ForEach-Object {
  $m = Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    path = $_
    run_name = $m.run_name
    seed = $m.seed
    best_val_accuracy = $m.best_val_accuracy
    test_accuracy = $m.test_accuracy
  }
} | Format-Table -AutoSize
```

Expected: the table reproduces the frozen seed `42` and `7` RGB values without changing any files.

- [ ] **Step 2: Define the final five-seed order used for this stage**

Use this exact seed order:

```text
42, 7, 123, 0, 2025
```

Expected: the stability pass follows the spec exactly and avoids ad hoc seed changes mid-run.

- [ ] **Step 3: Record the result-table schema before running new seeds**

Use this schema for the final summary:

```text
seed | baseline_test | nonlinear_test | lift | baseline_best_val | nonlinear_best_val
```

Expected: every seed pair is logged with the same fields, so the five-seed summary is comparable and complete.

### Task 2: Run the `seed=123` paired RGB comparison

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json`

- [ ] **Step 1: Run the baseline arm for `seed=123`**

Run:

```powershell
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 123 --experiment-stage cifar10-rgb-baseline --run-name cifar10_rgb_baseline_10ep_seed123
```

Expected: training completes successfully and writes the `seed123` baseline checkpoint plus manifest.

- [ ] **Step 2: Run the nonlinear arm for `seed=123`**

Run:

```powershell
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 123 --experiment-stage cifar10-rgb-nonlinear --run-name cifar10_rgb_incoherent_back_10ep_seed123 --activation-type incoherent_intensity --activation-placement back --activation-preset balanced
```

Expected: training completes successfully and writes the `seed123` nonlinear checkpoint plus manifest.

- [ ] **Step 3: Read both `seed=123` manifests and compute the lift**

Run:

```powershell
$b = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json' -Raw | ConvertFrom-Json
$n = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json' -Raw | ConvertFrom-Json
[PSCustomObject]@{
  seed = 123
  baseline_test = $b.test_accuracy
  nonlinear_test = $n.test_accuracy
  lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  baseline_best_val = $b.best_val_accuracy
  nonlinear_best_val = $n.best_val_accuracy
} | Format-List
```

Expected: one explicit seed-level summary for `123` is available before moving to the next seed.

### Task 3: Run the `seed=0` paired RGB comparison

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed0.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed0.json`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed0.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed0.json`

- [ ] **Step 1: Run the baseline arm for `seed=0`**

Run:

```powershell
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 0 --experiment-stage cifar10-rgb-baseline --run-name cifar10_rgb_baseline_10ep_seed0
```

Expected: training completes successfully and writes the `seed0` baseline checkpoint plus manifest.

- [ ] **Step 2: Run the nonlinear arm for `seed=0`**

Run:

```powershell
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 0 --experiment-stage cifar10-rgb-nonlinear --run-name cifar10_rgb_incoherent_back_10ep_seed0 --activation-type incoherent_intensity --activation-placement back --activation-preset balanced
```

Expected: training completes successfully and writes the `seed0` nonlinear checkpoint plus manifest.

- [ ] **Step 3: Read both `seed=0` manifests and compute the lift**

Run:

```powershell
$b = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed0.json' -Raw | ConvertFrom-Json
$n = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed0.json' -Raw | ConvertFrom-Json
[PSCustomObject]@{
  seed = 0
  baseline_test = $b.test_accuracy
  nonlinear_test = $n.test_accuracy
  lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  baseline_best_val = $b.best_val_accuracy
  nonlinear_best_val = $n.best_val_accuracy
} | Format-List
```

Expected: one explicit seed-level summary for `0` is available before moving to the next seed.

### Task 4: Run the `seed=2025` paired RGB comparison

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed2025.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed2025.json`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed2025.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed2025.json`

- [ ] **Step 1: Run the baseline arm for `seed=2025`**

Run:

```powershell
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 2025 --experiment-stage cifar10-rgb-baseline --run-name cifar10_rgb_baseline_10ep_seed2025
```

Expected: training completes successfully and writes the `seed2025` baseline checkpoint plus manifest.

- [ ] **Step 2: Run the nonlinear arm for `seed=2025`**

Run:

```powershell
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 --size 200 --layers 5 --batch-size 64 --lr 0.01 --seed 2025 --experiment-stage cifar10-rgb-nonlinear --run-name cifar10_rgb_incoherent_back_10ep_seed2025 --activation-type incoherent_intensity --activation-placement back --activation-preset balanced
```

Expected: training completes successfully and writes the `seed2025` nonlinear checkpoint plus manifest.

- [ ] **Step 3: Read both `seed=2025` manifests and compute the lift**

Run:

```powershell
$b = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed2025.json' -Raw | ConvertFrom-Json
$n = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed2025.json' -Raw | ConvertFrom-Json
[PSCustomObject]@{
  seed = 2025
  baseline_test = $b.test_accuracy
  nonlinear_test = $n.test_accuracy
  lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  baseline_best_val = $b.best_val_accuracy
  nonlinear_best_val = $n.best_val_accuracy
} | Format-List
```

Expected: one explicit seed-level summary for `2025` is available before the five-seed aggregate summary.

### Task 5: Compute the five-seed summary, perform stage review, and update the Obsidian logs

**Files:**
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed0.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed0.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed2025.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed2025.json`
- Modify: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018.md`
- Modify: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018 · 非线性层方案.md`

- [ ] **Step 1: Compute the five-seed aggregate table and summary metrics**

Run:

```powershell
$pairs = @(
  @{ seed = 42; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json' },
  @{ seed = 7; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json' },
  @{ seed = 123; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json' },
  @{ seed = 0; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed0.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed0.json' },
  @{ seed = 2025; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed2025.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed2025.json' }
)
$rows = foreach ($pair in $pairs) {
  $b = Get-Content -LiteralPath $pair.baseline -Raw | ConvertFrom-Json
  $n = Get-Content -LiteralPath $pair.nonlinear -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    seed = $pair.seed
    baseline_test = [double]$b.test_accuracy
    nonlinear_test = [double]$n.test_accuracy
    lift = [double]$n.test_accuracy - [double]$b.test_accuracy
    baseline_best_val = $b.best_val_accuracy
    nonlinear_best_val = $n.best_val_accuracy
  }
}
$rows | Format-Table -AutoSize
[PSCustomObject]@{
  baseline_mean = [math]::Round((($rows | Measure-Object baseline_test -Average).Average), 4)
  nonlinear_mean = [math]::Round((($rows | Measure-Object nonlinear_test -Average).Average), 4)
  lift_mean = [math]::Round((($rows | Measure-Object lift -Average).Average), 4)
  lift_min = [math]::Round((($rows | Measure-Object lift -Minimum).Minimum), 4)
  lift_max = [math]::Round((($rows | Measure-Object lift -Maximum).Maximum), 4)
  lift_spread = [math]::Round(((($rows | Measure-Object lift -Maximum).Maximum) - (($rows | Measure-Object lift -Minimum).Minimum)), 4)
} | Format-List
```

Expected: the output contains a complete five-seed table plus mean and spread values needed for the stage conclusion.

- [ ] **Step 2: Try the preferred independent review path**

Run:

```powershell
claude -p "Please review the current RGB CIFAR-10 5-seed stability stage in C:\Users\Jiangqianxian\source\repos\d2nn. Focus on manifest naming consistency, paired baseline/nonlinear integrity, CLI/config drift, and any issue that would block a stability claim."
```

Expected: either usable review output, or a clear login / unavailable failure that can be honestly recorded.

- [ ] **Step 3: If Claude review is unavailable, run the fallback read-only review**

Run:

```powershell
$pairs = @(
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed0.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed0.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed2025.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed2025.json'
)
$pairs | ForEach-Object {
  $m = Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    path = Split-Path $_ -Leaf
    run_name = $m.run_name
    seed = $m.seed
    activation_type = $m.activation_type
    activation_positions = ($m.activation_positions -join ',')
    best_val_accuracy = $m.best_val_accuracy
    test_accuracy = $m.test_accuracy
  }
} | Format-Table -AutoSize
```

Expected: one manual review surface that makes naming, seed pairing, and activation consistency easy to inspect.

- [ ] **Step 4: Append the result summary and review conclusion to the main reproduction log**

Add a new progress-log entry that includes:

```text
- completed RGB CIFAR-10 10 epoch seeds 123 / 0 / 2025
- listed all five baseline and nonlinear test accuracies
- reported five-seed mean baseline, mean nonlinear, mean lift, and lift spread
- recorded whether the stability condition was met
- recorded whether Claude review produced usable output or fallback review was used
```

Expected: the main note remains the authoritative chronological log.

- [ ] **Step 5: Append the stage interpretation to the nonlinear note**

Add a nonlinear-note entry that includes:

```text
- this was a stability-confirmation pass, not a new mechanism pass
- RGB CIFAR-10 now has five paired seeds under the fixed 10-epoch setup
- record whether the line is stable, directional but variable, or inconclusive
- state the next decision gate: extend epoch budget or stop at this stronger RGB evidence node
```

Expected: the nonlinear note remains the authoritative planning anchor for what comes next.

- [ ] **Step 6: Stop at the stage node and decide whether a new freeze is justified**

Decision rule:

```text
If the five-seed result is stable and materially stronger than the current 2-seed freeze, stop and treat this as a new review/git/release decision point.
If not, stop and report the instability or ambiguity clearly before any longer-epoch extension.
```

Expected: the session ends at the correct stage boundary instead of silently rolling into the next phase.
