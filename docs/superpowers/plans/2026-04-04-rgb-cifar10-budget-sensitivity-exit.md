# RGB CIFAR-10 Budget Sensitivity Exit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one final `20 epoch` RGB CIFAR-10 budget-sensitivity pass for the current nonlinear line, then review and log the final exit statement for the nonlinear-validation phase.

**Architecture:** Reuse the existing RGB CIFAR-10 training path unchanged and vary only the training budget from `10 epoch` to `20 epoch`. Execute three paired seed runs (`42`, `7`, `123`) for `phase-only baseline` versus `incoherent_intensity + back`, compute the summary, perform stage review, and append the result to the two Obsidian notes.

**Tech Stack:** Python 3.13 virtualenv, PyTorch, PowerShell, Git, Obsidian Markdown

---

### Task 1: Reconfirm the 10-epoch reference point and define the 20-epoch exit matrix

**Files:**
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json`

- [ ] **Step 1: Re-read the reference 10-epoch values for seeds 42, 7, and 123**

Run:

```powershell
$pairs = @(
  @{ seed = 42; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep.json' },
  @{ seed = 7; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed7.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed7.json' },
  @{ seed = 123; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_10ep_seed123.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_10ep_seed123.json' }
)
$rows = foreach ($pair in $pairs) {
  $b = Get-Content -LiteralPath $pair.baseline -Raw | ConvertFrom-Json
  $n = Get-Content -LiteralPath $pair.nonlinear -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    seed = $pair.seed
    baseline_test = $b.test_accuracy
    nonlinear_test = $n.test_accuracy
    lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  }
}
$rows | Format-Table -AutoSize
```

Expected: the table restates the `10 epoch` reference point that the new `20 epoch` pass will be compared against.

- [ ] **Step 2: Lock the exact 20-epoch matrix**

Use this exact six-run matrix:

```text
baseline 20ep: seeds 42 / 7 / 123
nonlinear 20ep: seeds 42 / 7 / 123
```

Expected: the exit pass remains bounded and does not expand to more seeds or extra variants.

- [ ] **Step 3: Lock the naming scheme**

Use these exact run-name patterns:

```text
cifar10_rgb_baseline_20ep_seed42
cifar10_rgb_baseline_20ep_seed7
cifar10_rgb_baseline_20ep_seed123
cifar10_rgb_incoherent_back_20ep_seed42
cifar10_rgb_incoherent_back_20ep_seed7
cifar10_rgb_incoherent_back_20ep_seed123
```

Expected: the 20-epoch artifacts remain disjoint from the existing `10ep` RGB stage.

### Task 2: Run the 20-epoch baseline and nonlinear pair for seed 42

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed42.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed42.json`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed42.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed42.json`

- [ ] **Step 1: Run the 20-epoch baseline arm for seed 42**

Run:

```powershell
$env:HOME='C:\Users\Jiangqianxian'
$env:USERPROFILE='C:\Users\Jiangqianxian'
$env:MPLCONFIGDIR='C:\Users\Jiangqianxian\source\repos\d2nn\.codex_tmp\mpl'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Start-Process -FilePath 'C:\Users\Jiangqianxian\source\repos\d2nn\.venv\Scripts\python.exe' -ArgumentList @(
  'train.py','--task','classification','--dataset','cifar10-rgb',
  '--epochs','20','--size','200','--layers','5','--batch-size','64','--lr','0.01',
  '--seed','42','--experiment-stage','cifar10-rgb-budget-exit',
  '--run-name','cifar10_rgb_baseline_20ep_seed42'
) -WorkingDirectory 'C:\Users\Jiangqianxian\source\repos\d2nn' -NoNewWindow -Wait -PassThru
```

Expected: exit code `0` and a new `baseline_20ep_seed42` checkpoint plus manifest.

- [ ] **Step 2: Run the 20-epoch nonlinear arm for seed 42**

Run:

```powershell
$env:HOME='C:\Users\Jiangqianxian'
$env:USERPROFILE='C:\Users\Jiangqianxian'
$env:MPLCONFIGDIR='C:\Users\Jiangqianxian\source\repos\d2nn\.codex_tmp\mpl'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Start-Process -FilePath 'C:\Users\Jiangqianxian\source\repos\d2nn\.venv\Scripts\python.exe' -ArgumentList @(
  'train.py','--task','classification','--dataset','cifar10-rgb',
  '--epochs','20','--size','200','--layers','5','--batch-size','64','--lr','0.01',
  '--seed','42','--experiment-stage','cifar10-rgb-budget-exit',
  '--run-name','cifar10_rgb_incoherent_back_20ep_seed42',
  '--activation-type','incoherent_intensity',
  '--activation-placement','back',
  '--activation-preset','balanced'
 ) -WorkingDirectory 'C:\Users\Jiangqianxian\source\repos\d2nn' -NoNewWindow -Wait -PassThru
```

Expected: exit code `0` and a new `incoherent_back_20ep_seed42` checkpoint plus manifest.

- [ ] **Step 3: Read the two seed-42 20-epoch manifests**

Run:

```powershell
$b = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed42.json' -Raw | ConvertFrom-Json
$n = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed42.json' -Raw | ConvertFrom-Json
[PSCustomObject]@{
  seed = 42
  baseline_test = $b.test_accuracy
  nonlinear_test = $n.test_accuracy
  lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  baseline_best_val = $b.best_val_accuracy
  nonlinear_best_val = $n.best_val_accuracy
} | Format-List
```

Expected: one explicit seed-42 summary for the `20 epoch` pass.

### Task 3: Run the 20-epoch baseline and nonlinear pair for seed 7

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed7.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed7.json`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed7.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed7.json`

- [ ] **Step 1: Run the 20-epoch baseline arm for seed 7**

Run:

```powershell
$env:HOME='C:\Users\Jiangqianxian'
$env:USERPROFILE='C:\Users\Jiangqianxian'
$env:MPLCONFIGDIR='C:\Users\Jiangqianxian\source\repos\d2nn\.codex_tmp\mpl'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Start-Process -FilePath 'C:\Users\Jiangqianxian\source\repos\d2nn\.venv\Scripts\python.exe' -ArgumentList @(
  'train.py','--task','classification','--dataset','cifar10-rgb',
  '--epochs','20','--size','200','--layers','5','--batch-size','64','--lr','0.01',
  '--seed','7','--experiment-stage','cifar10-rgb-budget-exit',
  '--run-name','cifar10_rgb_baseline_20ep_seed7'
) -WorkingDirectory 'C:\Users\Jiangqianxian\source\repos\d2nn' -NoNewWindow -Wait -PassThru
```

Expected: exit code `0` and a new `baseline_20ep_seed7` checkpoint plus manifest.

- [ ] **Step 2: Run the 20-epoch nonlinear arm for seed 7**

Run:

```powershell
$env:HOME='C:\Users\Jiangqianxian'
$env:USERPROFILE='C:\Users\Jiangqianxian'
$env:MPLCONFIGDIR='C:\Users\Jiangqianxian\source\repos\d2nn\.codex_tmp\mpl'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Start-Process -FilePath 'C:\Users\Jiangqianxian\source\repos\d2nn\.venv\Scripts\python.exe' -ArgumentList @(
  'train.py','--task','classification','--dataset','cifar10-rgb',
  '--epochs','20','--size','200','--layers','5','--batch-size','64','--lr','0.01',
  '--seed','7','--experiment-stage','cifar10-rgb-budget-exit',
  '--run-name','cifar10_rgb_incoherent_back_20ep_seed7',
  '--activation-type','incoherent_intensity',
  '--activation-placement','back',
  '--activation-preset','balanced'
 ) -WorkingDirectory 'C:\Users\Jiangqianxian\source\repos\d2nn' -NoNewWindow -Wait -PassThru
```

Expected: exit code `0` and a new `incoherent_back_20ep_seed7` checkpoint plus manifest.

- [ ] **Step 3: Read the two seed-7 20-epoch manifests**

Run:

```powershell
$b = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed7.json' -Raw | ConvertFrom-Json
$n = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed7.json' -Raw | ConvertFrom-Json
[PSCustomObject]@{
  seed = 7
  baseline_test = $b.test_accuracy
  nonlinear_test = $n.test_accuracy
  lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  baseline_best_val = $b.best_val_accuracy
  nonlinear_best_val = $n.best_val_accuracy
} | Format-List
```

Expected: one explicit seed-7 summary for the `20 epoch` pass.

### Task 4: Run the 20-epoch baseline and nonlinear pair for seed 123

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed123.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed123.json`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed123.pth`
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed123.json`

- [ ] **Step 1: Run the 20-epoch baseline arm for seed 123**

Run:

```powershell
$env:HOME='C:\Users\Jiangqianxian'
$env:USERPROFILE='C:\Users\Jiangqianxian'
$env:MPLCONFIGDIR='C:\Users\Jiangqianxian\source\repos\d2nn\.codex_tmp\mpl'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Start-Process -FilePath 'C:\Users\Jiangqianxian\source\repos\d2nn\.venv\Scripts\python.exe' -ArgumentList @(
  'train.py','--task','classification','--dataset','cifar10-rgb',
  '--epochs','20','--size','200','--layers','5','--batch-size','64','--lr','0.01',
  '--seed','123','--experiment-stage','cifar10-rgb-budget-exit',
  '--run-name','cifar10_rgb_baseline_20ep_seed123'
) -WorkingDirectory 'C:\Users\Jiangqianxian\source\repos\d2nn' -NoNewWindow -Wait -PassThru
```

Expected: exit code `0` and a new `baseline_20ep_seed123` checkpoint plus manifest.

- [ ] **Step 2: Run the 20-epoch nonlinear arm for seed 123**

Run:

```powershell
$env:HOME='C:\Users\Jiangqianxian'
$env:USERPROFILE='C:\Users\Jiangqianxian'
$env:MPLCONFIGDIR='C:\Users\Jiangqianxian\source\repos\d2nn\.codex_tmp\mpl'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
Start-Process -FilePath 'C:\Users\Jiangqianxian\source\repos\d2nn\.venv\Scripts\python.exe' -ArgumentList @(
  'train.py','--task','classification','--dataset','cifar10-rgb',
  '--epochs','20','--size','200','--layers','5','--batch-size','64','--lr','0.01',
  '--seed','123','--experiment-stage','cifar10-rgb-budget-exit',
  '--run-name','cifar10_rgb_incoherent_back_20ep_seed123',
  '--activation-type','incoherent_intensity',
  '--activation-placement','back',
  '--activation-preset','balanced'
 ) -WorkingDirectory 'C:\Users\Jiangqianxian\source\repos\d2nn' -NoNewWindow -Wait -PassThru
```

Expected: exit code `0` and a new `incoherent_back_20ep_seed123` checkpoint plus manifest.

- [ ] **Step 3: Read the two seed-123 20-epoch manifests**

Run:

```powershell
$b = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed123.json' -Raw | ConvertFrom-Json
$n = Get-Content -LiteralPath 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed123.json' -Raw | ConvertFrom-Json
[PSCustomObject]@{
  seed = 123
  baseline_test = $b.test_accuracy
  nonlinear_test = $n.test_accuracy
  lift = [math]::Round(($n.test_accuracy - $b.test_accuracy), 4)
  baseline_best_val = $b.best_val_accuracy
  nonlinear_best_val = $n.best_val_accuracy
} | Format-List
```

Expected: one explicit seed-123 summary for the `20 epoch` pass.

### Task 5: Summarize the 20-epoch pass, perform stage review, and write the final exit statement

**Files:**
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed42.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed42.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed7.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed123.json`
- Read: `C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed123.json`
- Modify: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018.md`
- Modify: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018 · 非线性层方案.md`

- [ ] **Step 1: Compute the three-seed 20-epoch summary**

Run:

```powershell
$pairs = @(
  @{ seed = 42; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed42.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed42.json' },
  @{ seed = 7; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed7.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed7.json' },
  @{ seed = 123; baseline = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed123.json'; nonlinear = 'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed123.json' }
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

Expected: a complete three-seed `20 epoch` summary is available for the final exit statement.

- [ ] **Step 2: Try the preferred external review path**

Run:

```powershell
claude -p "Please review the current RGB CIFAR-10 20-epoch exit pass in C:\Users\Jiangqianxian\source\repos\d2nn. Focus on 20ep naming consistency, paired baseline/nonlinear integrity across seeds 42/7/123, config drift, and whether the final written conclusion would match the data."
```

Expected: either usable review output or a concrete failure reason that can be logged honestly.

- [ ] **Step 3: If Claude review is unavailable, run fallback read-only review**

Run:

```powershell
$paths = @(
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed42.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed42.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed7.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed7.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_baseline_20ep_seed123.json',
  'C:\Users\Jiangqianxian\source\repos\d2nn\checkpoints\best_cifar10_rgb.cifar10_rgb_incoherent_back_20ep_seed123.json'
)
$paths | ForEach-Object {
  $m = Get-Content -LiteralPath $_ -Raw | ConvertFrom-Json
  [PSCustomObject]@{
    path = Split-Path $_ -Leaf
    run_name = $m.run_name
    seed = $m.seed
    experiment_stage = $m.experiment_stage
    activation_type = $m.activation_type
    activation_positions = ($m.activation_positions -join ',')
    best_val_accuracy = $m.best_val_accuracy
    test_accuracy = $m.test_accuracy
  }
} | Sort-Object seed, activation_type | Format-Table -AutoSize
```

Expected: naming, pairing, and config consistency can be checked directly from the manifests.

- [ ] **Step 4: Append the 20-epoch results to the main reproduction log**

Add a progress-log entry that includes:

```text
- all three 20-epoch baseline/nonlinear pairs
- the 20-epoch mean baseline, mean nonlinear, mean lift, and spread
- whether the longer budget made RGB materially stronger or not
- whether external Claude review produced usable output or fallback review was used
```

Expected: the main note records the final chronological stage exit.

- [ ] **Step 5: Append the final stage interpretation to the nonlinear note**

Add a note entry that includes one of these outcomes:

```text
- RGB became sufficiently stronger under longer budget, so the nonlinear-validation phase is closed through RGB
or
- RGB remained budget-sensitive / not fully stabilized, so the nonlinear-validation phase still ends here with that qualified conclusion
```

Expected: the nonlinear note contains the final exit statement for this phase.

- [ ] **Step 6: Stop at the stage boundary**

Decision rule:

```text
After the 20-epoch pass and review, stop.
Do not continue directly into repo cleanup, README rewrite, or new experiments in the same execution block.
```

Expected: this plan ends exactly at the nonlinear-phase exit node.
