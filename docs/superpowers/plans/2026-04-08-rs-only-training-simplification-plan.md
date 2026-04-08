# RS-Only Training Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the main classifier propagation/training path with an RS-only design, add composite classification loss plus detector-contrast reporting/visualization, and keep imaging and artifact flows explicitly versioned.

**Architecture:** Keep `forward()` tensor-only for lightweight inference, and add a structured helper for metrics/history to avoid duplicate forward passes in training and visualization. Route checkpoint/version/history behavior through shared artifact helpers, while keeping plotting helpers in `tasks.py` and `visualize.py` as a thin CLI wrapper.

**Tech Stack:** Python, PyTorch, matplotlib, pytest, uv

---

### Task 1: Lock The Public Contract And Shared Metadata

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tests\test_d2nn_core.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\artifacts.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\train.py`
- Test: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tests\test_d2nn_core.py`

- [ ] **Step 1: Write failing tests for versioning, run naming, and new CLI knobs**

```python
def test_parser_accepts_composite_loss_flags():
    parser = build_parser()
    args = parser.parse_args(["--alpha", "1.0", "--beta", "0.1", "--gamma", "0.01"])
    assert args.alpha == 1.0
    assert args.beta == 0.1
    assert args.gamma == 0.01


def test_derive_experiment_run_name_includes_loss_config_without_activation():
    run_name = derive_experiment_run_name(
        experiment_stage="rs_only",
        activation_type="none",
        seed=42,
        loss_config={"alpha": 1.0, "beta": 0.1, "gamma": 0.01},
    )
    assert "alpha-1" in run_name
    assert "beta-0p1" in run_name
    assert "gamma-0p01" in run_name


def test_checkpoint_version_mismatch_is_rejected():
    manifest = {"model_version": "asm_v1"}
    with self.assertRaisesRegex(ValueError, "expected rs_v1"):
        ensure_checkpoint_version(manifest, expected_version="rs_v1", checkpoint_path="demo.pth")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_d2nn_core.py -k "composite_loss_flags or run_name_includes_loss_config or checkpoint_version_mismatch" -v`
Expected: FAIL because the parser, artifact helpers, and version gate do not exist yet.

- [ ] **Step 3: Implement the minimal shared metadata and CLI changes**

```python
# train.py
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--gamma", type=float, default=0.01)


# artifacts.py
def ensure_checkpoint_version(manifest, expected_version, checkpoint_path):
    found = (manifest or {}).get("model_version")
    if found != expected_version:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has model_version={found!r}; expected {expected_version!r}"
        )


def derive_experiment_run_name(..., loss_config=None):
    ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_d2nn_core.py -k "composite_loss_flags or run_name_includes_loss_config or checkpoint_version_mismatch" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_d2nn_core.py artifacts.py train.py
git commit -m "test: lock rs-only cli and metadata contract"
```

### Task 2: Replace The Classifier Public Path With RS-Only Structured Helpers

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tests\test_d2nn_core.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\d2nn.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\artifacts.py`
- Test: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tests\test_d2nn_core.py`

- [ ] **Step 1: Write failing tests for the RS-only classifier contract**

```python
def test_classifier_forward_stays_tensor_only():
    model = D2NN(...)
    logits = model(torch.rand(2, 1, 28, 28))
    assert logits.shape == (2, 10)


def test_classifier_forward_with_metrics_returns_logits_intensity_and_contrast():
    model = D2NN(...)
    result = model.forward_with_metrics(torch.rand(2, 1, 28, 28), target=torch.tensor([0, 1]))
    assert set(result) >= {"scores", "logits", "intensity", "contrast"}


def test_detector_contrast_matches_formula():
    scores = torch.tensor([[4.0, 1.0, 0.5]])
    target = torch.tensor([0])
    contrast = detector_contrast(scores, target)
    expected = (4.0 - 1.0) / (4.0 + 1.0 + 1e-8)
    assert torch.allclose(contrast, torch.tensor([expected]))


def test_rs_plane_wave_smoke_is_nearly_uniform_in_center_crop():
    intensity = propagate_uniform_plane_wave_rs(...)
    center = intensity[..., 8:-8, 8:-8]
    rel_std = center.std() / center.mean()
    assert rel_std < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_d2nn_core.py -k "forward_stays_tensor_only or forward_with_metrics or detector_contrast_matches_formula or plane_wave_smoke" -v`
Expected: FAIL because RS-only helpers and contrast logic do not exist yet.

- [ ] **Step 3: Implement the minimal RS-only classifier path**

```python
class DetectorLayer(nn.Module):
    def forward(self, intensity):
        ...

    def contrast(self, scores, target, eps=1e-8):
        target_energy = scores.gather(1, target[:, None]).squeeze(1)
        masked = scores.clone()
        masked.scatter_(1, target[:, None], float("-inf"))
        other_energy = masked.max(dim=1).values
        return (target_energy - other_energy) / (target_energy + other_energy + eps)


class D2NN(...):
    def forward(self, x):
        return self.forward_with_metrics(x)["scores"]

    def forward_with_metrics(self, x, target=None):
        intensity = self.output_intensity(x)
        scores = self.detect_from_intensity(intensity)
        logits = normalized_scores_to_logits(scores)
        result = {"scores": scores, "logits": logits, "intensity": intensity}
        if target is not None:
            result["contrast"] = self.detector_layer.contrast(scores, target)
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_d2nn_core.py -k "forward_stays_tensor_only or forward_with_metrics or detector_contrast_matches_formula or plane_wave_smoke" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_d2nn_core.py d2nn.py artifacts.py
git commit -m "feat: add rs-only classifier metrics contract"
```

### Task 3: Add Composite Loss, History Persistence, And Visualization Wiring

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tests\test_d2nn_core.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tasks.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\visualize.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\artifacts.py`
- Test: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\tests\test_d2nn_core.py`

- [ ] **Step 1: Write failing tests for composite loss, history, and headless plotting**

```python
def test_composite_classification_loss_reports_all_terms():
    outputs = {"scores": torch.tensor([[2.0, 1.0]]), "logits": torch.log(torch.tensor([[0.6, 0.4]]))}
    terms = classification_composite_loss(outputs, torch.tensor([0]), alpha=1.0, beta=0.1, gamma=0.01, model=model)
    assert set(terms) == {"total", "mse", "ce", "reg"}


def test_manifest_history_records_accuracy_and_contrast():
    history = build_empty_history()
    append_history(history, split="train", total=1.0, mse=0.8, ce=0.1, reg=0.1, accuracy=90.0, contrast=0.5)
    assert history["train"]["accuracy"] == [90.0]
    assert history["train"]["contrast"] == [0.5]


def test_plot_sample_output_patterns_is_headless_safe():
    matplotlib.use("Agg", force=True)
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_d2nn_core.py -k "composite_classification_loss_reports_all_terms or manifest_history_records_accuracy_and_contrast or plot_sample_output_patterns_is_headless_safe" -v`
Expected: FAIL because the composite-loss helper, history recording, and backend-safe plotting are not implemented yet.

- [ ] **Step 3: Implement the minimal training/visualization changes**

```python
def classification_composite_loss(result, target, model, alpha=1.0, beta=0.1, gamma=0.01):
    mse = ...
    ce = F.cross_entropy(result["logits"], target)
    reg = phase_smoothness_regularizer(model)
    total = alpha * mse + beta * ce + gamma * reg
    return {"total": total, "mse": mse, "ce": ce, "reg": reg}


history = {
    "train": {"loss": [], "mse": [], "ce": [], "reg": [], "accuracy": [], "contrast": []},
    "val": {"loss": [], "mse": [], "ce": [], "reg": [], "accuracy": [], "contrast": []},
}


import matplotlib
matplotlib.use("Agg", force=True)
```

- [ ] **Step 4: Run targeted tests, then the full suite**

Run: `uv run python -m pytest tests/test_d2nn_core.py -k "composite_classification_loss_reports_all_terms or manifest_history_records_accuracy_and_contrast or plot_sample_output_patterns_is_headless_safe" -v`
Expected: PASS

Run: `uv run python -m pytest`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_d2nn_core.py tasks.py visualize.py artifacts.py
git commit -m "feat: add composite loss history and contrast visualization"
```

### Task 4: Final Verification And Stage Review

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\docs\superpowers\specs\2026-04-08-rs-only-training-simplification-design.md`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\.worktrees\rs-only-simplification\docs\superpowers\plans\2026-04-08-rs-only-training-simplification-plan.md`
- Modify: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018.md`

- [ ] **Step 1: Re-run the full verification command**

Run: `uv run python -m pytest`
Expected: PASS with zero failures

- [ ] **Step 2: Run focused CLI smoke checks**

Run: `uv run python train.py --help`
Expected: help output includes `--alpha`, `--beta`, and `--gamma`

Run: `uv run python visualize.py --help`
Expected: help output still succeeds and exposes the visualization entrypoint

- [ ] **Step 3: Perform an independent stage review**

Run an independent review pass against the diff in this worktree, focusing on:

```text
- RS-only public contract
- composite loss stability
- manifest/version/run-name consistency
- visualization/history wiring
- imaging regression risk
```

- [ ] **Step 4: Record the review outcome in the reproduction log**

Add a short dated entry to:

```text
C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018.md
```

The entry should summarize:

```text
- RS-only simplification stage reached
- verification command and result
- review path used
- whether any blocking issues remain
```

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-08-rs-only-training-simplification-design.md docs/superpowers/plans/2026-04-08-rs-only-training-simplification-plan.md
git commit -m "docs: record rs-only simplification verification"
```
