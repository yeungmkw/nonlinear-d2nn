# Reproduction Deliverable Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current D2NN repository into a clearer reproduction handoff package with a stronger README, explicit paper-vs-repo result reporting, and automated smoke verification.

**Architecture:** Keep the six-file runtime layout unchanged and focus only on deliverable-facing assets. Add tests that require README sections and CLI smoke coverage, then update the documentation and verification surface until those tests pass.

**Tech Stack:** Python 3.13, unittest, PyTorch CLI entrypoints, Markdown docs

---

### Task 1: Add failing tests for deliverable-facing requirements

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\tests\test_d2nn_core.py`

- [ ] **Step 1: Write failing README assertions**

Add tests that require the README to contain:
- a “推荐复现路径” section
- a “论文对照结果” section
- a “推荐 checkpoint” section

- [ ] **Step 2: Write failing CLI smoke assertions**

Add tests that execute:
- `train.py --help`
- `visualize.py --help`
- `export_phase_plate.py --help`

and assert exit code 0 plus expected keywords in stdout.

- [ ] **Step 3: Run the new tests and verify they fail for the expected reason**

Run:
`uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_readme_has_deliverable_sections tests.test_d2nn_core.D2NNCoreTests.test_cli_entrypoints_expose_help -v`

Expected:
- README test fails because the required section labels are not all present
- CLI smoke test may already pass; if so, keep it as a regression guard and treat the README test as the red test for this task

### Task 2: Upgrade README into a real handoff document

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\README.md`

- [ ] **Step 1: Add a scoped repository positioning section**

State clearly what is reproduced now, what is approximate, and what is outside scope.

- [ ] **Step 2: Add a “推荐复现路径” section**

Document three explicit paths:
- classification baseline
- imaging baseline
- phase-plate export baseline

- [ ] **Step 3: Add a “推荐 checkpoint” section**

Explicitly warn against using `best_mnist.pth` as the paper-faithful 5-layer baseline and point to the recommended checkpoints instead.

- [ ] **Step 4: Add a “论文对照结果” section**

Include a compact table with:
- paper target/result
- current repo result
- alignment note

- [ ] **Step 5: Run the README section test**

Run:
`uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_readme_has_deliverable_sections -v`

Expected: `PASS`

### Task 3: Keep smoke verification explicit and automated

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\tests\test_d2nn_core.py`

- [ ] **Step 1: Tighten the CLI smoke expectations**

Assert that stdout contains the correct high-level descriptions:
- `D2NN training`
- `D2NN visualization`
- `Export phase masks / height maps`

- [ ] **Step 2: Run the CLI smoke test**

Run:
`uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_cli_entrypoints_expose_help -v`

Expected: `PASS`

### Task 4: Final verification and note sync

**Files:**
- Modify via Obsidian CLI: `Papers/复现 · Lin et al. 2018.md`

- [ ] **Step 1: Run the full test suite**

Run:
`uv run python -m unittest discover -s tests -v`

Expected: all tests pass

- [ ] **Step 2: Re-read README**

Run:
`Get-Content README.md -Encoding utf8`

Expected: the README reads as a handoff document rather than a scratchpad

- [ ] **Step 3: Append one short log line to the Obsidian note**

Record that the repository was polished into a more deliverable reproduction package with stronger README guidance and automated smoke checks.
