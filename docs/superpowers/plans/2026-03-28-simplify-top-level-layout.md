# Simplify Top-Level Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the top-level Python surface to roughly six main files while preserving the existing training, visualization, export, and testing behavior.

**Architecture:** Merge task-specific helpers into a single `tasks.py` and merge shared artifact/config/visualization/export helpers into a single `artifacts.py`. Keep `d2nn.py`, `train.py`, `visualize.py`, and `export_phase_plate.py` as the user-facing runtime surface, and remove obsolete thin wrappers once the unified modules are wired in.

**Tech Stack:** Python 3.13, PyTorch, torchvision, unittest, Obsidian CLI for note sync

---

### Task 1: Lock in the new module boundaries with tests

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\tests\test_d2nn_core.py`

- [x] **Step 1: Write the failing test**

Add import-based checks that require `artifacts.py` to re-export shared helper functions and require `tasks.py` to expose both classification and imaging task entrypoints.

- [x] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_artifacts_module_reexports_shared_helpers tests.test_d2nn_core.D2NNCoreTests.test_tasks_module_exposes_both_task_families -v`
Expected: `ModuleNotFoundError` for `artifacts` and `tasks`

### Task 2: Build the consolidated helper module

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\artifacts.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\export_phase_plate.py`

- [ ] **Step 1: Implement `artifacts.py`**

Move or re-export the contents currently spread across `experiment_config.py`, `artifact_utils.py`, `viz_utils.py`, and `phase_plate_utils.py` into one module with stable names.

- [ ] **Step 2: Point `export_phase_plate.py` at `artifacts.py`**

Update imports so the export entrypoint only depends on `artifacts.py` plus `d2nn.py`.

- [ ] **Step 3: Run the targeted tests**

Run: `uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_artifacts_module_reexports_shared_helpers -v`
Expected: `PASS`

### Task 3: Build the consolidated task module

**Files:**
- Create: `C:\Users\Jiangqianxian\source\repos\d2nn\tasks.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\train.py`
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\visualize.py`

- [ ] **Step 1: Implement `tasks.py`**

Merge the contents of `classification_task.py` and `imaging_task.py` into one module while preserving the four task entry functions used by the tests.

- [ ] **Step 2: Point `train.py` and `visualize.py` at `tasks.py`**

Keep the CLI behavior unchanged and only swap import targets plus dispatch wiring.

- [ ] **Step 3: Run the targeted tests**

Run: `uv run python -m unittest tests.test_d2nn_core.D2NNCoreTests.test_tasks_module_exposes_both_task_families -v`
Expected: `PASS`

### Task 4: Remove obsolete top-level wrappers and helper fragments

**Files:**
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\classification_task.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\imaging_task.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\artifact_utils.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\experiment_config.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\phase_plate_utils.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\viz_utils.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\train_imager.py`
- Delete: `C:\Users\Jiangqianxian\source\repos\d2nn\visualize_imager.py`

- [ ] **Step 1: Delete the obsolete files once imports are clean**

Only remove files after the unified modules are in use and targeted tests are green.

- [ ] **Step 2: Re-run the full test suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: all tests pass

- [ ] **Step 3: Verify the three runtime entrypoints**

Run:
- `uv run python train.py --help`
- `uv run python visualize.py --help`
- `uv run python export_phase_plate.py --help`

Expected: all three commands exit successfully

### Task 5: Sync docs and Obsidian note to the six-file layout

**Files:**
- Modify: `C:\Users\Jiangqianxian\source\repos\d2nn\README.md`
- Modify via Obsidian CLI: `Papers/复现 · Lin et al. 2018.md`

- [ ] **Step 1: Update README**

Replace the old multi-file structure description with the six-file layout and note that the imaging wrappers were removed.

- [ ] **Step 2: Update the Obsidian note**

Record the final six-file layout and the reason for the merge using `obsidian` CLI only.

- [ ] **Step 3: Sanity-check both documents**

Run:
- `Get-Content README.md -Encoding utf8`
- `obsidian read vault="Wayne Yang" path="Papers/复现 · Lin et al. 2018.md"`

Expected: new layout is described consistently and renders without control-character corruption
