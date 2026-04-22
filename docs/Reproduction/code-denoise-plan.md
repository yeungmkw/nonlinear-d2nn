# Code Denoise Plan

> Source date: 2026-04-21
> Scope: maintenance-noise reduction for the current D2NN codebase.

## Decision

Use a behavior-preserving refactoring path:

1. Clean immediate tool-reported noise.
2. Keep the current file frame intact.
3. Reduce noisy, high-complexity functions inside their current files.
4. Extract private helpers only when they clarify existing code in the same file.
5. Avoid module moves, public API moves, and test-suite file splits.
6. Add lightweight lint/complexity checks only after the cleanup path is stable.

This sequence is intentionally conservative. The current training, visualization, and export behavior should remain unchanged throughout.

## Evidence Base

- Fowler's refactoring guidance emphasizes small behavior-preserving transformations protected by tests: https://martinfowler.com/books/refactoring.html
- Lacerda et al. connect code smells and refactoring with understandability, maintainability, testability, complexity, functionality, and reusability: https://arxiv.org/abs/2004.10777
- Palomba et al. report that long or complex smelly code is common and associated with higher change- and fault-proneness: https://link.springer.com/article/10.1007/s10664-017-9535-z
- Tufano et al. report that many code smells are introduced when artifacts are first created and often survive, favoring early cleanup while boundaries are still visible: https://www.cs.wm.edu/~mtufano/publications/J3.pdf
- Garousi and Kucuk survey test smells and support treating test code as first-class maintainability work: https://doi.org/10.1016/j.jss.2017.12.013

## Current Repository Signals

- `uv run python -m pytest --collect-only -q` collected 172 tests, all under `tests/test_d2nn_core.py::D2NNCoreTests`.
- `uvx ruff check .` initially found one unused import in `tests/test_d2nn_core.py`.
- `uvx radon raw ...` reported:
  - `tests/test_d2nn_core.py`: 2972 LOC
  - `tasks.py`: 1139 LOC
  - `artifacts.py`: 751 LOC
  - `d2nn.py`: 682 LOC
  - `train.py`: 528 LOC
- `uvx radon cc -s -a ...` reported the largest hotspots:
  - `artifacts.derive_experiment_run_name`: D/28
  - `artifacts.resolve_optics`: D/22
  - `tasks.resolve_activation_config`: C/14
  - `tasks.plot_quantization_sensitivity`: C/14
  - `tasks.plot_classification_history`: C/13
  - `tasks.build_experiment_grid`: C/13
  - `train_core._run_classification_epoch`: C/13

## Completion Standard

- Public entrypoints remain stable:
  - `train.py`
  - `visualize.py`
  - `export_phase_plate.py`
  - `export_fmnist5_phaseonly_aligned_final.py`
- `train.py` remains the single public training entrypoint.
- No checkpoint or manifest semantics change as part of cleanup-only work.
- `uvx ruff check .` passes.
- `uv run python -m pytest -q` passes.
- After meaningful stage nodes, run an independent review and record the conclusion in `docs/Reproduction/lin-2018-main-log.md`.
- The existing top-level file structure stays effectively unchanged.

Review order for meaningful stage nodes:

1. Try Claude Code review first when `claude` is available locally.
2. Run Codex CLI review mode for a second independent pass when practical.
3. Use Superpowers subagent review for task-level spec and code-quality checks.
4. If any reviewer cannot produce usable output, state that clearly and fall back to the next review path.

## Phase 1: Immediate Tooling Noise

Status: done.

Actions:

- Remove the unused `build_classification_test_loader` import from `tests/test_d2nn_core.py`.
- Verify with:

```powershell
uvx ruff check .
uv run python -m pytest -q
```

## Phase 2: In-File Complexity Reduction

Goal: reduce the highest complexity hotspots without moving code across files.

Status: started.

Initial targets:

- `artifacts.py`
  - `OpticalConfig.with_overrides`: reduced from B/8 to A/3 by using dataclass replacement with a filtered override map.
  - `derive_experiment_run_name`: reduced from D/28 to A/4 by extracting same-file private helpers.
  - `resolve_optics`: reduced from D/22 to B/7 by extracting same-file private validation helpers.
  - `build_fabrication_readiness_summary`: reduced from B/6 to A/1 by extracting same-file array max, optional float, and clipping-count helpers.
  - `build_layer_stats`: reduced from B/6 to A/3 by extracting same-file layer-entry and optional thickness-stat helpers.
- `tasks.py`
  - `resolve_activation_config`: reduced from C/14 to A/5 by extracting same-file private source-resolution helpers.
  - `plot_quantization_sensitivity`: reduced from C/14 to B/6 by extracting same-file phase-restore, quantized-evaluation, title, and bar-annotation helpers.
  - `plot_sample_output_patterns`: reduced from C/12 to A/4 by extracting same-file sample-collection, input-rendering, and target-label helpers.
  - `plot_classification_history`: reduced from C/13 to A/3 by extracting same-file history-series and metric-plot helpers.
  - `build_experiment_grid`: reduced from C/13 to A/5 by replacing repeated branches with a same-file table-driven variant registry.
  - `activation_preset_hparams`: reduced from B/6 to A/4 by replacing repeated activation preset branches with a same-file preset-family registry.
  - `resolve_activation_positions_from_alias`: reduced from B/7 to A/4 by replacing repeated placement branches with a same-file alias resolver registry.
  - `parse_int_sequence`: reduced from B/7 to A/5 by using one normalized parsing path for strings and iterables.
  - `resolve_propagation_config`: reduced from B/7 to A/5 by extracting same-file explicit-or-manifest value resolution.
  - `format_activation_diagnostics`: reduced from B/7 to A/3 by extracting same-file diagnostic field specifications and per-layer formatting.
  - `plot_output_energy`: reduced from B/6 to A/2 by extracting same-file energy accumulation and grid-plot helpers, with a direct figure-write regression test.
  - `plot_confusion_matrix`: reduced from B/7 to A/2 by extracting same-file confusion accumulation, axis setup, and cell-annotation helpers, with a direct figure-write regression test.
- `train_core.py`
  - `_run_classification_epoch`: reduced from C/13 to B/6 by extracting same-file finite-check, batch-accumulation, and epoch-summary helpers.
- `d2nn.py`
  - `normalize_activation_positions`: reduced from C/11 to A/3 by extracting same-file parsing, validation, and de-duplication helpers.
  - `build_activation_module`: reduced from B/7 to A/4 by replacing repeated activation-type branches with a same-file activation-class registry.
  - `_parse_activation_positions`: reduced from B/6 to A/4 by using one normalized parsing path for strings and iterables.
- `export_fmnist5_phaseonly_aligned_final.py`
  - `build_validation_summary`: reduced from C/17 to A/5 by extracting same-file missing-file, metadata-issue, clipping-warning, and status helpers.
  - `resolve_lab_inputs`: reduced from C/11 to A/5 by extracting same-file CLI/config merge and missing-field helpers.

Rules:

- Do not create new production modules.
- Do not move public functions between files.
- Prefer small private helpers in the same file.
- Preserve function signatures unless all callers and tests already support the change.
- Run the focused test names covering the touched function, then the full suite.

## Phase 3: Local Test Noise Reduction

Goal: make the existing single test file easier to scan without splitting it.

Allowed changes:

- Remove unused imports.
- Consolidate repeated constants or tiny local helper functions inside `tests/test_d2nn_core.py`.
- Keep all current tests in `tests/test_d2nn_core.py`.
- Keep the collection count stable unless a removal is explicitly justified.

Do not:

- Create new test files.
- Convert the current `unittest.TestCase` frame to another framework.
- Hide important setup behind broad fixtures that make test behavior harder to read.

## Phase 4: In-File Documentation and Naming Cleanup

Goal: clarify existing code without adding narrative noise.

Allowed changes:

- Rename private local variables when the current name obscures intent.
- Add short comments only before dense non-obvious blocks.
- Keep public names, CLI flags, manifest keys, and checkpoint names stable.

Verification:

```powershell
uvx ruff check .
uv run python -m pytest -q
```

## Phase 5: Lightweight Gates

After the in-file cleanup stabilizes, add `ruff` to the dev dependency group and document the standard local checks.

Preferred check set:

```powershell
uv run ruff check .
uv run python -m pytest -q
```

Do not add broad formatting churn in the same stage as logic cleanup.

## Non-Goals

- Do not change training semantics.
- Do not rename CLI flags.
- Do not change checkpoint or manifest schema.
- Do not combine cleanup with new nonlinear-layer mechanisms.
- Do not split existing files as part of this cleanup pass.
- Do not add new production modules as part of this cleanup pass.
- Do not rewrite the whole project.
