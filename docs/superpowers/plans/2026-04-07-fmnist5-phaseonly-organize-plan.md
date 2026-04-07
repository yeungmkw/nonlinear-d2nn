# FMNIST5 Phaseonly Organize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, repo-tracked official entry for the current `fmnist5-phaseonly-aligned` fabrication line and document which download-folder assets are redundant versus still worth keeping.

**Architecture:** Keep the existing code and experiment layout mostly unchanged. Add a small set of documentation/index files plus one tracked official artifact directory for the current phase-only fabrication line, regenerated from the current physics-aligned checkpoint/export flow. Audit download-folder duplicates separately and record recommendations without deleting user files automatically.

**Tech Stack:** Python (`uv`, `pytest`), PowerShell, Markdown docs, existing D2NN training/export scripts

---

### Task 1: Add Official Artifact Naming And Lookup Docs

**Files:**
- Create: `docs/INDEX.md`
- Create: `docs/official-artifacts/README.md`
- Test: manual content verification

- [ ] Draft `docs/INDEX.md` with short sections for code entrypoints, baseline docs, fabrication docs, official artifacts, and ignored runtime outputs.
- [ ] Draft `docs/official-artifacts/README.md` with the naming rule centered on `fmnist5-phaseonly-aligned`, plus a short rule that only current official artifacts go here.
- [ ] Verify both docs point to the current baseline note and lightpath/handoff notes.
- [ ] Commit the doc-only changes.

### Task 2: Regenerate The Official Physics-Aligned Artifact Set

**Files:**
- Modify: local ignored runtime outputs under `checkpoints/`, `exports/`, `figures/`
- Test: `uv run python train.py ...`, `uv run python visualize.py ...`, `uv run python export_phase_plate.py ...`

- [ ] Retrain `best_fashion_mnist.fmnist5-phaseonly-aligned.pth` with the frozen optics and run name recorded in the baseline note.
- [ ] Run the understanding-report export for that checkpoint so the official artifact set has matching visual references.
- [ ] Run the phase export dry-run for the regenerated checkpoint.
- [ ] Verify the regenerated checkpoint, manifest, figure set, and export package all exist locally before copying tracked files.

### Task 3: Create The Repo-Tracked Official Artifact Directory

**Files:**
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/README.md`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/phase_masks.npy`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/height_map.npy`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/height_map_manufacturable.npy`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/height_map_quantized.npy`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/thickness_map.npy`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/metadata.json`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/report.md`
- Create: `docs/official-artifacts/fmnist5-phaseonly-aligned/layers/*.csv`
- Test: file presence + hash/metadata spot checks

- [ ] Copy the directly reusable phase/export files from the regenerated local export package into the tracked `docs/official-artifacts/fmnist5-phaseonly-aligned/` directory.
- [ ] Exclude STL files from git-tracked official artifacts and explain why in the local `README.md`.
- [ ] Write the local `README.md` so it states source checkpoint, regeneration command, what is included, what is intentionally excluded, and where to find the fabrication handoff note.
- [ ] Verify the copied files match the regenerated export package.
- [ ] Commit the tracked official artifact directory.

### Task 4: Audit Download-Folder Duplicates Without Deleting Anything

**Files:**
- Create: `docs/official-artifacts/download-audit-2026-04-07.md`
- Test: hash comparison results included in the audit note

- [ ] Compare the known download-folder phase assets against the repo-tracked official artifact set and existing release asset zips.
- [ ] Use an independent reviewer/agent for the “can delete / keep / unresolved” classification before finalizing the note.
- [ ] Write the audit note with three buckets: covered by repo official artifacts, keep as release/archive assets, unresolved/manual-check.
- [ ] Do not delete any download-folder files; only record recommendations.
- [ ] Commit the audit note.

### Task 5: Final Verification And Stage Review

**Files:**
- Modify: `C:\Users\Jiangqianxian\iCloudDrive\iCloud~md~obsidian\Wayne Yang\Papers\复现 · Lin et al. 2018.md`
- Test: `uv run python -m pytest -q`

- [ ] Run the full test suite in the worktree.
- [ ] Run an independent stage review, preferring local Claude Code if it returns usable output.
- [ ] Record the stage result in the main Obsidian reproduction log.
- [ ] Prepare the branch for merge after the review passes.
