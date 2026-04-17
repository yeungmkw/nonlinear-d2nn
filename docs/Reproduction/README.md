# Reproduction

This directory isolates fast-moving reproduction logs from the more stable reference docs under `docs/`.

Read these files in this order when rebuilding project context:

1. `docs/INDEX.md`
2. `docs/Reproduction/lin-2018-main-log.md`
3. `docs/Reproduction/nonlinear-layer-plan.md`

Scope rules:

- `lin-2018-main-log.md` tracks the main reproduction line, timeline, and stage decisions.
- `nonlinear-layer-plan.md` tracks the nonlinear branch separately so the main log does not become the second dumping ground.
- The current active lab-validation path is the single-layer workflow under `docs/fabrication/lab-single-layer-workflow.md` and the reusable official artifact bundle under `docs/official-artifacts/`.
- The frozen `fmnist5-phaseonly-aligned` 5-layer bundle is preserved as historical support material; the editable baseline note and phase-mask CSVs are not first-stop references.
- Stable fabrication and baseline references still live outside this directory under `docs/baselines/`, `docs/fabrication/`, and `docs/official-artifacts/`.

The files here are now the preferred in-repo references.
The old Obsidian notes should be treated as archive material unless a missing update needs to be recovered.
