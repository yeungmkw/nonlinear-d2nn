# Reproduction

This directory isolates fast-moving reproduction logs from the more stable reference docs under `docs/`.

Read these files in this order when rebuilding project context:

1. `docs/INDEX.md`
2. `docs/Reproduction/lin-2018-main-log.md`
3. `docs/Reproduction/nonlinear-layer-plan.md`
4. `docs/baselines/fashion-mnist-phase-only-5layer-physics-aligned.md`

Scope rules:

- `lin-2018-main-log.md` tracks the main reproduction line, timeline, and stage decisions.
- `nonlinear-layer-plan.md` tracks the nonlinear branch separately so the main log does not become the second dumping ground.
- Stable fabrication and baseline references still live outside this directory under `docs/baselines/`, `docs/fabrication/`, and `docs/official-artifacts/`.

The files here are now the preferred in-repo references.
The old Obsidian notes should be treated as archive material unless a missing update needs to be recovered.
