# D2NN Docs Index

This index separates active references from historical baselines.
Current lab-validation starts from the single-layer workflow; the frozen `fmnist5-phaseonly-aligned` bundle stays available as a reusable historical artifact set under `docs/official-artifacts/fmnist5-phaseonly-aligned/`.

## Active Code Entrypoints

- `train.py`
- `visualize.py`
- `export_phase_plate.py`

Training architecture rule:

- `train.py` is the unified public training entrypoint.
- `train_core.py` keeps shared epoch/evaluation internals.
- `tasks.py` keeps task-specific builders and layered helpers.

## Active Single-Layer Lab/Fabrication References

- `docs/fabrication/lab-single-layer-workflow.md`
- `docs/fabrication/fashion-mnist-phase-only-lightpath-protocol.md`
- `docs/fabrication/fashion-mnist-phase-only-lab-handoff.md`

## Active Official Artifacts

- `docs/official-artifacts/README.md`
- `docs/official-artifacts/fmnist5-phaseonly-aligned/`
- `export_fmnist5_phaseonly_aligned_final.py`
- `fabrication/fmnist5-phaseonly-aligned.lab.template.json`

## Active Reproduction Context

- `docs/Reproduction/lin-2018-main-log.md`
- `docs/Reproduction/README.md`
- `docs/Reproduction/nonlinear-layer-plan.md`

## Historical Baselines / Frozen 5-Layer Fabrication Line

- `docs/baselines/fashion-mnist-phase-only-5layer-physics-aligned.md`
- The old editable baseline note and phase-mask CSV bundle were removed; keep using the official frozen bundle above for the preserved historical reference.

## Local Runtime Outputs

These are useful during runs, but they are not the first place to look for current reference material.

- `checkpoints/`
- `exports/`
- `figures/`

## Planning Records

- `docs/superpowers/`
