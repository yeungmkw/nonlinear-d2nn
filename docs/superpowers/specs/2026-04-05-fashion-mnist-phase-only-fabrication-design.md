# Fashion-MNIST Phase-Only Fabrication And Lightpath Calibration Design

**Status:** proposed  
**Date:** 2026-04-05  
**Primary task line:** `Fashion-MNIST phase-only 5-layer baseline`  
**Current frozen reference:** `checkpoints/best_fashion_mnist.baseline_5layer.pth`  
**Intended fabrication target:** retrained physics-aligned baseline after Commit `94bb34b`  
**Related baseline note:** `docs/baselines/fashion-mnist-phase-only-5layer-baseline.md`

## Goal

Turn the completed nonlinear-validation stage into a fabrication-oriented execution path that keeps the nonlinear line scientifically relevant while making the first real optical experiment easier to interpret and debug.

The first physical target is not the current nonlinear best checkpoint. The first physical target is the `Fashion-MNIST phase-only` 5-layer baseline so the project can establish a clean simulation-to-lightpath calibration loop before attempting any device-level nonlinear reproduction.

## Why This Direction

The current repository already supports:

- a frozen `Fashion-MNIST phase-only` baseline,
- reusable phase export tooling,
- nonlinear checkpoints that outperform or match the baseline numerically,
- documented nonlinear conclusions across Fashion-MNIST, grayscale CIFAR-10, and RGB CIFAR-10.

What is still missing is a low-ambiguity physical validation path. If the first fabrication target is chosen from the nonlinear line, the experiment immediately mixes three sources of uncertainty:

1. phase-mask fabrication error,
2. optical alignment and readout mismatch,
3. missing or partially implemented nonlinear device behavior.

That combination makes first-round failures hard to interpret. A `phase-only` baseline removes the third uncertainty and gives the project a better first physical checkpoint.

## Design Decisions

### Decision 1: Freeze the first fabrication target as Fashion-MNIST phase-only

The first fabrication task line is:

- task: `classification`
- dataset: `Fashion-MNIST`
- model type: `phase-only`
- layer count: `5`
- reference checkpoint: `checkpoints/best_fashion_mnist.baseline_5layer.pth`
- fabrication checkpoint: to be regenerated on top of the fixed physical backbone introduced by Commit `94bb34b`

This keeps the first physical target directly comparable to the nonlinear line while avoiding the need to physically implement the current layerwise nonlinear mechanism on day one.

### Decision 2: First optical experiment validates field/output-pattern consistency, not end-to-end classification

The first round of hardware work should answer:

`Does the fabricated phase-only stack produce output-plane energy distributions that follow the same spatial trends as simulation?`

It should not try to answer first:

`Does the physical system already reproduce full classification accuracy?`

This reduces debugging ambiguity. Classification remains the second-stage target after the physical propagation path is calibrated.

### Decision 3: Preserve the nonlinear line as a reference branch, not the first fabrication branch

The nonlinear line remains important. It is kept as:

- the main numerical research result,
- the control comparison for later physical extensions,
- the source of hypotheses about what should change once nonlinear hardware is introduced.

It is not discarded, but it is not the first fabrication deliverable.

## Scope

### In Scope

- freezing the first fabrication checkpoint,
- documenting the simulation-to-lightpath mapping,
- generating a fully traceable fabrication package,
- generating interpretation-oriented visualizations for the chosen baseline,
- running fabrication-readiness checks on exported phase data,
- defining the success criteria for the first optical calibration round.

### Out of Scope

- new nonlinear mechanism ablations,
- new position ablations,
- longer training sweeps,
- new seed sweeps,
- direct fabrication of `incoherent_intensity + back`,
- implementing physical nonlinear devices in this phase,
- claiming hardware classification closure in the first measurement round.

## Execution Architecture

The work is split into five tightly scoped outputs.

### 1. Baseline Freeze Output

Purpose: remove checkpoint ambiguity before any downstream work.

Required output:

- one canonical `Fashion-MNIST phase-only` fabrication line,
- one regenerated physics-aligned checkpoint for that line,
- one written note that explains why this target is chosen over the nonlinear best line.

Primary source:

- `docs/baselines/fashion-mnist-phase-only-5layer-baseline.md`

### 2. Network Understanding Output

Purpose: satisfy the advisor's requirement to understand what the network learned before fabrication.

Required output:

- 5 learned phase-mask visualizations,
- a small set of representative output-plane or propagation visualizations,
- a concise comparison between `phase-only` and `incoherent_intensity + back`,
- a quantization-sensitivity summary tied to fabrication precision.

Boundary:

This is not a reopened nonlinear research phase. It exists only to support baseline selection, fabrication settings, and optical interpretation.

### 3. Simulation-to-Lightpath Mapping Output

Purpose: define how each simulation quantity maps to the first physical experiment.

Required content:

- input pattern loading method,
- layer spacing mapping,
- phase plate ordering and orientation,
- output-plane measurement location,
- detector-region or output-region interpretation,
- normalization and comparison rule between simulation and measured intensity maps.

This output should let someone set up the experiment without reopening the training code.

### 4. Fabrication Package Output

Purpose: transform the frozen checkpoint into a manufacturing-ready package with traceable parameters.

Required package contents:

- `phase_masks.npy`
- `height_map.npy`
- `height_map_manufacturable.npy`
- `thickness_map.npy`
- per-layer CSV exports
- `report.md`
- `metadata.json`
- `stl/` if required by the fabrication workflow

Required frozen parameters:

- wavelength,
- layer distance,
- pixel size,
- refractive index,
- ambient index,
- base thickness,
- maximum relief,
- quantization levels.

### 5. Fabrication-Readiness Output

Purpose: decide whether the current exported design is manufacturable before any handoff.

Required checks:

- maximum relief against process limit,
- clipping severity after relief limiting,
- quantization adequacy,
- aperture and pixel pitch compatibility with the available optical setup.

If these checks fail, the first adjustment target is fabrication/material configuration, not retraining.

## Data Flow

1. Start from the frozen `Fashion-MNIST phase-only` reference and retrain the physics-aligned baseline on top of Commit `94bb34b`.
2. Produce interpretation-oriented visualizations for human understanding.
3. Write the simulation-to-lightpath mapping note.
4. Export the fabrication package from the regenerated checkpoint using the existing phase export toolchain.
5. Evaluate fabrication readiness on the exported package.
6. Use the resulting package and mapping note for first-round optical calibration.
7. Only after physical propagation is credible, move on to classification closure and later nonlinear hardware comparison.

## Error Handling And Decision Rules

### If the chosen checkpoint is ambiguous

Stop and resolve the canonical target before generating any new fabrication output.

### If phase exports are traceable but not manufacturable

Do not reopen training first. Revisit:

- material refractive index assumptions,
- base thickness,
- maximum relief,
- quantization level,
- fabrication process limits.

### If fabricated output does not match simulation trends

Debug in this order:

1. plate ordering and orientation,
2. layer spacing,
3. output-plane distance,
4. input loading method,
5. measurement normalization,
6. fabrication fidelity.

Do not interpret this as a nonlinear-theory failure because the first experiment is intentionally phase-only.

### If physical propagation matches simulation but classification is weak

Treat that as a second-stage readout or system-integration issue, not a failure of the first fabrication phase.

## Verification Strategy

The design is considered ready for implementation when the following are all true:

- the canonical fabrication checkpoint is named explicitly,
- the first-round optical success criterion is explicitly `output-pattern consistency`,
- the nonlinear line is explicitly preserved as a later comparison branch,
- fabrication outputs and frozen parameters are explicitly listed,
- manufacturability checks are defined,
- no step depends on adding new nonlinear model behavior first.

## Success Criteria

This design succeeds if it enables the team to do the following without reopening the project scope:

1. freeze one physical baseline,
2. explain why it is the first physical target,
3. export a fabrication package with traceable assumptions,
4. run a first optical experiment whose failure modes are diagnosable,
5. preserve a clean path back to the nonlinear line after calibration.

## Immediate Next Step (Execution Checklist)

**Phase 1: Code & Baseline Alignment (Pre-Fabrication Params)**
- [ ] 1. **Retrain Baseline:** Based on the fixed physical backbone (Commit 94bb34b, with Zero-padding ASM), retrain the `Fashion-MNIST phase-only` 5-layer baseline model.
- [ ] 2. **Network Visuals:** Run `visualize.py` on the new model to generate phase distribution maps and confusion matrices, satisfying the `Network Understanding Output`.
- [ ] 3. **Export Dry-Run:** Run `export_phase_plate.py --export-stl` using the new checkpoint to verify that the generation path for Numpy, CSV, and STL files is clear.
- [ ] 4. **Mapping Note Draft:** Write the "Simulation-to-Lightpath Mapping Protocol," pre-defining input loading methods, layer spacing mapping, and measurement normalization standards for physical experiments.

**Phase 2: Fabrication Handoff (Requires Lab Parameters)**
- [ ] 5. **Parameter Sync:** Obtain precise fabrication parameters from the lab (wavelength, pixel size, refractive index, base thickness, maximum relief height, quantization levels).
- [ ] 6. **Final Export:** Substitute the real parameters into `export_phase_plate.py` to generate the final Fabrication Package.
