# Fashion-MNIST Phase-Only 5-Layer Physics-Aligned Baseline

## Identity

- Task: classification
- Dataset: Fashion-MNIST
- Architecture: phase-only D2NN, 5 layers, 200 x 200 pixels per layer
- Experiment stage: `fabrication_baseline`
- Seed: `42`
- Checkpoint target: `checkpoints/best_fashion_mnist.baseline_5layer_physics_aligned.pth`
- Manifest target: `checkpoints/best_fashion_mnist.baseline_5layer_physics_aligned.json`
- Source commit: `94bb34b`
- Planning/spec alignment commit: `7763241`
- Frozen artifact name: `baseline_5layer_physics_aligned`

## Optical Configuration

- Wavelength: `0.00075 m`
- Layer distance: `0.03 m`
- Pixel size: `0.0004 m`
- Number of layers: `5`
- Layer resolution: `200 x 200`

## Verification

- Best validation accuracy: `85.34%`
- Test accuracy: `84.53%`
- Test loss: `0.025320793786644934`
- The regenerated checkpoint and manifest now anchor the fabrication baseline on top of the zero-padding ASM backbone introduced in Commit `94bb34b`.

## Understanding Report

- Baseline figures: `figures/fashion_mnist/baseline_5layer_physics_aligned/`
- Generated artifacts: `phase_masks.png`, `output_energy.png`, `confusion_matrix.png`, `sample_output_patterns.png`, `quantization_sensitivity.png`
- Quantization sensitivity on the regenerated baseline: `84.53%` at full precision, `83.97%` at `8` levels, `84.57%` at `16` levels, `84.48%` at `32` levels, `84.55%` at `64` levels, `84.51%` at `128` levels
- Interpretation: `16` through `128` uniform phase levels stay effectively flat for this baseline, while `8` levels is the first noticeable drop.

## Nonlinear Reference

- Comparison checkpoint: `checkpoints/best_fashion_mnist.incoherent_back_20ep.pth`
- Comparison figure directory: `figures/fashion_mnist/incoherent_back_20ep/`
- Run name: `incoherent_back_20ep`
- Activation type: `incoherent_intensity`
- Activation positions: `[5]`
- Best validation accuracy: `87.84%`
- Test accuracy: `87.61%`
- Purpose: retained nonlinear reference branch for later comparison, not the first fabrication target
- Interpretation: the nonlinear branch still scores higher numerically (`87.61%` vs `84.53%`), but the fabrication path stays anchored to the phase-only baseline so the first optical loop isolates propagation, alignment, and fabrication effects before nonlinear hardware is introduced.

## Reproduction Command

```bash
uv run python train.py \
  --task classification \
  --dataset fashion-mnist \
  --epochs 20 \
  --layers 5 \
  --size 200 \
  --wavelength 0.00075 \
  --layer-distance 0.03 \
  --pixel-size 0.0004 \
  --run-name baseline_5layer_physics_aligned \
  --experiment-stage fabrication_baseline \
  --seed 42
```

## Notes

This file tracks the regenerated physics-aligned baseline that feeds the current fabrication workflow. Future optical and nonlinear comparisons should refer back to this record before changing the task line or optical assumptions.
