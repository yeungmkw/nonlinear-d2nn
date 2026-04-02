# Fashion-MNIST Phase-Only 5-Layer Baseline

This document freezes the baseline used as the control group before adding any nonlinear layer.

## Identity

- Task: classification
- Dataset: Fashion-MNIST
- Architecture: phase-only D2NN, 5 layers, 200x200 pixels per layer
- Local checkpoint: `checkpoints/best_fashion_mnist.baseline_5layer.pth`
- Local manifest: `checkpoints/best_fashion_mnist.baseline_5layer.json`
- Source checkpoint: `checkpoints/best_fashion_mnist.pth`
- Source commit: `fa6a6d27746d15652b860483d8c4a1c559839ea4`
- Frozen artifact name: `baseline_5layer`

## Optical Configuration

- Wavelength: `0.75 mm`
- Layer distance: `30 mm`
- Pixel size: `0.4 mm`
- Number of layers: `5`
- Layer resolution: `200 x 200`

## Verification

- Architecture inferred from the copied checkpoint: `5 layers`, `size = 200`
- Freeze method: copied from the validated Fashion-MNIST checkpoint after the propagation-path refactor
- Purpose: stable control group for nonlinear-mechanism and placement ablations
- Editable phase-mask exports are versioned under `docs/baselines/fashion-mnist-phase-only-5layer-phase-masks/`

## Reproduction Command

```bash
uv run python train.py \
  --task classification \
  --dataset fashion-mnist \
  --epochs 20 \
  --layers 5 \
  --size 200 \
  --run-name baseline_5layer \
  --experiment-stage baseline \
  --seed 42
```

## Notes

- The checkpoint itself is intentionally gitignored because trained artifacts are stored locally.
- This document is the tracked baseline record inside the repository.
- The checked-in CSV phase masks preserve an editable reference for the pre-nonlinear stage even after later code diverges.
- Later nonlinear experiments should compare against this baseline before changing any other variable.
