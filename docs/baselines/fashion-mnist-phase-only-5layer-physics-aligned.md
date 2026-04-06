# Fashion-MNIST Phase-Only 5-Layer Physics-Aligned Baseline

## Identity

- Task: classification
- Dataset: Fashion-MNIST
- Architecture: phase-only D2NN, 5 layers, 200 x 200 pixels per layer
- Checkpoint target: `checkpoints/best_fashion_mnist.baseline_5layer_physics_aligned.pth`
- Manifest target: `checkpoints/best_fashion_mnist.baseline_5layer_physics_aligned.json`
- Source commit: `94bb34b`
- Planning/spec alignment commit: `7763241`
- Frozen artifact name: `baseline_5layer_physics_aligned`

## Reproduction Command

```bash
uv run python train.py \
  --task classification \
  --dataset fashion-mnist \
  --epochs 20 \
  --layers 5 \
  --size 200 \
  --run-name baseline_5layer_physics_aligned \
  --experiment-stage fabrication_baseline \
  --seed 42
```

## Notes

This file tracks the physics-aligned baseline that will be used by the fabrication workflow. Accuracy lines are added after retraining so the tracked record stays aligned with the regenerated checkpoint and manifest rather than with the earlier baseline artifact.
