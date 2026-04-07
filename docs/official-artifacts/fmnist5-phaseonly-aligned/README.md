# fmnist5-phaseonly-aligned

This is the current repo-tracked official artifact bundle for the first `Fashion-MNIST` phase-only fabrication line.

## Source

- Source checkpoint: `checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.pth`
- Source manifest snapshot: `source_checkpoint_manifest.json`
- Regeneration command:

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
  --run-name fmnist5-phaseonly-aligned \
  --experiment-stage fabrication_baseline \
  --seed 42
```

## Included

- `phase_masks.npy`
- `height_map.npy`
- `height_map_manufacturable.npy`
- `height_map_quantized.npy`
- `thickness_map.npy`
- `metadata.json`
- `report.md`
- `source_checkpoint_manifest.json`
- `layers/`

These are the directly reusable files that are most likely to be called again during fabrication preparation, review, and parameter checking.

## Intentionally Excluded

- `stl/`

The STL files are still available from the local export package and can be regenerated from the same checkpoint/export command, but they are excluded here to keep the git-tracked official artifact bundle compact and easier to carry inside the repository.

## Related Docs

- Baseline note: `docs/baselines/fashion-mnist-phase-only-5layer-physics-aligned.md`
- Lightpath protocol: `docs/fabrication/fashion-mnist-phase-only-lightpath-protocol.md`
- Lab handoff: `docs/fabrication/fashion-mnist-phase-only-lab-handoff.md`
