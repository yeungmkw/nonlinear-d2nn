# Fashion-MNIST Phase-Only Lab Handoff

## Purpose

This note is the single handoff page for the first lab-facing fabrication round of the regenerated `Fashion-MNIST phase-only 5-layer physics-aligned` baseline. Use it in two phases:

- before entering the lab: confirm which parameters are still missing
- after lab parameters are known: run the final export and decide whether the design is ready for fabrication

## Frozen Target

- Baseline note: `docs/baselines/fashion-mnist-phase-only-5layer-physics-aligned.md`
- Lightpath protocol: `docs/fabrication/fashion-mnist-phase-only-lightpath-protocol.md`
- Repo-tracked official artifact bundle: `docs/official-artifacts/fmnist5-phaseonly-aligned/`
- Checkpoint: `checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.pth`
- Manifest: `checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.json`
- Task: `classification`
- Dataset: `Fashion-MNIST`
- Layers: `5`
- Size: `200`
- Wavelength: `0.00075`
- Layer distance: `0.03`
- Pixel size: `0.0004`

## Lab Parameter Checklist

These items must be confirmed before the final fabrication export is treated as manufacturing-ready.

| Parameter | Status | Current value | What to confirm in lab |
|---|---|---:|---|
| `refractive_index` | pending | `1.7227` dry-run only | final material index used for the real plate |
| `ambient_index` | pending | `1.0` dry-run only | whether the real environment is still air or another surrounding medium |
| `base_thickness_um` | pending | `500.0` dry-run only | minimum printable or machinable substrate thickness |
| `max_relief_um` | required | unset | actual process relief ceiling |
| `quantization_levels` | pending | `256` dry-run only | fabrication-side usable height quantization |
| orientation labeling | pending | not frozen | how row `0` / column `0` and front/back are marked on the fabricated part |
| mechanical aperture constraint | pending | not frozen | whether the full `200 x 200` at `0.4 mm` pitch fits the real mount and clear aperture |

## Final Export Command Template

Replace the angle-bracket placeholders only after the lab values are confirmed.

```bash
uv run python export_phase_plate.py \
  --task classification \
  --checkpoint checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.pth \
  --output-dir exports/fmnist5-phaseonly-aligned-final_<YYYYMMDD> \
  --size 200 \
  --layers 5 \
  --wavelength 0.00075 \
  --layer-distance 0.03 \
  --pixel-size 0.0004 \
  --refractive-index <REFRACTIVE_INDEX> \
  --ambient-index <AMBIENT_INDEX> \
  --base-thickness-um <BASE_THICKNESS_UM> \
  --max-relief-um <MAX_RELIEF_UM> \
  --quantization-levels <QUANTIZATION_LEVELS> \
  --export-stl
```

Expected outputs under the chosen export root:

- `phase_masks.npy`
- `height_map.npy`
- `height_map_manufacturable.npy`
- `thickness_map.npy`
- `height_map_quantized.npy`
- `report.md`
- `metadata.json`
- `layers/`
- `stl/`

## Pre-Fabrication Checks

Run these checks immediately after the final export.

1. Confirm `metadata.json` still points to `checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.pth`.
2. Confirm `report.md` records the intended `refractive_index`, `ambient_index`, `base_thickness_um`, `max_relief_um`, and `quantization_levels`.
3. Check whether `clipped_pixels` is still acceptably low after the real `max_relief_um` is applied.
4. Compare `raw height max` against `current exported relief max` to see how much phase structure is being flattened by the real relief limit.
5. If clipping becomes large after setting the real relief ceiling, adjust fabrication/material parameters first rather than retraining the network.
6. Confirm the quantized height map still matches the intended fabrication resolution; the regenerated baseline was flat from `16` to `128` levels in simulation, but the real process limit should still be recorded.
7. Confirm the total aperture implied by `200 x 0.4 mm = 80 mm` is compatible with the real mount and optical clear aperture.
8. Confirm the lab can preserve orientation labeling for row `0` / column `0`; if not, treat orientation as a calibration variable in the first measurement round.

## First Lab Session SOP

Use this as the minimum first-session checklist after fabrication files are frozen.

1. Bring the final export package together with `docs/fabrication/fashion-mnist-phase-only-lightpath-protocol.md`.
2. Mount the five plates in ascending layer order from layer `01` to layer `05`.
3. Preserve the exported array indexing reference across every plate; do not silently remap orientation between fabrication and mounting.
4. Keep every inter-layer spacing at `30 mm`.
5. Place the detector one additional `30 mm` beyond layer `05`.
6. Use the same Fashion-MNIST input preprocessing assumed by the model: `ToTensor()` only, then centered `66 x 66` amplitude embedding inside the `200 x 200` field.
7. Compare the normalized full output-plane intensity map first.
8. Use detector-mask hotspots only as a secondary sanity check, not as the only success criterion.
9. If the measured output disagrees with simulation, debug in this order: orientation, layer spacing, detector distance, input loading, normalization, then fabrication fidelity.

## Stop Conditions

Do not hand the design off as fabrication-ready if any of the following remains unresolved:

- `max_relief_um` is still unknown
- orientation labeling cannot be preserved from export to physical plate
- real aperture or mount constraints conflict with the `80 mm` design footprint
- clipping becomes severe after applying the real relief ceiling

## Notes

- The current dry-run package remains a traceability reference, not the final fabrication package.
- This handoff note is intentionally phase-only. The nonlinear `incoherent_intensity + back` line remains a later comparison branch rather than the first lab fabrication target.
- For lookup, use `fmnist5-phaseonly-aligned` as the short official name for this fabrication line.
