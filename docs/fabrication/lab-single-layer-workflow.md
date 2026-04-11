# Lab Single-Layer Workflow

This note defines the minimum lab-facing single-layer workflow without overwriting the existing 5-layer paper-faithful path.

## Goal

Freeze a `1-layer` classification path that uses:

- `wavelength = 852 nm`
- `pixel_size = 1 um`
- `input_distance = 491.302 mm`
- `output_distance = 575.304 mm`
- formal BMP export for SLM loading

## Presets

The lab presets live in `[artifacts.py](../../artifacts.py)` and are selected through `[train.py](../../train.py)`:

- `lab852_f10`
  - `wavelength = 852e-9`
  - `pixel_size = 1e-6`
  - `input_distance = 491.302e-3`
  - `output_distance = 575.304e-3`
  - provisional `layer_distance ~= 1.1737 mm`
- `lab852_f5`
  - `wavelength = 852e-9`
  - `pixel_size = 1e-6`
  - `input_distance = 491.302e-3`
  - `output_distance = 575.304e-3`
  - provisional `layer_distance ~= 2.3474 mm`

The input/output distances above are the current measured optical path values and now override the older Fresnel-number estimate in the code defaults.
The remaining `layer_distance` field is still being used only as a provisional inter-layer spacing placeholder until the multi-layer spacing is confirmed separately.

## Minimum Training Command

```bash
uv run python train.py --task classification --dataset fashion-mnist \
    --epochs 1 --size 200 --layers 1 --seed 42 \
    --optics-preset lab852_f10 --experiment-stage lab_single_layer
```

Recommended first smoke target:

- `dataset = Fashion-MNIST`
- `layers = 1`
- `size = 200`
- `optics-preset = lab852_f10`
- `phase-only`, meaning no activation flags
- use the measured `input_distance/output_distance` from the preset rather than the older single-distance heuristic

## Visualization

```bash
uv run python visualize.py --task classification --dataset fashion-mnist \
    --checkpoint checkpoints/<single-layer-run>.pth --no-show
```

`visualize.py` reads the checkpoint manifest optical config, so lab checkpoints should be visualized under the same optics used at train time.
Do not separate the lab checkpoint `.pth` from its adjacent `.json` manifest unless you also plan to pass the full optics manually.

## Export

```bash
uv run python export_phase_plate.py --task classification \
    --checkpoint checkpoints/<single-layer-run>.pth --export-bmp
```

Formal BMP output uses:

- one file per layer
- `8-bit grayscale BMP`
- filename shape: `layer_01_phase_8bit.bmp`
- mapping rule: wrapped phase in `[0, 2pi)` linearly mapped to grayscale `[0, 255]`

Expected minimum artifacts for a single-layer run:

- from training: checkpoint `.pth` and manifest `.json`
- from visualization: `phase_masks.png`
- from export: `phase_masks.npy` and `bmp/layer_01_phase_8bit.bmp`

## Not In Scope Yet

Do not treat this note as approval to:

- replace the 5-layer paper baseline
- retrain the lab multi-layer path
- freeze a multi-layer lab handoff package
- merge the old fabrication handoff note into this one

The current priority is a stable `train -> visualize -> export -> BMP` single-layer loop.
