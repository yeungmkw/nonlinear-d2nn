# D2NN Phase-Only Baseline

A PyTorch-based reproduction codebase for the Diffractive Deep Neural Network (D2NN) architecture introduced by [Lin et al., *Science* 2018](https://doi.org/10.1126/science.aat8084).

This branch focuses on the phase-only baseline pipeline for D2NN classification, imaging, and phase-plate export. It is intended as the cleaner baseline line before the nonlinear extension branch.

## Current Status

- The phase-only baseline is available for MNIST and Fashion-MNIST classification workflows.
- The imaging pipeline is available for STL10 and custom `imagefolder` datasets.
- This branch is baseline-only and does not include nonlinear activation mechanisms.

## Features

- **Phase-only Forward Propagation**: Numerical simulation of free-space angular spectrum propagation with phase-only modulation.
- **Task Support**:
  - Image classification on MNIST and Fashion-MNIST
  - Imaging experiments on STL10 or custom `imagefolder` datasets
- **Physical Export Toolchain**: Converts trained phase masks into thickness / height-map representations and optional STL exports for fabrication-oriented workflows.

## Installation

Requirements:
- Python 3.11+
- PyTorch 2.0+

Clone the repository and install dependencies using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Clone the repository
git clone https://github.com/yeungmkw/nonlinear-d2nn.git
cd nonlinear-d2nn

# Switch to the phase-only baseline branch
git checkout phase-only-baseline

# Install dependencies
uv sync
```

## Quick Start

### 1. Training a Model
The `train.py` script is the main entrypoint for both classification and imaging workflows. Standard torchvision datasets are downloaded automatically when supported; `imagefolder` mode requires `--image-root`.

**Train a 5-layer classification model on Fashion-MNIST:**
```bash
uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5
```

**Train an imaging lens on STL10 natural images:**
```bash
uv run python train.py --task imaging --dataset stl10 --epochs 10 --size 200 --layers 5 --image-size 64 --batch-size 4
```

### 2. Inference and Visualization
After training a checkpoint, you can visualize detector outputs, confusion matrices, reconstructed images, or phase-mask structures.

```bash
uv run python visualize.py --task classification --dataset fashion-mnist \
    --checkpoint checkpoints/best_fashion_mnist.pth
```

### 3. Phase Plate Manufacturing Export
Given an existing checkpoint, export learned phase shifts into physical parameters (height maps, wrapped phase masks) and optional STL meshes based on specified material settings.

```bash
uv run python export_phase_plate.py --task classification \
    --checkpoint checkpoints/best_fashion_mnist.pth --export-stl
```

*Outputs are generated under `exports/<checkpoint_name>/`, including a Markdown report, `.npy` arrays, layer-wise `.csv` data, and optional `.stl` files.*

## Project Structure

- `d2nn.py`: Core logic for optical field propagation (`D2NN`, `D2NNImager`) and phase-only modeling.
- `train.py`: Unified training entrypoint parsing CLI parameters for both classification and imaging.
- `visualize.py`: Inference visualization logic for test sets and phase mask plots.
- `export_phase_plate.py`: Converts model tensor weights to physical dimensions.
- `tasks.py`: Dataset loading, split rules, and task-specific training / evaluation workflows.
- `artifacts.py`: Checkpoint handling, manifest helpers, optics presets, and export utilities.

## Known Limitations & Scope

- **Numerical Simulation Only**: This repository serves as a numerical verification and architectural exploration framework. It does not integrate physical optical tabletop experiments.
- **Imaging Scope**: The current imaging examples use STL10 or custom image folders rather than the paper's ImageNet-style natural-image showcase.
- **Baseline-only Scope**: This branch preserves the phase-only baseline and does not include the nonlinear activation extension line.
- **Fabrication-aware Training**: Physical boundaries and quantizations can be exported after training, but fabrication constraints are not yet deeply integrated into the optimization loop.

## References

- **Core paper**: Lin, X., Rivenson, Y., Yardimci, N. T., Veli, M., Luo, Y., Jarrahi, M., & Ozcan, A. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004-1008. [10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084)
