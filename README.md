# Nonlinear-D2NN

A PyTorch-based reproduction and extension codebase for the Diffractive Deep Neural Network (D2NN) architecture introduced by [Lin et al., *Science* 2018](https://doi.org/10.1126/science.aat8084).

This repository focuses on numerical simulation, experiment management, and phase-plate export for D2NN classification and imaging tasks. In addition to the phase-only baseline, it includes configurable nonlinear field activations for mechanism and placement studies.

## Current Status

- The phase-only reproduction baseline is in place for classification and imaging workflows.
- The current nonlinear line in this repository centers on `incoherent_intensity + back` as the strongest activation / placement combination found so far.
- Nonlinear transfer experiments have already been run on Fashion-MNIST, grayscale CIFAR-10, and RGB CIFAR-10.

## Features

- **Phase-only Forward Propagation**: Numerical simulation of free-space angular spectrum propagation with phase-only modulation.
- **Task Support**:
  - Image classification on MNIST, Fashion-MNIST, CIFAR-10 (grayscale), and CIFAR-10 (RGB)
  - Imaging experiments on STL10 or custom `imagefolder` datasets
- **Nonlinear Activation Extensions**: Configurable optical activation mechanisms (`coherent_amplitude`, `coherent_phase`, `incoherent_intensity`) and placement ablations (`front`, `mid`, `back`, `all`).
- **Physical Export Toolchain**: Converts trained phase masks into thickness / height-map representations and optional STL exports for fabrication-oriented workflows.

## Installation

Requirements:
- Python 3.11+
- PyTorch 2.0+ 

Clone the repository and install dependencies using [uv](https://github.com/astral-sh/uv) (recommended):

Some checkpoints and larger experiment artifacts may be distributed through GitHub Releases instead of being committed directly to the repository.

```bash
# Clone the repository
git clone https://github.com/yeungmkw/nonlinear-d2nn.git
cd nonlinear-d2nn

# Install dependencies using uv
uv sync

# Include test dependencies and notebook support when needed
uv sync --dev --extra notebook
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

### 2. Exploring Nonlinear Activations
You can enable optical nonlinearities after selected diffractive layers using either explicit layer indices (`--activation-positions`) or placement aliases (`--activation-placement`).

```bash
uv run python train.py --task classification --dataset cifar10-rgb --epochs 10 \
    --size 200 --layers 5 \
    --activation-type incoherent_intensity --activation-placement back \
    --activation-preset balanced
```

### 3. Inference and Visualization
After training a checkpoint, you can visualize detector outputs, confusion matrices, reconstructed images, or phase-mask structures.

```bash
uv run python visualize.py --task classification --dataset fashion-mnist \
    --checkpoint checkpoints/best_fashion_mnist.pth
```

### 4. Phase Plate Manufacturing Export
Given an existing checkpoint, export learned phase shifts into physical parameters (height maps, wrapped phase masks) and optional STL meshes based on specified material settings.

```bash
uv run python export_phase_plate.py --task classification \
    --checkpoint checkpoints/best_fashion_mnist.pth --export-stl
```
*Outputs are generated under `exports/<checkpoint_name>/`, including a Markdown report, `.npy` arrays, layer-wise `.csv` data, and optional `.stl` files.*

## Project Structure

- `d2nn.py`: Core logic for optical field propagation (`D2NNBase`, `D2NN`, `D2NNImager`) and nonlinear layers.
- `train.py`: Unified training entrypoint parsing CLI parameters for both classification and imaging.
- `visualize.py`: Inference visualization logic for test sets and phase mask plots.
- `export_phase_plate.py`: Converts model tensor weights to physical dimensions.
- `tasks.py`: Dataset loading, split rules, activation configuration, and task-specific training / evaluation workflows.
- `artifacts.py`: Checkpoint handling, manifest helpers, optics presets, and export utilities.

## Known Limitations & Scope

- **Numerical Simulation Only**: This repository serves as a numerical verification & architectural exploration framework. It does not integrate physical optical tabletop experiments.
- **Physical Accuracy Limitations**: The propagation simulation operates under paraxial approximations using the Angular Spectrum Method (ASM); highly skewed diffraction setups may require more rigorous physics routines.
- **Imaging Scope**: The current imaging examples use STL10 or custom image folders rather than the paper's ImageNet-style natural-image showcase.
- **Fabrication-aware Training**: Physical boundaries and quantizations can be exported after training, but fabrication constraints are not yet deeply integrated into the optimization loop.

## References

- **Core paper**: Lin, X., Rivenson, Y., Yardimci, N. T., Veli, M., Luo, Y., Jarrahi, M., & Ozcan, A. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004-1008. [10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084)
- **Nonlinear follow-up**: Yan, T., Yang, J., Zheng, Z., et al. Multilayer nonlinear diffraction neural networks with programmable and fast ReLU activation function. *Nature Communications* (2025). [Article](https://www.nature.com/articles/s41467-025-65275-0)
- **Nonlinear follow-up**: Wetzstein, G., et al. Reprogrammable Electro-Optic Nonlinear Activation Functions for Optical Neural Networks (2019). [arXiv:1903.04579](https://arxiv.org/abs/1903.04579)
- **Nonlinear follow-up**: Wang, R., et al. A surface-normal photodetector as nonlinear activation function in diffractive optical neural networks (2023). [arXiv:2305.03627](https://arxiv.org/abs/2305.03627)

