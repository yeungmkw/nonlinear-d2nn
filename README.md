# nonlinear-d2nn

A PyTorch implementation of the Diffractive Deep Neural Network (D2NN) architecture, originally proposed by [Lin et al., *Science* 2018](https://doi.org/10.1126/science.aat8084). 

This repository provides a robust numerical simulation framework for D2NNs, capable of end-to-end training and inference for both classification and lens imaging tasks. It also features extensions for studying the impact of non-linear activation mechanisms within optical layers, and tools for exporting trained model weights directly into physically manufacturable 3D phase plate formats (e.g., STL).

## Features

- **Phase-only Forward Propagation**: A faithful numerical simulation of free-space angular spectrum propagation and phase-only modulation.
- **Task Support**: 
  - Image Classification (MNIST, Fashion-MNIST, CIFAR-10)
  - Imaging Lens Simulation (STL10)
- **Non-linear Activation Extensions**: Configurable optical activation mechanisms (`coherent_amplitude`, `coherent_phase`, `incoherent_intensity`) and placement ablations (`front`, `mid`, `back`, `all`).
- **Physical Export Toolchain**: Extracts checkpoint weights to physical height/thickness representations and generates ready-to-print 3D STL files for fabrication.

## Installation

Requirements:
- Python 3.11+
- PyTorch 2.0+ 

Clone the repository and install dependencies using [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
# Clone the repository
git clone https://github.com/yeungmkw/nonlinear-d2nn.git
cd nonlinear-d2nn

# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Quick Start

### 1. Training a Model
The `train.py` script acts as the entrypoint for all model configurations. Datasets are downloaded automatically.

**Train a 5-layer classification model on Fashion-MNIST:**
```bash
uv run python train.py --task classification --dataset fashion-mnist --epochs 20 --size 200 --layers 5
```

**Train an imaging lens on STL10 natural images:**
```bash
uv run python train.py --task imaging --dataset stl10 --epochs 10 --size 200 --layers 5 --image-size 64 --batch-size 4
```

### 2. Exploring Non-linear Activations
You can introduce optical non-linearities at a specific layer (`--activation-positions`) using the designated physical mechanism.

```bash
uv run python train.py --task classification --dataset cifar10 --epochs 20 \
    --activation-type incoherent_intensity --activation-positions 5 \
    --activation-threshold 0.1 --activation-responsivity 1.0
```

### 3. Inference and Visualization
Evaluate a checkpoint and visualize its energy distribution, confusion matrix, or phase mask structures.

```bash
uv run python visualize.py --task classification --dataset fashion-mnist \
    --checkpoint checkpoints/best_fashion_mnist.pth
```

### 4. Phase Plate Manufacturing Export
Export learned phase shifts into physical parameters (thickness, phase masks) and 3D STL models based on specified material physics.

```bash
uv run python export_phase_plate.py --task classification \
    --checkpoint checkpoints/best_fashion_mnist.pth --export-stl
```
*Outputs are generated under `exports/<checkpoint_name>/`, including a comprehensive `.md` report, `.npy` arrays, layer-wise `.csv` data, and `.stl` files.*

## Project Structure

- `d2nn.py`: Core logic for optical field propagation (`D2NNBase`, `D2NN`, `D2NNImager`) and non-linear layers.
- `train.py`: Unified training entrypoint parsing CLI parameters for both classification and imaging.
- `visualize.py`: Inference visualization logic for test sets and phase mask plots.
- `export_phase_plate.py`: Converts model tensor weights to physical dimensions.
- `tasks.py`: Dataset loading, split rules, test loops, and training workflows.
- `artifacts.py`: Checkpoint saving workflows, metadata manifests, and default optical parameters.

## Known Limitations & Scope

- **Numerical Simulation Only**: This repository serves as a numerical verification & architectural exploration framework. It does not integrate physical optical tabletop experiments.
- **Physical Accuracy Limitations**: The propagation simulation operates under paraxial approximations using the Angular Spectrum Method (ASM); highly skewed diffraction setups may require more rigorous physics routines.
- **Fabrication-aware Training**: While physical boundaries and quantizations can be exported post-training, strict manufacturing penalty constraints during the gradient descent phase (`fabrication-aware training`) are not fully integrated.

## References

- **Original Architecture**: Lin, X., Rivenson, Y., Yardimci, N. T., Veli, M., Luo, Y., Jarrahi, M., & Ozcan, A. (2018). All-optical machine learning using diffractive deep neural networks. *Science*, 361(6406), 1004-1008. [10.1126/science.aat8084](https://doi.org/10.1126/science.aat8084)

## License

MIT License. See [LICENSE](LICENSE) for details.
