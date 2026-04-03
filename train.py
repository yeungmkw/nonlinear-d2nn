"""
D2NN unified training entrypoint.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch

from tasks import (
    execute_experiment_grid,
    format_experiment_grid_commands,
    run_classification_training,
    run_imaging_training,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(description="D2NN training")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "imaging"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="classification: mnist/fashion-mnist/cifar10-gray; imaging: stl10/imagefolder",
    )
    parser.add_argument("--image-root", type=str, default=None, help="root for imagefolder mode")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--size", type=int, default=200, help="Network pixel resolution (NxN)")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--print-experiment-grid",
        type=str,
        default=None,
        choices=[
            "coherent_amplitude_positions",
            "coherent_amplitude_presets",
            "coherent_phase_presets",
            "coherent_activation_mechanisms",
            "incoherent_intensity_presets",
            "activation_mechanisms",
        ],
        help="print a predefined experiment command grid and exit",
    )
    parser.add_argument(
        "--run-experiment-grid",
        type=str,
        default=None,
        choices=[
            "coherent_amplitude_positions",
            "coherent_amplitude_presets",
            "coherent_phase_presets",
            "coherent_activation_mechanisms",
            "incoherent_intensity_presets",
            "activation_mechanisms",
        ],
        help="run a predefined experiment grid sequentially",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="optional experiment suffix used to keep checkpoints/manifests separate",
    )
    parser.add_argument(
        "--experiment-stage",
        type=str,
        default="baseline",
        help="high-level experiment stage label recorded in manifests",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for splits, loaders, and training")
    parser.add_argument(
        "--activation-type",
        type=str,
        default="none",
        choices=["none", "identity", "coherent_amplitude", "coherent_phase", "incoherent_intensity"],
        help="optional field activation inserted after selected diffractive layers",
    )
    parser.add_argument(
        "--activation-positions",
        type=str,
        default=None,
        help="comma-separated 1-based layer indices after which activations are inserted",
    )
    parser.add_argument(
        "--activation-placement",
        type=str,
        default=None,
        choices=["front", "mid", "back", "all"],
        help="named placement alias resolved from the current layer count",
    )
    parser.add_argument(
        "--activation-preset",
        type=str,
        default=None,
        choices=["conservative", "balanced", "aggressive"],
        help="optional preset for activation hyperparameters (applies to all activation types)",
    )
    parser.add_argument("--activation-threshold", type=float, default=None)
    parser.add_argument("--activation-temperature", type=float, default=None)
    parser.add_argument("--activation-gain-min", type=float, default=None)
    parser.add_argument("--activation-gain-max", type=float, default=None)
    parser.add_argument("--activation-gamma", type=float, default=None)
    parser.add_argument("--activation-responsivity", type=float, default=None)
    parser.add_argument("--activation-emission-phase-mode", type=str, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--layer-distance", type=float, default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "imaging" and args.dataset == "mnist":
        args.dataset = "stl10"
    if args.print_experiment_grid:
        for command in format_experiment_grid_commands(args.print_experiment_grid, args):
            print(command)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).parent
    data_dir = repo_root / "data"
    save_dir = repo_root / args.save_dir
    save_dir.mkdir(exist_ok=True)

    def run_single(run_args):
        seed_everything(run_args.seed)
        print(f"Seed: {run_args.seed}")
        if run_args.task == "classification":
            run_classification_training(run_args, device, data_dir, save_dir)
        else:
            run_imaging_training(run_args, device, data_dir, save_dir)

    if args.run_experiment_grid:
        execute_experiment_grid(args.run_experiment_grid, args, run_single)
        return

    seed_everything(args.seed)
    print(f"Seed: {args.seed}")

    if args.task == "classification":
        run_classification_training(args, device, data_dir, save_dir)
    else:
        run_imaging_training(args, device, data_dir, save_dir)


if __name__ == "__main__":
    main()
