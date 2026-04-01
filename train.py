"""
D2NN unified training entrypoint.
"""

import argparse
from pathlib import Path

import torch

from tasks import run_classification_training, run_imaging_training


def build_parser():
    parser = argparse.ArgumentParser(description="D2NN training")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "imaging"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="classification: mnist/fashion-mnist; imaging: stl10/imagefolder",
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
        "--run-name",
        type=str,
        default=None,
        help="optional experiment suffix used to keep checkpoints/manifests separate",
    )
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--layer-distance", type=float, default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "imaging" and args.dataset == "mnist":
        args.dataset = "stl10"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    repo_root = Path(__file__).parent
    data_dir = repo_root / "data"
    save_dir = repo_root / args.save_dir
    save_dir.mkdir(exist_ok=True)

    if args.task == "classification":
        run_classification_training(args, device, data_dir, save_dir)
    else:
        run_imaging_training(args, device, data_dir, save_dir)


if __name__ == "__main__":
    main()
