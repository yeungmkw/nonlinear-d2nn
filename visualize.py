"""
D2NN unified visualization entrypoint.
"""

import argparse
from pathlib import Path

from tasks import run_classification_visualization, run_imaging_visualization


def build_parser():
    parser = argparse.ArgumentParser(description="D2NN visualization")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "imaging"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="classification: mnist/fashion-mnist/cifar10-gray/cifar10-rgb; imaging: stl10/imagefolder",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None, help="dataset split seed for imagefolder visualization")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-show", action="store_true", help="Save figures without opening windows")
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--layer-distance", type=float, default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "imaging" and args.dataset == "mnist":
        args.dataset = "stl10"

    args.repo_root = Path(__file__).parent

    if args.task == "classification":
        run_classification_visualization(args)
    else:
        run_imaging_visualization(args)


if __name__ == "__main__":
    main()
