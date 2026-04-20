"""
Unified D2NN training entrypoint and lifecycle orchestration.

This module owns the public training CLI, runtime setup, task dispatch,
and end-to-end training flow. Shared epoch math stays in ``train_core.py``,
and task-specific builders/config helpers stay in layered internal modules.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from artifacts import (
    build_model_for_task,
    checkpoint_manifest_path,
    checkpoint_variant_path,
    derive_experiment_run_name,
    experiment_manifest_fields,
    resolve_training_optics_preset,
    save_manifest,
)
from tasks import (
    MODEL_VERSION,
    build_classification_transform,
    build_imaging_loaders,
    build_imaging_training_model,
    classification_split_lengths,
    execute_experiment_grid,
    evaluate_imaging,
    fit_classification_model,
    fit_imaging_model,
    format_experiment_grid_commands,
    get_classification_dataset_config,
    print_model_summary,
    resolve_activation_config,
    resolve_propagation_config,
)
from train_core import run_classification_epoch


EXPERIMENT_GRID_CHOICES = [
    "coherent_amplitude_positions",
    "coherent_amplitude_presets",
    "coherent_phase_presets",
    "coherent_activation_mechanisms",
    "incoherent_intensity_presets",
    "activation_mechanisms",
]

def validate_training_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    if args.optics_preset == "paper":
        return

    if args.task != "classification":
        raise ValueError("Non-paper optics presets are only supported for classification runs in the current lab path.")

    if args.layers != 1:
        raise ValueError("The current lab optics presets are restricted to single-layer (--layers 1) runs.")

    if args.size != 200:
        raise ValueError("The current lab optics presets are calibrated for size 200 and cannot be combined with --size overrides.")

    if any(
        value is not None
        for value in (
            args.wavelength,
            args.layer_distance,
            args.pixel_size,
            args.input_distance,
            args.output_distance,
        )
    ):
        raise ValueError(
            "When using a lab optics preset, do not override --wavelength/--layer-distance/--pixel-size/--input-distance/--output-distance manually."
        )


def resolve_loader_runtime_config(args, device):
    num_workers = int(args.num_workers)
    return {
        "device": str(device),
        "allow_tf32": bool(args.allow_tf32 and device.type == "cuda"),
        "num_workers": num_workers,
        "pin_memory": bool(args.pin_memory and device.type == "cuda"),
        "prefetch_factor": int(args.prefetch_factor) if num_workers > 0 else None,
    }


def resolve_run_identity(
    *,
    args,
    save_dir,
    checkpoint_name,
    activation_type,
    activation_positions,
    activation_hparams,
    propagation_backend,
    propagation_chunk_size,
    loss_config=None,
):
    resolved_run_name = derive_experiment_run_name(
        run_name=args.run_name,
        experiment_stage=args.experiment_stage,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        seed=args.seed,
        loss_config=loss_config,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        optics_preset=args.optics_preset,
        layer_count=args.layers,
    )
    checkpoint_path = checkpoint_variant_path(Path(save_dir) / checkpoint_name, resolved_run_name)
    return resolved_run_name, checkpoint_path


def build_common_manifest_fields(
    *,
    args,
    checkpoint_path,
    run_name,
    optics,
    activation_type,
    activation_positions,
    activation_hparams,
    propagation_backend,
    propagation_chunk_size,
    runtime_config,
    loss_config=None,
):
    return experiment_manifest_fields(
        checkpoint_path=checkpoint_path,
        run_name=run_name,
        experiment_stage=args.experiment_stage,
        seed=args.seed,
        optics=optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        model_version=MODEL_VERSION,
        loss_config=loss_config,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        runtime_config=runtime_config,
        optics_preset=args.optics_preset,
    )

# Backward-compatible alias kept for existing tests and legacy introspection.
_run_classification_epoch = run_classification_epoch


def run_classification_training(args, device, data_dir, save_dir):
    dataset_cfg = get_classification_dataset_config(args.dataset)
    optics = resolve_training_optics_preset("classification", args.optics_preset).with_overrides(
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(args)
    propagation_backend, propagation_chunk_size = resolve_propagation_config(args=args)
    loss_config = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    runtime_config = resolve_loader_runtime_config(args, device)

    print(f"Dataset: {dataset_cfg['display_name']}")

    transform = build_classification_transform(dataset_cfg)
    train_set = dataset_cfg["dataset_cls"](data_dir, train=True, download=True, transform=transform)
    test_set = dataset_cfg["dataset_cls"](data_dir, train=False, download=True, transform=transform)
    train_len, val_len = classification_split_lengths(len(train_set))
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": runtime_config["num_workers"],
        "pin_memory": runtime_config["pin_memory"],
    }
    if runtime_config["num_workers"] > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = runtime_config["prefetch_factor"]
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_common,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_common)
    test_loader = DataLoader(test_set, shuffle=False, **loader_common)

    model = build_model_for_task(
        "classification",
        optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_chunk_size=propagation_chunk_size,
        propagation_backend=propagation_backend,
    ).to(device)
    resolved_run_name, checkpoint_path = resolve_run_identity(
        args=args,
        save_dir=save_dir,
        checkpoint_name=dataset_cfg["checkpoint_name"],
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        loss_config=loss_config,
    )
    print_model_summary("D2NN", model, task="classification")

    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")

    best_checkpoint_metrics, best_state_dict, history, last_activation_stats = fit_classification_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        loss_config=loss_config,
        checkpoint_path=checkpoint_path,
    )
    if best_checkpoint_metrics["epoch"] != args.epochs:
        model.load_state_dict(best_state_dict if best_state_dict is not None else torch.load(checkpoint_path, weights_only=True))
    test_metrics = _run_classification_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        **loss_config,
    )
    manifest_data = {
        "task": "classification",
        "dataset": dataset_cfg["display_name"],
        "paper_target_accuracy": dataset_cfg["paper_target"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_val_accuracy": best_checkpoint_metrics["accuracy"],
        "best_val_contrast": best_checkpoint_metrics["contrast"],
        "best_epoch": best_checkpoint_metrics["epoch"],
        "test_accuracy": test_metrics["accuracy"],
        "test_contrast": test_metrics["contrast"],
        "test_loss": test_metrics["loss"],
        "history": history,
        **build_common_manifest_fields(
            args=args,
            checkpoint_path=checkpoint_path,
            run_name=resolved_run_name,
            optics=optics,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_backend=propagation_backend,
            propagation_chunk_size=propagation_chunk_size,
            runtime_config=runtime_config,
            loss_config=loss_config,
        ),
        "activation_diagnostics": last_activation_stats,
    }
    save_manifest(checkpoint_manifest_path(checkpoint_path), manifest_data)
    paper_target = dataset_cfg["paper_target"]
    print(
        f"\nTest accuracy: {test_metrics['accuracy']:.2f}% | Test contrast: {test_metrics['contrast']:.4f} "
        f"(paper target: {f'{paper_target:.2f}%' if paper_target is not None else 'n/a'}, saved to {checkpoint_path.name})"
    )


def run_imaging_training(args, device, data_dir, save_dir):
    runtime_config = resolve_loader_runtime_config(args, device)
    dataset_cfg, train_loader, val_loader, test_loader = build_imaging_loaders(args, data_dir, runtime_config)
    print(f"Dataset: {dataset_cfg['display_name']}")

    (
        model,
        optics,
        activation_type,
        activation_positions,
        activation_hparams,
        propagation_backend,
        propagation_chunk_size,
    ) = build_imaging_training_model(args, device)
    resolved_run_name, checkpoint_path = resolve_run_identity(
        args=args,
        save_dir=save_dir,
        checkpoint_name=dataset_cfg["checkpoint_name"],
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
    )
    print_model_summary("D2NNImager", model, task="imaging")
    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    best_val_loss, best_epoch, best_state_dict, last_activation_stats = fit_imaging_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )
    if best_epoch != args.epochs:
        model.load_state_dict(best_state_dict if best_state_dict is not None else torch.load(checkpoint_path, weights_only=True))
    test_loss = evaluate_imaging(model, test_loader, criterion, device)
    manifest_data = {
        "task": "imaging",
        "dataset": dataset_cfg["display_name"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "image_size": args.image_size,
        "input_fraction": args.input_fraction,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "test_mse": test_loss,
        **build_common_manifest_fields(
            args=args,
            checkpoint_path=checkpoint_path,
            run_name=resolved_run_name,
            optics=optics,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_backend=propagation_backend,
            propagation_chunk_size=propagation_chunk_size,
            runtime_config=runtime_config,
        ),
        "activation_diagnostics": last_activation_stats,
    }
    save_manifest(checkpoint_manifest_path(checkpoint_path), manifest_data)
    print(f"\nTest MSE: {test_loss:.4f} (saved to {checkpoint_path.name})")


def run_training_task(args, device, data_dir, save_dir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if device.type == "cuda":
        allow_tf32 = bool(args.allow_tf32)
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = allow_tf32
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        torch.backends.cudnn.benchmark = True
    print(f"Seed: {args.seed}")
    task_runners = {
        "classification": run_classification_training,
        "imaging": run_imaging_training,
    }
    task_runners[args.task](args, device, data_dir, save_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="D2NN training")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "imaging"])
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="classification: mnist/fashion-mnist/cifar10-gray/cifar10-rgb; imaging: stl10/imagefolder",
    )
    parser.add_argument("--image-root", type=str, default=None, help="root for imagefolder mode")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--size", type=int, default=200, help="Network pixel resolution (NxN)")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0, help="classification MSE loss weight")
    parser.add_argument("--beta", type=float, default=0.1, help="classification cross-entropy loss weight")
    parser.add_argument("--gamma", type=float, default=0.01, help="classification regularization loss weight")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--print-experiment-grid",
        type=str,
        default=None,
        choices=EXPERIMENT_GRID_CHOICES,
        help="print a predefined experiment command grid and exit",
    )
    parser.add_argument(
        "--run-experiment-grid",
        type=str,
        default=None,
        choices=EXPERIMENT_GRID_CHOICES,
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
    parser.add_argument(
        "--rs-backend",
        type=str,
        default="direct",
        choices=["direct", "fft"],
        help="Rayleigh-Sommerfeld propagation backend used during forward propagation",
    )
    parser.add_argument(
        "--propagation-chunk-size",
        type=int,
        default=None,
        help="direct-backend target chunk size; ignored by the FFT backend",
    )
    parser.add_argument("--allow-tf32", action="store_true", help="enable TF32 matmul/cuDNN acceleration on CUDA")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--pin-memory", action="store_true", help="pin host memory for CUDA transfers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--layer-distance", type=float, default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--input-distance", type=float, default=None)
    parser.add_argument("--output-distance", type=float, default=None)
    parser.add_argument(
        "--optics-preset",
        type=str,
        default="paper",
        choices=["paper", "lab852_f10", "lab852_f5"],
        help="named optics preset; lab presets are fixed to classification, --layers 1, --size 200, and no manual optics overrides",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.task == "imaging" and args.dataset == "mnist":
        args.dataset = "stl10"
    validate_training_args(args)
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

    if args.run_experiment_grid:
        execute_experiment_grid(
            args.run_experiment_grid,
            args,
            lambda run_args: run_training_task(run_args, device, data_dir, save_dir),
        )
        return

    run_training_task(args, device, data_dir, save_dir)


if __name__ == "__main__":
    main()
