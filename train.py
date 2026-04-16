"""
D2NN unified training entrypoint.
"""

import argparse
from pathlib import Path
import random

import numpy as np
import torch
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
    classification_split_lengths,
    execute_experiment_grid,
    format_experiment_grid_commands,
    format_activation_diagnostics,
    get_classification_dataset_config,
    model_activation_diagnostics,
    resolve_activation_config,
    resolve_propagation_config,
    run_imaging_training,
)
from train_core import (
    _run_classification_epoch,
    append_metric_history,
    build_metric_history,
)


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


def fit_classification_model(*, model, train_loader, val_loader, device, epochs, learning_rate, loss_config, checkpoint_path):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_checkpoint_key = (float("-inf"), float("-inf"), 0)
    best_state_dict = None
    last_activation_stats = {}
    history = build_metric_history()

    for epoch in range(1, epochs + 1):
        for split, loader, split_optimizer in (("train", train_loader, optimizer), ("val", val_loader, None)):
            metrics = _run_classification_epoch(
                model,
                loader,
                device,
                optimizer=split_optimizer,
                **loss_config,
            )
            append_metric_history(
                history,
                split=split,
                total=metrics["loss"],
                mse=metrics["mse"],
                ce=metrics["ce"],
                reg=metrics["reg"],
                accuracy=metrics["accuracy"],
                contrast=metrics["contrast"],
            )
            if split == "train":
                train_metrics = metrics
            else:
                val_metrics = metrics

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.2f}% contrast: {train_metrics['contrast']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.2f}% contrast: {val_metrics['contrast']:.4f}"
        )
        last_activation_stats = model_activation_diagnostics(model)
        if last_activation_stats:
            print(f"  activation stats: {format_activation_diagnostics(last_activation_stats)}")

        checkpoint_key = (val_metrics["accuracy"], val_metrics["contrast"], epoch)
        if checkpoint_key > best_checkpoint_key:
            current_state_dict = model.state_dict()
            torch.save(current_state_dict, checkpoint_path)
            best_state_dict = {key: value.detach().cpu().clone() for key, value in current_state_dict.items()} if epoch != epochs else None
            print(
                "  -> Saved best model "
                f"(val acc: {val_metrics['accuracy']:.2f}%, val contrast: {val_metrics['contrast']:.4f}, epoch: {epoch})"
            )
            best_checkpoint_key = checkpoint_key

        scheduler.step()

    return {
        "accuracy": best_checkpoint_key[0],
        "contrast": best_checkpoint_key[1],
        "epoch": best_checkpoint_key[2],
    }, best_state_dict, history, last_activation_stats


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
    num_workers = args.num_workers
    pin_memory = args.pin_memory and device.type == "cuda"
    prefetch_factor = args.prefetch_factor if num_workers > 0 else None
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
    checkpoint_path = checkpoint_variant_path(Path(save_dir) / dataset_cfg["checkpoint_name"], resolved_run_name)

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
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = prefetch_factor
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
    print(
        f"D2NN: {len(model.layers)} layers, {model.size}x{model.size} neurons/layer, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params"
    )

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
    save_manifest(
        checkpoint_manifest_path(checkpoint_path),
        {
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
            **experiment_manifest_fields(
                checkpoint_path=checkpoint_path,
                run_name=resolved_run_name,
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
                runtime_config={
                    "device": str(device),
                    "allow_tf32": args.allow_tf32 and device.type == "cuda",
                    "num_workers": num_workers,
                    "pin_memory": pin_memory,
                    "prefetch_factor": prefetch_factor,
                },
                optics_preset=args.optics_preset,
            ),
            "activation_diagnostics": last_activation_stats,
        },
    )
    paper_target = dataset_cfg["paper_target"]
    print(
        f"\nTest accuracy: {test_metrics['accuracy']:.2f}% | Test contrast: {test_metrics['contrast']:.4f} "
        f"(paper target: {f'{paper_target:.2f}%' if paper_target is not None else 'n/a'}, saved to {checkpoint_path.name})"
    )


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
    if args.task == "classification":
        run_classification_training(args, device, data_dir, save_dir)
    else:
        run_imaging_training(args, device, data_dir, save_dir)


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
