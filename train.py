"""
D2NN unified training entrypoint.
"""

import argparse
from dataclasses import dataclass
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
    run_imaging_training,
)
from train_core import (
    _run_classification_epoch,
    append_metric_history,
    build_metric_history,
    classification_composite_loss,
    d2nn_mse_loss,
    is_better_classification_checkpoint,
    phase_smoothness_regularizer,
)


EXPERIMENT_GRID_CHOICES = [
    "coherent_amplitude_positions",
    "coherent_amplitude_presets",
    "coherent_phase_presets",
    "coherent_activation_mechanisms",
    "incoherent_intensity_presets",
    "activation_mechanisms",
]


@dataclass(frozen=True)
class ClassificationRunConfig:
    dataset_cfg: dict
    optics: object
    activation_type: str
    activation_positions: tuple
    activation_hparams: dict
    loss_config: dict
    resolved_run_name: str | None
    checkpoint_path: Path
    manifest_path: Path
    propagation_backend: str
    propagation_chunk_size: int | None
    runtime_config: dict
    optics_preset: str


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runtime_config(args, device):
    return {
        "device": str(device),
        "allow_tf32": bool(getattr(args, "allow_tf32", False) and device.type == "cuda"),
        "num_workers": int(getattr(args, "num_workers", 0)),
        "pin_memory": bool(getattr(args, "pin_memory", False) and device.type == "cuda"),
        "prefetch_factor": int(getattr(args, "prefetch_factor", 2)) if getattr(args, "num_workers", 0) > 0 else None,
    }


def configure_runtime(args, device):
    if device.type != "cuda":
        return

    allow_tf32 = bool(args.allow_tf32)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    torch.backends.cudnn.benchmark = True


def validate_training_args(args):
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


def append_epoch_history(history, train_metrics, val_metrics):
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
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


def maybe_save_best_classification_checkpoint(model, checkpoint_path, val_metrics, epoch, best_checkpoint_metrics):
    candidate_metrics = {
        "accuracy": val_metrics["accuracy"],
        "contrast": val_metrics["contrast"],
        "epoch": epoch,
    }
    if not is_better_classification_checkpoint(candidate_metrics, best_checkpoint_metrics):
        return best_checkpoint_metrics

    torch.save(model.state_dict(), checkpoint_path)
    print(
        "  -> Saved best model "
        f"(val acc: {val_metrics['accuracy']:.2f}%, val contrast: {val_metrics['contrast']:.4f}, epoch: {epoch})"
    )
    return candidate_metrics


def save_classification_manifest(
    manifest_path,
    *,
    checkpoint_path,
    dataset_cfg,
    args,
    best_checkpoint_metrics,
    test_metrics,
    history,
    optics,
    activation_type,
    activation_positions,
    activation_hparams,
    resolved_run_name,
    loss_config,
    last_activation_stats,
    propagation_backend,
    propagation_chunk_size,
    runtime_config,
):
    save_manifest(
        manifest_path,
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
            runtime_config=runtime_config,
            optics_preset=args.optics_preset,
        ),
            "activation_diagnostics": last_activation_stats,
        },
    )


def build_classification_loaders(args, data_dir, dataset_cfg):
    transform = build_classification_transform(dataset_cfg)
    dataset_cls = dataset_cfg["dataset_cls"]
    train_set = dataset_cls(data_dir, train=True, download=True, transform=transform)
    test_set = dataset_cls(data_dir, train=False, download=True, transform=transform)
    train_len, val_len = classification_split_lengths(len(train_set))
    train_set, val_set = torch.utils.data.random_split(
        train_set,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": bool(args.pin_memory and torch.cuda.is_available()),
    }
    if args.num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = args.prefetch_factor

    return (
        DataLoader(
            train_set,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed),
            **loader_common,
        ),
        DataLoader(val_set, shuffle=False, **loader_common),
        DataLoader(test_set, shuffle=False, **loader_common),
    )


def build_classification_model(args, device):
    base_optics = resolve_training_optics_preset("classification", args.optics_preset)
    optics = base_optics.with_overrides(
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(args)
    model = build_model_for_task(
        "classification",
        optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_chunk_size=args.propagation_chunk_size,
        propagation_backend=args.rs_backend,
    ).to(device)
    return model, optics, activation_type, activation_positions, activation_hparams


def build_classification_run_config(
    args,
    save_dir,
    dataset_cfg,
    optics,
    activation_type,
    activation_positions,
    activation_hparams,
):
    loss_config = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    runtime_config = build_runtime_config(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    resolved_run_name = derive_experiment_run_name(
        run_name=args.run_name,
        experiment_stage=args.experiment_stage,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        seed=args.seed,
        loss_config=loss_config,
        propagation_backend=args.rs_backend,
        propagation_chunk_size=args.propagation_chunk_size,
        optics_preset=args.optics_preset,
        layer_count=args.layers,
    )
    checkpoint_path = checkpoint_variant_path(save_dir / dataset_cfg["checkpoint_name"], resolved_run_name)
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    return ClassificationRunConfig(
        dataset_cfg=dataset_cfg,
        optics=optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        loss_config=loss_config,
        resolved_run_name=resolved_run_name,
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
        propagation_backend=args.rs_backend,
        propagation_chunk_size=args.propagation_chunk_size,
        runtime_config=runtime_config,
        optics_preset=args.optics_preset,
    )


def fit_classification_model(args, model, train_loader, val_loader, device, checkpoint_path):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_checkpoint_metrics = {"accuracy": float("-inf"), "contrast": float("-inf"), "epoch": 0}
    last_activation_stats = {}
    history = build_metric_history()

    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_classification_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        val_metrics = _run_classification_epoch(
            model,
            val_loader,
            device,
            optimizer=None,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        append_epoch_history(history, train_metrics, val_metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_metrics['loss']:.4f} acc: {train_metrics['accuracy']:.2f}% contrast: {train_metrics['contrast']:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.2f}% contrast: {val_metrics['contrast']:.4f}"
        )
        last_activation_stats = model_activation_diagnostics(model)
        if last_activation_stats:
            print(f"  activation stats: {format_activation_diagnostics(last_activation_stats)}")

        best_checkpoint_metrics = maybe_save_best_classification_checkpoint(
            model,
            checkpoint_path,
            val_metrics,
            epoch,
            best_checkpoint_metrics,
        )

        scheduler.step()

    return best_checkpoint_metrics, history, last_activation_stats


def finalize_classification_run(
    args,
    model,
    test_loader,
    device,
    run_config,
    best_checkpoint_metrics,
    history,
    last_activation_stats,
):
    model.load_state_dict(torch.load(run_config.checkpoint_path, weights_only=True))
    test_metrics = _run_classification_epoch(
        model,
        test_loader,
        device,
        optimizer=None,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    save_classification_manifest(
        run_config.manifest_path,
        checkpoint_path=run_config.checkpoint_path,
        dataset_cfg=run_config.dataset_cfg,
        args=args,
        best_checkpoint_metrics=best_checkpoint_metrics,
        test_metrics=test_metrics,
        history=history,
        optics=run_config.optics,
        activation_type=run_config.activation_type,
        activation_positions=run_config.activation_positions,
        activation_hparams=run_config.activation_hparams,
        resolved_run_name=run_config.resolved_run_name,
        loss_config=run_config.loss_config,
        last_activation_stats=last_activation_stats,
        propagation_backend=run_config.propagation_backend,
        propagation_chunk_size=run_config.propagation_chunk_size,
        runtime_config=run_config.runtime_config,
    )
    paper_target = run_config.dataset_cfg["paper_target"]
    paper_target_text = f"{paper_target:.2f}%" if paper_target is not None else "n/a"
    print(
        f"\nTest accuracy: {test_metrics['accuracy']:.2f}% | "
        f"Test contrast: {test_metrics['contrast']:.4f} "
        f"(paper target: {paper_target_text}, saved to {run_config.checkpoint_path.name})"
    )


def run_classification_training(args, device, data_dir, save_dir):
    dataset_cfg = get_classification_dataset_config(args.dataset)
    print(f"Dataset: {dataset_cfg['display_name']}")

    train_loader, val_loader, test_loader = build_classification_loaders(args, data_dir, dataset_cfg)
    model, optics, activation_type, activation_positions, activation_hparams = build_classification_model(args, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"D2NN: {args.layers} layers, {args.size}x{args.size} neurons/layer, {total_params} trainable params")

    run_config = build_classification_run_config(
        args,
        save_dir,
        dataset_cfg,
        optics,
        activation_type,
        activation_positions,
        activation_hparams,
    )
    if run_config.resolved_run_name:
        print(f"Run name: {run_config.resolved_run_name}")

    best_checkpoint_metrics, history, last_activation_stats = fit_classification_model(
        args,
        model,
        train_loader,
        val_loader,
        device,
        run_config.checkpoint_path,
    )

    finalize_classification_run(
        args,
        model,
        test_loader,
        device,
        run_config,
        best_checkpoint_metrics,
        history,
        last_activation_stats,
    )


def run_training_task(args, device, data_dir, save_dir):
    seed_everything(args.seed)
    configure_runtime(args, device)
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
