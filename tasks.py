"""
Consolidated D2NN task helpers for classification and imaging workflows.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from artifacts import (
    CLASSIFIER_PAPER_OPTICS,
    IMAGER_PAPER_OPTICS,
    build_model_for_task,
    checkpoint_manifest_path,
    checkpoint_variant_path,
    configure_matplotlib_backend,
    derive_experiment_run_name,
    ensure_checkpoint_version,
    experiment_manifest_fields,
    load_checkpoint_state_dict,
    maybe_show,
    plot_phase_masks,
    quantize_phase_masks_uniform,
    read_checkpoint_manifest,
    resolve_optics,
    resolve_training_optics_preset,
    save_manifest,
)
from train_core import (
    append_metric_history,
    build_metric_history,
    classification_composite_loss,
    d2nn_mse_loss,
    evaluate_classification,
    is_better_classification_checkpoint,
    phase_smoothness_regularizer,
)


MODEL_VERSION = "rs_v1"


CLASSIFICATION_DATASETS = {
    "mnist": {
        "dataset_cls": datasets.MNIST,
        "display_name": "MNIST",
        "checkpoint_name": "best_mnist.pth",
        "paper_target": 91.75,
        "default_output_dir": "figures",
        "grayscale": False,
    },
    "fashion_mnist": {
        "dataset_cls": datasets.FashionMNIST,
        "display_name": "Fashion-MNIST",
        "checkpoint_name": "best_fashion_mnist.pth",
        "paper_target": 81.13,
        "default_output_dir": "figures/fashion_mnist",
        "grayscale": False,
    },
    "cifar10_gray": {
        "dataset_cls": datasets.CIFAR10,
        "display_name": "CIFAR-10 (grayscale)",
        "checkpoint_name": "best_cifar10_gray.pth",
        "paper_target": None,
        "default_output_dir": "figures/cifar10_gray",
        "grayscale": True,
    },
    "cifar10_rgb": {
        "dataset_cls": datasets.CIFAR10,
        "display_name": "CIFAR-10 (RGB)",
        "checkpoint_name": "best_cifar10_rgb.pth",
        "paper_target": None,
        "default_output_dir": "figures/cifar10_rgb",
        "grayscale": False,
    },
}


COHERENT_AMPLITUDE_PRESETS = {
    "conservative": {"threshold": 0.25, "temperature": 0.12, "gain_min": 0.4, "gain_max": 0.98},
    "balanced": {"threshold": 0.2, "temperature": 0.1, "gain_min": 0.25, "gain_max": 0.95},
    "aggressive": {"threshold": 0.15, "temperature": 0.08, "gain_min": 0.1, "gain_max": 0.9},
}


COHERENT_PHASE_PRESETS = {
    "conservative": {"gamma": 0.1},
    "balanced": {"gamma": 0.25},
    "aggressive": {"gamma": 0.5},
}


INCOHERENT_INTENSITY_PRESETS = {
    "conservative": {"responsivity": 0.5, "threshold": 0.15, "emission_phase_mode": "zero"},
    "balanced": {"responsivity": 1.0, "threshold": 0.1, "emission_phase_mode": "zero"},
    "aggressive": {"responsivity": 1.5, "threshold": 0.05, "emission_phase_mode": "zero"},
}


def parse_activation_positions(value):
    return parse_int_sequence(value)


def parse_int_sequence(value):
    if value in (None, "", ()):
        return ()

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return tuple(int(part) for part in parts)

    return tuple(int(part) for part in value)


def resolve_activation_positions_from_alias(alias, num_layers):
    if not alias or num_layers is None:
        return ()
    if alias == "front":
        return (1,)
    if alias == "mid":
        return ((int(num_layers) + 1) // 2,)
    if alias == "back":
        return (int(num_layers),)
    if alias == "all":
        return tuple(range(1, int(num_layers) + 1))
    raise ValueError(f"Unsupported activation placement alias: {alias}")


def activation_hparams_from_args(args):
    return {
        key: value
        for key, value in {
            "threshold": args.activation_threshold,
            "temperature": args.activation_temperature,
            "gain_min": args.activation_gain_min,
            "gain_max": args.activation_gain_max,
            "gamma": args.activation_gamma,
            "responsivity": args.activation_responsivity,
            "emission_phase_mode": args.activation_emission_phase_mode,
        }.items()
        if value is not None
    }


def activation_preset_hparams(args=None):
    if args is None:
        return {}
    activation_type = getattr(args, "activation_type", None)
    activation_preset = getattr(args, "activation_preset", None)
    if not activation_preset:
        return {}
    if activation_type == "coherent_amplitude":
        return dict(COHERENT_AMPLITUDE_PRESETS[activation_preset])
    if activation_type == "coherent_phase":
        return dict(COHERENT_PHASE_PRESETS[activation_preset])
    if activation_type == "incoherent_intensity":
        return dict(INCOHERENT_INTENSITY_PRESETS[activation_preset])
    return {}


def resolve_activation_config(args=None, manifest=None):
    explicit_type = getattr(args, "activation_type", None) if args is not None else None
    explicit_positions = getattr(args, "activation_positions", None) if args is not None else None
    explicit_placement = getattr(args, "activation_placement", None) if args is not None else None
    explicit_hparams = activation_hparams_from_args(args) if args is not None else {}
    preset_hparams = activation_preset_hparams(args)

    manifest = manifest or {}
    activation_type = explicit_type or manifest.get("activation_type") or "none"
    num_layers = getattr(args, "layers", None) if args is not None else None
    if num_layers is None:
        num_layers = ((manifest.get("optical_config") or {}).get("num_layers"))
    activation_positions = (
        parse_activation_positions(explicit_positions)
        if explicit_positions is not None
        else (
            resolve_activation_positions_from_alias(explicit_placement, num_layers)
            if explicit_placement is not None
            else parse_activation_positions(manifest.get("activation_positions"))
        )
    )
    activation_hparams = dict(manifest.get("activation_hparams") or {})
    activation_hparams.update(preset_hparams)
    activation_hparams.update(explicit_hparams)
    return activation_type, activation_positions, activation_hparams


def get_classification_dataset_config(dataset_name):
    dataset_key = dataset_name.lower().replace("-", "_")
    if dataset_key not in CLASSIFICATION_DATASETS:
        valid = ", ".join(sorted(CLASSIFICATION_DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {valid}")
    return CLASSIFICATION_DATASETS[dataset_key]


def build_classification_transform(dataset_cfg):
    transform_steps = []
    if dataset_cfg.get("grayscale"):
        transform_steps.append(transforms.Grayscale(num_output_channels=1))
    transform_steps.append(transforms.ToTensor())
    return transforms.Compose(transform_steps)


def classification_split_lengths(total_train_size, val_size=5000):
    if total_train_size <= val_size:
        raise ValueError(f"Training set size {total_train_size} must exceed val_size {val_size}")
    return total_train_size - val_size, val_size


def resolve_experiment_seed(explicit_seed, manifest=None, default=42):
    if explicit_seed is not None:
        return explicit_seed
    if manifest and manifest.get("seed") is not None:
        return manifest["seed"]
    return default


def resolve_propagation_config(args=None, manifest=None):
    manifest = manifest or {}
    explicit_backend = getattr(args, "rs_backend", None) if args is not None else None
    explicit_chunk_size = getattr(args, "propagation_chunk_size", None) if args is not None else None
    propagation_backend = explicit_backend or manifest.get("propagation_backend") or "direct"
    propagation_chunk_size = (
        explicit_chunk_size if explicit_chunk_size is not None else manifest.get("propagation_chunk_size")
    )
    return propagation_backend, propagation_chunk_size


def model_activation_diagnostics(model):
    diagnostics_fn = getattr(model, "activation_diagnostics", None)
    if diagnostics_fn is None:
        return {}
    return diagnostics_fn()


def format_activation_diagnostics(diagnostics):
    parts = []
    for position, stats in diagnostics.items():
        summary = []
        if "mean_gain" in stats:
            summary.append(f"gain={stats['mean_gain']:.3f}")
        if "mean_phase_shift" in stats:
            summary.append(f"dphi={stats['mean_phase_shift']:.3f}")
        if "mean_output_amplitude" in stats:
            summary.append(f"A={stats['mean_output_amplitude']:.3f}")
        if "mean_intensity" in stats:
            summary.append(f"I={stats['mean_intensity']:.3f}")
        if summary:
            parts.append(f"L{position} " + ", ".join(summary))
    return " | ".join(parts)


@torch.no_grad()
def plot_sample_output_patterns(model, dataset, device, sample_indices, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    sample_indices = parse_int_sequence(sample_indices)
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")

    model.eval()
    samples = []
    targets = []
    for index in sample_indices:
        sample, target = dataset[index]
        samples.append(sample)
        targets.append(target)

    inputs = torch.stack(samples).to(device)
    outputs = model.output_intensity(inputs).cpu()
    class_names = getattr(dataset, "classes", None)

    fig, axes = plt.subplots(len(sample_indices), 2, figsize=(8, 3 * len(sample_indices)), squeeze=False)
    for row, (index, sample, target) in enumerate(zip(sample_indices, samples, targets)):
        input_ax, output_ax = axes[row]
        sample_cpu = sample.detach().cpu()
        if sample_cpu.ndim == 3 and sample_cpu.shape[0] == 1:
            input_ax.imshow(sample_cpu[0].numpy(), cmap="gray")
        elif sample_cpu.ndim == 3 and sample_cpu.shape[0] in (3, 4):
            input_ax.imshow(sample_cpu[:3].permute(1, 2, 0).clamp(0, 1).numpy())
        else:
            input_ax.imshow(sample_cpu.squeeze().numpy(), cmap="gray")

        target_index = int(target) if not torch.is_tensor(target) else int(target.item())
        target_label = class_names[target_index] if class_names and target_index < len(class_names) else str(target_index)
        input_ax.set_title(f"Input {index} | {target_label}")
        input_ax.axis("off")

        output_ax.imshow(outputs[row].numpy(), cmap="magma")
        output_ax.set_title("Output intensity")
        output_ax.axis("off")

    fig.suptitle("Sample Output Patterns", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


@torch.no_grad()
def plot_quantization_sensitivity(
    model,
    optics,
    manifest,
    test_loader,
    dataset_cfg,
    device,
    levels,
    save_path=None,
    no_show=False,
):
    plt = configure_matplotlib_backend(no_show=no_show)
    levels = parse_int_sequence(levels)
    if not levels:
        raise ValueError("levels must not be empty")

    model.eval()
    baseline_metrics = evaluate_classification(model, test_loader, device)
    baseline_acc = baseline_metrics["accuracy"]
    original_phases = [layer.phase.detach().clone() for layer in model.layers]

    results = [("baseline", baseline_acc)]
    try:
        for level in levels:
            for layer, original in zip(model.layers, original_phases):
                layer.phase.copy_(original)

            quantized = quantize_phase_masks_uniform(model.export_phase_masks(wrap=True), level)
            for layer, phase_mask in zip(model.layers, quantized):
                layer.phase.copy_(phase_mask.to(device=layer.phase.device, dtype=layer.phase.dtype))

            metrics = evaluate_classification(model, test_loader, device)
            results.append((str(level), metrics["accuracy"]))
    finally:
        for layer, original in zip(model.layers, original_phases):
            layer.phase.copy_(original)

    labels = [name for name, _ in results]
    accuracies = [accuracy for _, accuracy in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(range(len(labels)), accuracies, color=["#2b6cb0"] + ["#c084fc"] * (len(labels) - 1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Phase quantization")
    ax.set_ylim(0, max(100.0, max(accuracies) * 1.15))
    ax.axhline(baseline_acc, color="#2b6cb0", linestyle="--", linewidth=1, alpha=0.5)

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    title_bits = [dataset_cfg.get("display_name", "Classification")]
    if manifest:
        run_name = manifest.get("run_name")
        if run_name:
            title_bits.append(str(run_name))
        seed = manifest.get("seed")
        if seed is not None:
            title_bits.append(f"seed={seed}")
    ax.set_title("Quantization Sensitivity\n" + " | ".join(title_bits))
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def plot_classification_history(history, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    train_history = history.get("train", {}) if history else {}
    val_history = history.get("val", {}) if history else {}
    train_accuracy = list(train_history.get("accuracy", ()))
    val_accuracy = list(val_history.get("accuracy", ()))
    train_contrast = list(train_history.get("contrast", ()))
    val_contrast = list(val_history.get("contrast", ()))
    epochs = list(range(1, max(len(train_accuracy), len(val_accuracy), len(train_contrast), len(val_contrast)) + 1))

    if not epochs:
        raise ValueError("history must contain at least one epoch of accuracy or contrast values")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    accuracy_ax, contrast_ax = axes

    if train_accuracy:
        accuracy_ax.plot(range(1, len(train_accuracy) + 1), train_accuracy, label="train", color="#2563eb", linewidth=2)
    if val_accuracy:
        accuracy_ax.plot(range(1, len(val_accuracy) + 1), val_accuracy, label="val", color="#dc2626", linewidth=2)
    accuracy_ax.set_title("Accuracy")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylabel("Accuracy (%)")
    accuracy_ax.grid(alpha=0.2)
    if train_accuracy or val_accuracy:
        accuracy_ax.legend()

    if train_contrast:
        contrast_ax.plot(range(1, len(train_contrast) + 1), train_contrast, label="train", color="#2563eb", linewidth=2)
    if val_contrast:
        contrast_ax.plot(range(1, len(val_contrast) + 1), val_contrast, label="val", color="#dc2626", linewidth=2)
    contrast_ax.set_title("Detector Contrast")
    contrast_ax.set_xlabel("Epoch")
    contrast_ax.set_ylabel("Contrast")
    contrast_ax.grid(alpha=0.2)
    if train_contrast or val_contrast:
        contrast_ax.legend()

    fig.suptitle("Classification Training History", fontsize=14)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def build_experiment_grid(grid_name, args):
    base = {
        "task": args.task,
        "dataset": args.dataset,
        "layers": args.layers,
        "size": args.size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "optics_preset": args.optics_preset,
    }

    if grid_name == "coherent_amplitude_positions":
        return [
            {
                **base,
                "experiment_stage": "placement_ablation",
                "activation_type": "coherent_amplitude",
                "activation_preset": "balanced",
                "activation_placement": placement,
            }
            for placement in ("front", "mid", "back", "all")
        ]

    if grid_name == "coherent_amplitude_presets":
        return [
            {
                **base,
                "experiment_stage": "mechanism_tuning",
                "activation_type": "coherent_amplitude",
                "activation_preset": preset,
                "activation_placement": "mid",
            }
            for preset in ("conservative", "balanced", "aggressive")
        ]

    if grid_name == "coherent_phase_presets":
        return [
            {
                **base,
                "experiment_stage": "mechanism_tuning",
                "activation_type": "coherent_phase",
                "activation_preset": preset,
                "activation_placement": "mid",
            }
            for preset in ("conservative", "balanced", "aggressive")
        ]

    if grid_name == "coherent_activation_mechanisms":
        return [
            {
                **base,
                "experiment_stage": "mechanism_ablation",
                "activation_type": activation_type,
                "activation_preset": "balanced",
                "activation_placement": "mid",
            }
            for activation_type in ("coherent_amplitude", "coherent_phase")
        ]

    if grid_name == "incoherent_intensity_presets":
        return [
            {
                **base,
                "experiment_stage": "mechanism_tuning",
                "activation_type": "incoherent_intensity",
                "activation_preset": preset,
                "activation_placement": "mid",
            }
            for preset in ("conservative", "balanced", "aggressive")
        ]

    if grid_name == "activation_mechanisms":
        return [
            {
                **base,
                "experiment_stage": "mechanism_ablation",
                "activation_type": activation_type,
                "activation_preset": "balanced",
                "activation_placement": "mid",
            }
            for activation_type in ("coherent_amplitude", "coherent_phase", "incoherent_intensity")
        ]

    raise ValueError(f"Unsupported experiment grid: {grid_name}")


def format_experiment_grid_commands(grid_name, args):
    commands = []
    for spec in build_experiment_grid(grid_name, args):
        command_parts = [
            "python train.py",
            f"--task {spec['task']}",
            f"--dataset {spec['dataset']}",
            f"--epochs {spec['epochs']}",
            f"--layers {spec['layers']}",
            f"--size {spec['size']}",
            f"--batch-size {spec['batch_size']}",
            f"--lr {spec['lr']}",
            f"--seed {spec['seed']}",
            f"--optics-preset {spec['optics_preset']}",
            f"--alpha {spec['alpha']}",
            f"--beta {spec['beta']}",
            f"--gamma {spec['gamma']}",
            f"--experiment-stage {spec['experiment_stage']}",
            f"--activation-type {spec['activation_type']}",
            f"--activation-preset {spec['activation_preset']}",
            f"--activation-placement {spec['activation_placement']}",
        ]
        commands.append(" ".join(command_parts))
    return commands


def execute_experiment_grid(grid_name, args, runner):
    for spec in build_experiment_grid(grid_name, args):
        spec_args = argparse.Namespace(**vars(args))
        spec_args.activation_positions = None
        for key, value in spec.items():
            setattr(spec_args, key, value)
        runner(spec_args)


@torch.no_grad()
def plot_output_energy(model, test_loader, device, class_names, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    model.eval()
    size = model.size
    num_classes = model.num_classes
    energy_maps = torch.zeros(num_classes, size, size)
    counts = torch.zeros(num_classes)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        intensity = model.output_intensity(data).cpu()
        target_cpu = target.cpu()

        for i in range(num_classes):
            mask = target_cpu == i
            if mask.any():
                energy_maps[i] += intensity[mask].sum(dim=0)
                counts[i] += mask.sum()

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(num_classes):
        ax = axes[i // 5, i % 5]
        avg = energy_maps[i] / max(counts[i], 1)
        ax.imshow(avg.numpy(), cmap="hot")
        ax.set_title(class_names[i])
        ax.axis("off")

    fig.suptitle("Average Output Energy per Class", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


@torch.no_grad()
def plot_confusion_matrix(model, test_loader, device, class_names, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    model.eval()
    num_classes = model.num_classes
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        for t, p in zip(target, pred):
            confusion[t.item(), p.item()] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion.numpy(), cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title("Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                str(confusion[i, j].item()),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > confusion.max() / 2 else "black",
                fontsize=8,
            )

    fig.colorbar(im, shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def run_classification_visualization(args):
    dataset_cfg = get_classification_dataset_config(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_checkpoint_manifest(args.checkpoint)
    ensure_checkpoint_version(
        manifest,
        expected_version=MODEL_VERSION,
        checkpoint_path=args.checkpoint,
        allow_missing=True,
    )
    state_dict = load_checkpoint_state_dict(args.checkpoint, map_location=device)

    optics = resolve_optics(
        CLASSIFIER_PAPER_OPTICS,
        state_dict=state_dict,
        manifest=manifest,
        checkpoint_path=args.checkpoint,
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(manifest=manifest)
    propagation_backend, propagation_chunk_size = resolve_propagation_config(args=args, manifest=manifest)
    model = build_model_for_task(
        "classification",
        optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = args.output_dir or dataset_cfg["default_output_dir"]
    out_dir = args.repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = build_classification_transform(dataset_cfg)
    dataset_cls = dataset_cfg["dataset_cls"]
    test_set = dataset_cls(args.repo_root / "data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
    class_names = [str(name) for name in test_set.classes]

    plot_phase_masks(model, save_path=out_dir / "phase_masks.png", no_show=args.no_show)
    plot_output_energy(
        model,
        test_loader,
        device,
        class_names,
        save_path=out_dir / "output_energy.png",
        no_show=args.no_show,
    )
    plot_confusion_matrix(
        model,
        test_loader,
        device,
        class_names,
        save_path=out_dir / "confusion_matrix.png",
        no_show=args.no_show,
    )
    history = (manifest or {}).get("history")
    if history:
        plot_classification_history(
            history,
            save_path=out_dir / "classification_history.png",
            no_show=args.no_show,
        )

    if getattr(args, "understanding_report", False):
        sample_indices = parse_int_sequence(args.sample_indices)
        quantization_levels = parse_int_sequence(args.quantization_levels)
        plot_sample_output_patterns(
            model,
            test_set,
            device,
            sample_indices,
            save_path=out_dir / "sample_output_patterns.png",
            no_show=args.no_show,
        )
        plot_quantization_sensitivity(
            model,
            optics,
            manifest,
            test_loader,
            dataset_cfg,
            device,
            quantization_levels,
            save_path=out_dir / "quantization_sensitivity.png",
            no_show=args.no_show,
        )


def build_imaging_dataset(dataset_name, data_dir, image_root, transform, seed):
    dataset_key = dataset_name.lower().replace("-", "_")

    if dataset_key == "stl10":
        train_full = datasets.STL10(data_dir, split="train", download=True, transform=transform)
        test_set = datasets.STL10(data_dir, split="test", download=True, transform=transform)
        train_len = int(len(train_full) * 0.9)
        val_len = len(train_full) - train_len
        train_set, val_set = random_split(
            train_full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(seed),
        )
        return {
            "display_name": "STL10",
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "checkpoint_name": "best_imager_stl10.pth",
        }

    if dataset_key == "imagefolder":
        if not image_root:
            raise ValueError("--image-root is required when --dataset imagefolder is used")
        full_set = datasets.ImageFolder(image_root, transform=transform)
        total = len(full_set)
        train_len = int(total * 0.8)
        val_len = int(total * 0.1)
        test_len = total - train_len - val_len
        train_set, val_set, test_set = random_split(
            full_set,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(seed),
        )
        return {
            "display_name": f"ImageFolder({image_root})",
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "checkpoint_name": "best_imager_imagefolder.pth",
        }

    raise ValueError("Unsupported imaging dataset. Use stl10 or imagefolder.")


def train_imaging_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total = 0

    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        target = model.build_target(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total += data.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}, loss: {loss.item():.4f}")

    return total_loss / max(total, 1)


@torch.no_grad()
def evaluate_imaging(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0

    for data, _ in loader:
        data = data.to(device)
        output = model(data)
        target = model.build_target(data)
        total_loss += criterion(output, target).item() * data.size(0)
        total += data.size(0)

    return total_loss / max(total, 1)


def resolve_imaging_optics(args):
    return resolve_optics(
        resolve_training_optics_preset("imaging", args.optics_preset),
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )


def run_imaging_training(args, device, data_dir, save_dir):
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    dataset_cfg = build_imaging_dataset(args.dataset, data_dir, args.image_root, transform, args.seed)
    print(f"Dataset: {dataset_cfg['display_name']}")

    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": getattr(args, "num_workers", 0),
        "pin_memory": bool(getattr(args, "pin_memory", False) and torch.cuda.is_available()),
    }
    if loader_common["num_workers"] > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = getattr(args, "prefetch_factor", 2)

    train_loader = DataLoader(
        dataset_cfg["train_set"],
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_common,
    )
    val_loader = DataLoader(dataset_cfg["val_set"], shuffle=False, **loader_common)
    test_loader = DataLoader(dataset_cfg["test_set"], shuffle=False, **loader_common)

    optics = resolve_imaging_optics(args)
    activation_type, activation_positions, activation_hparams = resolve_activation_config(args)
    propagation_backend, propagation_chunk_size = resolve_propagation_config(args=args)
    model = build_model_for_task(
        "imaging",
        optics,
        input_fraction=args.input_fraction,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"D2NNImager: {args.layers} layers, {args.size}x{args.size}, {total_params} trainable params")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    resolved_run_name = derive_experiment_run_name(
        run_name=args.run_name,
        experiment_stage=args.experiment_stage,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        seed=args.seed,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
        optics_preset=args.optics_preset,
        layer_count=args.layers,
    )
    checkpoint_path = checkpoint_variant_path(save_dir / dataset_cfg["checkpoint_name"], resolved_run_name)
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    best_val_loss = float("inf")
    last_activation_stats = {}
    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_imaging_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_imaging(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
        )
        last_activation_stats = model_activation_diagnostics(model)
        if last_activation_stats:
            print(f"  activation stats: {format_activation_diagnostics(last_activation_stats)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (val loss: {val_loss:.4f})")

        scheduler.step()

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loss = evaluate_imaging(model, test_loader, criterion, device)
    save_manifest(
        manifest_path,
        {
            "task": "imaging",
            "dataset": dataset_cfg["display_name"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": args.image_size,
            "input_fraction": args.input_fraction,
            "best_val_loss": best_val_loss,
            "test_mse": test_loss,
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
                propagation_backend=propagation_backend,
                propagation_chunk_size=propagation_chunk_size,
                optics_preset=args.optics_preset,
                runtime_config={
                    "device": str(device),
                    "allow_tf32": bool(getattr(args, "allow_tf32", False) and device.type == "cuda"),
                    "num_workers": int(getattr(args, "num_workers", 0)),
                    "pin_memory": bool(getattr(args, "pin_memory", False) and device.type == "cuda"),
                    "prefetch_factor": int(getattr(args, "prefetch_factor", 2))
                    if getattr(args, "num_workers", 0) > 0
                    else None,
                },
            ),
            "activation_diagnostics": last_activation_stats,
        },
    )
    print(f"\nTest MSE: {test_loss:.4f} (saved to {checkpoint_path.name})")


@torch.no_grad()
def plot_reconstructions(model, loader, num_samples, save_path=None, no_show=False, title_suffix=""):
    plt = configure_matplotlib_backend(no_show=no_show)
    inputs, _ = next(iter(loader))
    inputs = inputs.to(next(model.parameters()).device)
    outputs = model(inputs).cpu()
    targets = model.build_target(inputs).cpu()

    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 9))
    if num_samples == 1:
        axes = axes.reshape(3, 1)

    for idx in range(num_samples):
        axes[0, idx].imshow(inputs[idx, 0].cpu().numpy(), cmap="gray")
        axes[0, idx].set_title(f"Input {idx + 1}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(targets[idx].numpy(), cmap="gray")
        axes[1, idx].set_title("Target")
        axes[1, idx].axis("off")

        axes[2, idx].imshow(outputs[idx].numpy(), cmap="gray")
        axes[2, idx].set_title("Output")
        axes[2, idx].axis("off")

    fig.suptitle(f"D2NN Imager Reconstructions{title_suffix}", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def build_imaging_visualization_dataset(dataset_name, data_dir, image_root, transform, seed):
    dataset_key = dataset_name.lower().replace("-", "_")
    if dataset_key == "stl10":
        return datasets.STL10(data_dir, split="test", download=True, transform=transform), "STL10"
    if dataset_key == "imagefolder":
        if not image_root:
            raise ValueError("--image-root is required when --dataset imagefolder is used")
        full_set = datasets.ImageFolder(image_root, transform=transform)
        total = len(full_set)
        train_len = int(total * 0.8)
        val_len = int(total * 0.1)
        test_len = total - train_len - val_len
        _, _, test_set = random_split(
            full_set,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(seed),
        )
        return test_set, f"ImageFolder({image_root})"
    raise ValueError("Unsupported imaging dataset. Use stl10 or imagefolder.")


def run_imaging_visualization(args):
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    data_dir = args.repo_root / "data"
    manifest = read_checkpoint_manifest(args.checkpoint)
    split_seed = resolve_experiment_seed(args.seed, manifest)
    test_set, dataset_name = build_imaging_visualization_dataset(
        args.dataset,
        data_dir,
        args.image_root,
        transform,
        split_seed,
    )
    loader = DataLoader(test_set, batch_size=args.num_samples, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = load_checkpoint_state_dict(args.checkpoint, map_location=device)
    ensure_checkpoint_version(
        manifest,
        expected_version=MODEL_VERSION,
        checkpoint_path=args.checkpoint,
        allow_missing=True,
    )
    optics = resolve_optics(
        IMAGER_PAPER_OPTICS,
        state_dict=state_dict,
        manifest=manifest,
        checkpoint_path=args.checkpoint,
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(manifest=manifest)
    propagation_backend, propagation_chunk_size = resolve_propagation_config(args=args, manifest=manifest)
    model = build_model_for_task(
        "imaging",
        optics,
        input_fraction=args.input_fraction,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        propagation_backend=propagation_backend,
        propagation_chunk_size=propagation_chunk_size,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir = args.output_dir or "figures/imager"
    out_dir = args.repo_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_phase_masks(model, save_path=out_dir / "phase_masks.png", no_show=args.no_show)
    plot_reconstructions(
        model,
        loader,
        args.num_samples,
        save_path=out_dir / "sample_reconstructions.png",
        no_show=args.no_show,
        title_suffix=f" - {dataset_name}",
    )
