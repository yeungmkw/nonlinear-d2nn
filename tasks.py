"""
Layered task helpers for classification and imaging workflows.

This module carries task-specific builders, config resolution, and workflow
helpers that support the unified ``train.py`` entrypoint without forcing all
task details back into the top-level training module.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from artifacts import (
    CLASSIFIER_PAPER_OPTICS,
    IMAGER_PAPER_OPTICS,
    build_model_for_task,
    configure_matplotlib_backend,
    ensure_checkpoint_version,
    load_checkpoint_state_dict,
    maybe_show,
    plot_phase_masks,
    quantize_phase_masks_uniform,
    read_checkpoint_manifest,
    resolve_optics,
    resolve_training_optics_preset,
)
from train_core import (
    append_metric_history,
    build_metric_history,
    evaluate_classification,
    run_classification_epoch,
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
ACTIVATION_PRESETS = {
    "coherent_amplitude": COHERENT_AMPLITUDE_PRESETS,
    "coherent_phase": COHERENT_PHASE_PRESETS,
    "incoherent_intensity": INCOHERENT_INTENSITY_PRESETS,
}


def parse_activation_positions(value):
    return parse_int_sequence(value)


def parse_int_sequence(value):
    if value in (None, "", ()):
        return ()

    raw_values = value.split(",") if isinstance(value, str) else value
    return tuple(int(str(part).strip()) for part in raw_values if str(part).strip())


def _mid_activation_position(num_layers):
    return ((int(num_layers) + 1) // 2,)


def _all_activation_positions(num_layers):
    return tuple(range(1, int(num_layers) + 1))


ACTIVATION_PLACEMENT_ALIASES = {
    "front": lambda num_layers: (1,),
    "mid": _mid_activation_position,
    "back": lambda num_layers: (int(num_layers),),
    "all": _all_activation_positions,
}


def resolve_activation_positions_from_alias(alias, num_layers):
    if not alias or num_layers is None:
        return ()
    resolver = ACTIVATION_PLACEMENT_ALIASES.get(alias)
    if resolver is None:
        raise ValueError(f"Unsupported activation placement alias: {alias}")
    return resolver(num_layers)


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
    preset_family = ACTIVATION_PRESETS.get(activation_type)
    return dict(preset_family[activation_preset]) if preset_family is not None else {}


def _activation_num_layers(args=None, manifest=None):
    num_layers = getattr(args, "layers", None) if args is not None else None
    if num_layers is not None:
        return num_layers
    return ((manifest or {}).get("optical_config") or {}).get("num_layers")


def _resolve_activation_positions(args=None, manifest=None):
    explicit_positions = getattr(args, "activation_positions", None) if args is not None else None
    if explicit_positions is not None:
        return parse_activation_positions(explicit_positions)

    explicit_placement = getattr(args, "activation_placement", None) if args is not None else None
    if explicit_placement is not None:
        return resolve_activation_positions_from_alias(explicit_placement, _activation_num_layers(args, manifest))

    return parse_activation_positions((manifest or {}).get("activation_positions"))


def _merge_activation_hparams(args=None, manifest=None):
    activation_hparams = dict((manifest or {}).get("activation_hparams") or {})
    activation_hparams.update(activation_preset_hparams(args))
    if args is not None:
        activation_hparams.update(activation_hparams_from_args(args))
    return activation_hparams


def resolve_activation_config(args=None, manifest=None):
    explicit_type = getattr(args, "activation_type", None) if args is not None else None
    manifest = manifest or {}
    activation_type = explicit_type or manifest.get("activation_type") or "none"
    activation_positions = _resolve_activation_positions(args, manifest)
    activation_hparams = _merge_activation_hparams(args, manifest)
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


def build_classification_test_loader(data_dir, dataset_cfg, batch_size=64, num_workers=0):
    transform = build_classification_transform(dataset_cfg)
    dataset_cls = dataset_cfg["dataset_cls"]
    test_set = dataset_cls(data_dir, train=False, download=True, transform=transform)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    class_names = [str(name) for name in getattr(test_set, "classes", [])]
    return loader, class_names


def resolve_experiment_seed(explicit_seed, manifest=None, default=42):
    if explicit_seed is not None:
        return explicit_seed
    if manifest and manifest.get("seed") is not None:
        return manifest["seed"]
    return default


def _explicit_or_manifest(explicit_value, manifest, key, default=None):
    if explicit_value is not None:
        return explicit_value
    return manifest.get(key, default)


def _backend_explicit_or_manifest(explicit_backend, manifest):
    return explicit_backend or manifest.get("propagation_backend") or "direct"


def resolve_propagation_config(args=None, manifest=None):
    manifest = manifest or {}
    explicit_backend = getattr(args, "rs_backend", None) if args is not None else None
    explicit_chunk_size = getattr(args, "propagation_chunk_size", None) if args is not None else None
    propagation_backend = _backend_explicit_or_manifest(explicit_backend, manifest)
    propagation_chunk_size = _explicit_or_manifest(explicit_chunk_size, manifest, "propagation_chunk_size")
    return propagation_backend, propagation_chunk_size


def model_activation_diagnostics(model):
    diagnostics_fn = getattr(model, "activation_diagnostics", None)
    if diagnostics_fn is None:
        return {}
    return diagnostics_fn()


ACTIVATION_DIAGNOSTIC_FIELDS = (
    ("mean_gain", "gain"),
    ("mean_phase_shift", "dphi"),
    ("mean_output_amplitude", "A"),
    ("mean_intensity", "I"),
)


def _format_activation_layer_diagnostics(position, stats):
    summary = [
        f"{label}={stats[field_name]:.3f}"
        for field_name, label in ACTIVATION_DIAGNOSTIC_FIELDS
        if field_name in stats
    ]
    return f"L{position} " + ", ".join(summary) if summary else None


def format_activation_diagnostics(diagnostics):
    parts = []
    for position, stats in diagnostics.items():
        formatted = _format_activation_layer_diagnostics(position, stats)
        if formatted:
            parts.append(formatted)
    return " | ".join(parts)


def print_model_summary(label, model, *, task):
    body = f"{model.size}x{model.size} neurons/layer" if task == "classification" else f"{model.size}x{model.size}"
    print(f"{label}: {len(model.layers)} layers, {body}, {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params")


def fit_classification_model(*, model, train_loader, val_loader, device, epochs, learning_rate, loss_config, checkpoint_path):
    """Run the classification fit loop.

    Returns `(best_metrics, best_state_dict, history, last_activation_stats)`.
    `best_state_dict` is a CPU-cloned cache of the best checkpoint when the best
    epoch is not the final epoch; otherwise it is `None` and callers should skip
    restore if the final epoch is already best.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    best_checkpoint_key = (float("-inf"), float("-inf"), 0)
    best_state_dict = None
    last_activation_stats = {}
    history = build_metric_history()

    for epoch in range(1, epochs + 1):
        for split, loader, split_optimizer in (("train", train_loader, optimizer), ("val", val_loader, None)):
            metrics = run_classification_epoch(
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


def _collect_indexed_samples(dataset, sample_indices):
    samples = []
    targets = []
    for index in sample_indices:
        sample, target = dataset[index]
        samples.append(sample)
        targets.append(target)
    return samples, targets


def _plot_input_sample(ax, sample):
    sample_cpu = sample.detach().cpu()
    if sample_cpu.ndim == 3 and sample_cpu.shape[0] == 1:
        ax.imshow(sample_cpu[0].numpy(), cmap="gray")
        return
    if sample_cpu.ndim == 3 and sample_cpu.shape[0] in (3, 4):
        ax.imshow(sample_cpu[:3].permute(1, 2, 0).clamp(0, 1).numpy())
        return
    ax.imshow(sample_cpu.squeeze().numpy(), cmap="gray")


def _target_label(target, class_names):
    target_index = int(target) if not torch.is_tensor(target) else int(target.item())
    if class_names and target_index < len(class_names):
        return class_names[target_index]
    return str(target_index)


@torch.no_grad()
def plot_sample_output_patterns(model, dataset, device, sample_indices, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    sample_indices = parse_int_sequence(sample_indices)
    if not sample_indices:
        raise ValueError("sample_indices must not be empty")

    model.eval()
    samples, targets = _collect_indexed_samples(dataset, sample_indices)
    inputs = torch.stack(samples).to(device)
    outputs = model.output_intensity(inputs).cpu()
    class_names = getattr(dataset, "classes", None)

    fig, axes = plt.subplots(len(sample_indices), 2, figsize=(8, 3 * len(sample_indices)), squeeze=False)
    for row, (index, sample, target) in enumerate(zip(sample_indices, samples, targets)):
        input_ax, output_ax = axes[row]
        _plot_input_sample(input_ax, sample)
        input_ax.set_title(f"Input {index} | {_target_label(target, class_names)}")
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


def _clone_layer_phases(model):
    return [layer.phase.detach().clone() for layer in model.layers]


def _restore_layer_phases(model, original_phases):
    for layer, original in zip(model.layers, original_phases):
        layer.phase.copy_(original)


def _evaluate_quantized_level(model, test_loader, device, original_phases, level):
    _restore_layer_phases(model, original_phases)
    quantized = quantize_phase_masks_uniform(model.export_phase_masks(wrap=True), level)
    for layer, phase_mask in zip(model.layers, quantized):
        layer.phase.copy_(phase_mask.to(device=layer.phase.device, dtype=layer.phase.dtype))
    metrics = evaluate_classification(model, test_loader, device)
    return str(level), metrics["accuracy"]


def _quantization_title(dataset_cfg, manifest):
    title_bits = [dataset_cfg.get("display_name", "Classification")]
    if manifest:
        run_name = manifest.get("run_name")
        if run_name:
            title_bits.append(str(run_name))
        seed = manifest.get("seed")
        if seed is not None:
            title_bits.append(f"seed={seed}")
    return "Quantization Sensitivity\n" + " | ".join(title_bits)


def _annotate_accuracy_bars(ax, bars, accuracies):
    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )


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
    original_phases = _clone_layer_phases(model)

    results = [("baseline", baseline_acc)]
    try:
        for level in levels:
            results.append(_evaluate_quantized_level(model, test_loader, device, original_phases, level))
    finally:
        _restore_layer_phases(model, original_phases)

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

    _annotate_accuracy_bars(ax, bars, accuracies)
    ax.set_title(_quantization_title(dataset_cfg, manifest))
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def _history_metric_series(history, metric_name):
    train_history = history.get("train", {}) if history else {}
    val_history = history.get("val", {}) if history else {}
    return list(train_history.get(metric_name, ())), list(val_history.get(metric_name, ()))


def _plot_history_metric(ax, *, title, ylabel, train_values, val_values):
    if train_values:
        ax.plot(range(1, len(train_values) + 1), train_values, label="train", color="#2563eb", linewidth=2)
    if val_values:
        ax.plot(range(1, len(val_values) + 1), val_values, label="val", color="#dc2626", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    if train_values or val_values:
        ax.legend()


def plot_classification_history(history, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    train_accuracy, val_accuracy = _history_metric_series(history, "accuracy")
    train_contrast, val_contrast = _history_metric_series(history, "contrast")
    epochs = list(range(1, max(len(train_accuracy), len(val_accuracy), len(train_contrast), len(val_contrast)) + 1))

    if not epochs:
        raise ValueError("history must contain at least one epoch of accuracy or contrast values")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    accuracy_ax, contrast_ax = axes
    _plot_history_metric(
        accuracy_ax,
        title="Accuracy",
        ylabel="Accuracy (%)",
        train_values=train_accuracy,
        val_values=val_accuracy,
    )
    _plot_history_metric(
        contrast_ax,
        title="Detector Contrast",
        ylabel="Contrast",
        train_values=train_contrast,
        val_values=val_contrast,
    )

    fig.suptitle("Classification Training History", fontsize=14)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


EXPERIMENT_GRID_VARIANTS = {
    "coherent_amplitude_positions": {
        "experiment_stage": "placement_ablation",
        "activation_type": "coherent_amplitude",
        "activation_preset": "balanced",
        "activation_placements": ("front", "mid", "back", "all"),
    },
    "coherent_amplitude_presets": {
        "experiment_stage": "mechanism_tuning",
        "activation_type": "coherent_amplitude",
        "activation_presets": ("conservative", "balanced", "aggressive"),
        "activation_placement": "mid",
    },
    "coherent_phase_presets": {
        "experiment_stage": "mechanism_tuning",
        "activation_type": "coherent_phase",
        "activation_presets": ("conservative", "balanced", "aggressive"),
        "activation_placement": "mid",
    },
    "coherent_activation_mechanisms": {
        "experiment_stage": "mechanism_ablation",
        "activation_types": ("coherent_amplitude", "coherent_phase"),
        "activation_preset": "balanced",
        "activation_placement": "mid",
    },
    "incoherent_intensity_presets": {
        "experiment_stage": "mechanism_tuning",
        "activation_type": "incoherent_intensity",
        "activation_presets": ("conservative", "balanced", "aggressive"),
        "activation_placement": "mid",
    },
    "activation_mechanisms": {
        "experiment_stage": "mechanism_ablation",
        "activation_types": ("coherent_amplitude", "coherent_phase", "incoherent_intensity"),
        "activation_preset": "balanced",
        "activation_placement": "mid",
    },
}


def _experiment_grid_base(args):
    return {
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


def _experiment_grid_sweep_key(variant):
    if "activation_placements" in variant:
        return "activation_placement", variant["activation_placements"]
    if "activation_presets" in variant:
        return "activation_preset", variant["activation_presets"]
    if "activation_types" in variant:
        return "activation_type", variant["activation_types"]
    raise ValueError("Experiment grid variant is missing a sweep dimension")


def build_experiment_grid(grid_name, args):
    variant = EXPERIMENT_GRID_VARIANTS.get(grid_name)
    if variant is None:
        raise ValueError(f"Unsupported experiment grid: {grid_name}")

    sweep_key, sweep_values = _experiment_grid_sweep_key(variant)
    static_fields = {
        key: value
        for key, value in variant.items()
        if key not in ("activation_placements", "activation_presets", "activation_types")
    }
    base = _experiment_grid_base(args)
    return [{**base, **static_fields, sweep_key: value} for value in sweep_values]


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


def _empty_energy_accumulators(model):
    energy_maps = torch.zeros(model.num_classes, model.size, model.size)
    counts = torch.zeros(model.num_classes)
    return energy_maps, counts


def _accumulate_output_energy(model, loader, device, energy_maps, counts):
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        intensity = model.output_intensity(data).cpu()
        target_cpu = target.cpu()

        for class_index in range(model.num_classes):
            mask = target_cpu == class_index
            if mask.any():
                energy_maps[class_index] += intensity[mask].sum(dim=0)
                counts[class_index] += mask.sum()


def _plot_output_energy_grid(axes, energy_maps, counts, class_names):
    for class_index in range(len(class_names)):
        ax = axes[class_index // 5, class_index % 5]
        avg = energy_maps[class_index] / max(counts[class_index], 1)
        ax.imshow(avg.numpy(), cmap="hot")
        ax.set_title(class_names[class_index])
        ax.axis("off")


@torch.no_grad()
def plot_output_energy(model, test_loader, device, class_names, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    model.eval()
    energy_maps, counts = _empty_energy_accumulators(model)
    _accumulate_output_energy(model, test_loader, device, energy_maps, counts)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    _plot_output_energy_grid(axes, energy_maps, counts, class_names)

    fig.suptitle("Average Output Energy per Class", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def _build_confusion_matrix(model, loader, device):
    confusion = torch.zeros(model.num_classes, model.num_classes, dtype=torch.int64)
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        for true_label, predicted_label in zip(target, pred):
            confusion[true_label.item(), predicted_label.item()] += 1
    return confusion


def _configure_confusion_axes(ax, class_names):
    num_classes = len(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_title("Confusion Matrix")


def _validate_class_names(class_names, num_classes):
    if len(class_names) != num_classes:
        raise ValueError(f"class_names length {len(class_names)} does not match num_classes {num_classes}")


def _annotate_confusion_matrix(ax, confusion):
    threshold = confusion.max() / 2
    num_classes = confusion.shape[0]
    for true_index in range(num_classes):
        for predicted_index in range(num_classes):
            value = confusion[true_index, predicted_index]
            ax.text(
                predicted_index,
                true_index,
                str(value.item()),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )


@torch.no_grad()
def plot_confusion_matrix(model, test_loader, device, class_names, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    model.eval()
    _validate_class_names(class_names, model.num_classes)
    confusion = _build_confusion_matrix(model, test_loader, device)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion.numpy(), cmap="Blues")
    _configure_confusion_axes(ax, class_names)
    _annotate_confusion_matrix(ax, confusion)

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

    test_loader, class_names = build_classification_test_loader(
        args.repo_root / "data",
        dataset_cfg,
        batch_size=64,
        num_workers=0,
    )

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
            test_loader.dataset,
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


def build_imaging_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )


def build_imaging_loaders(args, data_dir, runtime_config):
    transform = build_imaging_transform(args.image_size)
    dataset_cfg = build_imaging_dataset(args.dataset, data_dir, args.image_root, transform, args.seed)
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": runtime_config["num_workers"],
        "pin_memory": runtime_config["pin_memory"],
    }
    if runtime_config["num_workers"] > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = runtime_config["prefetch_factor"]

    train_loader = DataLoader(
        dataset_cfg["train_set"],
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_common,
    )
    val_loader = DataLoader(dataset_cfg["val_set"], shuffle=False, **loader_common)
    test_loader = DataLoader(dataset_cfg["test_set"], shuffle=False, **loader_common)
    return dataset_cfg, train_loader, val_loader, test_loader


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


def build_imaging_training_model(args, device):
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
    return (
        model,
        optics,
        activation_type,
        activation_positions,
        activation_hparams,
        propagation_backend,
        propagation_chunk_size,
    )


def fit_imaging_model(*, model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs, checkpoint_path):
    """Run the imaging fit loop.

    Returns `(best_val_loss, best_epoch, best_state_dict, last_activation_stats)`.
    `best_state_dict` is a CPU-cloned cache of the best checkpoint when the best
    epoch is not the final epoch; otherwise it is `None` and callers should skip
    restore if the final epoch is already best.
    """
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    last_activation_stats = {}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_imaging_one_epoch(model, train_loader, optimizer, criterion, device)
        if not math.isfinite(train_loss):
            raise ValueError("non-finite training loss")
        val_loss = evaluate_imaging(model, val_loader, criterion, device)
        if not math.isfinite(val_loss):
            raise ValueError("non-finite validation loss")
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}"
        )
        last_activation_stats = model_activation_diagnostics(model)
        if last_activation_stats:
            print(f"  activation stats: {format_activation_diagnostics(last_activation_stats)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            current_state_dict = model.state_dict()
            torch.save(current_state_dict, checkpoint_path)
            best_state_dict = {key: value.detach().cpu().clone() for key, value in current_state_dict.items()} if epoch != epochs else None
            print(f"  -> Saved best model (val loss: {val_loss:.4f})")

        scheduler.step()

    return best_val_loss, best_epoch, best_state_dict, last_activation_stats


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
