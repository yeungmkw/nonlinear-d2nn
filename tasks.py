"""
Consolidated D2NN task helpers for classification and imaging workflows.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
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
    derive_experiment_run_name,
    experiment_manifest_fields,
    load_checkpoint_state_dict,
    maybe_show,
    plot_phase_masks,
    read_checkpoint_manifest,
    resolve_optics,
    save_manifest,
)


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


def d2nn_mse_loss(output, target, num_classes=10):
    output_norm = output / (output.sum(dim=1, keepdim=True) + 1e-8)
    target_onehot = F.one_hot(target, num_classes).float()
    return F.mse_loss(output_norm, target_onehot)


def resolve_experiment_seed(explicit_seed, manifest=None, default=42):
    if explicit_seed is not None:
        return explicit_seed
    if manifest and manifest.get("seed") is not None:
        return manifest["seed"]
    return default


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
        for key, value in spec.items():
            setattr(spec_args, key, value)
        runner(spec_args)


def train_classification_one_epoch(model, loader, optimizer, device, num_classes=10):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = d2nn_mse_loss(output, target, num_classes)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}, loss: {loss.item():.4f}")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate_classification(model, loader, device, num_classes=10):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += d2nn_mse_loss(output, target, num_classes).item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return total_loss / total, 100.0 * correct / total


def run_classification_training(args, device, data_dir, save_dir):
    dataset_cfg = get_classification_dataset_config(args.dataset)
    print(f"Dataset: {dataset_cfg['display_name']}")

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

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optics = CLASSIFIER_PAPER_OPTICS.with_overrides(
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(args)
    model = build_model_for_task(
        "classification",
        optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"D2NN: {args.layers} layers, {args.size}x{args.size} neurons/layer, {total_params} trainable params")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    resolved_run_name = derive_experiment_run_name(
        run_name=args.run_name,
        experiment_stage=args.experiment_stage,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        seed=args.seed,
    )
    checkpoint_path = checkpoint_variant_path(save_dir / dataset_cfg["checkpoint_name"], resolved_run_name)
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    best_val_acc = 0.0
    last_activation_stats = {}
    if resolved_run_name:
        print(f"Run name: {resolved_run_name}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_classification_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate_classification(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.2f}% | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.2f}%"
        )
        last_activation_stats = model_activation_diagnostics(model)
        if last_activation_stats:
            print(f"  activation stats: {format_activation_diagnostics(last_activation_stats)}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model (val acc: {val_acc:.2f}%)")

        scheduler.step()

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_loss, test_acc = evaluate_classification(model, test_loader, device)
    save_manifest(
        manifest_path,
        {
            "task": "classification",
            "dataset": dataset_cfg["display_name"],
            "paper_target_accuracy": dataset_cfg["paper_target"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            **experiment_manifest_fields(
                checkpoint_path=checkpoint_path,
                run_name=resolved_run_name,
                experiment_stage=args.experiment_stage,
                seed=args.seed,
                optics=optics,
                activation_type=activation_type,
                activation_positions=activation_positions,
                activation_hparams=activation_hparams,
            ),
            "activation_diagnostics": last_activation_stats,
        },
    )
    paper_target = dataset_cfg["paper_target"]
    paper_target_text = f"{paper_target:.2f}%" if paper_target is not None else "n/a"
    print(f"\nTest accuracy: {test_acc:.2f}% (paper target: {paper_target_text}, saved to {checkpoint_path.name})")


@torch.no_grad()
def plot_output_energy(model, test_loader, device, class_names, save_path=None, no_show=False):
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
    state_dict = load_checkpoint_state_dict(args.checkpoint, map_location=device)

    optics = resolve_optics(
        CLASSIFIER_PAPER_OPTICS,
        state_dict=state_dict,
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(manifest=manifest)
    model = build_model_for_task(
        "classification",
        optics,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
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
        IMAGER_PAPER_OPTICS,
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
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

    train_loader = DataLoader(
        dataset_cfg["train_set"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(dataset_cfg["val_set"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset_cfg["test_set"], batch_size=args.batch_size, shuffle=False, num_workers=0)

    optics = resolve_imaging_optics(args)
    activation_type, activation_positions, activation_hparams = resolve_activation_config(args)
    model = build_model_for_task(
        "imaging",
        optics,
        input_fraction=args.input_fraction,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
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
            ),
            "activation_diagnostics": last_activation_stats,
        },
    )
    print(f"\nTest MSE: {test_loss:.4f} (saved to {checkpoint_path.name})")


@torch.no_grad()
def plot_reconstructions(model, loader, num_samples, save_path=None, no_show=False, title_suffix=""):
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
    optics = resolve_optics(
        IMAGER_PAPER_OPTICS,
        state_dict=state_dict,
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
    )
    activation_type, activation_positions, activation_hparams = resolve_activation_config(manifest=manifest)
    model = build_model_for_task(
        "imaging",
        optics,
        input_fraction=args.input_fraction,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
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
