"""
D2NN unified training entrypoint.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from artifacts import (
    CLASSIFIER_PAPER_OPTICS,
    build_model_for_task,
    checkpoint_manifest_path,
    checkpoint_variant_path,
    derive_experiment_run_name,
    experiment_manifest_fields,
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def d2nn_mse_loss(output, target, num_classes=10):
    output_norm = output / (output.sum(dim=1, keepdim=True) + 1e-8)
    target_onehot = F.one_hot(target, num_classes).float()
    return F.mse_loss(output_norm, target_onehot)


def phase_smoothness_regularizer(model):
    penalties = []
    for layer in getattr(model, "layers", ()):
        phase = layer.phase
        horizontal_delta = torch.atan2(torch.sin(phase[:, 1:] - phase[:, :-1]), torch.cos(phase[:, 1:] - phase[:, :-1]))
        vertical_delta = torch.atan2(torch.sin(phase[1:, :] - phase[:-1, :]), torch.cos(phase[1:, :] - phase[:-1, :]))
        penalties.append(horizontal_delta.pow(2).mean())
        penalties.append(vertical_delta.pow(2).mean())
    if not penalties:
        reference = next(model.parameters(), None)
        device = reference.device if reference is not None else "cpu"
        return torch.tensor(0.0, device=device)
    return torch.stack(penalties).mean()


def classification_composite_loss(result, target, model, alpha=1.0, beta=0.1, gamma=0.01):
    scores = result["scores"]
    num_classes = scores.shape[1]
    mse = d2nn_mse_loss(scores, target, num_classes=num_classes)
    ce = F.cross_entropy(result["logits"], target)
    reg = phase_smoothness_regularizer(model)
    total = alpha * mse + beta * ce + gamma * reg
    return {"total": total, "mse": mse, "ce": ce, "reg": reg}


def build_metric_history():
    return {
        "train": {"loss": [], "mse": [], "ce": [], "reg": [], "accuracy": [], "contrast": []},
        "val": {"loss": [], "mse": [], "ce": [], "reg": [], "accuracy": [], "contrast": []},
    }


def append_metric_history(history, *, split, total, mse, ce, reg, accuracy, contrast):
    bucket = history[split]
    bucket["loss"].append(float(total))
    bucket["mse"].append(float(mse))
    bucket["ce"].append(float(ce))
    bucket["reg"].append(float(reg))
    bucket["accuracy"].append(float(accuracy))
    bucket["contrast"].append(float(contrast))


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


def is_better_classification_checkpoint(candidate, best):
    if candidate["accuracy"] != best["accuracy"]:
        return candidate["accuracy"] > best["accuracy"]
    if candidate["contrast"] != best["contrast"]:
        return candidate["contrast"] > best["contrast"]
    return candidate["epoch"] > best["epoch"]


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
            ),
            "activation_diagnostics": last_activation_stats,
        },
    )


def _run_classification_epoch(model, loader, device, *, optimizer=None, alpha=1.0, beta=0.1, gamma=0.01):
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_reg = 0.0
    total_contrast = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if training:
                optimizer.zero_grad()

            result = model.forward_with_metrics(data, target=target)
            loss_terms = classification_composite_loss(result, target, model, alpha=alpha, beta=beta, gamma=gamma)
            if training:
                loss_terms["total"].backward()
                optimizer.step()

            total_loss += loss_terms["total"].item() * data.size(0)
            total_mse += loss_terms["mse"].item() * data.size(0)
            total_ce += loss_terms["ce"].item() * data.size(0)
            total_reg += loss_terms["reg"].item() * data.size(0)
            total_contrast += result["contrast"].mean().item() * data.size(0)
            pred = result["scores"].argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

            if training and (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx + 1}/{len(loader)}, loss: {loss_terms['total'].item():.4f}")

    return {
        "loss": total_loss / total,
        "mse": total_mse / total,
        "ce": total_ce / total,
        "reg": total_reg / total,
        "accuracy": 100.0 * correct / total,
        "contrast": total_contrast / total,
    }


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

    return (
        DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
        ),
        DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0),
        DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0),
    )


def build_classification_model(args, device):
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
    resolved_run_name = derive_experiment_run_name(
        run_name=args.run_name,
        experiment_stage=args.experiment_stage,
        activation_type=activation_type,
        activation_positions=activation_positions,
        activation_hparams=activation_hparams,
        seed=args.seed,
        loss_config=loss_config,
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
