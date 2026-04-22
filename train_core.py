"""
Shared low-level training helpers for train/task entrypoints.

Keep batch/epoch numerical logic here so ``train.py`` can stay focused on
entrypoint orchestration instead of absorbing low-level training details.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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


def is_better_classification_checkpoint(candidate, best):
    if candidate["accuracy"] != best["accuracy"]:
        return candidate["accuracy"] > best["accuracy"]
    if candidate["contrast"] != best["contrast"]:
        return candidate["contrast"] > best["contrast"]
    return candidate["epoch"] > best["epoch"]


def _ensure_finite_tensors(named_tensors):
    for name, tensor in named_tensors:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite {name}")


def _empty_classification_totals():
    return {"loss": 0.0, "mse": 0.0, "ce": 0.0, "reg": 0.0, "contrast": 0.0, "correct": 0, "total": 0}


def _accumulate_classification_batch(totals, *, data, target, result, loss_terms):
    batch_size = data.size(0)
    pred = result["scores"].argmax(dim=1)
    totals["loss"] += loss_terms["total"].item() * batch_size
    totals["mse"] += loss_terms["mse"].item() * batch_size
    totals["ce"] += loss_terms["ce"].item() * batch_size
    totals["reg"] += loss_terms["reg"].item() * batch_size
    totals["contrast"] += result["contrast"].mean().item() * batch_size
    totals["correct"] += pred.eq(target).sum().item()
    totals["total"] += batch_size
    return pred


def _classification_epoch_summary(totals):
    total = totals["total"]
    return {
        "loss": totals["loss"] / total,
        "mse": totals["mse"] / total,
        "ce": totals["ce"] / total,
        "reg": totals["reg"] / total,
        "accuracy": 100.0 * totals["correct"] / total,
        "contrast": totals["contrast"] / total,
    }


def _run_classification_epoch(model, loader, device, *, optimizer=None, alpha=1.0, beta=0.1, gamma=0.01):
    training = optimizer is not None
    model.train(mode=training)
    totals = _empty_classification_totals()

    with torch.set_grad_enabled(training):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if training:
                optimizer.zero_grad()

            result = model.forward_with_metrics(data, target=target)
            _ensure_finite_tensors(
                (
                    ("scores", result["scores"]),
                    ("logits", result["logits"]),
                    ("contrast", result["contrast"]),
                )
            )

            loss_terms = classification_composite_loss(result, target, model, alpha=alpha, beta=beta, gamma=gamma)
            _ensure_finite_tensors(
                (
                    ("loss", loss_terms["total"]),
                    ("mse", loss_terms["mse"]),
                    ("ce", loss_terms["ce"]),
                    ("reg", loss_terms["reg"]),
                )
            )
            if training:
                loss_terms["total"].backward()
                optimizer.step()

            pred = _accumulate_classification_batch(totals, data=data, target=target, result=result, loss_terms=loss_terms)

            if training and (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx + 1}/{len(loader)}, loss: {loss_terms['total'].item():.4f}")

            del result, loss_terms, pred

    return _classification_epoch_summary(totals)


@torch.no_grad()
def evaluate_classification(model, loader, device, alpha=1.0, beta=0.1, gamma=0.01):
    return _run_classification_epoch(model, loader, device, optimizer=None, alpha=alpha, beta=beta, gamma=gamma)


run_classification_epoch = _run_classification_epoch
