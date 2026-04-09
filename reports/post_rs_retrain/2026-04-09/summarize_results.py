from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"
REPORT_DIR = Path(__file__).resolve().parent
SUMMARY_JSON = REPORT_DIR / "summary.json"
SUMMARY_MD = REPORT_DIR / "summary.md"
TARGET_STAGE = "post-rs-full-retrain"
TARGET_VERSION = "rs_v1"


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_target_manifests():
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        manifest = _load_manifest(path)
        if manifest.get("experiment_stage") != TARGET_STAGE:
            continue
        if manifest.get("model_version") != TARGET_VERSION:
            continue
        yield path, manifest


def _infer_variant(manifest: dict) -> str:
    activation_type = manifest.get("activation_type") or "none"
    positions = manifest.get("activation_positions") or []
    if activation_type == "none":
        return "phase_only"
    if activation_type == "incoherent_intensity" and positions == [5]:
        return "incoherent_back"
    return f"{activation_type}:{','.join(str(v) for v in positions) or 'none'}"


def _record_from_manifest(path: Path, manifest: dict) -> dict:
    return {
        "manifest_path": str(path),
        "checkpoint_path": manifest.get("checkpoint"),
        "dataset": manifest.get("dataset"),
        "run_name": manifest.get("run_name"),
        "seed": manifest.get("seed"),
        "variant": _infer_variant(manifest),
        "best_val_accuracy": manifest.get("best_val_accuracy"),
        "best_val_contrast": manifest.get("best_val_contrast"),
        "best_epoch": manifest.get("best_epoch"),
        "test_accuracy": manifest.get("test_accuracy"),
        "test_contrast": manifest.get("test_contrast"),
    }


def _group_key(record: dict) -> tuple[str, str]:
    return record["dataset"], record["variant"]


def build_summary() -> dict:
    records = [_record_from_manifest(path, manifest) for path, manifest in _iter_target_manifests()]
    grouped = defaultdict(list)
    for record in records:
        grouped[_group_key(record)].append(record)

    groups = []
    for (dataset, variant), items in sorted(grouped.items()):
        test_accuracies = [item["test_accuracy"] for item in items if item["test_accuracy"] is not None]
        test_contrasts = [item["test_contrast"] for item in items if item["test_contrast"] is not None]
        groups.append(
            {
                "dataset": dataset,
                "variant": variant,
                "num_runs": len(items),
                "mean_test_accuracy": mean(test_accuracies) if test_accuracies else None,
                "mean_test_contrast": mean(test_contrasts) if test_contrasts else None,
                "runs": sorted(items, key=lambda item: (item["seed"] is None, item["seed"], item["run_name"] or "")),
            }
        )

    return {
        "stage": TARGET_STAGE,
        "model_version": TARGET_VERSION,
        "num_completed_runs": len(records),
        "groups": groups,
    }


def write_summary(summary: dict) -> None:
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Post-RS Full Retrain Summary",
        "",
        f"- stage: `{summary['stage']}`",
        f"- model_version: `{summary['model_version']}`",
        f"- completed runs: `{summary['num_completed_runs']}`",
        "",
    ]

    if not summary["groups"]:
        lines.append("No completed retrain manifests found yet.")
    else:
        lines.extend(
            [
                "| Dataset | Variant | Runs | Mean Test Accuracy | Mean Test Contrast |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for group in summary["groups"]:
            acc = "n/a" if group["mean_test_accuracy"] is None else f"{group['mean_test_accuracy']:.2f}%"
            contrast = "n/a" if group["mean_test_contrast"] is None else f"{group['mean_test_contrast']:.4f}"
            lines.append(
                f"| {group['dataset']} | {group['variant']} | {group['num_runs']} | {acc} | {contrast} |"
            )

        lines.append("")
        lines.append("## Runs")
        lines.append("")
        for group in summary["groups"]:
            lines.append(f"### {group['dataset']} / {group['variant']}")
            lines.append("")
            lines.append("| Seed | Run Name | Best Val Accuracy | Best Val Contrast | Test Accuracy | Test Contrast |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
            for item in group["runs"]:
                best_val_acc = "n/a" if item["best_val_accuracy"] is None else f"{item['best_val_accuracy']:.2f}%"
                best_val_contrast = "n/a" if item["best_val_contrast"] is None else f"{item['best_val_contrast']:.4f}"
                test_acc = "n/a" if item["test_accuracy"] is None else f"{item['test_accuracy']:.2f}%"
                test_contrast = "n/a" if item["test_contrast"] is None else f"{item['test_contrast']:.4f}"
                lines.append(
                    f"| {item['seed']} | `{item['run_name']}` | {best_val_acc} | {best_val_contrast} | {test_acc} | {test_contrast} |"
                )
            lines.append("")

    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    write_summary(summary)
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
