from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output_dir = repo_root / "reports" / "post_rs_retrain" / datetime.now().strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Summarize completed post-RS retrain manifests")
    parser.add_argument("--checkpoints-dir", type=Path, default=repo_root / "checkpoints")
    parser.add_argument("--stage", type=str, default="post-rs-full-retrain")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args()


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_positions(positions: list[int], num_layers: int | None) -> str:
    if not positions:
        return "none"
    pos = tuple(int(value) for value in positions)
    if num_layers:
        if pos == (1,):
            return "front"
        if pos == (num_layers,):
            return "back"
        if pos == tuple(range(1, num_layers + 1)):
            return "all"
        middle = (num_layers + 1) // 2
        if pos == (middle,):
            return "mid"
    return ",".join(str(value) for value in pos)


def config_label(manifest: dict) -> str:
    activation_type = manifest.get("activation_type") or "none"
    if activation_type == "none":
        return "phase-only"

    optical_config = manifest.get("optical_config") or {}
    num_layers = optical_config.get("num_layers")
    placement = normalize_positions(manifest.get("activation_positions") or [], num_layers)
    return f"{activation_type}@{placement}"


def collect_runs(checkpoints_dir: Path, stage: str) -> list[dict]:
    runs: list[dict] = []
    for manifest_path in sorted(checkpoints_dir.glob("*.json")):
        manifest = read_manifest(manifest_path)
        if manifest.get("task") != "classification":
            continue
        if manifest.get("experiment_stage") != stage:
            continue

        run = {
            "dataset": manifest.get("dataset", "unknown"),
            "run_name": manifest.get("run_name") or manifest_path.stem,
            "seed": manifest.get("seed"),
            "config_label": config_label(manifest),
            "best_val_accuracy": manifest.get("best_val_accuracy"),
            "best_val_contrast": manifest.get("best_val_contrast"),
            "best_epoch": manifest.get("best_epoch"),
            "test_accuracy": manifest.get("test_accuracy"),
            "test_contrast": manifest.get("test_contrast"),
            "checkpoint": manifest.get("checkpoint"),
            "manifest": str(manifest_path),
        }
        runs.append(run)
    return runs


def build_aggregates(runs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["dataset"], run["config_label"])].append(run)

    aggregates: list[dict] = []
    for (dataset, label), group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: (item["seed"] is None, item["seed"]))
        test_acc = [value["test_accuracy"] for value in group if value["test_accuracy"] is not None]
        test_contrast = [value["test_contrast"] for value in group if value["test_contrast"] is not None]
        aggregates.append(
            {
                "dataset": dataset,
                "config_label": label,
                "num_runs": len(group),
                "seeds": [value["seed"] for value in group],
                "mean_test_accuracy": round(sum(test_acc) / len(test_acc), 4) if test_acc else None,
                "mean_test_contrast": round(sum(test_contrast) / len(test_contrast), 6) if test_contrast else None,
                "runs": group,
            }
        )
    return aggregates


def format_metric(value: float | None, precision: int) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def render_markdown(stage: str, runs: list[dict], aggregates: list[dict]) -> str:
    lines = [
        "# Post-RS Retrain Summary",
        "",
        f"- stage: `{stage}`",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
    ]

    if not runs:
        lines.extend(
            [
                "No completed classification manifests found for this stage.",
                "",
                "Expected source: `checkpoints/*.json` written by `train.py` after a run completes.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Aggregates",
            "",
            "| Dataset | Config | Runs | Mean test acc | Mean test contrast | Seeds |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for item in aggregates:
        seed_text = ",".join("-" if seed is None else str(seed) for seed in item["seeds"])
        lines.append(
            f"| {item['dataset']} | {item['config_label']} | {item['num_runs']} | "
            f"{format_metric(item['mean_test_accuracy'], 2)}% | {format_metric(item['mean_test_contrast'], 4)} | {seed_text} |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Dataset | Run | Seed | Best val acc | Best val contrast | Test acc | Test contrast | Checkpoint |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in runs:
        checkpoint_name = Path(run["checkpoint"]).name if run["checkpoint"] else "-"
        lines.append(
            f"| {run['dataset']} | {run['run_name']} | {run['seed'] if run['seed'] is not None else '-'} | "
            f"{format_metric(run['best_val_accuracy'], 2)}% | {format_metric(run['best_val_contrast'], 4)} | "
            f"{format_metric(run['test_accuracy'], 2)}% | {format_metric(run['test_contrast'], 4)} | {checkpoint_name} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.checkpoints_dir, args.stage)
    aggregates = build_aggregates(runs)
    payload = {
        "stage": args.stage,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "num_runs": len(runs),
        "runs": runs,
        "aggregates": aggregates,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(args.stage, runs, aggregates), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Wrote {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
