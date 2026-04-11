"""
Thin final-export wrapper for the current official
`fmnist5-phaseonly-aligned` fabrication line.

This script intentionally keeps the real export logic inside
`export_phase_plate.py` and only adds:
1) a frozen official preset,
2) lab-parameter validation,
3) a short post-export validation summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

from artifacts import CLASSIFIER_PAPER_OPTICS, build_model_for_task, save_manifest


OFFICIAL_PRESET = {
    "name": "fmnist5-phaseonly-aligned",
    "task": "classification",
    "checkpoint": Path("checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.pth"),
    "size": 200,
    "layers": 5,
    "wavelength": 852e-9,
    "layer_distance": 0.03,
    "pixel_size": 1e-6,
    "input_distance": 491.302e-3,
    "output_distance": 575.304e-3,
}

REQUIRED_EXPORT_FILES = (
    "phase_masks.npy",
    "height_map.npy",
    "height_map_manufacturable.npy",
    "thickness_map.npy",
    "height_map_quantized.npy",
    "report.md",
    "metadata.json",
)

WARN_CLIPPED_FRACTION_MAX = 0.01
OFFICIAL_ARTIFACT_DIR = Path("docs/official-artifacts/fmnist5-phaseonly-aligned")
REQUIRED_LAB_FIELDS = (
    "refractive_index",
    "ambient_index",
    "base_thickness_um",
    "max_relief_um",
    "quantization_levels",
)


def _load_source_manifest(repo_root: Path) -> tuple[Path, dict]:
    source_manifest_path = repo_root / OFFICIAL_ARTIFACT_DIR / "source_checkpoint_manifest.json"
    source_manifest = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    return source_manifest_path, source_manifest


def _assert_bootstrap_optics_match_official(source_manifest: dict) -> None:
    source_optics = dict(source_manifest.get("optical_config") or {})
    if not source_optics:
        return

    expected_fields = ("wavelength", "layer_distance", "pixel_size", "input_distance", "output_distance")
    mismatched = []
    for field_name in expected_fields:
        source_value = source_optics.get(field_name)
        expected_value = OFFICIAL_PRESET[field_name]
        if source_value is None or source_value != expected_value:
            mismatched.append((field_name, source_value, expected_value))

    if mismatched:
        details = ", ".join(
            f"{field_name}={source_value!r} (expected {expected_value!r})"
            for field_name, source_value, expected_value in mismatched
        )
        raise ValueError(
            "Cannot bootstrap the frozen export checkpoint from legacy optical config: "
            + details
            + ". Refresh the official checkpoint/artifacts under the current measured optics first."
        )


def build_default_output_dir(*, current_date: date | None = None) -> Path:
    current_date = current_date or date.today()
    return Path("exports") / f"{OFFICIAL_PRESET['name']}-final_{current_date:%Y%m%d}"


def build_export_command(
    *,
    python_executable: str,
    checkpoint_path: Path,
    output_dir: Path,
    refractive_index: float,
    ambient_index: float,
    base_thickness_um: float,
    max_relief_um: float,
    quantization_levels: int,
    export_stl: bool,
) -> list[str]:
    command = [
        python_executable,
        "-u",
        "export_phase_plate.py",
        "--task",
        OFFICIAL_PRESET["task"],
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--output-dir",
        output_dir.as_posix(),
        "--size",
        str(OFFICIAL_PRESET["size"]),
        "--layers",
        str(OFFICIAL_PRESET["layers"]),
        "--wavelength",
        str(OFFICIAL_PRESET["wavelength"]),
        "--layer-distance",
        str(OFFICIAL_PRESET["layer_distance"]),
        "--pixel-size",
        str(OFFICIAL_PRESET["pixel_size"]),
        "--input-distance",
        str(OFFICIAL_PRESET["input_distance"]),
        "--output-distance",
        str(OFFICIAL_PRESET["output_distance"]),
        "--refractive-index",
        str(refractive_index),
        "--ambient-index",
        str(ambient_index),
        "--base-thickness-um",
        str(base_thickness_um),
        "--max-relief-um",
        str(max_relief_um),
        "--quantization-levels",
        str(quantization_levels),
    ]
    if export_stl:
        command.append("--export-stl")
    return command


def resolve_checkpoint_path(repo_root: Path) -> Path:
    checkpoint_path = repo_root / OFFICIAL_PRESET["checkpoint"]
    if checkpoint_path.exists():
        return checkpoint_path
    if bootstrap_checkpoint_from_official_artifacts(repo_root, checkpoint_path):
        return checkpoint_path
    raise FileNotFoundError(
        "Official checkpoint is missing from the local workspace: "
        f"{OFFICIAL_PRESET['checkpoint'].as_posix()}"
    )


def bootstrap_checkpoint_from_official_artifacts(repo_root: Path, checkpoint_path: Path) -> bool:
    artifact_dir = repo_root / OFFICIAL_ARTIFACT_DIR
    phase_masks_path = artifact_dir / "phase_masks.npy"
    if not phase_masks_path.exists():
        return False
    _, source_manifest = _load_source_manifest(repo_root)
    _assert_bootstrap_optics_match_official(source_manifest)

    phase_masks = np.load(phase_masks_path)
    expected_shape = (
        OFFICIAL_PRESET["layers"],
        OFFICIAL_PRESET["size"],
        OFFICIAL_PRESET["size"],
    )
    if tuple(phase_masks.shape) != expected_shape:
        raise ValueError(
            "Official phase_masks.npy has unexpected shape: "
            f"expected {expected_shape}, got {tuple(phase_masks.shape)}"
        )

    optics = CLASSIFIER_PAPER_OPTICS.with_overrides(
        size=OFFICIAL_PRESET["size"],
        num_layers=OFFICIAL_PRESET["layers"],
        wavelength=OFFICIAL_PRESET["wavelength"],
        layer_distance=OFFICIAL_PRESET["layer_distance"],
        pixel_size=OFFICIAL_PRESET["pixel_size"],
        input_distance=OFFICIAL_PRESET["input_distance"],
        output_distance=OFFICIAL_PRESET["output_distance"],
    )
    model = build_model_for_task(OFFICIAL_PRESET["task"], optics)
    with torch.no_grad():
        for idx, layer in enumerate(model.layers):
            layer.phase.copy_(torch.from_numpy(phase_masks[idx]).to(dtype=layer.phase.dtype))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    _write_bootstrap_manifest(repo_root, checkpoint_path)
    return True


def _write_bootstrap_manifest(repo_root: Path, checkpoint_path: Path) -> None:
    source_manifest_path, source_manifest = _load_source_manifest(repo_root)

    payload = {
        "task": OFFICIAL_PRESET["task"],
        "dataset": source_manifest.get("dataset", "Fashion-MNIST"),
        "checkpoint": str(checkpoint_path.resolve()),
        "run_name": source_manifest.get("run_name", OFFICIAL_PRESET["name"]),
        "experiment_stage": source_manifest.get("experiment_stage", "fabrication_baseline"),
        "seed": source_manifest.get("seed", 42),
        "optical_config": {
            "wavelength": OFFICIAL_PRESET["wavelength"],
            "layer_distance": OFFICIAL_PRESET["layer_distance"],
            "pixel_size": OFFICIAL_PRESET["pixel_size"],
            "input_distance": OFFICIAL_PRESET["input_distance"],
            "output_distance": OFFICIAL_PRESET["output_distance"],
            "size": OFFICIAL_PRESET["size"],
            "num_layers": OFFICIAL_PRESET["layers"],
        },
        "bootstrap_source": {
            "phase_masks": str((repo_root / OFFICIAL_ARTIFACT_DIR / "phase_masks.npy").resolve()),
            "source_checkpoint_manifest": str(source_manifest_path.resolve()) if source_manifest_path.exists() else None,
            "method": "rebuilt checkpoint from official wrapped phase masks",
        },
    }
    save_manifest(checkpoint_path.with_suffix(".json"), payload)


def load_lab_config(path: Path) -> dict[str, float | bool]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    material = payload.get("material", {})
    process = payload.get("process", {})
    return {
        "refractive_index": material.get("refractive_index"),
        "ambient_index": material.get("ambient_index"),
        "base_thickness_um": process.get("base_thickness_um"),
        "max_relief_um": process.get("max_relief_um"),
        "quantization_levels": process.get("quantization_levels"),
        "export_stl": bool(process.get("stl_required", False)),
    }


def resolve_lab_inputs(args: argparse.Namespace) -> dict[str, float | int | bool]:
    config_values: dict[str, float | bool] = {}
    if args.lab_config is not None:
        config_values = load_lab_config(args.lab_config)

    resolved = {
        "refractive_index": args.refractive_index if args.refractive_index is not None else config_values.get("refractive_index"),
        "ambient_index": args.ambient_index if args.ambient_index is not None else config_values.get("ambient_index"),
        "base_thickness_um": args.base_thickness_um if args.base_thickness_um is not None else config_values.get("base_thickness_um"),
        "max_relief_um": args.max_relief_um if args.max_relief_um is not None else config_values.get("max_relief_um"),
        "quantization_levels": args.quantization_levels if args.quantization_levels is not None else config_values.get("quantization_levels"),
        "export_stl": bool(args.export_stl or config_values.get("export_stl", False)),
    }
    missing = [field for field in REQUIRED_LAB_FIELDS if resolved.get(field) is None]
    if missing:
        raise ValueError("Missing required lab inputs after merging CLI and lab config: " + ", ".join(missing))
    return resolved


def validate_lab_inputs(lab_inputs: dict[str, float | int | bool], parser: argparse.ArgumentParser) -> None:
    refractive_index = float(lab_inputs["refractive_index"])
    ambient_index = float(lab_inputs["ambient_index"])
    base_thickness_um = float(lab_inputs["base_thickness_um"])
    max_relief_um = float(lab_inputs["max_relief_um"])
    quantization_levels = int(lab_inputs["quantization_levels"])

    if refractive_index <= 0:
        parser.error("--refractive-index must be positive")
    if ambient_index <= 0:
        parser.error("--ambient-index must be positive")
    if refractive_index <= ambient_index:
        parser.error("--refractive-index must be greater than --ambient-index")
    if base_thickness_um < 0:
        parser.error("--base-thickness-um must be non-negative")
    if max_relief_um <= 0:
        parser.error("--max-relief-um must be positive")
    if quantization_levels < 2:
        parser.error("--quantization-levels must be at least 2")


def _read_metadata(export_root: Path) -> dict:
    metadata_path = export_root / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_validation_summary(export_root: Path) -> dict:
    export_root = Path(export_root)
    repo_root = Path(__file__).resolve().parent
    issues: list[str] = []
    missing_files = [name for name in REQUIRED_EXPORT_FILES if not (export_root / name).exists()]
    if missing_files:
        issues.extend(f"Missing required export file: {name}" for name in missing_files)

    metadata = _read_metadata(export_root) if (export_root / "metadata.json").exists() else {}
    readiness = metadata.get("fabrication_readiness", {})
    manufacturing = metadata.get("manufacturing", {})
    clipped_fraction = float(readiness.get("clipped_fraction", 0.0) or 0.0)
    clipped_pixels = int(readiness.get("clipped_pixels", 0) or 0)

    checkpoint_value = metadata.get("checkpoint")
    expected_checkpoint = (repo_root / OFFICIAL_PRESET["checkpoint"]).resolve()
    if checkpoint_value and Path(checkpoint_value).resolve() != expected_checkpoint:
        issues.append(
            "Export checkpoint does not match the frozen official preset: "
            f"{expected_checkpoint}"
        )

    max_relief_um = manufacturing.get("max_relief_um")
    if max_relief_um is None:
        issues.append("Export metadata is missing manufacturing.max_relief_um")

    if manufacturing.get("export_stl") and not (export_root / "stl").exists():
        issues.append("STL export was requested but the stl/ directory is missing")

    warnings: list[str] = []
    if clipped_fraction > 0:
        warnings.append(
            f"Clipping detected: {clipped_pixels} pixels ({clipped_fraction:.2%}) exceed the relief limit."
        )

    if missing_files or clipped_fraction > WARN_CLIPPED_FRACTION_MAX:
        status = "STOP"
    elif warnings:
        status = "WARN"
    else:
        status = "PASS"

    all_issues = issues + warnings
    return {
        "status": status,
        "preset": OFFICIAL_PRESET["name"],
        "export_root": str(export_root.resolve()),
        "checkpoint": checkpoint_value,
        "clipped_pixels": clipped_pixels,
        "clipped_fraction": clipped_fraction,
        "raw_height_max_um": _meters_to_um(readiness.get("raw_height_max_m")),
        "manufacturable_height_max_um": _meters_to_um(readiness.get("manufacturable_height_max_m")),
        "thickness_max_um": _meters_to_um(readiness.get("thickness_max_m")),
        "missing_files": missing_files,
        "issues": all_issues,
    }


def _meters_to_um(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) * 1e6


def write_validation_summary(export_root: Path) -> Path:
    summary = build_validation_summary(export_root)
    path = export_root / "validation_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Final export wrapper for the official fmnist5-phaseonly-aligned fabrication line"
    )
    parser.add_argument("--lab-config", type=Path, default=None)
    parser.add_argument("--refractive-index", type=float, default=None)
    parser.add_argument("--ambient-index", type=float, default=None)
    parser.add_argument("--base-thickness-um", type=float, default=None)
    parser.add_argument("--max-relief-um", type=float, default=None)
    parser.add_argument("--quantization-levels", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--export-stl", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        lab_inputs = resolve_lab_inputs(args)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    validate_lab_inputs(lab_inputs, parser)

    repo_root = Path(__file__).resolve().parent
    try:
        checkpoint_path = resolve_checkpoint_path(repo_root)
    except FileNotFoundError as exc:
        parser.exit(1, f"{exc}\n")
    output_dir = args.output_dir or build_default_output_dir()
    command = build_export_command(
        python_executable=sys.executable,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        refractive_index=float(lab_inputs["refractive_index"]),
        ambient_index=float(lab_inputs["ambient_index"]),
        base_thickness_um=float(lab_inputs["base_thickness_um"]),
        max_relief_um=float(lab_inputs["max_relief_um"]),
        quantization_levels=int(lab_inputs["quantization_levels"]),
        export_stl=bool(lab_inputs["export_stl"]),
    )

    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    export_root = output_dir / OFFICIAL_PRESET["checkpoint"].stem
    summary_path = write_validation_summary(export_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    print(f"Saved validation summary to: {summary_path}")
    print(f"Validation status: {summary['status']}")
    for issue in summary["issues"]:
        print(f"- {issue}")


if __name__ == "__main__":
    main()
