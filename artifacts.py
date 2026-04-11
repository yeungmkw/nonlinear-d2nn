"""
Consolidated shared helpers for D2NN artifacts, optics, visualization, and
manufacturing-oriented exports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from d2nn import D2NN, D2NNImager


@dataclass(frozen=True)
class OpticalConfig:
    wavelength: float
    layer_distance: float
    pixel_size: float
    input_distance: float
    output_distance: float
    size: int = 200
    num_layers: int = 5

    def with_overrides(
        self,
        *,
        wavelength: float | None = None,
        layer_distance: float | None = None,
        pixel_size: float | None = None,
        input_distance: float | None = None,
        output_distance: float | None = None,
        size: int | None = None,
        num_layers: int | None = None,
    ) -> "OpticalConfig":
        return OpticalConfig(
            wavelength=self.wavelength if wavelength is None else wavelength,
            layer_distance=self.layer_distance if layer_distance is None else layer_distance,
            pixel_size=self.pixel_size if pixel_size is None else pixel_size,
            input_distance=self.input_distance if input_distance is None else input_distance,
            output_distance=self.output_distance if output_distance is None else output_distance,
            size=self.size if size is None else size,
            num_layers=self.num_layers if num_layers is None else num_layers,
        )

    def classifier_model_kwargs(self, *, num_classes: int = 10) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "size": self.size,
            "num_classes": num_classes,
            "wavelength": self.wavelength,
            "layer_distance": self.layer_distance,
            "pixel_size": self.pixel_size,
            "input_distance": self.input_distance,
            "output_distance": self.output_distance,
        }

    def imager_model_kwargs(self, *, input_fraction: float = 0.5) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "size": self.size,
            "wavelength": self.wavelength,
            "layer_distance": self.layer_distance,
            "pixel_size": self.pixel_size,
            "input_distance": self.input_distance,
            "output_distance": self.output_distance,
            "input_fraction": input_fraction,
        }


CLASSIFIER_PAPER_OPTICS = OpticalConfig(
    wavelength=852e-9,
    layer_distance=30e-3,
    pixel_size=1e-6,
    input_distance=491.302e-3,
    output_distance=575.304e-3,
)

CLASSIFIER_LAB852_F10_OPTICS = OpticalConfig(
    wavelength=852e-9,
    layer_distance=1.17370892018779e-3,
    pixel_size=1e-6,
    input_distance=491.302e-3,
    output_distance=575.304e-3,
)

CLASSIFIER_LAB852_F5_OPTICS = OpticalConfig(
    wavelength=852e-9,
    layer_distance=2.34741784037559e-3,
    pixel_size=1e-6,
    input_distance=491.302e-3,
    output_distance=575.304e-3,
)

IMAGER_PAPER_OPTICS = OpticalConfig(
    wavelength=852e-9,
    layer_distance=4e-3,
    pixel_size=1e-6,
    input_distance=491.302e-3,
    output_distance=575.304e-3,
)


CLASSIFICATION_OPTICS_PRESETS = {
    "paper": CLASSIFIER_PAPER_OPTICS,
    "lab852_f10": CLASSIFIER_LAB852_F10_OPTICS,
    "lab852_f5": CLASSIFIER_LAB852_F5_OPTICS,
}

IMAGING_OPTICS_PRESETS = {
    "paper": IMAGER_PAPER_OPTICS,
}


def infer_optics_preset_hint(checkpoint_path, manifest=None) -> str | None:
    manifest_preset = (manifest or {}).get("optics_preset")
    if manifest_preset:
        return str(manifest_preset)

    if checkpoint_path is None:
        return None

    stem = Path(checkpoint_path).stem.lower().replace("-", "_")
    for preset_name in CLASSIFICATION_OPTICS_PRESETS:
        if preset_name != "paper" and preset_name in stem:
            return preset_name
    return None


def configure_matplotlib_backend(no_show=False):
    if no_show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def optical_config_dict(config: OpticalConfig) -> dict[str, Any]:
    return asdict(config)


def load_checkpoint_state_dict(checkpoint_path, map_location="cpu"):
    return torch.load(Path(checkpoint_path), map_location=map_location, weights_only=True)


def infer_architecture(state_dict):
    phase_keys = [key for key in state_dict if key.startswith("layers.") and key.endswith(".phase")]
    if not phase_keys:
        raise ValueError("Could not infer architecture from checkpoint: no layer phase keys found")

    layer_indices = sorted({int(key.split(".")[1]) for key in phase_keys})
    size = state_dict[phase_keys[0]].shape[0]
    return {"num_layers": len(layer_indices), "size": size}


def resolve_optics(
    base_optics: OpticalConfig,
    *,
    state_dict=None,
    manifest=None,
    checkpoint_path=None,
    size=None,
    num_layers=None,
    wavelength=None,
    layer_distance=None,
    pixel_size=None,
    input_distance=None,
    output_distance=None,
):
    manifest_optics = (manifest or {}).get("optical_config") or {}
    inferred = infer_architecture(state_dict) if state_dict is not None else {}
    preset_hint = infer_optics_preset_hint(checkpoint_path, manifest=manifest)
    missing_split_fields = [
        field_name
        for field_name, explicit_value in (
            ("input_distance", input_distance),
            ("output_distance", output_distance),
        )
        if explicit_value is None and manifest_optics and manifest_optics.get(field_name) is None
    ]

    if missing_split_fields:
        missing_summary = ", ".join(missing_split_fields)
        raise ValueError(
            f"Checkpoint {checkpoint_path or '<in-memory>'} uses a legacy manifest optical config that is missing "
            f"{missing_summary}. Supply the split distances explicitly or regenerate the checkpoint manifest with "
            "input/output propagation distances."
        )

    if preset_hint not in (None, "", "paper"):
        missing_fields = [
            field_name
            for field_name, explicit_value in (
                ("wavelength", wavelength),
                ("layer_distance", layer_distance),
                ("pixel_size", pixel_size),
                ("input_distance", input_distance),
                ("output_distance", output_distance),
            )
            if explicit_value is None and manifest_optics.get(field_name) is None
        ]
        if missing_fields:
            raise ValueError(
                f"Checkpoint {checkpoint_path} appears to use optics preset {preset_hint!r} but is missing its checkpoint "
                "manifest optical config. Restore the adjacent .json manifest or pass the optical distances and sizes explicitly."
            )

    return base_optics.with_overrides(
        size=inferred.get("size") if size is None else size,
        num_layers=manifest_optics.get("num_layers", inferred.get("num_layers")) if num_layers is None else num_layers,
        wavelength=manifest_optics.get("wavelength") if wavelength is None else wavelength,
        layer_distance=manifest_optics.get("layer_distance") if layer_distance is None else layer_distance,
        pixel_size=manifest_optics.get("pixel_size") if pixel_size is None else pixel_size,
        input_distance=manifest_optics.get("input_distance") if input_distance is None else input_distance,
        output_distance=manifest_optics.get("output_distance") if output_distance is None else output_distance,
    )


def build_model_for_task(
    task,
    optics,
    *,
    input_fraction=0.5,
    num_classes=10,
    activation_type="none",
    activation_positions=None,
    activation_hparams=None,
    propagation_chunk_size=None,
    propagation_backend="direct",
):
    if task == "classification":
        return D2NN(
            **optics.classifier_model_kwargs(num_classes=num_classes),
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_chunk_size=propagation_chunk_size,
            propagation_backend=propagation_backend,
        )
    if task == "imaging":
        return D2NNImager(
            **optics.imager_model_kwargs(input_fraction=input_fraction),
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_chunk_size=propagation_chunk_size,
            propagation_backend=propagation_backend,
        )
    raise ValueError(f"Unsupported task: {task}")


def resolve_training_optics_preset(task: str, preset_name: str) -> OpticalConfig:
    if task == "classification":
        presets = CLASSIFICATION_OPTICS_PRESETS
    elif task == "imaging":
        presets = IMAGING_OPTICS_PRESETS
    else:
        raise ValueError(f"Unsupported task: {task}")

    if preset_name not in presets:
        valid = ", ".join(sorted(presets))
        raise ValueError(f"Unsupported optics preset {preset_name!r} for task {task!r}. Expected one of: {valid}")
    return presets[preset_name]


def checkpoint_manifest_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".json")


def _sanitize_run_name(value: str | None) -> str:
    safe_name = re.sub(r"\s+", "_", str(value or "").strip())
    safe_name = re.sub(r'[<>:"/\\|?*]+', "-", safe_name)
    safe_name = re.sub(r"[-_]{2,}", "-", safe_name)
    return safe_name.strip("._-")


def _format_run_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        formatted = f"{value:.6g}"
        formatted = formatted.replace("-", "m").replace(".", "p")
        return formatted
    return _sanitize_run_name(str(value).replace("_", "-")).lower()


DEFAULT_CLASSIFICATION_LOSS_CONFIG = {"alpha": 1.0, "beta": 0.1, "gamma": 0.01}


def derive_experiment_run_name(
    *,
    run_name: str | None = None,
    experiment_stage: str | None = None,
    activation_type: str | None = None,
    activation_positions=None,
    activation_hparams: dict[str, Any] | None = None,
    seed: int | None = None,
    loss_config: dict[str, Any] | None = None,
    propagation_backend: str | None = None,
    propagation_chunk_size: int | None = None,
    optics_preset: str | None = None,
    layer_count: int | None = None,
):
    if run_name:
        return run_name

    loss_config = dict(loss_config or {})
    has_nondefault_loss_config = any(
        value is not None and value != DEFAULT_CLASSIFICATION_LOSS_CONFIG.get(key)
        for key, value in loss_config.items()
    )

    has_nondefault_propagation = (propagation_backend not in (None, "", "direct")) or (propagation_chunk_size is not None)
    has_nondefault_optics = optics_preset not in (None, "", "paper")
    has_nondefault_topology = layer_count not in (None, 5)

    if (
        activation_type in (None, "", "none")
        and not has_nondefault_loss_config
        and not has_nondefault_propagation
        and not has_nondefault_optics
        and not has_nondefault_topology
    ):
        return None

    stage_label = (experiment_stage or "nonlinear").replace("_", "-")
    activation_label = str(activation_type or "none").replace("_", "-")
    parts = [f"stage-{stage_label}", f"act-{activation_label}"]

    if optics_preset not in (None, "", "paper"):
        parts.append(f"optics-{_format_run_value(optics_preset)}")

    if layer_count not in (None, 5):
        parts.append(f"layers-{int(layer_count)}")

    if activation_positions:
        position_token = "-".join(str(int(position)) for position in activation_positions)
        parts.append(f"pos-{position_token}")

    ordered_hparams = (
        "threshold",
        "temperature",
        "gain_min",
        "gain_max",
        "gamma",
        "responsivity",
        "emission_phase_mode",
    )
    activation_hparams = activation_hparams or {}
    for key in ordered_hparams:
        value = activation_hparams.get(key)
        if value is None:
            continue
        token_key = key.replace("_", "-")
        parts.append(f"{token_key}-{_format_run_value(value)}")

    for key in ("alpha", "beta", "gamma"):
        value = loss_config.get(key)
        if value is None:
            continue
        parts.append(f"{key}-{_format_run_value(value)}")

    if propagation_backend not in (None, "", "direct"):
        parts.append(f"rs-{_format_run_value(propagation_backend)}")

    if propagation_chunk_size is not None:
        parts.append(f"chunk-{_format_run_value(int(propagation_chunk_size))}")

    if seed is not None:
        parts.append(f"seed-{seed}")

    safe_name = "__".join(_sanitize_run_name(part.lower()) for part in parts if part)
    return safe_name or None


def checkpoint_variant_path(checkpoint_path, run_name=None):
    checkpoint_path = Path(checkpoint_path)
    if not run_name:
        return checkpoint_path

    safe_name = _sanitize_run_name(str(run_name))
    if not safe_name:
        return checkpoint_path

    return checkpoint_path.with_name(f"{checkpoint_path.stem}.{safe_name}{checkpoint_path.suffix}")


def experiment_manifest_fields(
    *,
    checkpoint_path,
    run_name=None,
    experiment_stage=None,
    seed=None,
    optics: OpticalConfig | None = None,
    activation_type=None,
    activation_positions=None,
    activation_hparams=None,
    model_version=None,
    loss_config=None,
    propagation_backend=None,
    propagation_chunk_size=None,
    runtime_config=None,
    optics_preset=None,
):
    payload = {
        "checkpoint": str(Path(checkpoint_path)),
        "run_name": run_name,
        "experiment_stage": experiment_stage,
        "seed": seed,
        "activation_type": activation_type,
        "activation_positions": list(activation_positions or ()),
        "activation_hparams": dict(activation_hparams or {}),
        "model_version": model_version,
        "loss_config": dict(loss_config or {}),
        "propagation_backend": propagation_backend,
        "propagation_chunk_size": propagation_chunk_size,
        "runtime_config": dict(runtime_config or {}),
        "optics_preset": optics_preset,
    }
    if optics is not None:
        payload["optical_config"] = optical_config_dict(optics)
    return payload


def ensure_checkpoint_version(manifest, expected_version, checkpoint_path, allow_missing=False):
    found_version = (manifest or {}).get("model_version")
    if allow_missing and found_version is None:
        return
    if found_version != expected_version:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has model_version={found_version!r}; expected model_version={expected_version!r}"
        )


def read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_checkpoint_manifest(checkpoint_path):
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    if not manifest_path.exists():
        return None
    return read_manifest(manifest_path)


def maybe_show(no_show: bool) -> None:
    if not no_show:
        plt = configure_matplotlib_backend(no_show=no_show)
        plt.show()


def plot_phase_masks(model, save_path=None, no_show=False):
    plt = configure_matplotlib_backend(no_show=no_show)
    num_layers = len(model.layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4), constrained_layout=True)
    if num_layers == 1:
        axes = [axes]

    im = None
    for i, layer in enumerate(model.layers):
        phase = layer.phase.detach().cpu().numpy()
        im = axes[i].imshow(phase, cmap="twilight", vmin=-3.141592653589793, vmax=3.141592653589793)
        axes[i].set_title(f"Layer {i + 1}")
        axes[i].axis("off")

    fig.colorbar(im, ax=axes, label="Phase (rad)", shrink=0.8)
    fig.suptitle("Learned Phase Masks", fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    maybe_show(no_show)
    plt.close(fig)


def apply_manufacturing_profile(
    height_map: np.ndarray,
    *,
    base_thickness_m: float = 0.0,
    max_relief_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    manufacturable_relief = np.array(height_map, copy=True)
    if max_relief_m is not None:
        manufacturable_relief = np.clip(manufacturable_relief, 0.0, max_relief_m)
    thickness_map = manufacturable_relief + base_thickness_m
    return manufacturable_relief, thickness_map


def build_fabrication_readiness_summary(
    raw_height_map: np.ndarray,
    manufacturable_relief: np.ndarray,
    thickness_map: np.ndarray,
    *,
    max_relief_m: float | None,
    pixel_size_m: float,
) -> dict[str, Any]:
    raw_height_map = np.asarray(raw_height_map)
    manufacturable_relief = np.asarray(manufacturable_relief)
    thickness_map = np.asarray(thickness_map)
    clipped = np.abs(raw_height_map - manufacturable_relief) > 1e-12
    total_pixels = int(raw_height_map.size)
    clipped_pixels = int(np.count_nonzero(clipped))

    return {
        "has_relief_limit": max_relief_m is not None,
        "max_relief_m": None if max_relief_m is None else float(max_relief_m),
        "raw_height_max_m": float(raw_height_map.max()) if raw_height_map.size else 0.0,
        "manufacturable_height_max_m": float(manufacturable_relief.max()) if manufacturable_relief.size else 0.0,
        "thickness_max_m": float(thickness_map.max()) if thickness_map.size else 0.0,
        "clipped_fraction": float(clipped_pixels / total_pixels) if total_pixels else 0.0,
        "clipped_pixels": clipped_pixels,
        "total_pixels": total_pixels,
        "pixel_size_m": float(pixel_size_m),
    }


def quantize_height_map(height_map: np.ndarray, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be at least 2")

    max_height = float(height_map.max())
    if max_height <= 0:
        return np.zeros_like(height_map, dtype=np.uint16)

    normalized = np.clip(height_map / max_height, 0.0, 1.0)
    return np.rint(normalized * (levels - 1)).astype(np.uint16)


def quantize_phase_masks_uniform(phase_masks, levels: int):
    if levels < 2:
        raise ValueError("levels must be at least 2")

    phase_masks = torch.as_tensor(phase_masks)
    wrapped = torch.remainder(phase_masks, 2 * np.pi)
    step = (2 * np.pi) / levels
    quantized = torch.round(wrapped / step) * step
    return torch.remainder(quantized, 2 * np.pi)


def phase_masks_to_bmp_uint8(phase_masks: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(phase_masks), 2 * np.pi)
    scaled = np.rint((wrapped / (2 * np.pi)) * 255.0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def build_layer_stats(
    phase_masks: np.ndarray,
    relief_map: np.ndarray,
    thickness_map: np.ndarray | None = None,
) -> list[dict]:
    stats = []
    for idx in range(phase_masks.shape[0]):
        phase_layer = phase_masks[idx]
        relief_layer = relief_map[idx]
        thickness_layer = thickness_map[idx] if thickness_map is not None else None
        stats.append(
            {
                "layer": idx + 1,
                "phase_min_rad": float(phase_layer.min()),
                "phase_max_rad": float(phase_layer.max()),
                "height_min_m": float(relief_layer.min()),
                "height_max_m": float(relief_layer.max()),
                "height_mean_m": float(relief_layer.mean()),
                "thickness_min_m": float(thickness_layer.min()) if thickness_layer is not None else None,
                "thickness_max_m": float(thickness_layer.max()) if thickness_layer is not None else None,
                "thickness_mean_m": float(thickness_layer.mean()) if thickness_layer is not None else None,
            }
        )
    return stats


def save_layer_csvs(
    export_root: Path,
    phase_masks: np.ndarray,
    relief_map: np.ndarray,
    thickness_map: np.ndarray,
    quantized_map: np.ndarray,
) -> None:
    layers_dir = export_root / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(phase_masks.shape[0]):
        layer_id = f"layer_{idx + 1:02d}"
        np.savetxt(layers_dir / f"{layer_id}_phase_rad.csv", phase_masks[idx], delimiter=",", fmt="%.8f")
        np.savetxt(layers_dir / f"{layer_id}_height_um.csv", relief_map[idx] * 1e6, delimiter=",", fmt="%.6f")
        np.savetxt(layers_dir / f"{layer_id}_thickness_um.csv", thickness_map[idx] * 1e6, delimiter=",", fmt="%.6f")
        np.savetxt(layers_dir / f"{layer_id}_height_quantized.csv", quantized_map[idx], delimiter=",", fmt="%d")


def write_export_report(
    path: Path,
    *,
    checkpoint_name: str,
    task: str,
    num_layers: int,
    size: int,
    pixel_size_um: float,
    wavelength_um: float,
    quantization_levels: int,
    layer_stats: list[dict],
    fabrication_readiness: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Phase Plate Export Report",
        "",
        f"- checkpoint: `{checkpoint_name}`",
        f"- task: `{task}`",
        f"- layers: `{num_layers}`",
        f"- layer resolution: `{size} x {size}`",
        f"- pixel size: `{pixel_size_um:.3f} um`",
        f"- wavelength: `{wavelength_um:.3f} um`",
        f"- quantization levels: `{quantization_levels}`",
        "",
        "## Layer Ranges",
        "",
        "| Layer | Phase Min (rad) | Phase Max (rad) | Height Min (um) | Height Max (um) | Height Mean (um) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for item in layer_stats:
        lines.append(
            "| {layer} | {phase_min:.4f} | {phase_max:.4f} | {height_min:.4f} | {height_max:.4f} | {height_mean:.4f} |".format(
                layer=item["layer"],
                phase_min=item["phase_min_rad"],
                phase_max=item["phase_max_rad"],
                height_min=item["height_min_m"] * 1e6,
                height_max=item["height_max_m"] * 1e6,
                height_mean=item["height_mean_m"] * 1e6,
            )
        )

    if fabrication_readiness is not None:
        has_relief_limit = fabrication_readiness["has_relief_limit"]
        max_relief_m = fabrication_readiness["max_relief_m"]
        raw_height_max_m = fabrication_readiness["raw_height_max_m"]
        manufacturable_height_max_m = fabrication_readiness["manufacturable_height_max_m"]
        thickness_max_m = fabrication_readiness.get("thickness_max_m")
        clipped_pixels = fabrication_readiness["clipped_pixels"]
        total_pixels = fabrication_readiness["total_pixels"]
        clipped_fraction = fabrication_readiness["clipped_fraction"]
        pixel_size_m = fabrication_readiness["pixel_size_m"]
        lines.extend(
            [
                "",
                "## Fabrication Readiness",
                "",
                f"- relief limit enabled: `{str(has_relief_limit).lower()}`",
                f"- max relief: `{('none' if max_relief_m is None else f'{max_relief_m * 1e6:.3f} um')}`",
                f"- raw height max: `{raw_height_max_m * 1e6:.3f} um`",
                f"- manufacturable height max: `{manufacturable_height_max_m * 1e6:.3f} um`",
                f"- thickness max: `{thickness_max_m * 1e6:.3f} um`" if thickness_max_m is not None else "- thickness max: `n/a`",
                f"- clipped pixels: `{clipped_pixels}` of `{total_pixels}`",
                f"- clipped fraction: `{clipped_fraction:.2%}`",
                f"- pixel size: `{pixel_size_m * 1e6:.3f} um`",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_triangle(handle, v1, v2, v3):
    handle.write("  facet normal 0 0 0\n")
    handle.write("    outer loop\n")
    handle.write(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}\n")
    handle.write(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}\n")
    handle.write(f"      vertex {v3[0]:.6f} {v3[1]:.6f} {v3[2]:.6f}\n")
    handle.write("    endloop\n")
    handle.write("  endfacet\n")


def export_height_map_to_ascii_stl(
    path: Path,
    thickness_map: np.ndarray,
    *,
    pixel_size_m: float,
    xy_unit_scale: float = 1e3,
    z_unit_scale: float = 1e3,
) -> None:
    rows, cols = thickness_map.shape
    z = thickness_map * z_unit_scale
    step = pixel_size_m * xy_unit_scale

    def top_vertex(r, c):
        return (c * step, r * step, float(z[r, c]))

    def bottom_vertex(r, c):
        return (c * step, r * step, 0.0)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"solid {path.stem}\n")

        for r in range(rows - 1):
            for c in range(cols - 1):
                v00 = top_vertex(r, c)
                v01 = top_vertex(r, c + 1)
                v10 = top_vertex(r + 1, c)
                v11 = top_vertex(r + 1, c + 1)
                _write_triangle(handle, v00, v10, v11)
                _write_triangle(handle, v00, v11, v01)

                b00 = bottom_vertex(r, c)
                b01 = bottom_vertex(r, c + 1)
                b10 = bottom_vertex(r + 1, c)
                b11 = bottom_vertex(r + 1, c + 1)
                _write_triangle(handle, b00, b11, b10)
                _write_triangle(handle, b00, b01, b11)

        for c in range(cols - 1):
            t0 = top_vertex(0, c)
            t1 = top_vertex(0, c + 1)
            b0 = bottom_vertex(0, c)
            b1 = bottom_vertex(0, c + 1)
            _write_triangle(handle, b0, t1, t0)
            _write_triangle(handle, b0, b1, t1)

            t0 = top_vertex(rows - 1, c)
            t1 = top_vertex(rows - 1, c + 1)
            b0 = bottom_vertex(rows - 1, c)
            b1 = bottom_vertex(rows - 1, c + 1)
            _write_triangle(handle, b0, t0, t1)
            _write_triangle(handle, b0, t1, b1)

        for r in range(rows - 1):
            t0 = top_vertex(r, 0)
            t1 = top_vertex(r + 1, 0)
            b0 = bottom_vertex(r, 0)
            b1 = bottom_vertex(r + 1, 0)
            _write_triangle(handle, b0, t0, t1)
            _write_triangle(handle, b0, t1, b1)

            t0 = top_vertex(r, cols - 1)
            t1 = top_vertex(r + 1, cols - 1)
            b0 = bottom_vertex(r, cols - 1)
            b1 = bottom_vertex(r + 1, cols - 1)
            _write_triangle(handle, b0, t1, t0)
            _write_triangle(handle, b0, b1, t1)

        handle.write(f"endsolid {path.stem}\n")
