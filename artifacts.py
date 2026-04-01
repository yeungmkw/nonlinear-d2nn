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

import matplotlib.pyplot as plt
import numpy as np
import torch

from d2nn import D2NN, D2NNImager


@dataclass(frozen=True)
class OpticalConfig:
    wavelength: float
    layer_distance: float
    pixel_size: float
    size: int = 200
    num_layers: int = 5

    def with_overrides(
        self,
        *,
        wavelength: float | None = None,
        layer_distance: float | None = None,
        pixel_size: float | None = None,
        size: int | None = None,
        num_layers: int | None = None,
    ) -> "OpticalConfig":
        return OpticalConfig(
            wavelength=self.wavelength if wavelength is None else wavelength,
            layer_distance=self.layer_distance if layer_distance is None else layer_distance,
            pixel_size=self.pixel_size if pixel_size is None else pixel_size,
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
        }

    def imager_model_kwargs(self, *, input_fraction: float = 0.5) -> dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "size": self.size,
            "wavelength": self.wavelength,
            "layer_distance": self.layer_distance,
            "pixel_size": self.pixel_size,
            "input_fraction": input_fraction,
        }


CLASSIFIER_PAPER_OPTICS = OpticalConfig(
    wavelength=0.75e-3,
    layer_distance=30e-3,
    pixel_size=0.4e-3,
)

IMAGER_PAPER_OPTICS = OpticalConfig(
    wavelength=0.75e-3,
    layer_distance=4e-3,
    pixel_size=0.3e-3,
)


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
    size=None,
    num_layers=None,
    wavelength=None,
    layer_distance=None,
    pixel_size=None,
):
    inferred = infer_architecture(state_dict) if state_dict is not None else {}
    return base_optics.with_overrides(
        size=inferred.get("size") if size is None else size,
        num_layers=inferred.get("num_layers") if num_layers is None else num_layers,
        wavelength=wavelength,
        layer_distance=layer_distance,
        pixel_size=pixel_size,
    )


def build_model_for_task(task, optics, *, input_fraction=0.5, num_classes=10):
    if task == "classification":
        return D2NN(**optics.classifier_model_kwargs(num_classes=num_classes))
    if task == "imaging":
        return D2NNImager(**optics.imager_model_kwargs(input_fraction=input_fraction))
    raise ValueError(f"Unsupported task: {task}")


def checkpoint_manifest_path(checkpoint_path):
    return Path(checkpoint_path).with_suffix(".json")


def checkpoint_variant_path(checkpoint_path, run_name=None):
    checkpoint_path = Path(checkpoint_path)
    if not run_name:
        return checkpoint_path

    safe_name = re.sub(r"\s+", "_", str(run_name).strip())
    safe_name = re.sub(r'[<>:"/\\|?*]+', "-", safe_name)
    safe_name = re.sub(r'[-_]{2,}', "-", safe_name)
    safe_name = safe_name.strip("._-")
    if not safe_name:
        return checkpoint_path

    return checkpoint_path.with_name(f"{checkpoint_path.stem}.{safe_name}{checkpoint_path.suffix}")


def read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_checkpoint_manifest(checkpoint_path):
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    if not manifest_path.exists():
        return None
    return read_manifest(manifest_path)


def maybe_show(no_show: bool) -> None:
    if not no_show:
        plt.show()


def plot_phase_masks(model, save_path=None, no_show=False):
    num_layers = len(model.layers)
    fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
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
    plt.tight_layout()
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


def quantize_height_map(height_map: np.ndarray, levels: int) -> np.ndarray:
    if levels < 2:
        raise ValueError("levels must be at least 2")

    max_height = float(height_map.max())
    if max_height <= 0:
        return np.zeros_like(height_map, dtype=np.uint16)

    normalized = np.clip(height_map / max_height, 0.0, 1.0)
    return np.rint(normalized * (levels - 1)).astype(np.uint16)


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
