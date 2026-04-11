"""
Export trained D2NN phase masks into reusable numerical phase-plate artifacts.

This is the shared bridge between:
1) paper-faithful numerical reproduction,
2) fabrication-oriented phase plate generation,
3) later research extensions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from artifacts import (
    CLASSIFIER_PAPER_OPTICS,
    IMAGER_PAPER_OPTICS,
    apply_manufacturing_profile,
    build_fabrication_readiness_summary,
    build_layer_stats,
    build_model_for_task,
    export_height_map_to_ascii_stl,
    load_checkpoint_state_dict,
    optical_config_dict,
    phase_masks_to_bmp_uint8,
    quantize_height_map,
    read_checkpoint_manifest,
    resolve_optics,
    save_layer_csvs,
    save_manifest,
    write_export_report,
)
from d2nn import phase_to_height_map


def build_parser():
    parser = argparse.ArgumentParser(description="Export phase masks / height maps from a trained D2NN checkpoint")
    parser.add_argument("--task", choices=["classification", "imaging"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="exports")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--wavelength", type=float, default=None)
    parser.add_argument("--layer-distance", type=float, default=None)
    parser.add_argument("--pixel-size", type=float, default=None)
    parser.add_argument("--input-distance", type=float, default=None)
    parser.add_argument("--output-distance", type=float, default=None)
    parser.add_argument("--input-fraction", type=float, default=0.5)
    parser.add_argument("--refractive-index", type=float, default=1.7227)
    parser.add_argument("--ambient-index", type=float, default=1.0)
    parser.add_argument("--quantization-levels", type=int, default=256)
    parser.add_argument("--base-thickness-um", type=float, default=500.0)
    parser.add_argument("--max-relief-um", type=float, default=None)
    parser.add_argument("--export-stl", action="store_true")
    parser.add_argument("--export-bmp", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location="cpu")
    checkpoint_manifest = read_checkpoint_manifest(checkpoint_path)

    base_optics = CLASSIFIER_PAPER_OPTICS if args.task == "classification" else IMAGER_PAPER_OPTICS
    optics = resolve_optics(
        base_optics,
        state_dict=state_dict,
        manifest=checkpoint_manifest,
        checkpoint_path=checkpoint_path,
        size=args.size,
        num_layers=args.layers,
        wavelength=args.wavelength,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        input_distance=args.input_distance,
        output_distance=args.output_distance,
    )

    device = torch.device("cpu")
    model = build_model_for_task(args.task, optics, input_fraction=args.input_fraction).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    phase_masks = model.export_phase_masks(wrap=True).numpy()
    raw_height_map = phase_to_height_map(
        torch.from_numpy(phase_masks),
        wavelength=optics.wavelength,
        refractive_index=args.refractive_index,
        ambient_index=args.ambient_index,
    ).numpy()
    manufacturable_relief, thickness_map = apply_manufacturing_profile(
        raw_height_map,
        base_thickness_m=args.base_thickness_um * 1e-6,
        max_relief_m=None if args.max_relief_um is None else args.max_relief_um * 1e-6,
    )

    export_root = Path(args.output_dir) / checkpoint_path.stem
    export_root.mkdir(parents=True, exist_ok=True)

    phase_path = export_root / "phase_masks.npy"
    height_path = export_root / "height_map.npy"
    manufacturable_height_path = export_root / "height_map_manufacturable.npy"
    thickness_path = export_root / "thickness_map.npy"
    quantized_path = export_root / "height_map_quantized.npy"
    np.save(phase_path, phase_masks)
    np.save(height_path, raw_height_map)
    np.save(manufacturable_height_path, manufacturable_relief)
    np.save(thickness_path, thickness_map)

    quantized_height_map = quantize_height_map(manufacturable_relief, args.quantization_levels)
    np.save(quantized_path, quantized_height_map)
    save_layer_csvs(export_root, phase_masks, manufacturable_relief, thickness_map, quantized_height_map)

    if args.export_bmp:
        bmp_dir = export_root / "bmp"
        bmp_dir.mkdir(parents=True, exist_ok=True)
        bmp_layers = phase_masks_to_bmp_uint8(phase_masks)
        for idx, layer_bmp in enumerate(bmp_layers):
            Image.fromarray(layer_bmp, mode="L").save(bmp_dir / f"layer_{idx + 1:02d}_phase_8bit.bmp")

    if args.export_stl:
        stl_dir = export_root / "stl"
        stl_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(thickness_map.shape[0]):
            export_height_map_to_ascii_stl(
                stl_dir / f"layer_{idx + 1:02d}.stl",
                thickness_map[idx],
                pixel_size_m=optics.pixel_size,
            )

    layer_stats = build_layer_stats(phase_masks, manufacturable_relief, thickness_map)
    fabrication_readiness = build_fabrication_readiness_summary(
        raw_height_map,
        manufacturable_relief,
        thickness_map,
        max_relief_m=None if args.max_relief_um is None else args.max_relief_um * 1e-6,
        pixel_size_m=optics.pixel_size,
    )
    write_export_report(
        export_root / "report.md",
        checkpoint_name=checkpoint_path.name,
        task=args.task,
        num_layers=optics.num_layers,
        size=optics.size,
        pixel_size_um=optics.pixel_size * 1e6,
        wavelength_um=optics.wavelength * 1e6,
        quantization_levels=args.quantization_levels,
        layer_stats=layer_stats,
        fabrication_readiness=fabrication_readiness,
    )

    save_manifest(
        export_root / "metadata.json",
        {
            "task": args.task,
            "checkpoint": str(checkpoint_path.resolve()),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "phase_masks_file": str(phase_path.resolve()),
            "height_map_file": str(height_path.resolve()),
            "manufacturable_height_map_file": str(manufacturable_height_path.resolve()),
            "thickness_map_file": str(thickness_path.resolve()),
            "quantized_height_map_file": str(quantized_path.resolve()),
            "optical_config": optical_config_dict(optics),
            "material": {
                "refractive_index": args.refractive_index,
                "ambient_index": args.ambient_index,
            },
            "manufacturing": {
                "base_thickness_um": args.base_thickness_um,
                "max_relief_um": args.max_relief_um,
                "export_stl": args.export_stl,
                "export_bmp": args.export_bmp,
            },
            "bmp_export": {
                "enabled": args.export_bmp,
                "mapping": "[0, 2pi) phase linearly mapped to [0, 255] grayscale",
            },
            "quantization_levels": args.quantization_levels,
            "layer_stats": layer_stats,
            "fabrication_readiness": fabrication_readiness,
            "source_checkpoint_manifest": checkpoint_manifest,
            "notes": "height_map assumes transmissive phase-only plates with wrapped phase in [0, 2pi).",
        },
    )

    print(f"Saved phase masks to: {phase_path}")
    print(f"Saved raw height map to: {height_path}")
    print(f"Saved manufacturable height map to: {manufacturable_height_path}")
    print(f"Saved thickness map to: {thickness_path}")
    print(f"Saved quantized height map to: {quantized_path}")
    print(f"Saved per-layer CSVs under: {export_root / 'layers'}")
    print(f"Saved export report to: {export_root / 'report.md'}")
    if args.export_bmp:
        print(f"Saved per-layer BMP files under: {export_root / 'bmp'}")
    if args.export_stl:
        print(f"Saved per-layer STL files under: {export_root / 'stl'}")


if __name__ == "__main__":
    main()
