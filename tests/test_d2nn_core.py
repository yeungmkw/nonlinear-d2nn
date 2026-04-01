import math
import tempfile
import unittest
import importlib
import subprocess
import sys
from pathlib import Path

import torch

from artifacts import (
    CLASSIFIER_PAPER_OPTICS,
    IMAGER_PAPER_OPTICS,
    apply_manufacturing_profile,
    checkpoint_manifest_path,
    checkpoint_variant_path,
    export_height_map_to_ascii_stl,
    infer_architecture,
    read_manifest,
    resolve_optics,
    save_manifest,
    quantize_height_map,
    build_layer_stats,
)
from d2nn import D2NN, D2NNImager, phase_to_height_map


class D2NNCoreTests(unittest.TestCase):
    def test_readme_has_deliverable_sections(self):
        readme = Path(__file__).resolve().parents[1] / "README.md"
        content = readme.read_text(encoding="utf-8")
        for section in ("## 推荐复现路径", "## 推荐 checkpoint", "## 论文对照结果"):
            self.assertIn(section, content)

    def test_cli_entrypoints_expose_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        cases = [
            ("train.py", "D2NN training"),
            ("visualize.py", "D2NN visualization"),
            ("export_phase_plate.py", "Export phase masks / height maps"),
        ]
        for script_name, expected in cases:
            result = subprocess.run(
                [sys.executable, str(repo_root / script_name), "--help"],
                capture_output=True,
                text=True,
                check=False,
                cwd=repo_root,
            )
            self.assertEqual(result.returncode, 0, msg=f"{script_name} exited with {result.returncode}")
            self.assertIn(expected, result.stdout)

    def test_artifacts_module_reexports_shared_helpers(self):
        artifacts = importlib.import_module("artifacts")
        for name in (
            "CLASSIFIER_PAPER_OPTICS",
            "apply_manufacturing_profile",
            "checkpoint_manifest_path",
            "export_height_map_to_ascii_stl",
            "plot_phase_masks",
            "resolve_optics",
        ):
            self.assertTrue(hasattr(artifacts, name), f"artifacts missing {name}")

    def test_tasks_module_exposes_both_task_families(self):
        tasks = importlib.import_module("tasks")
        for name in (
            "run_classification_training",
            "run_classification_visualization",
            "run_imaging_training",
            "run_imaging_visualization",
        ):
            self.assertTrue(hasattr(tasks, name), f"tasks missing {name}")

    def test_classifier_detector_count_matches_classes(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=32, num_layers=2).classifier_model_kwargs())
        self.assertEqual(model.detector_masks.shape, (10, 32, 32))
        self.assertTrue(torch.all(model.detector_masks.sum(dim=(-2, -1)) > 0))

    def test_phase_export_is_wrapped(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=3).classifier_model_kwargs())
        phase_masks = model.export_phase_masks(wrap=True)
        self.assertEqual(phase_masks.shape, (3, 16, 16))
        self.assertTrue(torch.all(phase_masks >= 0))
        self.assertTrue(torch.all(phase_masks < 2 * math.pi + 1e-6))

    def test_imager_target_is_normalized(self):
        model = D2NNImager(**IMAGER_PAPER_OPTICS.with_overrides(size=20, num_layers=2).imager_model_kwargs())
        x = torch.rand(2, 1, 8, 8)
        target = model.build_target(x)
        self.assertEqual(target.shape, (2, 20, 20))
        self.assertTrue(torch.all(target >= 0))
        self.assertTrue(torch.allclose(target.amax(dim=(-2, -1)), torch.ones(2), atol=1e-5))

    def test_phase_to_height_map_requires_positive_delta_n(self):
        phases = torch.ones(2, 4, 4)
        with self.assertRaises(ValueError):
            phase_to_height_map(phases, wavelength=0.75e-3, refractive_index=1.0, ambient_index=1.0)

    def test_manifest_writer_outputs_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            save_manifest(path, {"hello": "world"})
            self.assertTrue(path.exists())
            self.assertIn('"hello": "world"', path.read_text(encoding="utf-8"))
            self.assertEqual(read_manifest(path), {"hello": "world"})

    def test_infer_architecture_from_checkpoint_state_dict(self):
        state_dict = {
            "layers.0.phase": torch.zeros(12, 12),
            "layers.1.phase": torch.zeros(12, 12),
            "layers.2.phase": torch.zeros(12, 12),
        }
        inferred = infer_architecture(state_dict)
        self.assertEqual(inferred, {"num_layers": 3, "size": 12})

    def test_checkpoint_manifest_path_swaps_suffix(self):
        path = checkpoint_manifest_path("checkpoints/best_mnist.pth")
        self.assertEqual(Path("checkpoints/best_mnist.json"), path)

    def test_checkpoint_variant_path_keeps_default_name_without_run_name(self):
        path = checkpoint_variant_path("checkpoints/best_mnist.pth", None)
        self.assertEqual(Path("checkpoints/best_mnist.pth"), path)

    def test_checkpoint_variant_path_appends_run_name_before_suffix(self):
        path = checkpoint_variant_path("checkpoints/best_mnist.pth", "baseline_5layer")
        self.assertEqual(Path("checkpoints/best_mnist.baseline_5layer.pth"), path)

    def test_checkpoint_variant_path_sanitizes_windows_unsafe_characters(self):
        path = checkpoint_variant_path("checkpoints/best_mnist.pth", "baseline: 5/layer?")
        self.assertEqual(Path("checkpoints/best_mnist.baseline-5-layer.pth"), path)

    def test_checkpoint_variant_path_ignores_blank_run_name_after_sanitizing(self):
        path = checkpoint_variant_path("checkpoints/best_mnist.pth", "   ")
        self.assertEqual(Path("checkpoints/best_mnist.pth"), path)

    def test_resolve_optics_uses_checkpoint_architecture_when_missing(self):
        state_dict = {
            "layers.0.phase": torch.zeros(24, 24),
            "layers.1.phase": torch.zeros(24, 24),
        }
        optics = resolve_optics(CLASSIFIER_PAPER_OPTICS, state_dict=state_dict)
        self.assertEqual(optics.size, 24)
        self.assertEqual(optics.num_layers, 2)

    def test_quantize_height_map_produces_expected_range(self):
        height_map = torch.tensor([[[0.0, 0.5], [1.0, 0.25]]]).numpy()
        quantized = quantize_height_map(height_map, levels=8)
        self.assertEqual(quantized.dtype, "uint16")
        self.assertEqual(int(quantized.min()), 0)
        self.assertEqual(int(quantized.max()), 7)

    def test_build_layer_stats_matches_layer_count(self):
        phase_masks = torch.zeros(3, 4, 4).numpy()
        height_map = torch.ones(3, 4, 4).numpy() * 2e-6
        stats = build_layer_stats(phase_masks, height_map, height_map + 1e-6)
        self.assertEqual(len(stats), 3)
        self.assertEqual(stats[0]["layer"], 1)
        self.assertAlmostEqual(stats[0]["height_mean_m"], 2e-6)

    def test_apply_manufacturing_profile_adds_base_and_clips_relief(self):
        height_map = torch.tensor([[[0.0, 3e-6], [7e-6, 10e-6]]]).numpy()
        relief, thickness = apply_manufacturing_profile(
            height_map,
            base_thickness_m=5e-6,
            max_relief_m=6e-6,
        )
        self.assertAlmostEqual(float(relief.max()), 6e-6)
        self.assertAlmostEqual(float(thickness.min()), 5e-6)
        self.assertAlmostEqual(float(thickness.max()), 11e-6)

    def test_export_height_map_to_ascii_stl_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "layer_01.stl"
            export_height_map_to_ascii_stl(
                path,
                thickness_map=torch.tensor([[1e-6, 2e-6], [3e-6, 4e-6]]).numpy(),
                pixel_size_m=1e-6,
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("solid layer_01", content)
            self.assertIn("facet normal", content)


if __name__ == "__main__":
    unittest.main()
