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
    build_layer_stats,
    checkpoint_manifest_path,
    derive_experiment_run_name,
    checkpoint_variant_path,
    experiment_manifest_fields,
    export_height_map_to_ascii_stl,
    infer_architecture,
    quantize_height_map,
    read_manifest,
    resolve_optics,
    save_manifest,
)
from d2nn import (
    CoherentAmplitudeActivation,
    CoherentPhaseActivation,
    D2NN,
    D2NNImager,
    IdentityActivation,
    IncoherentIntensityActivation,
    phase_to_height_map,
    safe_abs,
)
from train import build_parser
from tasks import (
    build_experiment_grid,
    d2nn_mse_loss,
    format_experiment_grid_commands,
    resolve_activation_config,
    resolve_experiment_seed,
)
from visualize import build_parser as build_visualize_parser


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

    def test_classifier_propagation_helpers_match_forward(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).classifier_model_kwargs())
        x = torch.rand(2, 1, 28, 28)
        field = model.propagate_field(x)
        intensity = model.output_intensity_from_field(field)
        detected = model.detect_from_intensity(intensity)
        self.assertEqual(field.shape, (2, 24, 24))
        self.assertEqual(intensity.shape, (2, 24, 24))
        self.assertTrue(torch.allclose(detected, model(x), atol=1e-5, rtol=1e-4))

    def test_imager_target_is_normalized(self):
        model = D2NNImager(**IMAGER_PAPER_OPTICS.with_overrides(size=20, num_layers=2).imager_model_kwargs())
        x = torch.rand(2, 1, 8, 8)
        target = model.build_target(x)
        self.assertEqual(target.shape, (2, 20, 20))
        self.assertTrue(torch.all(target >= 0))
        self.assertTrue(torch.allclose(target.amax(dim=(-2, -1)), torch.ones(2), atol=1e-5))

    def test_imager_propagation_helpers_match_forward(self):
        model = D2NNImager(**IMAGER_PAPER_OPTICS.with_overrides(size=20, num_layers=2).imager_model_kwargs())
        x = torch.rand(2, 1, 8, 8)
        field = model.propagate_field(x)
        intensity = model.output_intensity_from_field(field)
        expected = intensity / intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        self.assertEqual(field.shape, (2, 20, 20))
        self.assertEqual(intensity.shape, (2, 20, 20))
        self.assertTrue(torch.allclose(expected, model(x), atol=1e-5, rtol=1e-4))

    def test_safe_abs_avoids_zero_gradient_singularity(self):
        x = torch.zeros(2, 3, dtype=torch.cfloat)
        magnitude = safe_abs(x)
        self.assertEqual(magnitude.shape, (2, 3))
        self.assertTrue(torch.all(magnitude > 0))

    def test_identity_activation_preserves_complex_field(self):
        activation = IdentityActivation()
        u = torch.randn(2, 8, 8, dtype=torch.cfloat)
        self.assertTrue(torch.equal(activation(u), u))

    def test_coherent_amplitude_activation_preserves_phase_and_limits_gain(self):
        activation = CoherentAmplitudeActivation(threshold=0.2, temperature=0.05, gain_min=0.1, gain_max=0.9)
        u = torch.randn(2, 8, 8, dtype=torch.cfloat)
        out = activation(u)

        input_mag = safe_abs(u)
        output_mag = safe_abs(out)
        nonzero = input_mag > 1e-5
        phase_delta = torch.angle(out[nonzero] / u[nonzero])

        self.assertTrue(torch.all(output_mag <= input_mag * 0.90001))
        self.assertTrue(torch.all(output_mag >= input_mag * 0.09999))
        self.assertTrue(torch.allclose(phase_delta, torch.zeros_like(phase_delta), atol=1e-5, rtol=1e-5))

    def test_coherent_amplitude_activation_maps_low_and_high_intensity_to_expected_gain_band(self):
        activation = CoherentAmplitudeActivation(threshold=0.5, temperature=0.1, gain_min=0.2, gain_max=0.8)
        low = torch.full((1, 2, 2), 0.1 + 0j, dtype=torch.cfloat)
        high = torch.full((1, 2, 2), 2.0 + 0j, dtype=torch.cfloat)

        low_gain = safe_abs(activation(low)) / safe_abs(low)
        high_gain = safe_abs(activation(high)) / safe_abs(high)

        self.assertTrue(torch.all(low_gain < high_gain))
        self.assertTrue(torch.all(low_gain > 0.19))
        self.assertTrue(torch.all(high_gain < 0.81))

    def test_coherent_amplitude_activation_records_last_stats(self):
        activation = CoherentAmplitudeActivation(threshold=0.2, temperature=0.1, gain_min=0.1, gain_max=0.9)
        u = torch.full((1, 2, 2), 1.0 + 0j, dtype=torch.cfloat)
        _ = activation(u)

        self.assertIn("mean_intensity", activation.last_stats)
        self.assertIn("mean_gain", activation.last_stats)
        self.assertGreater(activation.last_stats["mean_intensity"], 0.0)
        self.assertGreaterEqual(activation.last_stats["mean_gain"], 0.1)
        self.assertLessEqual(activation.last_stats["mean_gain"], 0.9)

    def test_coherent_phase_activation_preserves_magnitude_and_applies_intensity_phase_shift(self):
        activation = CoherentPhaseActivation(gamma=0.5)
        u = torch.tensor([[[1.0 + 0j, 1.0j]]], dtype=torch.cfloat)

        out = activation(u)
        expected_shift = 0.5 * (safe_abs(u) ** 2)
        phase_delta = torch.angle(out / u)

        self.assertTrue(torch.allclose(safe_abs(out), safe_abs(u), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(phase_delta, expected_shift, atol=1e-5, rtol=1e-5))

    def test_coherent_phase_activation_records_phase_stats(self):
        activation = CoherentPhaseActivation(gamma=0.25)
        u = torch.full((1, 2, 2), 2.0 + 0j, dtype=torch.cfloat)
        _ = activation(u)

        self.assertIn("mean_intensity", activation.last_stats)
        self.assertIn("mean_phase_shift", activation.last_stats)
        self.assertGreater(activation.last_stats["mean_phase_shift"], 0.0)

    def test_incoherent_intensity_activation_discards_input_phase_for_equal_intensity(self):
        activation = IncoherentIntensityActivation(responsivity=1.0, threshold=0.1, emission_phase_mode="zero")
        u = torch.tensor([[[1.0 + 0j, 1.0j]]], dtype=torch.cfloat)

        out = activation(u)

        self.assertTrue(torch.allclose(safe_abs(out[..., 0]), safe_abs(out[..., 1]), atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(torch.angle(out), torch.zeros_like(out.real), atol=1e-6, rtol=1e-6))

    def test_incoherent_intensity_activation_records_output_amplitude_stats(self):
        activation = IncoherentIntensityActivation(responsivity=0.5, threshold=0.1, emission_phase_mode="zero")
        u = torch.full((1, 2, 2), 2.0 + 0j, dtype=torch.cfloat)
        _ = activation(u)

        self.assertIn("mean_intensity", activation.last_stats)
        self.assertIn("mean_output_amplitude", activation.last_stats)
        self.assertGreater(activation.last_stats["mean_output_amplitude"], 0.0)
        self.assertEqual(activation.last_stats["emission_phase_mode"], "zero")

    def test_classifier_identity_activation_matches_baseline_forward(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=3)
        baseline = D2NN(**optics.classifier_model_kwargs())
        nonlinear = D2NN(
            **optics.classifier_model_kwargs(),
            activation_type="identity",
            activation_positions=(1, 3),
        )
        nonlinear.load_state_dict(baseline.state_dict(), strict=True)

        x = torch.rand(2, 1, 28, 28)
        self.assertTrue(torch.allclose(nonlinear(x), baseline(x), atol=1e-6, rtol=1e-5))

    def test_baseline_checkpoint_loads_into_default_activation_model(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=3)
        original = D2NN(**optics.classifier_model_kwargs())
        restored = D2NN(**optics.classifier_model_kwargs())
        restored.load_state_dict(original.state_dict(), strict=True)

        x = torch.rand(2, 1, 28, 28)
        self.assertEqual(restored(x).shape, (2, 10))
        self.assertEqual(restored(x).dtype, torch.float32)
        self.assertTrue(torch.allclose(restored(x), original(x), atol=1e-6, rtol=1e-5))

    def test_identity_activation_supports_backward_pass(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2)
        model = D2NN(
            **optics.classifier_model_kwargs(),
            activation_type="identity",
            activation_positions=(1,),
        )
        x = torch.rand(4, 1, 28, 28)
        target = torch.tensor([0, 1, 2, 3])
        loss = d2nn_mse_loss(model(x), target, num_classes=10)
        loss.backward()
        self.assertTrue(all(layer.phase.grad is not None for layer in model.layers))

    def test_classifier_coherent_amplitude_activation_supports_backward_pass(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2)
        model = D2NN(
            **optics.classifier_model_kwargs(),
            activation_type="coherent_amplitude",
            activation_positions=(1,),
            activation_hparams={"threshold": 0.2, "temperature": 0.1, "gain_min": 0.1, "gain_max": 0.9},
        )
        x = torch.rand(4, 1, 28, 28)
        target = torch.tensor([0, 1, 2, 3])
        loss = d2nn_mse_loss(model(x), target, num_classes=10)
        loss.backward()
        self.assertTrue(all(layer.phase.grad is not None for layer in model.layers))

    def test_classifier_coherent_phase_activation_supports_backward_pass(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2)
        model = D2NN(
            **optics.classifier_model_kwargs(),
            activation_type="coherent_phase",
            activation_positions=(1,),
            activation_hparams={"gamma": 0.25},
        )
        x = torch.rand(4, 1, 28, 28)
        target = torch.tensor([0, 1, 2, 3])
        loss = d2nn_mse_loss(model(x), target, num_classes=10)
        loss.backward()
        self.assertTrue(all(layer.phase.grad is not None for layer in model.layers))

    def test_classifier_incoherent_intensity_activation_supports_backward_pass(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2)
        model = D2NN(
            **optics.classifier_model_kwargs(),
            activation_type="incoherent_intensity",
            activation_positions=(1,),
            activation_hparams={"responsivity": 1.0, "threshold": 0.1, "emission_phase_mode": "zero"},
        )
        x = torch.rand(4, 1, 28, 28)
        target = torch.tensor([0, 1, 2, 3])
        loss = d2nn_mse_loss(model(x), target, num_classes=10)
        loss.backward()
        self.assertTrue(all(layer.phase.grad is not None for layer in model.layers))

    def test_model_activation_diagnostics_reports_configured_positions(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=3)
        model = D2NN(
            **optics.classifier_model_kwargs(),
            activation_type="coherent_amplitude",
            activation_positions=(1, 3),
            activation_hparams={"threshold": 0.2, "temperature": 0.1, "gain_min": 0.1, "gain_max": 0.9},
        )
        x = torch.rand(2, 1, 28, 28)
        _ = model(x)

        diagnostics = model.activation_diagnostics()
        self.assertEqual(set(diagnostics.keys()), {"1", "3"})
        self.assertIn("mean_gain", diagnostics["1"])
        self.assertIn("mean_intensity", diagnostics["3"])

    def test_imager_identity_activation_keeps_shape_and_dtype(self):
        optics = IMAGER_PAPER_OPTICS.with_overrides(size=20, num_layers=3)
        baseline = D2NNImager(**optics.imager_model_kwargs())
        nonlinear = D2NNImager(
            **optics.imager_model_kwargs(),
            activation_type="identity",
            activation_positions=(2,),
        )
        nonlinear.load_state_dict(baseline.state_dict(), strict=True)

        x = torch.rand(2, 1, 8, 8)
        baseline_out = baseline(x)
        nonlinear_out = nonlinear(x)
        self.assertEqual(nonlinear_out.shape, baseline_out.shape)
        self.assertEqual(nonlinear_out.dtype, baseline_out.dtype)
        self.assertTrue(torch.allclose(nonlinear_out, baseline_out, atol=1e-6, rtol=1e-5))

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

    def test_derive_experiment_run_name_keeps_explicit_run_name(self):
        run_name = derive_experiment_run_name(
            run_name="manual_name",
            experiment_stage="mechanism_ablation",
            activation_type="coherent_amplitude",
            activation_positions=(1, 3),
            activation_hparams={"threshold": 0.2},
            seed=42,
        )
        self.assertEqual(run_name, "manual_name")

    def test_derive_experiment_run_name_returns_none_for_phase_only_default(self):
        run_name = derive_experiment_run_name(
            run_name=None,
            experiment_stage="baseline",
            activation_type="none",
            activation_positions=(),
            activation_hparams={},
            seed=42,
        )
        self.assertIsNone(run_name)

    def test_derive_experiment_run_name_encodes_nonlinear_identity(self):
        run_name = derive_experiment_run_name(
            run_name=None,
            experiment_stage="mechanism_ablation",
            activation_type="coherent_amplitude",
            activation_positions=(1, 3),
            activation_hparams={"threshold": 0.2, "temperature": 0.1, "gain_min": 0.1, "gain_max": 0.9},
            seed=42,
        )
        self.assertIn("mechanism-ablation", run_name)
        self.assertIn("act-coherent-amplitude", run_name)
        self.assertIn("pos-1-3", run_name)
        self.assertIn("seed-42", run_name)
        self.assertIn("threshold-0p2", run_name)

    def test_checkpoint_variant_path_sanitizes_windows_unsafe_characters(self):
        path = checkpoint_variant_path("checkpoints/best_mnist.pth", "baseline: 5/layer?")
        self.assertEqual(Path("checkpoints/best_mnist.baseline-5-layer.pth"), path)

    def test_checkpoint_variant_path_ignores_blank_run_name_after_sanitizing(self):
        path = checkpoint_variant_path("checkpoints/best_mnist.pth", "   ")
        self.assertEqual(Path("checkpoints/best_mnist.pth"), path)

    def test_experiment_manifest_fields_include_identity_metadata(self):
        payload = experiment_manifest_fields(
            checkpoint_path=Path("checkpoints/best_fashion_mnist.pth"),
            run_name="baseline_5layer",
            experiment_stage="baseline",
            seed=42,
            optics=CLASSIFIER_PAPER_OPTICS.with_overrides(size=200, num_layers=5),
            activation_type="none",
            activation_positions=(),
            activation_hparams={},
        )
        self.assertEqual(payload["checkpoint"], str(Path("checkpoints/best_fashion_mnist.pth")))
        self.assertEqual(payload["run_name"], "baseline_5layer")
        self.assertEqual(payload["experiment_stage"], "baseline")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["optical_config"]["num_layers"], 5)
        self.assertEqual(payload["activation_type"], "none")
        self.assertEqual(payload["activation_positions"], [])
        self.assertEqual(payload["activation_hparams"], {})

    def test_train_parser_accepts_seed_and_experiment_stage(self):
        args = build_parser().parse_args(["--seed", "7", "--experiment-stage", "placement_ablation"])
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.experiment_stage, "placement_ablation")

    def test_train_parser_accepts_activation_arguments(self):
        args = build_parser().parse_args(
            [
                "--print-experiment-grid",
                "coherent_amplitude_positions",
                "--activation-type",
                "incoherent_intensity",
                "--activation-preset",
                "balanced",
                "--activation-placement",
                "mid",
                "--activation-positions",
                "1,3",
                "--activation-responsivity",
                "1.2",
                "--activation-emission-phase-mode",
                "zero",
            ]
        )
        self.assertEqual(args.print_experiment_grid, "coherent_amplitude_positions")
        self.assertEqual(args.activation_type, "incoherent_intensity")
        self.assertEqual(args.activation_preset, "balanced")
        self.assertEqual(args.activation_placement, "mid")
        self.assertEqual(args.activation_positions, "1,3")
        self.assertAlmostEqual(args.activation_responsivity, 1.2)
        self.assertEqual(args.activation_emission_phase_mode, "zero")

    def test_visualize_parser_accepts_seed(self):
        args = build_visualize_parser().parse_args(["--checkpoint", "checkpoints/demo.pth", "--seed", "11"])
        self.assertEqual(args.seed, 11)

    def test_resolve_experiment_seed_prefers_explicit_seed_then_manifest_then_default(self):
        self.assertEqual(resolve_experiment_seed(11, {"seed": 7}), 11)
        self.assertEqual(resolve_experiment_seed(None, {"seed": 7}), 7)
        self.assertEqual(resolve_experiment_seed(None, None), 42)

    def test_resolve_activation_config_prefers_args_then_manifest_then_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--activation-type", "identity", "--activation-positions", "1,3"])
        activation_type, positions, hparams = resolve_activation_config(
            args,
            {
                "activation_type": "none",
                "activation_positions": [2],
                "activation_hparams": {"threshold": 0.2},
            },
        )
        self.assertEqual(activation_type, "identity")
        self.assertEqual(positions, (1, 3))
        self.assertEqual(hparams, {"threshold": 0.2})

        activation_type, positions, hparams = resolve_activation_config(
            None,
            {
                "activation_type": "identity",
                "activation_positions": [2],
                "activation_hparams": {"threshold": 0.2},
            },
        )
        self.assertEqual(activation_type, "identity")
        self.assertEqual(positions, (2,))
        self.assertEqual(hparams, {"threshold": 0.2})

    def test_resolve_activation_config_applies_coherent_amplitude_preset_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--activation-type",
                "coherent_amplitude",
                "--activation-preset",
                "balanced",
                "--activation-positions",
                "2",
            ]
        )
        activation_type, positions, hparams = resolve_activation_config(args, None)
        self.assertEqual(activation_type, "coherent_amplitude")
        self.assertEqual(positions, (2,))
        self.assertEqual(
            hparams,
            {
                "threshold": 0.2,
                "temperature": 0.1,
                "gain_min": 0.25,
                "gain_max": 0.95,
            },
        )

    def test_resolve_activation_config_lets_explicit_args_override_preset(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--activation-type",
                "coherent_amplitude",
                "--activation-preset",
                "balanced",
                "--activation-threshold",
                "0.3",
            ]
        )
        _, _, hparams = resolve_activation_config(args, None)
        self.assertEqual(hparams["threshold"], 0.3)
        self.assertEqual(hparams["temperature"], 0.1)
        self.assertEqual(hparams["gain_min"], 0.25)
        self.assertEqual(hparams["gain_max"], 0.95)

    def test_resolve_activation_config_applies_coherent_phase_preset_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--activation-type",
                "coherent_phase",
                "--activation-preset",
                "balanced",
                "--activation-positions",
                "2",
            ]
        )
        activation_type, positions, hparams = resolve_activation_config(args, None)
        self.assertEqual(activation_type, "coherent_phase")
        self.assertEqual(positions, (2,))
        self.assertEqual(hparams, {"gamma": 0.25})

    def test_resolve_activation_config_lets_explicit_gamma_override_phase_preset(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--activation-type",
                "coherent_phase",
                "--activation-preset",
                "balanced",
                "--activation-gamma",
                "0.4",
            ]
        )
        _, _, hparams = resolve_activation_config(args, None)
        self.assertEqual(hparams, {"gamma": 0.4})

    def test_resolve_activation_config_applies_incoherent_intensity_preset_defaults(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--activation-type",
                "incoherent_intensity",
                "--activation-preset",
                "balanced",
                "--activation-positions",
                "2",
            ]
        )
        activation_type, positions, hparams = resolve_activation_config(args, None)
        self.assertEqual(activation_type, "incoherent_intensity")
        self.assertEqual(positions, (2,))
        self.assertEqual(hparams, {"responsivity": 1.0, "threshold": 0.1, "emission_phase_mode": "zero"})

    def test_resolve_activation_config_lets_explicit_incoherent_args_override_preset(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--activation-type",
                "incoherent_intensity",
                "--activation-preset",
                "balanced",
                "--activation-threshold",
                "0.2",
                "--activation-responsivity",
                "1.4",
            ]
        )
        _, _, hparams = resolve_activation_config(args, None)
        self.assertEqual(hparams["threshold"], 0.2)
        self.assertEqual(hparams["responsivity"], 1.4)
        self.assertEqual(hparams["emission_phase_mode"], "zero")

    def test_resolve_activation_config_maps_mid_placement_alias(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--layers",
                "5",
                "--activation-type",
                "coherent_amplitude",
                "--activation-placement",
                "mid",
            ]
        )
        activation_type, positions, _ = resolve_activation_config(args, None)
        self.assertEqual(activation_type, "coherent_amplitude")
        self.assertEqual(positions, (3,))

    def test_resolve_activation_config_explicit_positions_override_placement_alias(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--layers",
                "5",
                "--activation-type",
                "coherent_amplitude",
                "--activation-placement",
                "mid",
                "--activation-positions",
                "2,4",
            ]
        )
        _, positions, _ = resolve_activation_config(args, None)
        self.assertEqual(positions, (2, 4))

    def test_build_experiment_grid_returns_coherent_amplitude_position_sweep(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
                "--seed",
                "7",
            ]
        )
        grid = build_experiment_grid("coherent_amplitude_positions", args)
        self.assertEqual(len(grid), 4)
        self.assertEqual([item["activation_placement"] for item in grid], ["front", "mid", "back", "all"])
        self.assertTrue(all(item["activation_type"] == "coherent_amplitude" for item in grid))
        self.assertTrue(all(item["activation_preset"] == "balanced" for item in grid))
        self.assertTrue(all(item["experiment_stage"] == "placement_ablation" for item in grid))
        self.assertTrue(all(item["seed"] == 7 for item in grid))

    def test_build_experiment_grid_returns_coherent_phase_preset_sweep(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        grid = build_experiment_grid("coherent_phase_presets", args)
        self.assertEqual(len(grid), 3)
        self.assertTrue(all(item["activation_type"] == "coherent_phase" for item in grid))
        self.assertEqual([item["activation_preset"] for item in grid], ["conservative", "balanced", "aggressive"])
        self.assertTrue(all(item["activation_placement"] == "mid" for item in grid))
        self.assertTrue(all(item["experiment_stage"] == "mechanism_tuning" for item in grid))

    def test_build_experiment_grid_returns_mechanism_ablation_commands(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        grid = build_experiment_grid("coherent_activation_mechanisms", args)
        self.assertEqual(len(grid), 2)
        self.assertEqual([item["activation_type"] for item in grid], ["coherent_amplitude", "coherent_phase"])
        self.assertTrue(all(item["activation_preset"] == "balanced" for item in grid))
        self.assertTrue(all(item["activation_placement"] == "mid" for item in grid))
        self.assertTrue(all(item["experiment_stage"] == "mechanism_ablation" for item in grid))

    def test_build_experiment_grid_returns_incoherent_preset_sweep(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        grid = build_experiment_grid("incoherent_intensity_presets", args)
        self.assertEqual(len(grid), 3)
        self.assertTrue(all(item["activation_type"] == "incoherent_intensity" for item in grid))
        self.assertEqual([item["activation_preset"] for item in grid], ["conservative", "balanced", "aggressive"])
        self.assertTrue(all(item["activation_placement"] == "mid" for item in grid))
        self.assertTrue(all(item["experiment_stage"] == "mechanism_tuning" for item in grid))

    def test_build_experiment_grid_returns_all_activation_mechanisms(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        grid = build_experiment_grid("activation_mechanisms", args)
        self.assertEqual(len(grid), 3)
        self.assertEqual(
            [item["activation_type"] for item in grid],
            ["coherent_amplitude", "coherent_phase", "incoherent_intensity"],
        )
        self.assertTrue(all(item["activation_preset"] == "balanced" for item in grid))
        self.assertTrue(all(item["activation_placement"] == "mid" for item in grid))
        self.assertTrue(all(item["experiment_stage"] == "mechanism_ablation" for item in grid))

    def test_format_experiment_grid_commands_renders_train_commands(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        commands = format_experiment_grid_commands("coherent_amplitude_positions", args)
        self.assertEqual(len(commands), 4)
        self.assertIn("python train.py", commands[0])
        self.assertIn("--activation-placement front", commands[0])
        self.assertIn("--activation-preset balanced", commands[0])

    def test_format_experiment_grid_commands_renders_phase_and_mechanism_variants(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        phase_commands = format_experiment_grid_commands("coherent_phase_presets", args)
        mechanism_commands = format_experiment_grid_commands("coherent_activation_mechanisms", args)
        self.assertIn("--activation-type coherent_phase", phase_commands[0])
        self.assertIn("--activation-preset conservative", phase_commands[0])
        self.assertIn("--experiment-stage mechanism_ablation", mechanism_commands[0])
        self.assertIn("--activation-type coherent_amplitude", mechanism_commands[0])
        self.assertIn("--activation-type coherent_phase", mechanism_commands[1])

    def test_format_experiment_grid_commands_renders_incoherent_variants(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--task",
                "classification",
                "--dataset",
                "fashion-mnist",
                "--layers",
                "5",
            ]
        )
        preset_commands = format_experiment_grid_commands("incoherent_intensity_presets", args)
        mechanism_commands = format_experiment_grid_commands("activation_mechanisms", args)
        self.assertIn("--activation-type incoherent_intensity", preset_commands[0])
        self.assertIn("--activation-preset conservative", preset_commands[0])
        self.assertIn("--activation-type incoherent_intensity", mechanism_commands[2])

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
