import math
import tempfile
import unittest
import importlib
import subprocess
import sys
import json
from datetime import date
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
    checkpoint_manifest_path,
    configure_matplotlib_backend,
    derive_experiment_run_name,
    ensure_checkpoint_version,
    checkpoint_variant_path,
    experiment_manifest_fields,
    export_height_map_to_ascii_stl,
    infer_architecture,
    quantize_height_map,
    read_manifest,
    resolve_optics,
    save_manifest,
    write_export_report,
    plot_phase_masks,
)
from d2nn import (
    CoherentAmplitudeActivation,
    CoherentPhaseActivation,
    DetectorLayer,
    D2NN,
    D2NNImager,
    DiffractiveNetwork,
    IdentityActivation,
    IncoherentIntensityActivation,
    RayleighSommerfeldPropagation,
    detector_contrast,
    field_intensity,
    normalized_detector_logits,
    phase_to_height_map,
    safe_abs,
)
from train import build_parser
from tasks import (
    append_metric_history,
    build_classification_transform,
    build_experiment_grid,
    build_metric_history,
    classification_split_lengths,
    classification_composite_loss,
    d2nn_mse_loss,
    execute_experiment_grid,
    format_experiment_grid_commands,
    get_classification_dataset_config,
    is_better_classification_checkpoint,
    plot_classification_history,
    plot_quantization_sensitivity,
    plot_sample_output_patterns,
    resolve_activation_config,
    resolve_experiment_seed,
)
from visualize import build_parser as build_visualize_parser


class D2NNCoreTests(unittest.TestCase):
    def test_readme_has_public_repo_sections(self):
        readme = Path(__file__).resolve().parents[1] / "README.md"
        content = readme.read_text(encoding="utf-8")
        for section in (
            "## Features",
            "## Quick Start",
            "## Project Structure",
            "## Known Limitations & Scope",
            "## References",
        ):
            self.assertIn(section, content)

    def test_cli_entrypoints_expose_help(self):
        repo_root = Path(__file__).resolve().parents[1]
        cases = [
            ("train.py", "D2NN training"),
            ("visualize.py", "D2NN visualization"),
            ("export_phase_plate.py", "Export phase masks / height maps"),
            ("export_fmnist5_phaseonly_aligned_final.py", "Final export wrapper"),
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

    def test_train_module_exposes_classification_training_core(self):
        train_module = importlib.import_module("train")
        for name in (
            "classification_composite_loss",
            "train_classification_one_epoch",
            "evaluate_classification",
            "run_classification_training",
        ):
            self.assertTrue(hasattr(train_module, name), f"train missing {name}")

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

    def test_get_classification_dataset_config_supports_cifar10_gray_alias(self):
        cfg = get_classification_dataset_config("cifar10-gray")
        self.assertEqual(cfg["display_name"], "CIFAR-10 (grayscale)")
        self.assertEqual(cfg["checkpoint_name"], "best_cifar10_gray.pth")

    def test_get_classification_dataset_config_supports_cifar10_rgb_alias(self):
        cfg = get_classification_dataset_config("cifar10-rgb")
        self.assertEqual(cfg["display_name"], "CIFAR-10 (RGB)")
        self.assertEqual(cfg["checkpoint_name"], "best_cifar10_rgb.pth")

    def test_build_classification_transform_converts_cifar10_gray_to_single_channel(self):
        transform = build_classification_transform(get_classification_dataset_config("cifar10_gray"))
        image = Image.new("RGB", (32, 32), color=(255, 0, 0))
        tensor = transform(image)
        self.assertEqual(tensor.shape, (1, 32, 32))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_build_classification_transform_preserves_cifar10_rgb_channels(self):
        transform = build_classification_transform(get_classification_dataset_config("cifar10_rgb"))
        image = Image.new("RGB", (32, 32), color=(255, 128, 0))
        tensor = transform(image)
        self.assertEqual(tensor.shape, (3, 32, 32))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_classification_split_lengths_are_dataset_aware(self):
        self.assertEqual(classification_split_lengths(60000), (55000, 5000))
        self.assertEqual(classification_split_lengths(50000), (45000, 5000))

    def test_classifier_detector_count_matches_classes(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=32, num_layers=2).classifier_model_kwargs())
        self.assertEqual(model.detector_masks.shape, (10, 32, 32))
        self.assertTrue(torch.all(model.detector_masks.sum(dim=(-2, -1)) > 0))

    def test_diffractive_layer_uses_rs_propagation_modules(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).classifier_model_kwargs())
        self.assertIsInstance(model.network, DiffractiveNetwork)
        self.assertIsInstance(model.layers[0].propagation, RayleighSommerfeldPropagation)
        self.assertIsInstance(model.network.output_propagation, RayleighSommerfeldPropagation)
        self.assertFalse(hasattr(model.layers[0], "H"))
        self.assertFalse(hasattr(model, "H_out"))
        self.assertIsInstance(model.detector_layer, DetectorLayer)

    def test_classifier_even_detector_size_is_respected(self):
        detector_size = 4
        masks = D2NN._build_detectors(size=32, num_classes=1, detector_size=detector_size)
        active = torch.where(masks[0] > 0)
        width = int(active[0].max() - active[0].min() + 1)
        height = int(active[1].max() - active[1].min() + 1)
        self.assertEqual(width, detector_size)
        self.assertEqual(height, detector_size)

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

    def test_rs_plane_wave_smoke_is_nearly_uniform_in_center_crop(self):
        propagation = RayleighSommerfeldPropagation(size=64, wavelength=0.75e-3, distance=1e-4, pixel_size=0.4e-3)
        field = torch.ones(1, 64, 64, dtype=torch.cfloat)

        propagated = propagation(field)
        intensity = field_intensity(propagated)
        center = intensity[:, 16:-16, 16:-16]
        rel_std = center.std() / center.mean().clamp_min(1e-8)

        self.assertLess(float(rel_std), 1e-3)

    def test_normalized_detector_logits_are_based_on_relative_energy(self):
        scores = torch.tensor([[4.0, 1.0, 1.0]], dtype=torch.float32)
        logits = normalized_detector_logits(scores)
        expected = torch.log(torch.tensor([[4.0, 1.0, 1.0]]) / 6.0 + 1e-8)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6, rtol=1e-6))

    def test_detector_contrast_matches_requested_formula(self):
        scores = torch.tensor([[4.0, 1.0, 0.5], [1.0, 3.0, 2.0]], dtype=torch.float32)
        target = torch.tensor([0, 2])
        contrast = detector_contrast(scores, target)
        expected = torch.tensor(
            [
                (4.0 - 1.0) / (4.0 + 1.0 + 1e-8),
                (2.0 - 3.0) / (2.0 + 3.0 + 1e-8),
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(contrast, expected, atol=1e-6, rtol=1e-6))

    def test_classifier_forward_with_metrics_returns_auxiliary_outputs(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).classifier_model_kwargs())
        x = torch.rand(2, 1, 28, 28)
        target = torch.tensor([0, 1])

        result = model.forward_with_metrics(x, target=target)

        self.assertEqual(set(result.keys()), {"scores", "logits", "intensity", "contrast"})
        self.assertEqual(result["scores"].shape, (2, 10))
        self.assertEqual(result["logits"].shape, (2, 10))
        self.assertEqual(result["intensity"].shape, (2, 24, 24))
        self.assertEqual(result["contrast"].shape, (2,))

    def test_classifier_forward_remains_lightweight_without_metrics_helper(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).classifier_model_kwargs())
        x = torch.rand(2, 1, 28, 28)

        with unittest.mock.patch.object(model, "forward_with_metrics", side_effect=AssertionError("do not call")):
            output = model(x)

        self.assertEqual(output.shape, (2, 10))

    def test_classifier_rgb_embed_input_places_channels_into_disjoint_regions(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=30, num_layers=2).classifier_model_kwargs())
        x = torch.zeros(1, 3, 12, 12)
        x[:, 0].fill_(1.0)
        x[:, 1].fill_(2.0)
        x[:, 2].fill_(3.0)

        field = model._embed_input(x)

        self.assertEqual(field.shape, (1, 30, 30))
        self.assertEqual(field.dtype, torch.cfloat)

        amplitude = field.abs()[0]
        occupied_cols = torch.where(amplitude.sum(dim=0) > 0)[0]
        self.assertGreaterEqual(occupied_cols.numel(), 3)

        transitions = torch.where(torch.diff(occupied_cols) > 1)[0]
        self.assertGreaterEqual(transitions.numel(), 2)

    def test_classifier_accepts_rgb_batches(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).classifier_model_kwargs())
        x = torch.rand(2, 3, 32, 32)
        logits = model(x)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(logits.dtype, torch.float32)

    def test_rgb_embedding_raises_value_error_for_small_target_size(self):
        from d2nn import embed_rgb_amplitude_image
        x = torch.zeros(1, 3, 12, 12)
        with self.assertRaisesRegex(ValueError, "too small"):
            embed_rgb_amplitude_image(x, size=10, target_size=3)

    def test_imager_accepts_rgb_inputs(self):
        model = D2NNImager(**IMAGER_PAPER_OPTICS.with_overrides(size=24, num_layers=2).imager_model_kwargs())
        x = torch.rand(2, 3, 8, 8)
        target = model.build_target(x)
        self.assertEqual(target.shape, (2, 24, 24))
        output = model(x)
        self.assertEqual(output.shape, (2, 24, 24))

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

    def test_field_intensity_matches_squared_magnitude_without_eps_bias(self):
        x = torch.tensor([[3.0 + 4.0j, 0.0 + 1.0j]], dtype=torch.cfloat)
        intensity = field_intensity(x)
        expected = torch.tensor([[25.0, 1.0]])
        self.assertTrue(torch.equal(intensity, expected))

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

    def test_checkpoint_load_ignores_stale_transfer_buffers_and_detector_masks(self):
        optics = CLASSIFIER_PAPER_OPTICS.with_overrides(size=24, num_layers=2)
        model = D2NN(**optics.classifier_model_kwargs())
        restored = D2NN(**optics.classifier_model_kwargs())
        state_dict = model.state_dict()
        state_dict["layers.0.H"] = torch.zeros(24, 24, dtype=torch.cfloat)
        state_dict["layers.1.H"] = torch.zeros(24, 24, dtype=torch.cfloat)
        state_dict["H_out"] = torch.zeros(24, 24, dtype=torch.cfloat)
        state_dict["detector_masks"] = torch.zeros_like(restored.detector_masks)

        restored.load_state_dict(state_dict, strict=True)

        self.assertTrue(torch.equal(restored.detector_masks, restored._build_detectors(24, 10, None)))
        self.assertIsInstance(restored.layers[0].propagation, RayleighSommerfeldPropagation)
        self.assertIsInstance(restored.network.output_propagation, RayleighSommerfeldPropagation)
        x = torch.rand(2, 1, 28, 28)
        self.assertEqual(restored(x).shape, (2, 10))

    def test_main_d2nn_module_no_longer_uses_torch_fft(self):
        source = Path(importlib.import_module("d2nn").__file__).read_text(encoding="utf-8")
        self.assertNotIn("torch.fft", source)

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

    def test_derive_experiment_run_name_encodes_loss_config_without_activation(self):
        run_name = derive_experiment_run_name(
            run_name=None,
            experiment_stage="rs_only",
            activation_type="none",
            activation_positions=(),
            activation_hparams={},
            seed=42,
            loss_config={"alpha": 1.0, "beta": 0.2, "gamma": 0.01},
        )
        self.assertIn("alpha-1", run_name)
        self.assertIn("beta-0p2", run_name)
        self.assertIn("gamma-0p01", run_name)
        self.assertIn("seed-42", run_name)

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

    def test_experiment_manifest_fields_include_model_version_and_loss_config(self):
        payload = experiment_manifest_fields(
            checkpoint_path=Path("checkpoints/best_fashion_mnist.pth"),
            run_name="rs_only",
            experiment_stage="baseline",
            seed=42,
            optics=CLASSIFIER_PAPER_OPTICS.with_overrides(size=200, num_layers=5),
            activation_type="none",
            activation_positions=(),
            activation_hparams={},
            model_version="rs_v1",
            loss_config={"alpha": 1.0, "beta": 0.1, "gamma": 0.01},
        )
        self.assertEqual(payload["model_version"], "rs_v1")
        self.assertEqual(payload["loss_config"], {"alpha": 1.0, "beta": 0.1, "gamma": 0.01})

    def test_derive_experiment_run_name_keeps_canonical_baseline_for_default_loss_config(self):
        run_name = derive_experiment_run_name(
            experiment_stage="baseline",
            activation_type="none",
            seed=42,
            loss_config={"alpha": 1.0, "beta": 0.1, "gamma": 0.01},
        )
        self.assertIsNone(run_name)

    def test_train_parser_accepts_seed_and_experiment_stage(self):
        args = build_parser().parse_args(["--seed", "7", "--experiment-stage", "placement_ablation"])
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.experiment_stage, "placement_ablation")

    def test_train_parser_accepts_composite_loss_arguments(self):
        args = build_parser().parse_args(["--alpha", "1.0", "--beta", "0.1", "--gamma", "0.01"])
        self.assertEqual(args.alpha, 1.0)
        self.assertEqual(args.beta, 0.1)
        self.assertEqual(args.gamma, 0.01)

    def test_ensure_checkpoint_version_rejects_mismatch(self):
        with self.assertRaisesRegex(ValueError, "expected model_version='rs_v1'"):
            ensure_checkpoint_version({"model_version": "asm_v1"}, expected_version="rs_v1", checkpoint_path="demo.pth")

    def test_ensure_checkpoint_version_accepts_match(self):
        ensure_checkpoint_version({"model_version": "rs_v1"}, expected_version="rs_v1", checkpoint_path="demo.pth")

    def test_ensure_checkpoint_version_allows_missing_manifest_when_requested(self):
        ensure_checkpoint_version(None, expected_version="rs_v1", checkpoint_path="demo.pth", allow_missing=True)
        ensure_checkpoint_version({}, expected_version="rs_v1", checkpoint_path="demo.pth", allow_missing=True)

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

    def test_train_parser_accepts_run_experiment_grid(self):
        args = build_parser().parse_args(["--run-experiment-grid", "activation_mechanisms"])
        self.assertEqual(args.run_experiment_grid, "activation_mechanisms")

    def test_visualize_parser_accepts_seed(self):
        args = build_visualize_parser().parse_args(["--checkpoint", "checkpoints/demo.pth", "--seed", "11"])
        self.assertEqual(args.seed, 11)

    def test_visualize_parser_understanding_report_flags(self):
        args = build_visualize_parser().parse_args(
            [
                "--checkpoint",
                "checkpoints/demo.pth",
                "--understanding-report",
                "--sample-indices",
                "4,5,6",
                "--quantization-levels",
                "4,8",
            ]
        )
        self.assertTrue(args.understanding_report)
        self.assertEqual(args.sample_indices, "4,5,6")
        self.assertEqual(args.quantization_levels, "4,8")

    def test_visualize_parser_help_mentions_cifar10_rgb(self):
        help_text = build_visualize_parser().format_help()
        self.assertIn("cifar10-rgb", help_text)

    def test_quantize_phase_masks_uniform_preserves_wrapped_range(self):
        import artifacts

        phase_masks = torch.tensor(
            [
                [[-0.2, 0.0], [math.pi, 2 * math.pi + 0.2]],
                [[-4 * math.pi, 3 * math.pi / 2], [5 * math.pi, 7 * math.pi / 2]],
            ],
            dtype=torch.float32,
        )

        quantized = artifacts.quantize_phase_masks_uniform(phase_masks, levels=8)

        self.assertEqual(quantized.shape, phase_masks.shape)
        self.assertTrue(torch.all(quantized >= 0))
        self.assertTrue(torch.all(quantized < 2 * math.pi + 1e-6))

    def test_plot_sample_output_patterns_writes_understanding_report_figure(self):
        from torch.utils.data import TensorDataset

        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=2).classifier_model_kwargs())
        dataset = TensorDataset(
            torch.rand(3, 1, 28, 28),
            torch.tensor([0, 1, 2]),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "sample_output_patterns.png"
            plot_sample_output_patterns(model, dataset, torch.device("cpu"), [0, 2], save_path=save_path, no_show=True)
            self.assertTrue(save_path.exists())

    def test_plot_quantization_sensitivity_restores_model_phases(self):
        from torch.utils.data import DataLoader, TensorDataset

        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=2).classifier_model_kwargs())
        original_phases = [layer.phase.detach().clone() for layer in model.layers]
        dataset = TensorDataset(
            torch.rand(4, 1, 28, 28),
            torch.tensor([0, 1, 2, 3]),
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        manifest = {"run_name": "demo"}
        dataset_cfg = {"display_name": "Demo"}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "quantization_sensitivity.png"
            plot_quantization_sensitivity(
                model,
                CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=2),
                manifest,
                loader,
                dataset_cfg,
                torch.device("cpu"),
                [4, 8],
                save_path=save_path,
                no_show=True,
            )
            self.assertTrue(save_path.exists())

        for layer, original in zip(model.layers, original_phases):
            self.assertTrue(torch.allclose(layer.phase.detach(), original))

    def test_plot_quantization_sensitivity_restores_model_phases_on_failure(self):
        from torch.utils.data import DataLoader, TensorDataset

        tasks_module = importlib.import_module("tasks")
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=2).classifier_model_kwargs())
        original_phases = [layer.phase.detach().clone() for layer in model.layers]
        dataset = TensorDataset(
            torch.rand(4, 1, 28, 28),
            torch.tensor([0, 1, 2, 3]),
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        manifest = {"run_name": "demo"}
        dataset_cfg = {"display_name": "Demo"}

        with unittest.mock.patch.object(
            tasks_module,
            "evaluate_classification",
            side_effect=[
                {"loss": 0.0, "mse": 0.0, "ce": 0.0, "reg": 0.0, "accuracy": 50.0, "contrast": 0.1},
                RuntimeError("boom"),
            ],
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                tasks_module.plot_quantization_sensitivity(
                    model,
                    CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=2),
                    manifest,
                    loader,
                    dataset_cfg,
                    torch.device("cpu"),
                    [4, 8],
                    no_show=True,
                )

        for layer, original in zip(model.layers, original_phases):
            self.assertTrue(torch.allclose(layer.phase.detach(), original))

    def test_resolve_experiment_seed_prefers_explicit_seed_then_manifest_then_default(self):
        self.assertEqual(resolve_experiment_seed(11, {"seed": 7}), 11)
        self.assertEqual(resolve_experiment_seed(None, {"seed": 7}), 7)
        self.assertEqual(resolve_experiment_seed(None, None), 42)

    def test_classification_composite_loss_reports_all_terms(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=16, num_layers=2).classifier_model_kwargs())
        outputs = {
            "scores": torch.tensor([[3.0, 1.0] + [0.0] * 8], dtype=torch.float32),
            "logits": torch.log(torch.tensor([[0.75, 0.25] + [1e-8] * 8], dtype=torch.float32)),
        }
        target = torch.tensor([0])

        terms = classification_composite_loss(outputs, target, model, alpha=1.0, beta=0.1, gamma=0.01)

        self.assertEqual(set(terms.keys()), {"total", "mse", "ce", "reg"})
        self.assertGreater(float(terms["total"].detach()), 0.0)
        self.assertGreaterEqual(float(terms["reg"].detach()), 0.0)

    def test_phase_regularizer_treats_2pi_equivalent_neighbors_as_identical(self):
        from tasks import phase_smoothness_regularizer

        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=4, num_layers=1).classifier_model_kwargs())
        with torch.no_grad():
            model.layers[0].phase.zero_()
            model.layers[0].phase[:, 2:] = 2 * math.pi

        regularizer = phase_smoothness_regularizer(model)

        self.assertLess(float(regularizer.detach()), 1e-6)

    def test_configure_matplotlib_backend_only_forces_agg_for_no_show(self):
        with unittest.mock.patch("artifacts.matplotlib.use") as use_mock:
            configure_matplotlib_backend(no_show=False)
            use_mock.assert_not_called()

        with unittest.mock.patch("artifacts.matplotlib.use") as use_mock:
            configure_matplotlib_backend(no_show=True)
            use_mock.assert_called_once_with("Agg", force=True)

    def test_plot_phase_masks_uses_backend_helper(self):
        model = D2NN(**CLASSIFIER_PAPER_OPTICS.with_overrides(size=8, num_layers=2).classifier_model_kwargs())

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "phase_masks.png"
            with unittest.mock.patch("artifacts.configure_matplotlib_backend", wraps=configure_matplotlib_backend) as backend_mock:
                plot_phase_masks(model, save_path=save_path, no_show=True)
            backend_mock.assert_called_once_with(no_show=True)
            self.assertTrue(save_path.exists())

    def test_metric_history_tracks_accuracy_and_contrast(self):
        history = build_metric_history()
        append_metric_history(
            history,
            split="train",
            total=1.2,
            mse=1.0,
            ce=0.1,
            reg=0.1,
            accuracy=87.5,
            contrast=0.4,
        )
        self.assertEqual(history["train"]["accuracy"], [87.5])
        self.assertEqual(history["train"]["contrast"], [0.4])
        self.assertEqual(history["train"]["loss"], [1.2])

    def test_plot_classification_history_writes_accuracy_and_contrast_figure(self):
        history = build_metric_history()
        append_metric_history(
            history,
            split="train",
            total=1.2,
            mse=1.0,
            ce=0.1,
            reg=0.1,
            accuracy=87.5,
            contrast=0.4,
        )
        append_metric_history(
            history,
            split="val",
            total=1.0,
            mse=0.8,
            ce=0.1,
            reg=0.1,
            accuracy=89.0,
            contrast=0.5,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "classification_history.png"
            plot_classification_history(history, save_path=save_path, no_show=True)
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)

    def test_checkpoint_selection_prefers_accuracy_then_contrast_then_later_epoch(self):
        best = {"accuracy": 90.0, "contrast": 0.3, "epoch": 4}
        better_contrast = {"accuracy": 90.0, "contrast": 0.4, "epoch": 3}
        later_tie = {"accuracy": 90.0, "contrast": 0.4, "epoch": 5}
        worse = {"accuracy": 89.0, "contrast": 0.9, "epoch": 99}

        self.assertTrue(is_better_classification_checkpoint(better_contrast, best))
        self.assertTrue(is_better_classification_checkpoint(later_tie, better_contrast))
        self.assertFalse(is_better_classification_checkpoint(worse, later_tie))

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

    def test_resolve_activation_config_maps_all_placement_aliases(self):
        parser = build_parser()
        
        args = parser.parse_args(["--layers", "5", "--activation-type", "coherent_amplitude", "--activation-placement", "front"])
        _, positions_front, _ = resolve_activation_config(args, None)
        self.assertEqual(positions_front, (1,))
        
        args = parser.parse_args(["--layers", "5", "--activation-type", "coherent_amplitude", "--activation-placement", "back"])
        _, positions_back, _ = resolve_activation_config(args, None)
        self.assertEqual(positions_back, (5,))
        
        args = parser.parse_args(["--layers", "5", "--activation-type", "coherent_amplitude", "--activation-placement", "all"])
        _, positions_all, _ = resolve_activation_config(args, None)
        self.assertEqual(positions_all, (1, 2, 3, 4, 5))

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
                "--alpha",
                "1.0",
                "--beta",
                "0.1",
                "--gamma",
                "0.01",
            ]
        )
        commands = format_experiment_grid_commands("coherent_amplitude_positions", args)
        self.assertEqual(len(commands), 4)
        self.assertIn("python train.py", commands[0])
        self.assertIn("--activation-placement front", commands[0])
        self.assertIn("--activation-preset balanced", commands[0])
        self.assertIn("--alpha 1.0", commands[0])
        self.assertIn("--beta 0.1", commands[0])
        self.assertIn("--gamma 0.01", commands[0])

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

    def test_execute_experiment_grid_invokes_callback_for_each_spec(self):
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
        seen = []

        def runner(spec_args):
            seen.append((spec_args.activation_type, spec_args.activation_preset, spec_args.activation_placement))

        execute_experiment_grid("activation_mechanisms", args, runner)
        self.assertEqual(
            seen,
            [
                ("coherent_amplitude", "balanced", "mid"),
                ("coherent_phase", "balanced", "mid"),
                ("incoherent_intensity", "balanced", "mid"),
            ],
        )

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

    def test_build_fabrication_readiness_summary_reports_clipping(self):
        raw_height_map = torch.tensor([[[0.0, 2e-6], [5e-6, 8e-6]]]).numpy()
        manufacturable_relief = torch.tensor([[[0.0, 2e-6], [5e-6, 6e-6]]]).numpy()
        thickness_map = manufacturable_relief + 1e-5

        summary = build_fabrication_readiness_summary(
            raw_height_map,
            manufacturable_relief,
            thickness_map,
            max_relief_m=6e-6,
            pixel_size_m=1e-6,
        )

        self.assertTrue(summary["has_relief_limit"])
        self.assertAlmostEqual(summary["max_relief_m"], 6e-6)
        self.assertAlmostEqual(summary["raw_height_max_m"], 8e-6)
        self.assertAlmostEqual(summary["manufacturable_height_max_m"], 6e-6)
        self.assertEqual(summary["clipped_pixels"], 1)
        self.assertEqual(summary["total_pixels"], 4)
        self.assertAlmostEqual(summary["clipped_fraction"], 0.25)
        self.assertAlmostEqual(summary["pixel_size_m"], 1e-6)

    def test_write_export_report_includes_fabrication_readiness_section(self):
        layer_stats = [
            {
                "layer": 1,
                "phase_min_rad": 0.0,
                "phase_max_rad": 1.0,
                "height_min_m": 0.0,
                "height_max_m": 1e-6,
                "height_mean_m": 0.5e-6,
                "thickness_min_m": 1e-5,
                "thickness_max_m": 1.1e-5,
                "thickness_mean_m": 1.05e-5,
            }
        ]
        readiness = {
            "has_relief_limit": True,
            "max_relief_m": 6e-6,
            "raw_height_max_m": 8e-6,
            "manufacturable_height_max_m": 6e-6,
            "clipped_fraction": 0.25,
            "clipped_pixels": 1,
            "total_pixels": 4,
            "pixel_size_m": 1e-6,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            write_export_report(
                report_path,
                checkpoint_name="demo.pth",
                task="classification",
                num_layers=1,
                size=4,
                pixel_size_um=0.4,
                wavelength_um=0.75,
                quantization_levels=256,
                layer_stats=layer_stats,
                fabrication_readiness=readiness,
            )
            content = report_path.read_text(encoding="utf-8")

        self.assertIn("## Fabrication Readiness", content)
        self.assertIn("clipped fraction", content)
        self.assertIn("demo.pth", content)

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

    def test_final_export_wrapper_default_output_dir_uses_official_date_stamp(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        output_dir = wrapper.build_default_output_dir(current_date=date(2026, 4, 7))
        self.assertEqual(output_dir, Path("exports/fmnist5-phaseonly-aligned-final_20260407"))

    def test_final_export_wrapper_builds_expected_export_command(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        command = wrapper.build_export_command(
            python_executable="python",
            checkpoint_path=wrapper.OFFICIAL_PRESET["checkpoint"],
            output_dir=Path("exports/final"),
            refractive_index=1.7227,
            ambient_index=1.0,
            base_thickness_um=500.0,
            max_relief_um=900.0,
            quantization_levels=256,
            export_stl=True,
        )
        self.assertEqual(command[0], "python")
        self.assertEqual(command[2], "export_phase_plate.py")
        self.assertIn("--checkpoint", command)
        self.assertIn("checkpoints/best_fashion_mnist.fmnist5-phaseonly-aligned.pth", command)
        self.assertIn("--task", command)
        self.assertIn("classification", command)
        self.assertIn("--wavelength", command)
        self.assertIn("0.00075", command)
        self.assertIn("--layer-distance", command)
        self.assertIn("0.03", command)
        self.assertIn("--pixel-size", command)
        self.assertIn("0.0004", command)
        self.assertIn("--export-stl", command)

    def test_final_export_wrapper_loads_lab_config(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "lab.json"
            config_path.write_text(
                json.dumps(
                    {
                        "material": {
                            "refractive_index": 1.8,
                            "ambient_index": 1.0,
                        },
                        "process": {
                            "base_thickness_um": 500.0,
                            "max_relief_um": 950.0,
                            "quantization_levels": 256,
                            "stl_required": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = wrapper.load_lab_config(config_path)
            self.assertEqual(config["refractive_index"], 1.8)
            self.assertEqual(config["ambient_index"], 1.0)
            self.assertEqual(config["base_thickness_um"], 500.0)
            self.assertEqual(config["max_relief_um"], 950.0)
            self.assertEqual(config["quantization_levels"], 256)
            self.assertTrue(config["export_stl"])

    def test_final_export_wrapper_loads_lab_config_with_utf8_bom(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "lab.json"
            config_path.write_text(
                json.dumps(
                    {
                        "material": {
                            "refractive_index": 1.8,
                            "ambient_index": 1.0,
                        },
                        "process": {
                            "base_thickness_um": 500.0,
                            "max_relief_um": 950.0,
                            "quantization_levels": 256,
                            "stl_required": False,
                        },
                    }
                ),
                encoding="utf-8-sig",
            )

            config = wrapper.load_lab_config(config_path)
            self.assertEqual(config["refractive_index"], 1.8)
            self.assertFalse(config["export_stl"])

    def test_final_export_wrapper_cli_values_override_lab_config(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        parser = wrapper.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "lab.json"
            config_path.write_text(
                json.dumps(
                    {
                        "material": {
                            "refractive_index": 1.8,
                            "ambient_index": 1.0,
                        },
                        "process": {
                            "base_thickness_um": 500.0,
                            "max_relief_um": 950.0,
                            "quantization_levels": 256,
                            "stl_required": False,
                        },
                    }
                ),
                encoding="utf-8",
            )

            args = parser.parse_args(
                [
                    "--lab-config",
                    str(config_path),
                    "--max-relief-um",
                    "1000",
                    "--export-stl",
                ]
            )
            resolved = wrapper.resolve_lab_inputs(args)
            self.assertEqual(resolved["refractive_index"], 1.8)
            self.assertEqual(resolved["max_relief_um"], 1000.0)
            self.assertTrue(resolved["export_stl"])

    def test_final_export_wrapper_requires_complete_lab_inputs_after_merge(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        parser = wrapper.build_parser()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "lab.json"
            config_path.write_text(json.dumps({"material": {"refractive_index": 1.8}}), encoding="utf-8")
            args = parser.parse_args(["--lab-config", str(config_path)])
            with self.assertRaises(ValueError):
                wrapper.resolve_lab_inputs(args)

    def test_final_export_wrapper_resolves_official_checkpoint_in_repo_root(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            checkpoint_path = repo_root / wrapper.OFFICIAL_PRESET["checkpoint"]
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"stub")

            resolved = wrapper.resolve_checkpoint_path(repo_root)
            self.assertEqual(resolved, checkpoint_path)

    def test_final_export_wrapper_bootstraps_checkpoint_from_official_phase_masks(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            artifact_dir = repo_root / "docs" / "official-artifacts" / "fmnist5-phaseonly-aligned"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.save(artifact_dir / "phase_masks.npy", np.zeros((5, 200, 200), dtype=np.float32))
            (artifact_dir / "source_checkpoint_manifest.json").write_text(
                json.dumps(
                    {
                        "task": "classification",
                        "dataset": "Fashion-MNIST",
                        "run_name": "fmnist5-phaseonly-aligned",
                        "experiment_stage": "fabrication_baseline",
                        "seed": 42,
                    }
                ),
                encoding="utf-8",
            )

            resolved = wrapper.resolve_checkpoint_path(repo_root)
            self.assertEqual(resolved, repo_root / wrapper.OFFICIAL_PRESET["checkpoint"])
            self.assertTrue(resolved.exists())
            manifest_path = resolved.with_suffix(".json")
            self.assertTrue(manifest_path.exists())

    def test_final_export_wrapper_raises_clear_error_when_checkpoint_and_artifacts_are_missing(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            with self.assertRaises(FileNotFoundError) as ctx:
                wrapper.resolve_checkpoint_path(repo_root)
        self.assertIn("best_fashion_mnist.fmnist5-phaseonly-aligned.pth", str(ctx.exception))

    def test_final_export_wrapper_validation_summary_is_pass_when_unclipped(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir)
            for rel_path in (
                "phase_masks.npy",
                "height_map.npy",
                "height_map_manufacturable.npy",
                "thickness_map.npy",
                "height_map_quantized.npy",
                "report.md",
                "metadata.json",
            ):
                path = export_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stub", encoding="utf-8")

            metadata = {
                "checkpoint": str(wrapper.OFFICIAL_PRESET["checkpoint"]),
                "manufacturing": {
                    "base_thickness_um": 500.0,
                    "max_relief_um": 900.0,
                    "export_stl": False,
                },
                "quantization_levels": 256,
                "fabrication_readiness": {
                    "clipped_pixels": 0,
                    "clipped_fraction": 0.0,
                    "raw_height_max_m": 8e-4,
                    "manufacturable_height_max_m": 8e-4,
                    "thickness_max_m": 1.3e-3,
                },
            }
            (export_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            summary = wrapper.build_validation_summary(export_root)
            self.assertEqual(summary["status"], "PASS")
            self.assertFalse(summary["issues"])

    def test_final_export_wrapper_validation_summary_is_warn_for_minor_clipping(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir)
            for rel_path in (
                "phase_masks.npy",
                "height_map.npy",
                "height_map_manufacturable.npy",
                "thickness_map.npy",
                "height_map_quantized.npy",
                "report.md",
                "metadata.json",
            ):
                path = export_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stub", encoding="utf-8")

            metadata = {
                "checkpoint": str(wrapper.OFFICIAL_PRESET["checkpoint"]),
                "manufacturing": {
                    "base_thickness_um": 500.0,
                    "max_relief_um": 900.0,
                    "export_stl": False,
                },
                "quantization_levels": 256,
                "fabrication_readiness": {
                    "clipped_pixels": 10,
                    "clipped_fraction": 0.005,
                    "raw_height_max_m": 8e-4,
                    "manufacturable_height_max_m": 7.5e-4,
                    "thickness_max_m": 1.25e-3,
                },
            }
            (export_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            summary = wrapper.build_validation_summary(export_root)
            self.assertEqual(summary["status"], "WARN")
            self.assertTrue(summary["issues"])

    def test_final_export_wrapper_validation_summary_is_stop_for_large_clipping_or_missing_files(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir)
            (export_root / "metadata.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(wrapper.OFFICIAL_PRESET["checkpoint"]),
                        "manufacturing": {
                            "base_thickness_um": 500.0,
                            "max_relief_um": 900.0,
                            "export_stl": False,
                        },
                        "quantization_levels": 256,
                        "fabrication_readiness": {
                            "clipped_pixels": 2000,
                            "clipped_fraction": 0.05,
                            "raw_height_max_m": 8e-4,
                            "manufacturable_height_max_m": 6e-4,
                            "thickness_max_m": 1.1e-3,
                        },
                    }
                ),
                encoding="utf-8",
            )

            summary = wrapper.build_validation_summary(export_root)
            self.assertEqual(summary["status"], "STOP")
            self.assertGreaterEqual(len(summary["issues"]), 2)

    def test_final_export_wrapper_validation_summary_flags_same_name_wrong_checkpoint_path(self):
        wrapper = importlib.import_module("export_fmnist5_phaseonly_aligned_final")
        with tempfile.TemporaryDirectory() as tmpdir:
            export_root = Path(tmpdir)
            for rel_path in (
                "phase_masks.npy",
                "height_map.npy",
                "height_map_manufacturable.npy",
                "thickness_map.npy",
                "height_map_quantized.npy",
                "report.md",
                "metadata.json",
            ):
                path = export_root / rel_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("stub", encoding="utf-8")

            wrong_checkpoint = Path("D:/other/checkpoints") / wrapper.OFFICIAL_PRESET["checkpoint"].name
            metadata = {
                "checkpoint": str(wrong_checkpoint),
                "manufacturing": {
                    "base_thickness_um": 500.0,
                    "max_relief_um": 900.0,
                    "export_stl": False,
                },
                "quantization_levels": 256,
                "fabrication_readiness": {
                    "clipped_pixels": 0,
                    "clipped_fraction": 0.0,
                    "raw_height_max_m": 8e-4,
                    "manufacturable_height_max_m": 8e-4,
                    "thickness_max_m": 1.3e-3,
                },
            }
            (export_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            summary = wrapper.build_validation_summary(export_root)
            self.assertIn("does not match the frozen official preset", " ".join(summary["issues"]))


if __name__ == "__main__":
    unittest.main()
