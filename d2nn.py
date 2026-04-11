"""
Diffractive Deep Neural Network (D2NN)

Reproducing: "All-optical machine learning using diffractive deep neural networks"
Lin et al., Science 361, 1004-1008 (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def field_intensity(u):
    """Squared complex magnitude without stabilization bias."""
    return u.real**2 + u.imag**2


def safe_abs(u, eps=1e-8):
    """Stable complex magnitude used when amplitude is required."""
    return torch.sqrt(field_intensity(u) + eps)


def normalized_detector_logits(scores, eps=1e-8):
    """Convert detector scores into CE-compatible logits based on relative energy."""
    normalized = scores / scores.sum(dim=1, keepdim=True).clamp_min(eps)
    return torch.log(normalized.clamp_min(eps))


def detector_contrast(scores, target, eps=1e-8):
    """Target-vs-strongest-other detector contrast in (-1, 1)."""
    target_energy = scores.gather(1, target.view(-1, 1)).squeeze(1)
    masked_scores = scores.clone()
    masked_scores.scatter_(1, target.view(-1, 1), float("-inf"))
    max_other = masked_scores.max(dim=1).values
    return (target_energy - max_other) / (target_energy + max_other + eps)


def _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key):
    """Ignore deterministic buffers that are regenerated from model config."""
    while key in missing_keys:
        missing_keys.remove(key)
    while key in unexpected_keys:
        unexpected_keys.remove(key)


def checkpointed_module_forward(module, u):
    """
    Recompute memory-heavy propagation during backward when gradients are enabled.

    RS direct-space propagation creates a large autograd graph at 200x200 resolution.
    Checkpointing keeps the training path numerically identical while trading compute
    for memory so the default paper-sized model remains trainable on 8 GB GPUs.
    """
    if not torch.is_grad_enabled() or not u.requires_grad or not getattr(module, "training", False):
        return module(u)
    return activation_checkpoint(lambda x: module(x), u, use_reentrant=False)


def embed_amplitude_image(x, size, target_size=None):
    """Embed a grayscale image batch into the center of a complex optical field."""
    batch = x.shape[0]
    field = torch.zeros(batch, size, size, dtype=torch.cfloat, device=x.device)

    image = x.squeeze(1)
    target_size = target_size or size // 3
    image_resized = F.interpolate(
        image.unsqueeze(1), size=(target_size, target_size), mode="bilinear", align_corners=False
    ).squeeze(1)

    offset = (size - target_size) // 2
    field[:, offset : offset + target_size, offset : offset + target_size] = image_resized.to(torch.cfloat)
    return field


def embed_rgb_amplitude_image(x, size, target_size=None):
    """Embed an RGB image batch into three fixed horizontal amplitude subregions."""
    batch, channels, _, _ = x.shape
    if channels != 3:
        raise ValueError("RGB embedding expects exactly 3 channels")

    field = torch.zeros(batch, size, size, dtype=torch.cfloat, device=x.device)
    target_size = target_size or size // 3
    patch_size = target_size // 4
    if patch_size < 1:
        raise ValueError(f"Target size ({target_size}) is too small to embed RGB channels.")

    resized = F.interpolate(x, size=(patch_size, patch_size), mode="bilinear", align_corners=False)

    total_width = 3 * patch_size
    gap_count = 4
    gap = max((target_size - total_width) // gap_count, 0)
    window_offset = (size - target_size) // 2
    offset_y = window_offset + max((target_size - patch_size) // 2, 0)
    start_x = window_offset + gap

    for channel_idx in range(3):
        offset_x = start_x + channel_idx * (patch_size + gap)
        field[:, offset_y : offset_y + patch_size, offset_x : offset_x + patch_size] = resized[:, channel_idx].to(
            torch.cfloat
        )

    return field


def collect_phase_masks(layers, wrap=True):
    """Collect phase masks from a stack of diffractive layers."""
    phase_masks = torch.stack([layer.phase.detach().cpu() for layer in layers], dim=0)
    if wrap:
        phase_masks = torch.remainder(phase_masks, 2 * np.pi)
    return phase_masks


def phase_to_height_map(phase_masks, wavelength, refractive_index, ambient_index=1.0):
    """
    Convert phase delay to thickness for a transmissive phase plate.

    height = phase * wavelength / (2*pi*(n_material - n_ambient))
    """
    delta_n = refractive_index - ambient_index
    if delta_n <= 0:
        raise ValueError("refractive_index must be greater than ambient_index")
    return phase_masks * wavelength / (2 * np.pi * delta_n)


class FieldActivationBase(nn.Module):
    """Base interface for optional field activations inserted between layers."""

    def __init__(self):
        super().__init__()
        self.last_stats = {}

    def forward(self, u):
        raise NotImplementedError


class IdentityActivation(FieldActivationBase):
    """No-op activation used to verify nonlinear placement plumbing."""

    def forward(self, u):
        return u


class CoherentAmplitudeActivation(FieldActivationBase):
    """Intensity-dependent amplitude gate that preserves the complex phase."""

    def __init__(self, threshold=0.1, temperature=0.1, gain_min=0.0, gain_max=1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= gain_min <= gain_max <= 1.0:
            raise ValueError("gain range must satisfy 0 <= gain_min <= gain_max <= 1")

        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

    def forward(self, u):
        intensity = field_intensity(u)
        gate = torch.sigmoid((intensity - self.threshold) / self.temperature)
        gain = self.gain_min + (self.gain_max - self.gain_min) * gate
        self.last_stats = {
            "mean_intensity": float(intensity.mean().detach().cpu()),
            "mean_gain": float(gain.mean().detach().cpu()),
            "max_gain": float(gain.max().detach().cpu()),
            "min_gain": float(gain.min().detach().cpu()),
        }
        return gain.to(dtype=u.real.dtype) * u


class CoherentPhaseActivation(FieldActivationBase):
    """Intensity-dependent phase modulation that preserves the field magnitude."""

    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, u):
        intensity = field_intensity(u)
        phase_shift = self.gamma * intensity
        self.last_stats = {
            "mean_intensity": float(intensity.mean().detach().cpu()),
            "mean_phase_shift": float(phase_shift.mean().detach().cpu()),
            "max_phase_shift": float(phase_shift.max().detach().cpu()),
            "min_phase_shift": float(phase_shift.min().detach().cpu()),
        }
        return u * torch.exp(1j * phase_shift)


class IncoherentIntensityActivation(FieldActivationBase):
    """Intensity-only activation that discards input phase and emits a new field."""

    def __init__(self, responsivity=1.0, threshold=0.1, emission_phase_mode="zero"):
        super().__init__()
        if responsivity <= 0:
            raise ValueError("responsivity must be positive")
        if emission_phase_mode != "zero":
            raise ValueError("Only emission_phase_mode='zero' is currently supported")

        self.responsivity = float(responsivity)
        self.threshold = float(threshold)
        self.emission_phase_mode = emission_phase_mode

    def forward(self, u):
        intensity = field_intensity(u)
        emitted_amplitude = torch.relu(self.responsivity * intensity - self.threshold)
        self.last_stats = {
            "mean_intensity": float(intensity.mean().detach().cpu()),
            "mean_output_amplitude": float(emitted_amplitude.mean().detach().cpu()),
            "max_output_amplitude": float(emitted_amplitude.max().detach().cpu()),
            "min_output_amplitude": float(emitted_amplitude.min().detach().cpu()),
            "emission_phase_mode": self.emission_phase_mode,
        }
        if self.emission_phase_mode == "zero":
            return emitted_amplitude.to(torch.cfloat)

        emitted_phase = torch.zeros_like(emitted_amplitude)
        return emitted_amplitude.to(torch.cfloat) * torch.exp(1j * emitted_phase)


def normalize_activation_positions(positions, num_layers):
    """Normalize 1-based layer indices used for inter-layer activation placement."""
    if positions in (None, "", ()):
        return ()

    if isinstance(positions, str):
        parts = [part.strip() for part in positions.split(",") if part.strip()]
        positions = tuple(int(part) for part in parts)
    else:
        positions = tuple(int(position) for position in positions)

    normalized = []
    for position in positions:
        if position < 1 or position > num_layers:
            raise ValueError(f"activation position {position} is outside 1..{num_layers}")
        if position not in normalized:
            normalized.append(position)
    return tuple(normalized)


def build_activation_module(activation_type, activation_hparams=None):
    activation_hparams = dict(activation_hparams or {})
    if activation_type in (None, "none"):
        return None
    if activation_type == "identity":
        return IdentityActivation()
    if activation_type == "coherent_amplitude":
        return CoherentAmplitudeActivation(**activation_hparams)
    if activation_type == "coherent_phase":
        return CoherentPhaseActivation(**activation_hparams)
    if activation_type == "incoherent_intensity":
        return IncoherentIntensityActivation(**activation_hparams)
    raise ValueError(f"Unsupported activation type: {activation_type}")


class RayleighSommerfeldPropagation(nn.Module):
    """Rayleigh-Sommerfeld propagation using either direct integration or FFT convolution."""

    def __init__(self, size, wavelength, distance, pixel_size, chunk_size=None, backend="direct"):
        super().__init__()
        self.size = int(size)
        self.wavelength = float(wavelength)
        self.distance = float(distance)
        self.pixel_size = float(pixel_size)
        self.backend = str(backend or "direct").lower()
        if self.backend not in {"direct", "fft"}:
            raise ValueError(f"Unsupported RS propagation backend: {backend}")

        self.chunk_size = int(chunk_size) if chunk_size is not None else self._default_chunk_size(self.size)
        self.fft_shape = (3 * self.size - 2, 3 * self.size - 2)
        self._fft_kernel_cache = {}

        coords = (torch.arange(self.size, dtype=torch.float32) - (self.size - 1) / 2.0) * self.pixel_size
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
        self.register_buffer("source_points", points, persistent=False)
        self.register_buffer("target_points", points.clone(), persistent=False)
        self.register_buffer("relative_kernel", self._build_relative_kernel(), persistent=False)

    @staticmethod
    def _default_chunk_size(size):
        if size <= 24:
            return size * size
        if size <= 64:
            return 256
        return 64

    def _kernel_chunk(self, target_points, dtype):
        source_points = self.source_points.to(device=target_points.device, dtype=target_points.dtype)
        dx = target_points[:, None, 0] - source_points[None, :, 0]
        dy = target_points[:, None, 1] - source_points[None, :, 1]
        r2 = dx.square() + dy.square() + self.distance**2
        r = torch.sqrt(r2.clamp_min(1e-12))
        phase = torch.exp(1j * (2 * np.pi / self.wavelength) * r)
        kernel = (-1j * (self.pixel_size**2 / self.wavelength)) * (self.distance / r2) * phase
        return kernel.to(dtype=dtype)

    def _build_relative_kernel(self):
        delta_coords = torch.arange(-(self.size - 1), self.size, dtype=torch.float32) * self.pixel_size
        grid_y, grid_x = torch.meshgrid(delta_coords, delta_coords, indexing="ij")
        r2 = grid_x.square() + grid_y.square() + self.distance**2
        r = torch.sqrt(r2.clamp_min(1e-12))
        phase = torch.exp(1j * (2 * np.pi / self.wavelength) * r)
        kernel = (-1j * (self.pixel_size**2 / self.wavelength)) * (self.distance / r2) * phase
        return kernel.to(torch.cfloat)

    def _fft_kernel(self, device, dtype):
        cache_key = (str(device), dtype)
        kernel = self._fft_kernel_cache.get(cache_key)
        if kernel is not None:
            return kernel

        padded_kernel = torch.zeros(self.fft_shape, dtype=dtype, device=device)
        relative_kernel = self.relative_kernel.to(device=device, dtype=dtype)
        padded_kernel[: relative_kernel.shape[0], : relative_kernel.shape[1]] = relative_kernel
        kernel = torch.fft.fft2(padded_kernel)
        self._fft_kernel_cache[cache_key] = kernel
        return kernel

    def _forward_fft(self, u):
        kernel_fft = self._fft_kernel(u.device, u.dtype)
        padded_field = torch.zeros((u.shape[0], *self.fft_shape), dtype=u.dtype, device=u.device)
        padded_field[:, : self.size, : self.size] = u
        propagated = torch.fft.ifft2(torch.fft.fft2(padded_field) * kernel_fft.unsqueeze(0))
        start = self.size - 1
        end = start + self.size
        return propagated[:, start:end, start:end]

    def forward(self, u):
        batch, height, width = u.shape
        if height != self.size or width != self.size:
            raise ValueError(f"Expected a {self.size}x{self.size} field, got {height}x{width}")

        if self.backend == "fft":
            return self._forward_fft(u)

        flat_field = u.reshape(batch, -1)
        outputs = []
        for start in range(0, self.target_points.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, self.target_points.shape[0])
            kernel = self._kernel_chunk(self.target_points[start:end], dtype=u.dtype)
            outputs.append(flat_field @ kernel.transpose(0, 1))
        return torch.cat(outputs, dim=1).reshape(batch, self.size, self.size)


class DiffractiveLayer(nn.Module):
    """Single diffractive layer with learnable phase modulation."""

    def __init__(
        self,
        size,
        wavelength,
        layer_distance,
        pixel_size,
        propagate=True,
        propagation_chunk_size=None,
        propagation_backend="direct",
    ):
        super().__init__()
        self.size = int(size)
        self.wavelength = float(wavelength)
        self.layer_distance = float(layer_distance)
        self.pixel_size = float(pixel_size)
        self.propagate = bool(propagate)

        self.phase = nn.Parameter(torch.randn(size, size) * 0.05)
        self.propagation = None
        if self.propagate:
            self.propagation = RayleighSommerfeldPropagation(
                size=size,
                wavelength=wavelength,
                distance=layer_distance,
                pixel_size=pixel_size,
                chunk_size=propagation_chunk_size,
                backend=propagation_backend,
            )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}H"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key)

    def forward(self, u):
        modulation = torch.exp(1j * self.phase)
        u = u * modulation.unsqueeze(0)
        if self.propagation is None:
            return u
        return checkpointed_module_forward(self.propagation, u)


class DiffractiveNetwork:
    """Applies diffractive layers, optional activations, then a final free-space propagation."""

    def __init__(self, layers, input_propagation, output_propagation, activations=None):
        self.layers = layers
        self.input_propagation = input_propagation
        self.output_propagation = output_propagation
        self.activations = activations if activations is not None else {}

    def __call__(self, u):
        u = checkpointed_module_forward(self.input_propagation, u)
        for layer_idx, layer in enumerate(self.layers, start=1):
            u = layer(u)
            key = str(layer_idx)
            if key in self.activations:
                u = self.activations[key](u)
        return checkpointed_module_forward(self.output_propagation, u)


class DetectorLayer(nn.Module):
    """Maps output-plane intensity to detector energies and contrast/logit diagnostics."""

    def __init__(self, size, num_classes, detector_size=None):
        super().__init__()
        self.size = int(size)
        self.num_classes = int(num_classes)
        self.detector_size = detector_size
        self.register_buffer(
            "masks",
            self.build_detector_masks(size=size, num_classes=num_classes, detector_size=detector_size),
            persistent=False,
        )

    @staticmethod
    def build_detector_masks(size, num_classes, detector_size=None):
        if detector_size is None:
            detector_size = max(size // 15, 3)

        masks = torch.zeros(num_classes, size, size)
        center = size // 2
        radius = size // 4

        for index in range(num_classes):
            angle = 2 * np.pi * index / num_classes - np.pi / 2
            cx = int(center + radius * np.cos(angle))
            cy = int(center + radius * np.sin(angle))
            half_low = detector_size // 2
            half_high = detector_size - half_low
            x_start = max(cx - half_low, 0)
            x_end = min(cx + half_high, size)
            y_start = max(cy - half_low, 0)
            y_end = min(cy + half_high, size)
            masks[index, x_start:x_end, y_start:y_end] = 1.0

        return masks

    def forward(self, intensity):
        batch = intensity.shape[0]
        flat_intensity = intensity.reshape(batch, -1)
        flat_masks = self.masks.reshape(self.num_classes, -1).to(dtype=intensity.dtype)
        return flat_intensity @ flat_masks.transpose(0, 1)

    def logits(self, scores, eps=1e-8):
        return normalized_detector_logits(scores, eps=eps)

    def contrast(self, scores, target, eps=1e-8):
        return detector_contrast(scores, target, eps=eps)


class D2NNBase(nn.Module):
    """Base optical forward model providing shared layers, propagation, and diagnostics."""

    def __init__(
        self,
        num_layers,
        size,
        wavelength,
        layer_distance,
        pixel_size,
        input_distance,
        output_distance,
        activation_type="none",
        activation_positions=None,
        activation_hparams=None,
        propagation_chunk_size=None,
        propagation_backend="direct",
    ):
        super().__init__()
        self.size = size
        self.activation_type = activation_type or "none"
        self.activation_positions = normalize_activation_positions(activation_positions, num_layers)
        self.activation_hparams = dict(activation_hparams or {})

        self.layers = nn.ModuleList(
            [
                DiffractiveLayer(
                    size,
                    wavelength,
                    layer_distance,
                    pixel_size,
                    propagate=layer_index < (num_layers - 1),
                    propagation_chunk_size=propagation_chunk_size,
                    propagation_backend=propagation_backend,
                )
                for layer_index in range(num_layers)
            ]
        )
        self.activations = nn.ModuleDict()
        if self.activation_type != "none":
            for position in self.activation_positions:
                activation = build_activation_module(self.activation_type, self.activation_hparams)
                if activation is not None:
                    self.activations[str(position)] = activation
        self.input_propagation = RayleighSommerfeldPropagation(
            size=size,
            wavelength=wavelength,
            distance=input_distance,
            pixel_size=pixel_size,
            chunk_size=propagation_chunk_size,
            backend=propagation_backend,
        )
        self.output_propagation = RayleighSommerfeldPropagation(
            size=size,
            wavelength=wavelength,
            distance=output_distance,
            pixel_size=pixel_size,
            chunk_size=propagation_chunk_size,
            backend=propagation_backend,
        )
        self.network = DiffractiveNetwork(self.layers, self.input_propagation, self.output_propagation, self.activations)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}H_out"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key)

    def propagate_embedded_field(self, u):
        return self.network(u)

    def propagate_field(self, x):
        return self.propagate_embedded_field(self._embed_input(x))

    @staticmethod
    def output_intensity_from_field(field):
        return field_intensity(field)

    def output_intensity(self, x):
        return self.output_intensity_from_field(self.propagate_field(x))

    def export_phase_masks(self, wrap=True):
        return collect_phase_masks(self.layers, wrap=wrap)

    def activation_diagnostics(self):
        return {key: dict(module.last_stats) for key, module in self.activations.items() if module.last_stats}

    def _embed_input(self, x):
        raise NotImplementedError("Subclasses must implement input embedding.")


class D2NN(D2NNBase):
    """Diffractive classifier."""

    def __init__(
        self,
        num_layers=5,
        size=200,
        num_classes=10,
        wavelength=852e-9,
        layer_distance=30e-3,
        pixel_size=1e-6,
        input_distance=491.302e-3,
        output_distance=575.304e-3,
        detector_size=None,
        activation_type="none",
        activation_positions=None,
        activation_hparams=None,
        propagation_chunk_size=None,
        propagation_backend="direct",
    ):
        super().__init__(
            num_layers=num_layers,
            size=size,
            wavelength=wavelength,
            layer_distance=layer_distance,
            pixel_size=pixel_size,
            input_distance=input_distance,
            output_distance=output_distance,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_chunk_size=propagation_chunk_size,
            propagation_backend=propagation_backend,
        )
        self.num_classes = num_classes
        self.detector_layer = DetectorLayer(size=size, num_classes=num_classes, detector_size=detector_size)

    _build_detectors = staticmethod(DetectorLayer.build_detector_masks)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}detector_masks"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key)

    def forward(self, x):
        intensity = self.output_intensity(x)
        return self.detect_from_intensity(intensity)

    def forward_with_metrics(self, x, target=None):
        intensity = self.output_intensity(x)
        scores = self.detect_from_intensity(intensity)
        result = {
            "scores": scores,
            "logits": self.detector_layer.logits(scores),
            "intensity": intensity,
        }
        if target is not None:
            result["contrast"] = self.detector_layer.contrast(scores, target)
        return result

    def detect_from_intensity(self, intensity):
        return self.detector_layer(intensity)

    @property
    def detector_masks(self):
        return self.detector_layer.masks

    def _embed_input(self, x):
        channels = x.shape[1]
        if channels == 1:
            return embed_amplitude_image(x, self.size, target_size=self.size // 3)
        if channels == 3:
            return embed_rgb_amplitude_image(x, self.size, target_size=self.size // 3)
        raise ValueError(f"Unsupported classification input channels: {channels}")


class D2NNImager(D2NNBase):
    """D2NN variant for image reconstruction / imaging-lens experiments."""

    def __init__(
        self,
        num_layers=5,
        size=200,
        wavelength=852e-9,
        layer_distance=4e-3,
        pixel_size=1e-6,
        input_distance=491.302e-3,
        output_distance=575.304e-3,
        input_fraction=0.5,
        activation_type="none",
        activation_positions=None,
        activation_hparams=None,
        propagation_chunk_size=None,
        propagation_backend="direct",
    ):
        super().__init__(
            num_layers=num_layers,
            size=size,
            wavelength=wavelength,
            layer_distance=layer_distance,
            pixel_size=pixel_size,
            input_distance=input_distance,
            output_distance=output_distance,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
            propagation_chunk_size=propagation_chunk_size,
            propagation_backend=propagation_backend,
        )
        self.input_fraction = input_fraction

    def forward(self, x):
        intensity = self.output_intensity(x)
        max_vals = intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return intensity / max_vals

    def propagate(self, x):
        return self.output_intensity(x)

    def _embed_input(self, x):
        target_size = max(int(self.size * self.input_fraction), 1)
        channels = x.shape[1]
        if channels == 1:
            return embed_amplitude_image(x, self.size, target_size=target_size)
        if channels == 3:
            return embed_rgb_amplitude_image(x, self.size, target_size=target_size)
        raise ValueError(f"Unsupported imaging input channels: {channels}")

    def build_target(self, x):
        target = self._embed_input(x).abs()
        max_vals = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return target / max_vals
