"""
Diffractive Deep Neural Network (D2NN)

Reproducing: "All-optical machine learning using diffractive deep neural networks"
Lin et al., Science 361, 1004-1008 (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def field_intensity(u):
    """Squared complex magnitude without stabilization bias."""
    return u.real**2 + u.imag**2


def safe_abs(u, eps=1e-8):
    """Stable complex magnitude used when amplitude is required."""
    return torch.sqrt(field_intensity(u) + eps)


def padded_grid_size(size):
    """Use a doubled simulation grid to approximate linear convolution."""
    return 2 * size


def build_transfer_function(size, wavelength, layer_distance, pixel_size):
    """Build a batched ASM transfer function on the requested simulation grid."""
    fx = torch.fft.fftfreq(size, d=pixel_size)
    fy = torch.fft.fftfreq(size, d=pixel_size)
    FX, FY = torch.meshgrid(fx, fy, indexing="ij")

    sq = (1.0 / wavelength) ** 2 - FX**2 - FY**2
    propagating = sq > 0
    kz = torch.sqrt(torch.clamp(sq, min=0))
    return (torch.exp(1j * 2 * np.pi * kz * layer_distance) * propagating).unsqueeze(0)


def _pad_complex_field(u, target_size):
    """Zero-pad the last two spatial dimensions to the requested square size."""
    current_size = u.shape[-1]
    if current_size == target_size:
        return u
    if current_size > target_size:
        raise ValueError(f"Cannot pad from size {current_size} down to smaller target size {target_size}")

    pad_total = target_size - current_size
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    padded = torch.zeros(*u.shape[:-2], target_size, target_size, dtype=u.dtype, device=u.device)
    padded[..., pad_before : target_size - pad_after, pad_before : target_size - pad_after] = u
    return padded


def _crop_center(u, target_size):
    """Crop the centered square window from the last two spatial dimensions."""
    current_size = u.shape[-1]
    if current_size == target_size:
        return u
    if current_size < target_size:
        raise ValueError(f"Cannot crop from size {current_size} up to larger target size {target_size}")

    start = (current_size - target_size) // 2
    end = start + target_size
    return u[..., start:end, start:end]


def propagate_with_transfer(u, transfer_function):
    """Propagate on the padded grid and crop back to the original field size."""
    if transfer_function.ndim == 2:
        transfer_function = transfer_function.unsqueeze(0)

    padded = _pad_complex_field(u, transfer_function.shape[-1])
    spectrum = torch.fft.fft2(padded)
    propagated = torch.fft.ifft2(spectrum * transfer_function)
    return _crop_center(propagated, u.shape[-1])


def _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key):
    """Ignore deterministic buffers that are regenerated from model config."""
    while key in missing_keys:
        missing_keys.remove(key)
    while key in unexpected_keys:
        unexpected_keys.remove(key)


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


def propagate_output_field(u, layers, transfer_function, activations=None):
    """Propagate a complex field through diffractive layers to the output plane."""
    for layer_idx, layer in enumerate(layers, start=1):
        u = layer(u)
        if activations is not None:
            key = str(layer_idx)
            if key in activations:
                u = activations[key](u)

    return propagate_with_transfer(u, transfer_function)


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


class DiffractiveLayer(nn.Module):
    """Single diffractive layer with learnable phase modulation."""

    def __init__(self, size, wavelength, layer_distance, pixel_size):
        super().__init__()
        self.size = size
        self.padded_size = padded_grid_size(size)
        self.wavelength = wavelength
        self.layer_distance = layer_distance
        self.pixel_size = pixel_size

        self.phase = nn.Parameter(torch.randn(size, size) * 0.05)
        self.register_buffer("H", self._build_transfer_function())

    def _build_transfer_function(self):
        return build_transfer_function(self.padded_size, self.wavelength, self.layer_distance, self.pixel_size)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}H"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key)

    def forward(self, u):
        modulation = torch.exp(1j * self.phase)
        u = u * modulation.unsqueeze(0)
        return propagate_with_transfer(u, self.H)


class D2NNBase(nn.Module):
    """Base optical forward model providing shared layers, propagation, and diagnostics."""

    def __init__(
        self,
        num_layers,
        size,
        wavelength,
        layer_distance,
        pixel_size,
        activation_type="none",
        activation_positions=None,
        activation_hparams=None,
    ):
        super().__init__()
        self.size = size
        self.activation_type = activation_type or "none"
        self.activation_positions = normalize_activation_positions(activation_positions, num_layers)
        self.activation_hparams = dict(activation_hparams or {})

        self.layers = nn.ModuleList(
            [DiffractiveLayer(size, wavelength, layer_distance, pixel_size) for _ in range(num_layers)]
        )
        self.activations = nn.ModuleDict()
        if self.activation_type != "none":
            for position in self.activation_positions:
                activation = build_activation_module(self.activation_type, self.activation_hparams)
                if activation is not None:
                    self.activations[str(position)] = activation

    @staticmethod
    def _build_output_transfer(size, wavelength, layer_distance, pixel_size):
        return build_transfer_function(padded_grid_size(size), wavelength, layer_distance, pixel_size)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}H_out"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key)

    def propagate_embedded_field(self, u):
        return propagate_output_field(u, self.layers, self.H_out, self.activations)

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
        wavelength=0.75e-3,
        layer_distance=30e-3,
        pixel_size=0.4e-3,
        detector_size=None,
        activation_type="none",
        activation_positions=None,
        activation_hparams=None,
    ):
        super().__init__(
            num_layers=num_layers,
            size=size,
            wavelength=wavelength,
            layer_distance=layer_distance,
            pixel_size=pixel_size,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
        )
        self.num_classes = num_classes

        self.register_buffer(
            "H_out",
            self._build_output_transfer(size, wavelength, layer_distance, pixel_size),
        )
        self.register_buffer("detector_masks", self._build_detectors(size, num_classes, detector_size))

    @staticmethod
    def _build_detectors(size, num_classes, detector_size=None):
        if detector_size is None:
            detector_size = max(size // 15, 3)

        masks = torch.zeros(num_classes, size, size)
        center = size // 2
        radius = size // 4

        for i in range(num_classes):
            angle = 2 * np.pi * i / num_classes - np.pi / 2
            cx = int(center + radius * np.cos(angle))
            cy = int(center + radius * np.sin(angle))
            half_low = detector_size // 2
            half_high = detector_size - half_low
            x_start = max(cx - half_low, 0)
            x_end = min(cx + half_high, size)
            y_start = max(cy - half_low, 0)
            y_end = min(cy + half_high, size)
            masks[i, x_start:x_end, y_start:y_end] = 1.0

        return masks

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}detector_masks"
        state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        _discard_state_dict_key_compatibility(missing_keys, unexpected_keys, key)

    def forward(self, x):
        return self.detect_from_intensity(self.output_intensity(x))

    def detect_from_intensity(self, intensity):
        batch = intensity.shape[0]
        class_intensities = torch.zeros(batch, self.num_classes, device=intensity.device)
        for i in range(self.num_classes):
            mask = self.detector_masks[i]
            class_intensities[:, i] = (intensity * mask.unsqueeze(0)).sum(dim=(-2, -1))
        return class_intensities

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
        wavelength=0.75e-3,
        layer_distance=4e-3,
        pixel_size=0.3e-3,
        input_fraction=0.5,
        activation_type="none",
        activation_positions=None,
        activation_hparams=None,
    ):
        super().__init__(
            num_layers=num_layers,
            size=size,
            wavelength=wavelength,
            layer_distance=layer_distance,
            pixel_size=pixel_size,
            activation_type=activation_type,
            activation_positions=activation_positions,
            activation_hparams=activation_hparams,
        )
        self.input_fraction = input_fraction

        self.register_buffer(
            "H_out",
            self._build_output_transfer(size, wavelength, layer_distance, pixel_size),
        )

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
