"""
Diffractive Deep Neural Network (D2NN)

Reproducing: "All-optical machine learning using diffractive deep neural networks"
Lin et al., Science 361, 1004-1008 (2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def propagate_output_field(u, layers, transfer_function):
    """Propagate a complex field through diffractive layers to the output plane."""
    for layer in layers:
        u = layer(u)

    spectrum = torch.fft.fft2(u)
    spectrum = spectrum * transfer_function.unsqueeze(0)
    return torch.fft.ifft2(spectrum)


class DiffractiveLayer(nn.Module):
    """Single diffractive layer with learnable phase modulation."""

    def __init__(self, size, wavelength, layer_distance, pixel_size):
        super().__init__()
        self.size = size
        self.wavelength = wavelength
        self.layer_distance = layer_distance
        self.pixel_size = pixel_size

        self.phase = nn.Parameter(torch.randn(size, size) * 0.05)
        self.register_buffer("H", self._build_transfer_function())

    def _build_transfer_function(self):
        fx = torch.fft.fftfreq(self.size, d=self.pixel_size)
        fy = torch.fft.fftfreq(self.size, d=self.pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")

        sq = (1.0 / self.wavelength) ** 2 - FX**2 - FY**2
        propagating = sq > 0
        kz = torch.sqrt(torch.clamp(sq, min=0))
        return torch.exp(1j * 2 * np.pi * kz * self.layer_distance) * propagating

    def forward(self, u):
        modulation = torch.exp(1j * self.phase)
        u = u * modulation.unsqueeze(0)

        spectrum = torch.fft.fft2(u)
        spectrum = spectrum * self.H.unsqueeze(0)
        return torch.fft.ifft2(spectrum)


class D2NN(nn.Module):
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
    ):
        super().__init__()
        self.size = size
        self.num_classes = num_classes

        self.layers = nn.ModuleList(
            [DiffractiveLayer(size, wavelength, layer_distance, pixel_size) for _ in range(num_layers)]
        )
        self.register_buffer(
            "H_out",
            self._build_output_transfer(size, wavelength, layer_distance, pixel_size),
        )
        self.register_buffer("detector_masks", self._build_detectors(size, num_classes, detector_size))

    @staticmethod
    def _build_output_transfer(size, wavelength, layer_distance, pixel_size):
        fx = torch.fft.fftfreq(size, d=pixel_size)
        fy = torch.fft.fftfreq(size, d=pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")
        sq = (1.0 / wavelength) ** 2 - FX**2 - FY**2
        propagating = sq > 0
        kz = torch.sqrt(torch.clamp(sq, min=0))
        return torch.exp(1j * 2 * np.pi * kz * layer_distance) * propagating

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
            half = detector_size // 2
            x_start = max(cx - half, 0)
            x_end = min(cx + half + 1, size)
            y_start = max(cy - half, 0)
            y_end = min(cy + half + 1, size)
            masks[i, x_start:x_end, y_start:y_end] = 1.0

        return masks

    def forward(self, x):
        return self.detect_from_intensity(self.output_intensity(x))

    def propagate_embedded_field(self, u):
        return propagate_output_field(u, self.layers, self.H_out)

    def propagate_field(self, x):
        return self.propagate_embedded_field(self._embed_input(x))

    @staticmethod
    def output_intensity_from_field(field):
        return torch.abs(field) ** 2

    def output_intensity(self, x):
        return self.output_intensity_from_field(self.propagate_field(x))

    def detect_from_intensity(self, intensity):
        batch = intensity.shape[0]
        class_intensities = torch.zeros(batch, self.num_classes, device=intensity.device)
        for i in range(self.num_classes):
            mask = self.detector_masks[i]
            class_intensities[:, i] = (intensity * mask.unsqueeze(0)).sum(dim=(-2, -1))
        return class_intensities

    def _embed_input(self, x):
        batch = x.shape[0]
        u = torch.zeros(batch, self.size, self.size, dtype=torch.cfloat, device=x.device)

        image = x.squeeze(1)
        target_size = self.size // 3
        image_resized = F.interpolate(
            image.unsqueeze(1), size=(target_size, target_size), mode="bilinear", align_corners=False
        ).squeeze(1)

        offset = (self.size - target_size) // 2
        u[:, offset : offset + target_size, offset : offset + target_size] = image_resized.to(torch.cfloat)
        return u

    def export_phase_masks(self, wrap=True):
        return collect_phase_masks(self.layers, wrap=wrap)


class D2NNImager(nn.Module):
    """D2NN variant for image reconstruction / imaging-lens experiments."""

    def __init__(
        self,
        num_layers=5,
        size=200,
        wavelength=0.75e-3,
        layer_distance=4e-3,
        pixel_size=0.3e-3,
        input_fraction=0.5,
    ):
        super().__init__()
        self.size = size
        self.input_fraction = input_fraction

        self.layers = nn.ModuleList(
            [DiffractiveLayer(size, wavelength, layer_distance, pixel_size) for _ in range(num_layers)]
        )
        self.register_buffer(
            "H_out",
            D2NN._build_output_transfer(size, wavelength, layer_distance, pixel_size),
        )

    def forward(self, x):
        intensity = self.output_intensity(x)
        max_vals = intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return intensity / max_vals

    def propagate_embedded_field(self, u):
        return propagate_output_field(u, self.layers, self.H_out)

    def propagate_field(self, x):
        return self.propagate_embedded_field(self._embed_input(x))

    @staticmethod
    def output_intensity_from_field(field):
        return torch.abs(field) ** 2

    def output_intensity(self, x):
        return self.output_intensity_from_field(self.propagate_field(x))

    def propagate(self, x):
        return self.output_intensity(x)

    def _embed_input(self, x):
        target_size = max(int(self.size * self.input_fraction), 1)
        return embed_amplitude_image(x, self.size, target_size=target_size)

    def build_target(self, x):
        target = self._embed_input(x).abs()
        max_vals = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return target / max_vals

    def export_phase_masks(self, wrap=True):
        return collect_phase_masks(self.layers, wrap=wrap)
