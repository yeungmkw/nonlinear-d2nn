"""
Diffractive Deep Neural Network (D2NN)

Reproducing: "All-optical machine learning using diffractive deep neural networks"
Lin et al., Science 361, 1004-1008 (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class DiffractiveLayer(nn.Module):
    """单个衍射层：每个像素点作为一个神经元，可学习的相位调制。"""

    def __init__(self, size, wavelength, layer_distance, pixel_size):
        """
        Args:
            size: 层的像素数 (size x size)
            wavelength: 工作波长 (m)
            layer_distance: 到下一层的距离 (m)
            pixel_size: 像素间距 (m)
        """
        super().__init__()
        self.size = size
        self.wavelength = wavelength
        self.layer_distance = layer_distance
        self.pixel_size = pixel_size

        # 可学习参数：每个神经元的相位值
        self.phase = nn.Parameter(torch.randn(size, size) * 0.05)

        # 预计算自由空间传播的传递函数 (Angular Spectrum Method)
        self.register_buffer("H", self._build_transfer_function())

    def _build_transfer_function(self):
        """角谱法 (Angular Spectrum Method) 的传递函数。"""
        k = 2 * np.pi / self.wavelength
        fx = torch.fft.fftfreq(self.size, d=self.pixel_size)
        fy = torch.fft.fftfreq(self.size, d=self.pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")

        # 传播相位: sqrt(k^2 - kx^2 - ky^2) * z
        sq = (1.0 / self.wavelength) ** 2 - FX**2 - FY**2
        # 倏逝波置零
        propagating = sq > 0
        kz = torch.sqrt(torch.clamp(sq, min=0))
        H = torch.exp(1j * 2 * np.pi * kz * self.layer_distance) * propagating
        return H

    def forward(self, u):
        """
        前向传播：相位调制 -> 自由空间衍射传播。

        Args:
            u: 输入复数光场 (batch, size, size)
        Returns:
            传播后的复数光场 (batch, size, size)
        """
        # 相位调制
        modulation = torch.exp(1j * self.phase)  # (size, size)
        u = u * modulation.unsqueeze(0)  # (batch, size, size)

        # 角谱传播
        U = torch.fft.fft2(u)
        U = U * self.H.unsqueeze(0)
        u = torch.fft.ifft2(U)
        return u


class D2NN(nn.Module):
    """完整的衍射深度神经网络。"""

    def __init__(
        self,
        num_layers=5,
        size=200,
        num_classes=10,
        wavelength=0.75e-3,       # 0.4 THz -> 0.75 mm
        layer_distance=30e-3,     # 3 cm
        pixel_size=0.4e-3,        # 0.4 mm
        detector_size=None,
    ):
        """
        Args:
            num_layers: 衍射层数
            size: 每层像素分辨率 (size x size)
            num_classes: 分类数
            wavelength: 工作波长 (m)
            layer_distance: 层间距 (m)
            pixel_size: 像素间距 (m)
            detector_size: 每个检测器区域的像素数 (自动计算)
        """
        super().__init__()
        self.size = size
        self.num_classes = num_classes

        # 构建衍射层
        self.layers = nn.ModuleList([
            DiffractiveLayer(size, wavelength, layer_distance, pixel_size)
            for _ in range(num_layers)
        ])

        # 最后一层到输出平面的传播 (无调制)
        self.register_buffer(
            "H_out",
            self._build_output_transfer(size, wavelength, layer_distance, pixel_size),
        )

        # 检测器区域：将输出平面划分为 num_classes 个区域
        self.register_buffer("detector_masks", self._build_detectors(size, num_classes, detector_size))

    @staticmethod
    def _build_output_transfer(size, wavelength, layer_distance, pixel_size):
        k = 2 * np.pi / wavelength
        fx = torch.fft.fftfreq(size, d=pixel_size)
        fy = torch.fft.fftfreq(size, d=pixel_size)
        FX, FY = torch.meshgrid(fx, fy, indexing="ij")
        sq = (1.0 / wavelength) ** 2 - FX**2 - FY**2
        propagating = sq > 0
        kz = torch.sqrt(torch.clamp(sq, min=0))
        H = torch.exp(1j * 2 * np.pi * kz * layer_distance) * propagating
        return H

    @staticmethod
    def _build_detectors(size, num_classes, detector_size=None):
        """
        在输出平面上创建环形排列的检测器区域。

        论文中 10 个检测器排列在输出平面上（类似钟表刻度）。
        每个检测器是一个方形区域，收集该区域的总光强。
        """
        if detector_size is None:
            detector_size = max(size // 15, 3)

        masks = torch.zeros(num_classes, size, size)
        center = size // 2
        radius = size // 4  # 检测器到中心的距离

        for i in range(num_classes):
            angle = 2 * np.pi * i / num_classes - np.pi / 2  # 从12点钟方向开始
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
        """
        Args:
            x: 输入图像 (batch, 1, H, W)，值域 [0, 1]
        Returns:
            class_intensities: 各检测器区域的总光强 (batch, num_classes)
        """
        batch = x.shape[0]

        # 将输入图像嵌入到网络输入平面中央（振幅编码）
        u = self._embed_input(x)

        # 逐层衍射传播
        for layer in self.layers:
            u = layer(u)

        # 最后传播到输出平面
        U = torch.fft.fft2(u)
        U = U * self.H_out.unsqueeze(0)
        u = torch.fft.ifft2(U)

        # 计算输出平面的光强
        intensity = torch.abs(u) ** 2  # (batch, size, size)

        # 各检测器区域收集光强
        class_intensities = torch.zeros(batch, self.num_classes, device=x.device)
        for i in range(self.num_classes):
            mask = self.detector_masks[i]  # (size, size)
            class_intensities[:, i] = (intensity * mask.unsqueeze(0)).sum(dim=(-2, -1))

        return class_intensities

    def _embed_input(self, x):
        """将输入图像 (batch, 1, 28, 28) 嵌入到 (batch, size, size) 的复数光场中。"""
        batch = x.shape[0]
        u = torch.zeros(batch, self.size, self.size, dtype=torch.cfloat, device=x.device)

        # 将 28x28 放缩到合适大小，嵌入中央
        img = x.squeeze(1)  # (batch, 28, 28)
        target_size = self.size // 3  # 输入占网络面积的约 1/3
        img_resized = F.interpolate(
            img.unsqueeze(1), size=(target_size, target_size), mode="bilinear", align_corners=False
        ).squeeze(1)

        offset = (self.size - target_size) // 2
        u[:, offset : offset + target_size, offset : offset + target_size] = img_resized.to(torch.cfloat)
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
        intensity = self.propagate(x)
        max_vals = intensity.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return intensity / max_vals

    def propagate(self, x):
        u = self._embed_input(x)
        for layer in self.layers:
            u = layer(u)

        U = torch.fft.fft2(u)
        U = U * self.H_out.unsqueeze(0)
        u = torch.fft.ifft2(U)
        return torch.abs(u) ** 2

    def _embed_input(self, x):
        target_size = max(int(self.size * self.input_fraction), 1)
        return embed_amplitude_image(x, self.size, target_size=target_size)

    def build_target(self, x):
        target = self._embed_input(x).abs()
        max_vals = target.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return target / max_vals

    def export_phase_masks(self, wrap=True):
        return collect_phase_masks(self.layers, wrap=wrap)
