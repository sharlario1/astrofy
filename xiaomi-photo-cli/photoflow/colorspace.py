from __future__ import annotations

import cv2
import numpy as np


def _dt(x: np.ndarray):
    return np.float64 if x.dtype == np.float64 else np.float32


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(_dt(x))


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(_dt(x))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """float32 RGB [0,1] (sRGB) -> Lab (L 0..100). Accepts HxWx3 or Nx3."""
    a = np.ascontiguousarray(rgb, dtype=np.float32)
    flat = a.ndim == 2
    if flat:
        a = a.reshape(-1, 1, 3)
    out = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    return out.reshape(-1, 3) if flat else out


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(lab, dtype=np.float32)
    flat = a.ndim == 2
    if flat:
        a = a.reshape(-1, 1, 3)
    out = cv2.cvtColor(a, cv2.COLOR_LAB2RGB)
    out = np.clip(out, 0.0, 1.0)
    return out.reshape(-1, 3) if flat else out


def delta_e76(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    d = rgb_to_lab(rgb_a) - rgb_to_lab(rgb_b)
    return np.sqrt((d * d).sum(axis=-1))


_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab_precise(rgb: np.ndarray) -> np.ndarray:
    """Exact float64 sRGB(D65) -> Lab. Smooth (unlike OpenCV's table-based float path), which the
    calibration optimiser needs for clean finite-difference gradients. Nx3 in, Nx3 out."""
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64))
    xyz = (lin @ _M.T) / _WHITE
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)
