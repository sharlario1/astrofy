from __future__ import annotations

import cv2
import numpy as np


def apply_grain(rgb: np.ndarray, cfg: dict, strength: float, seed: int) -> np.ndarray:
    s = strength / 100.0
    h, w = rgb.shape[:2]
    sigma = max(0.5, cfg["sigma_frac"] * max(h, w))
    rng = np.random.default_rng(seed ^ 0x6A09E667)
    n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    n /= max(float(n.std()), 1e-6)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    bell = 0.35 + 0.65 * np.clip(4 * luma * (1 - luma), 0, 1)      # less grain in deep shadows / clipped highlights
    delta = (cfg["amplitude"] * s / 100.0) * n * bell
    return np.clip(rgb + delta[..., None], 0, 1).astype(np.float32)
