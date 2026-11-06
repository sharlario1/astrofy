"""Iris/sclera local contrast + fine sharpening, halo-free.

Halo control: (1) each region (iris / sclera) is processed against a *masked, normalised* blur so
the dark iris never leaks into the sclera reference (the usual halo source), (2) detail is soft
limited, (3) the result is clamped to the local min/max of the original (no overshoot ring).
"""
from __future__ import annotations

import cv2
import numpy as np

from .colorspace import lab_to_rgb, rgb_to_lab
from .masks import FaceMasks


def _soft(d, lim):
    return lim * np.tanh(d / lim)


def _masked_blur(L: np.ndarray, region: np.ndarray, sigma: float) -> np.ndarray:
    hard = (region > 0.05).astype(np.float32)
    num = cv2.GaussianBlur(L * hard, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(hard, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), L)


def apply_eyes(rgb: np.ndarray, masks: FaceMasks, cfg: dict, strength: float) -> np.ndarray:
    s = strength / 100.0
    region_union = np.maximum(masks.iris, masks.sclera)
    if region_union.max() < 1e-3:
        return rgb
    y0, y1, x0, x1 = masks.roi
    roi = rgb[y0:y1, x0:x1]
    lab = rgb_to_lab(roi)
    L = lab[..., 0]
    r = max(masks.iris_r, 2.0)
    s_lc = max(1.5, cfg["local_sigma"] * r)
    s_sh = float(np.clip(cfg["sharpen_sigma_frac"] * masks.face_w, cfg["sharpen_sigma_min"], cfg["sharpen_sigma_max"]))
    lim = cfg["detail_limit"]
    delta = np.zeros_like(L)
    for region, g_lc, g_sh in ((masks.iris, cfg["iris_local_contrast"], cfg["iris_sharpen"]),
                               (masks.sclera, cfg["sclera_local_contrast"], cfg["sclera_sharpen"])):
        if region.max() < 1e-3:
            continue
        d_lc = L - _masked_blur(L, region, s_lc)
        d_sh = L - _masked_blur(L, region, s_sh)
        delta += region * (g_lc * _soft(d_lc, lim) + g_sh * _soft(d_sh, lim))
    new = L + s * delta
    k = max(1, int(round(0.004 * masks.face_w)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    lo, hi = cv2.erode(L, ker), cv2.dilate(L, ker)
    new = np.clip(new, lo, hi)
    lab[..., 0] = new
    out_roi = lab_to_rgb(lab)
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.where((region_union > 1e-3)[..., None], out_roi, roi)
    return out
