"""Skin texture: frequency separation + procedural micro-texture, inside the soft skin mask only."""
from __future__ import annotations

import cv2
import numpy as np

from .colorspace import lab_to_rgb, rgb_to_lab
from .masks import FaceMasks


def _g(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def _unit_noise(rng, shape, sigma: float) -> np.ndarray:
    n = _g(rng.standard_normal(shape).astype(np.float32), sigma)
    return n / max(float(n.std()), 1e-6)


def _soft(d: np.ndarray, lim: float) -> np.ndarray:
    return lim * np.tanh(d / lim)


def apply_skin_texture(rgb: np.ndarray, masks: FaceMasks, cfg: dict, strength: float, seed: int) -> np.ndarray:
    s = strength / 100.0
    y0, y1, x0, x1 = masks.roi
    roi = rgb[y0:y1, x0:x1]
    m = masks.skin * s
    if m.max() < 1e-4:
        return rgb
    S = masks.face_w
    sig = lambda frac: max(cfg["min_sigma_px"], frac * S)
    sf, sm = sig(cfg["fine_sigma"]), sig(cfg["medium_sigma"])
    lab = rgb_to_lab(roi)
    L = lab[..., 0]
    rng = np.random.default_rng(seed)

    # 1) frequency separation: boost existing fine + medium detail (soft-limited -> no halos/overshoot)
    g1, g2 = _g(L, sf), _g(L, sm)
    lim = cfg["hf_limit"]
    boost = cfg["hf_gain_fine"] * _soft(L - g1, lim) + cfg["hf_gain_medium"] * _soft(g1 - g2, lim)

    # 2) procedural micro-texture: grain-like micro-relief + pore dots (bandpass noise, dark lobes only)
    micro = 0.65 * _unit_noise(rng, L.shape, sig(cfg["micro_sigma"])) + 0.35 * _unit_noise(rng, L.shape, sig(cfg["micro_sigma2"]))
    micro /= max(float(micro.std()), 1e-6)
    sp = sig(cfg["pore_sigma"])
    w = rng.standard_normal(L.shape).astype(np.float32)
    bp = _g(w, sp) - _g(w, 2 * sp)
    bp /= max(float(bp.std()), 1e-6)
    pores = -np.maximum(bp - 1.0, 0.0) + 0.0833          # zero-mean: E[relu(z-1)] = 0.0833
    tex = cfg["micro_amplitude"] * micro + cfg["pore_amplitude"] * pores

    # 3) modulation: midtones get most texture; smooth areas get more than already-textured ones
    Lc = np.clip(L / 100.0, 0, 1)
    lm = cfg["luminance_modulation"]
    w_lum = (1 - lm) + lm * np.clip(4 * Lc * (1 - Lc), 0, 1)
    energy = np.sqrt(np.maximum(_g((L - g1) ** 2, max(1.0, 2 * sm)), 0))
    sb = cfg["smooth_bias"]
    w_smooth = (1 - sb) + sb / (1 + (energy / 2.0) ** 2)

    lab[..., 0] = L + m * (boost + w_lum * w_smooth * tex)
    cv = cfg["chroma_variation"]
    if cv > 0:
        cs = sig(cfg["chroma_sigma"])
        lab[..., 1] += m * w_lum * cv * _unit_noise(rng, L.shape, cs)
        lab[..., 2] += m * w_lum * cv * 0.7 * _unit_noise(rng, L.shape, cs)
    out_roi = lab_to_rgb(lab)
    out = rgb.copy()
    out[y0:y1, x0:x1] = np.where((masks.skin > 1e-4)[..., None], out_roi, roi)
    return out
