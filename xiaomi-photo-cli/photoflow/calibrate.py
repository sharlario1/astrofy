"""Fit the tone model + LUT blend to original/edited pairs made in the phone editor, in Lab space.

Model (same order as the pipeline):  y = tone(x; ev, offset, gain);  out = y + beta*(LUT(y) - y)
Residual = Lab(out) - Lab(edited), robust (soft-L1) least squares over sampled pixels of
downscaled pairs (downscaling averages out the phone's local sharpening/retouching, which this
global model cannot represent anyway).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .colorspace import rgb_to_lab_precise as rgb_to_lab
from .imageio import IMAGE_EXTS, load_image
from .lut import Lut3D, apply_lut
from .pipeline import lut_blend_factor, resolve_lut
from .tone import ToneParams, apply_tone, tone_params

log = logging.getLogger("photoflow")


def pair_images(before: Path, after: Path) -> list[tuple[Path, Path]]:
    def index(d: Path):
        return {p.stem.lower(): p for p in sorted(Path(d).iterdir()) if p.suffix.lower() in IMAGE_EXTS}
    b, a = index(before), index(after)
    return [(b[k], a[k]) for k in sorted(b) if k in a]


def sample_pairs(pairs, max_side: int = 384, per_image: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, Y, used, skipped = [], [], [], []
    for pb, pa in pairs:
        try:
            ib, ia = load_image(pb).rgb, load_image(pa).rgb
        except Exception as e:  # noqa: BLE001
            skipped.append((pb.name, f"unreadable: {e}"))
            continue
        ab, aa = ib.shape[1] / ib.shape[0], ia.shape[1] / ia.shape[0]
        if abs(ab / aa - 1) > 0.02:
            skipped.append((pb.name, f"aspect differs ({ab:.3f} vs {aa:.3f}): the phone cropped/rotated it"))
            continue
        s = min(1.0, max_side / max(ia.shape[:2]))
        size = (max(8, round(ia.shape[1] * s)), max(8, round(ia.shape[0] * s)))
        ib = cv2.resize(ib, size, interpolation=cv2.INTER_AREA)
        ia = cv2.resize(ia, size, interpolation=cv2.INTER_AREA)
        n = ib.shape[0] * ib.shape[1]
        idx = rng.choice(n, size=min(per_image, n), replace=False)
        X.append(ib.reshape(-1, 3)[idx])
        Y.append(ia.reshape(-1, 3)[idx])
        used.append(pb.name)
    if not X:
        raise ValueError("no usable pairs (match files by name: same stem in BEFORE_DIR and AFTER_DIR)")
    return np.concatenate(X).astype(np.float64), np.concatenate(Y).astype(np.float64), used, skipped


def model(X: np.ndarray, theta, pivot: float, lut: Lut3D | None) -> np.ndarray:
    ev, off, gain = theta[:3]
    y = apply_tone(X, ToneParams(ev, off, gain, pivot))
    if lut is not None and len(theta) > 3 and theta[3] > 0:
        y = np.clip(y + theta[3] * (apply_lut(y, lut) - y), 0, 1)
    return y


def de76_stats(A: np.ndarray, B: np.ndarray) -> dict:
    d = rgb_to_lab(A) - rgb_to_lab(B)
    e = np.sqrt((d ** 2).sum(1))
    return {"mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
            "mean_abs_dL": float(np.abs(d[:, 0]).mean())}


@dataclass
class CalibrationResult:
    ev: float
    offset: float
    gain: float
    beta: float | None
    per_pt: dict
    lut_strength: float | None
    before: dict
    after: dict
    identity: dict
    n_pairs: int
    n_pixels: int
    skipped: list
    notes: list


def calibrate(before_dir: Path, after_dir: Path, preset: dict, max_side: int = 384, per_image: int = 3000,
              seed: int = 0) -> CalibrationResult:
    pairs = pair_images(before_dir, after_dir)
    X, Y, used, skipped = sample_pairs(pairs, max_side, per_image, seed)
    notes: list[str] = []
    tn, ln = preset["steps"]["tone"], preset["steps"]["lut_nat"]
    pivot = tn["mapping"].get("pivot", 0.5)
    lut, _ = resolve_lut(preset)
    fit_lut = not lut.is_placeholder
    if not fit_lut:
        notes.append("No real LUT configured: LUT blend NOT fitted and not written; the tone fit absorbs "
                     "whatever the phone's NAT filter did. Run make-hald/lut-from-hald first for a clean fit.")
    lut_use = lut if fit_lut else None

    p0 = tone_params(tn)
    beta0 = lut_blend_factor(ln, False) if fit_lut else 0.0
    start_preset = np.array([p0.ev, p0.offset, p0.gain] + ([beta0] if fit_lut else []))
    lo = [-1.5, -0.25, 0.5] + ([0.0] if fit_lut else [])
    hi = [1.5, 0.25, 2.0] + ([2.0] if fit_lut else [])
    Ylab = rgb_to_lab(Y)
    sq = np.sqrt(len(X))

    prior_scale = np.array([0.1, 0.02, 0.1] + ([0.5] if fit_lut else []))
    LAM = 0.05      # weak ridge toward the preset: only breaks ties between near-equivalent EV/offset/gain combos

    def resid(th):
        data = ((rgb_to_lab(model(X, th, pivot, lut_use)) - Ylab) / sq).ravel()
        return np.concatenate([data, LAM * (th - start_preset) / prior_scale])

    starts = [start_preset, np.array([0, 0, 1] + ([0.5] if fit_lut else []), float)]
    best = None
    for s0 in starts:
        s0 = np.clip(s0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(resid, s0, bounds=(lo, hi), loss="soft_l1", f_scale=3.0 / sq, xtol=1e-12, ftol=1e-12, gtol=1e-12,
                          x_scale=[0.1, 0.02, 0.1] + ([0.3] if fit_lut else []))
        if best is None or r.cost < best.cost:
            best = r
    th = best.x
    before = de76_stats(model(X, start_preset, pivot, lut_use), Y)
    after = de76_stats(model(X, th, pivot, lut_use), Y)
    identity = de76_stats(X, Y)

    pts, scale = tn["ui_points"], float(tn["strength"]) / 100.0
    per_pt = {}
    for key, val, name, base in (("exposure", th[0], "exposure_ev_per_pt", 0.0),
                                 ("brightness", th[1], "brightness_per_pt", 0.0),
                                 ("contrast", th[2] - 1.0, "contrast_per_pt", 0.0)):
        denom = pts[key] * scale
        if abs(denom) < 1e-9:
            notes.append(f"ui_points.{key} is 0, mapping.{name} left unchanged")
            continue
        per_pt[name] = float(val / denom)
    beta = float(th[3]) if fit_lut else None
    lut_strength = float(np.clip(beta * max(ln["lut_baked_strength"], 1.0) / 1.0, 0, 100)) if fit_lut else None
    if fit_lut and beta * ln["lut_baked_strength"] > 100:
        notes.append("fitted LUT blend exceeds 100% of the baked LUT; strength clipped to 100 "
                     "(set allow_extrapolation or re-bake the Hald at a higher NAT strength)")
    if after["mean"] > 4.0:
        notes.append(f"residual dE76 {after['mean']:.1f} is high: the phone edit has local/non-global components "
                     "(AI Beautify smoothing, face retouch, auto-HDR) that a global model cannot reproduce.")
    if len(used) < 8:
        notes.append("fewer than 8 usable pairs: the fit may overfit; use 20+ varied photos.")
    notes.append("With a single recipe the three slider constants are only identified as a combined effect "
                 "(shared proportionally); change slider values later and they will extrapolate approximately.")
    return CalibrationResult(float(th[0]), float(th[1]), float(th[2]), beta, per_pt, lut_strength,
                             before, after, identity, len(used), len(X), skipped, notes)
