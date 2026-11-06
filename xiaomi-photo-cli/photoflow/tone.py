"""'AI Beautify' tone model: exposure + brightness + contrast, scaled by intensity.

   lin   = srgb_to_linear(x) * 2^ev                     (exposure, photographic EV)
   y     = linear_to_srgb(lin) + offset                 (brightness, display-referred)
   y     = (y - pivot) * gain + pivot                   (contrast about a pivot)
with  ev     = exposure_ev_per_pt  * exposure_pts   * scale
      offset = brightness_per_pt   * brightness_pts * scale
      gain   = 1 + contrast_per_pt * contrast_pts   * scale,     scale = strength / 100.
The per-point constants live in the preset and are what `calibrate` fits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colorspace import linear_to_srgb, srgb_to_linear


@dataclass
class ToneParams:
    ev: float = 0.0
    offset: float = 0.0
    gain: float = 1.0
    pivot: float = 0.5


def tone_params(cfg: dict) -> ToneParams:
    s = float(cfg["strength"]) / 100.0
    pts, m = cfg["ui_points"], cfg["mapping"]
    return ToneParams(
        ev=m["exposure_ev_per_pt"] * pts["exposure"] * s,
        offset=m["brightness_per_pt"] * pts["brightness"] * s,
        gain=1.0 + m["contrast_per_pt"] * pts["contrast"] * s,
        pivot=m.get("pivot", 0.5),
    )


def apply_tone(rgb: np.ndarray, p: ToneParams) -> np.ndarray:
    if p.ev == 0.0 and p.offset == 0.0 and p.gain == 1.0:
        return rgb
    dt = np.float64 if rgb.dtype == np.float64 else np.float32
    y = linear_to_srgb(srgb_to_linear(rgb) * dt(2.0 ** p.ev)) if p.ev != 0.0 else np.clip(rgb, 0, 1)
    y = y + dt(p.offset)
    y = (y - p.pivot) * dt(p.gain) + p.pivot
    return np.clip(y, 0.0, 1.0).astype(dt)
