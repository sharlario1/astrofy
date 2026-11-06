"""Fixed-order pipeline: crop -> [img2img] -> skin_texture -> eyes -> tone -> lut_nat -> grain."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .config import resolve_path
from .crop import crop_box
from .eyes import apply_eyes
from .grain import apply_grain
from .lut import blend_lut, load_lut
from .masks import compute_face_masks
from .skin import apply_skin_texture
from .tone import apply_tone, tone_params

log = logging.getLogger("photoflow")
_warned: set[str] = set()


@dataclass
class PipelineResult:
    rgb: np.ndarray
    alpha: np.ndarray | None
    n_faces: int = 0
    applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: bool = False
    timings: dict = field(default_factory=dict)


def lut_blend_factor(cfg: dict, is_placeholder: bool) -> float:
    """Effective blend of LUT(x) into x. target strength / strength already baked into the LUT."""
    target = float(cfg["strength"]) / 100.0
    baked = 1.0 if is_placeholder else max(float(cfg["lut_baked_strength"]), 1.0) / 100.0
    b = target / baked
    return float(np.clip(b, 0.0, 2.0 if cfg["allow_extrapolation"] else 1.0))


def resolve_lut(preset: dict):
    cfg = preset["steps"]["lut_nat"]
    path = resolve_path(cfg.get("lut_path"), preset)
    if path is not None and not path.exists():
        raise FileNotFoundError(f"steps.lut_nat.lut_path not found: {path}")
    return load_lut(path), path


def _active(cfg: dict) -> bool:
    return bool(cfg.get("enabled", True)) and float(cfg.get("strength", 100)) > 0


def process_array(rgb: np.ndarray, alpha, preset: dict, detector, parser, seed: int) -> PipelineResult:
    steps = preset["steps"]
    res = PipelineResult(rgb=rgb, alpha=alpha)
    t = time.perf_counter()

    def tick(name):
        nonlocal t
        now = time.perf_counter()
        res.timings[name] = round(now - t, 3)
        t = now

    # 1. crop
    c = steps["crop"]
    if _active(c):
        y0, y1, x0, x1 = crop_box(*res.rgb.shape[:2], c["per_side_pct"], c["strength"])
        res.rgb = np.ascontiguousarray(res.rgb[y0:y1, x0:x1])
        if res.alpha is not None:
            res.alpha = np.ascontiguousarray(res.alpha[y0:y1, x0:x1])
        res.applied.append("crop")
    tick("crop")

    # optional img2img (ComfyUI)
    i2 = steps["img2img"]
    if _active(i2):
        from .comfy import img2img_step
        wf = resolve_path(i2["workflow"], preset)
        res.rgb = img2img_step(res.rgb, i2, i2["strength"], seed, wf)
        res.applied.append("img2img")
        tick("img2img")

    # faces
    faces = detector.detect(res.rgb) if (detector is not None) else []
    res.n_faces = len(faces)
    tick("detect")
    if not faces:
        res.notes.append("no_face")
        if preset["face"]["on_no_face"] == "skip":
            res.skipped = True
            return res
    H, W = res.rgb.shape[:2]
    all_masks = []
    for f in faces:
        m = compute_face_masks(f, (H, W), steps["mask"])
        if parser is not None:
            y0, y1, x0, x1 = m.roi
            m.skin = m.skin * parser.skin_mask(res.rgb[y0:y1, x0:x1])
        all_masks.append(m)
    tick("masks")

    # 2. skin texture
    st = steps["skin_texture"]
    if faces and _active(st):
        for i, m in enumerate(all_masks):
            res.rgb = apply_skin_texture(res.rgb, m, st, st["strength"], (seed + 7919 * i) & 0xFFFFFFFF)
        res.applied.append("skin_texture")
    tick("skin_texture")

    # 3. eyes
    ey = steps["eyes"]
    if faces and _active(ey):
        for m in all_masks:
            res.rgb = apply_eyes(res.rgb, m, ey, ey["strength"])
        res.applied.append("eyes")
    tick("eyes")

    # 4. tone
    tn = steps["tone"]
    if _active(tn):
        res.rgb = apply_tone(res.rgb, tone_params(tn))
        res.applied.append("tone")
    tick("tone")

    # 5. LUT
    ln = steps["lut_nat"]
    if _active(ln):
        lut, path = resolve_lut(preset)
        if lut.is_placeholder:
            res.notes.append("placeholder_lut")
            if "placeholder_lut" not in _warned:
                _warned.add("placeholder_lut")
                log.warning("lut_nat: no LUT configured -> using the PLACEHOLDER approximation curve "
                            "(not the Xiaomi NAT look). See README: make-hald / lut-from-hald.")
        res.rgb = blend_lut(res.rgb, lut, lut_blend_factor(ln, lut.is_placeholder))
        res.applied.append("lut_nat")
    tick("lut_nat")

    # 6. grain
    gr = steps["grain"]
    if _active(gr):
        res.rgb = apply_grain(res.rgb, gr, gr["strength"], seed)
        res.applied.append("grain")
    tick("grain")
    return res
