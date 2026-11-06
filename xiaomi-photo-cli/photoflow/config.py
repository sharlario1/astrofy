"""Preset loading, defaults, validation, per-photo jitter and config hashing."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from . import PIPELINE_VERSION

log = logging.getLogger("photoflow")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESET_DIR = PROJECT_ROOT / "presets"

STEP_ORDER = ["crop", "img2img", "skin_texture", "eyes", "tone", "lut_nat", "grain"]

DEFAULTS: dict[str, Any] = {
    "name": "default",
    "output": {
        "format": "auto",            # auto (png->png, everything else->jpeg) | jpeg | png
        "jpeg_quality": 95,
        "jpeg_subsampling": 0,       # 0 = 4:4:4
        "png_compress_level": 6,
    },
    "face": {
        "max_faces": 4,
        "min_confidence": 0.5,
        "detect_max_side": 1600,     # detection runs on a downscaled copy
        "model_path": None,          # default: models/face_landmarker.task
        "parsing_onnx": None,        # optional face-parsing ONNX (user supplied); see README
        "on_no_face": "global_only", # global_only (crop/tone/LUT/grain still run) | skip (no output)
    },
    "steps": {
        "crop": {"enabled": True, "strength": 100, "per_side_pct": 4.0},
        "img2img": {
            "enabled": False, "strength": 100,
            "server": "127.0.0.1:8188",
            "workflow": "workflows/img2img_lowdenoise_example.json",
            "checkpoint": "", "denoise": 0.18, "max_side": 1024, "timeout_s": 600,
            "positive": "photo, natural skin texture, pores, detailed, sharp focus",
            "negative": "plastic skin, smooth, airbrushed, blurry, painting, cgi",
        },
        "skin_texture": {
            "enabled": True, "strength": 100,
            # All radii are fractions of face width (oval bbox) -> resolution independent
            "fine_sigma": 0.0035, "medium_sigma": 0.010,
            "hf_gain_fine": 0.8, "hf_gain_medium": 0.4, "hf_limit": 6.0,   # L* units
            "pore_sigma": 0.0016, "micro_sigma": 0.0013, "micro_sigma2": 0.0030,
            "pore_amplitude": 2.0, "micro_amplitude": 1.2,                 # L* units
            "chroma_variation": 0.5, "chroma_sigma": 0.006,
            "luminance_modulation": 0.75,   # 0 = flat, 1 = full midtone bell curve
            "smooth_bias": 0.7,             # 0 = ignore existing texture, 1 = add only where skin is smooth
            "min_sigma_px": 0.6,
        },
        "mask": {  # skin / eye mask geometry (fractions of face width)
            "oval_erode": 0.020, "feather": 0.012,
            "eye_dilate": 0.012, "brow_dilate": 0.008, "lip_dilate": 0.015,
        },
        "eyes": {
            "enabled": True, "strength": 100,
            "iris_local_contrast": 0.35, "iris_sharpen": 0.7,
            "sclera_local_contrast": 0.12, "sclera_sharpen": 0.35,
            "local_sigma": 0.8,          # x iris radius
            "sharpen_sigma_frac": 0.0016, "sharpen_sigma_min": 0.6, "sharpen_sigma_max": 1.6,
            "detail_limit": 5.0,         # soft clip, L* units
            "sclera_erode": 0.20,        # x iris radius, keeps lids/lashes out
        },
        "tone": {
            "enabled": True, "strength": 66,      # "AI Beautify" intensity 0-100 -> scale 0.66
            "ui_points": {"exposure": 3.0, "brightness": 15.0, "contrast": 15.0},
            # Slider-point -> physical mapping. PLACEHOLDER GUESSES until `calibrate` runs.
            "mapping": {
                "exposure_ev_per_pt": 0.02,    # EV per slider point
                "brightness_per_pt": 0.0025,   # additive, display-referred [0..1] per point
                "contrast_per_pt": 0.004,      # gain about pivot = 1 + k*points
                "pivot": 0.5,
            },
        },
        "lut_nat": {
            "enabled": True, "strength": 50,       # target effective blend, percent
            "lut_path": None,                      # .cube or Hald PNG; None -> placeholder curve
            # Strength the LUT was ALREADY baked at on the phone. 100 = full-strength NAT.
            # If you processed the Hald at NAT 50, set 50 here, otherwise the 50% is applied twice.
            "lut_baked_strength": 100,
            "allow_extrapolation": False,
        },
        "grain": {"enabled": False, "strength": 20, "sigma_frac": 0.0004, "amplitude": 2.5},
    },
    "jitter": {
        "enabled": False, "seed": 0,
        # +/- relative variation of each step strength, per photo (deterministic per file hash)
        "relative": {"skin_texture": 0.10, "eyes": 0.10, "tone": 0.05, "lut_nat": 0.08, "grain": 0.25},
        "tone_points": 1.0,   # +/- absolute slider points on exposure/brightness/contrast
    },
    "qc": {
        "blur_min": 25.0,           # Laplacian variance of the 256px face crop
        "bg_noise_max": 2.0,        # robust noise sigma (0-255) outside the face
        "bg_skin_ratio_max": 1.6,   # background noise / skin noise
        "bg_noise_floor": 1.0,      # ratio test only applies above this
        "max_faces_ok": 1,
    },
    "calibration": {"date": None, "n_pairs": None, "de76_mean_before": None, "de76_mean_after": None},
}


def deep_merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            log.warning("preset: unknown key '%s%s' (ignored by the pipeline)", path, k)
            out[k] = v
        elif isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


def validate(p: dict) -> None:
    for name, s in p["steps"].items():
        if "strength" in s and not (0 <= float(s["strength"]) <= 100):
            raise ValueError(f"steps.{name}.strength must be 0-100, got {s['strength']}")
    if not (0 <= p["steps"]["crop"]["per_side_pct"] < 40):
        raise ValueError("steps.crop.per_side_pct must be in [0, 40)")
    if p["output"]["format"] not in ("auto", "jpeg", "png"):
        raise ValueError("output.format must be auto|jpeg|png")
    if p["face"]["on_no_face"] not in ("global_only", "skip"):
        raise ValueError("face.on_no_face must be global_only|skip")
    if p["output"]["jpeg_subsampling"] not in (0, 1, 2):
        raise ValueError("output.jpeg_subsampling must be 0 (4:4:4), 1 (4:2:2) or 2 (4:2:0)")


def resolve_preset_path(preset: str | Path) -> Path:
    p = Path(preset)
    if p.suffix.lower() in (".yaml", ".yml") or p.exists():
        if p.exists():
            return p
    cand = PRESET_DIR / f"{preset}.yaml"
    if cand.exists():
        return cand
    raise FileNotFoundError(f"Preset '{preset}' not found (looked for {p} and {cand})")


def load_preset(preset: str | Path | None) -> dict:
    if preset is None:
        p = copy.deepcopy(DEFAULTS)
    else:
        path = resolve_preset_path(preset)
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        p = deep_merge(DEFAULTS, data)
        p["_path"] = str(path)
        # relative LUT / model / workflow paths are relative to the preset file's project root
    validate(p)
    return p


def resolve_path(value: str | None, preset: dict) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    for base in (Path.cwd(), PROJECT_ROOT):
        if (base / p).exists():
            return base / p
    return PROJECT_ROOT / p


def apply_jitter(preset: dict, file_sha256: str) -> dict:
    """Return an effective preset with deterministic per-photo random variation."""
    j = preset["jitter"]
    if not j.get("enabled"):
        return preset
    eff = copy.deepcopy(preset)
    rng = np.random.default_rng((int(file_sha256[:16], 16) ^ int(j.get("seed", 0))) & 0xFFFFFFFFFFFFFFFF)
    for step, rel in j["relative"].items():
        if step in eff["steps"] and "strength" in eff["steps"][step]:
            f = 1.0 + rng.uniform(-rel, rel)
            eff["steps"][step]["strength"] = float(np.clip(eff["steps"][step]["strength"] * f, 0, 100))
    tp = float(j.get("tone_points", 0.0))
    if tp:
        for k in eff["steps"]["tone"]["ui_points"]:
            eff["steps"]["tone"]["ui_points"][k] += float(rng.uniform(-tp, tp))
    return eff


def config_hash(preset: dict, extra_files: list[Path | None] = ()) -> str:
    """Hash of everything that influences pixels: effective preset, LUT bytes, pipeline version."""
    clean = {k: v for k, v in preset.items() if k not in ("_path", "name", "calibration", "qc")}
    h = hashlib.sha256()
    h.update(PIPELINE_VERSION.encode())
    h.update(json.dumps(clean, sort_keys=True, default=str).encode())
    for f in extra_files:
        if f and Path(f).exists():
            h.update(Path(f).read_bytes())
    return h.hexdigest()


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()
