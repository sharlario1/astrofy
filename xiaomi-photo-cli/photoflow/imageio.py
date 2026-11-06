"""Image loading/saving with metadata passthrough.

Metadata policy: bytes of EXIF / ICC / XMP (and PNG text chunks) are copied from the
input untouched. The single exception: if EXIF Orientation != 1 the pixels are physically
rotated upright (face detection needs it) and the Orientation tag is reset to 1, otherwise
viewers would rotate twice. Nothing is ever added (no Software tag, no fake camera EXIF).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, PngImagePlugin

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ORIENTATION_TAG = 0x0112


@dataclass
class LoadedImage:
    rgb: np.ndarray                    # float32 HxWx3 in [0,1], encoded (sRGB-ish) values
    alpha: np.ndarray | None = None    # float32 HxW or None
    meta: dict = field(default_factory=dict)
    src_format: str = ""
    orientation_fixed: bool = False


def _to_float(img: Image.Image) -> np.ndarray:
    if img.mode in ("I;16", "I;16L", "I;16B", "I"):
        return (np.asarray(img, dtype=np.float32) / 65535.0)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_image(path: Path) -> LoadedImage:
    with Image.open(path) as im:
        fmt = (im.format or "").upper()
        info = dict(im.info)
        exif_bytes = info.get("exif")
        orientation = 1
        try:
            orientation = int(im.getexif().get(ORIENTATION_TAG, 1) or 1)
        except Exception:
            pass
        meta: dict = {}
        if exif_bytes:
            meta["exif"] = bytes(exif_bytes)
        if info.get("icc_profile"):
            meta["icc_profile"] = info["icc_profile"]
        if info.get("xmp"):
            meta["xmp"] = info["xmp"]
        if info.get("dpi"):
            meta["dpi"] = info["dpi"]
        if fmt == "PNG":
            text = {k: v for k, v in getattr(im, "text", {}).items() if isinstance(v, str)}
            if text:
                meta["png_text"] = text
        fixed = False
        if orientation != 1:
            im = ImageOps.exif_transpose(im)
            fixed = True
            if exif_bytes:
                ex = Image.Exif()
                ex.load(exif_bytes)
                ex[ORIENTATION_TAG] = 1
                meta["exif"] = ex.tobytes()
        alpha = None
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.0
            rgb_img = rgba.convert("RGB")
        elif im.mode in ("I;16", "I;16L", "I;16B", "I"):
            g = _to_float(im)
            rgb_img = None
            rgb = np.repeat(g[..., None], 3, axis=2)
        else:
            rgb_img = im.convert("RGB")
        if rgb_img is not None:
            rgb = _to_float(rgb_img)
    return LoadedImage(rgb=np.ascontiguousarray(rgb, dtype=np.float32), alpha=alpha,
                       meta=meta, src_format=fmt, orientation_fixed=fixed)


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def resolve_format(fmt: str, src_format: str, has_alpha: bool) -> str:
    if has_alpha:
        return "png"
    if fmt == "auto":
        return "png" if src_format == "PNG" else "jpeg"
    return fmt


def save_image(path: Path, rgb: np.ndarray, meta: dict, fmt: str, out_cfg: dict,
               alpha: np.ndarray | None = None) -> None:
    """Atomic write (tmp + os.replace) so an interrupted run never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arr = to_uint8(rgb)
    if fmt == "jpeg":
        im = Image.fromarray(arr, "RGB")
        kw = dict(format="JPEG", quality=int(out_cfg["jpeg_quality"]),
                  subsampling=int(out_cfg["jpeg_subsampling"]), optimize=True)
        for k in ("exif", "icc_profile", "xmp", "dpi"):
            if k in meta:
                kw[k] = meta[k]
        im.save(tmp, **kw)
    else:
        if alpha is not None:
            im = Image.fromarray(np.dstack([arr, to_uint8(alpha)]), "RGBA")
        else:
            im = Image.fromarray(arr, "RGB")
        kw = dict(format="PNG", compress_level=int(out_cfg["png_compress_level"]))
        for k in ("exif", "icc_profile", "dpi"):
            if k in meta:
                kw[k] = meta[k]
        if meta.get("png_text"):
            pi = PngImagePlugin.PngInfo()
            for k, v in meta["png_text"].items():
                pi.add_text(k, v)
            kw["pnginfo"] = pi
        im.save(tmp, **kw)
    os.replace(tmp, path)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                  and ".photoflow" not in p.parts)
