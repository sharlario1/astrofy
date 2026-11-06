"""3D LUT engine: .cube / Hald PNG IO, identity Hald generation, trilinear application, placeholder curve.

Layout (shared by Hald and .cube): flat index = r + g*N + b*N^2 (red fastest), so a flat list
reshapes to table[b, g, r, 3]. A Hald of level L is an (L^3 x L^3) image holding N = L^2 cube nodes
per axis, row-major, so level 8 -> 512x512 px, N = 64.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("photoflow")


@dataclass
class Lut3D:
    table: np.ndarray                 # float32 (N, N, N, 3) indexed [b, g, r]
    domain_min: tuple = (0.0, 0.0, 0.0)
    domain_max: tuple = (1.0, 1.0, 1.0)
    title: str = ""
    is_placeholder: bool = False

    @property
    def size(self) -> int:
        return self.table.shape[0]


# ---------------------------------------------------------------- Hald
def identity_table(n: int) -> np.ndarray:
    g = np.linspace(0, 1, n, dtype=np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")     # axes: b, g, r
    return np.stack([r, gg, b], axis=-1)


def make_identity_hald(level: int = 8) -> Image.Image:
    n = level * level
    side = level ** 3
    flat = identity_table(n).reshape(-1, 3)
    arr = np.clip(flat * 255.0 + 0.5, 0, 255).astype(np.uint8).reshape(side, side, 3)
    return Image.fromarray(arr, "RGB")


def hald_level_from_side(side: int) -> int:
    level = round(side ** (1 / 3))
    if level ** 3 != side:
        raise ValueError(f"{side}px is not a Hald CLUT side (expected level^3, e.g. 512 for level 8)")
    return level


def read_hald(path: Path) -> Lut3D:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width != im.height:
            raise ValueError(f"Hald must be square, got {im.size}")
        level = hald_level_from_side(im.width)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    n = level * level
    return Lut3D(arr.reshape(n, n, n, 3), title=Path(path).stem)


# ---------------------------------------------------------------- .cube
def read_cube(path: Path) -> Lut3D:
    size = None
    dmin, dmax = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    title = ""
    rows: list[list[float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            tok = line.split()
            key = tok[0].upper()
            if key == "TITLE":
                title = line.partition(" ")[2].strip().strip('"')
            elif key == "LUT_3D_SIZE":
                size = int(tok[1])
            elif key == "LUT_1D_SIZE":
                raise ValueError("1D .cube LUTs are not supported (need LUT_3D_SIZE)")
            elif key == "DOMAIN_MIN":
                dmin = tuple(float(v) for v in tok[1:4])
            elif key == "DOMAIN_MAX":
                dmax = tuple(float(v) for v in tok[1:4])
            elif key[0] in "+-.0123456789":
                rows.append([float(v) for v in tok[:3]])
    if not size or len(rows) != size ** 3:
        raise ValueError(f"{path}: expected {size}^3 rows, found {len(rows)}")
    table = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)
    return Lut3D(table, dmin, dmax, title)


def write_cube(path: Path, lut: Lut3D, title: str | None = None, comments: list[str] = ()) -> None:
    n = lut.size
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for c in comments:
            fh.write(f"# {c}\n")
        fh.write(f'TITLE "{title or lut.title or "photoflow"}"\n')
        fh.write(f"LUT_3D_SIZE {n}\n")
        fh.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        np.savetxt(fh, lut.table.reshape(-1, 3), fmt="%.6f")


def smooth_lut(lut: Lut3D, sigma: float) -> Lut3D:
    """Gaussian-smooth the cube (in node units) to suppress JPEG/quantisation noise of a phone-processed Hald."""
    if sigma <= 0:
        return lut
    from scipy.ndimage import gaussian_filter
    t = np.stack([gaussian_filter(lut.table[..., c], sigma, mode="nearest") for c in range(3)], axis=-1)
    return Lut3D(t.astype(np.float32), lut.domain_min, lut.domain_max, lut.title)


# ---------------------------------------------------------------- apply
def apply_lut(rgb: np.ndarray, lut: Lut3D, chunk: int = 1_500_000) -> np.ndarray:
    """Trilinear interpolation. rgb float32 [..., 3] in [0,1] -> same shape."""
    shape = rgb.shape
    flat = rgb.reshape(-1, 3)
    n = lut.size
    tflat = lut.table.reshape(-1, 3)
    dmin = np.asarray(lut.domain_min, np.float32)
    span = np.asarray(lut.domain_max, np.float32) - dmin
    dt = np.float64 if flat.dtype == np.float64 else np.float32
    out = np.empty(flat.shape, dtype=dt)
    for s in range(0, flat.shape[0], chunk):
        x = (np.clip(flat[s:s + chunk], 0, 1) - dmin.astype(dt)) / span.astype(dt)
        p = np.clip(x, 0, 1) * (n - 1)
        i0 = np.minimum(p.astype(np.int32), n - 2)
        f = (p - i0).astype(dt)
        r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
        fr, fg, fb = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        base = r0 + n * g0 + n * n * b0
        c = lambda dr, dg, db: tflat[base + dr + n * dg + n * n * db]
        c00 = c(0, 0, 0) * (1 - fr) + c(1, 0, 0) * fr
        c10 = c(0, 1, 0) * (1 - fr) + c(1, 1, 0) * fr
        c01 = c(0, 0, 1) * (1 - fr) + c(1, 0, 1) * fr
        c11 = c(0, 1, 1) * (1 - fr) + c(1, 1, 1) * fr
        out[s:s + chunk] = (c00 * (1 - fg) + c10 * fg) * (1 - fb) + (c01 * (1 - fg) + c11 * fg) * fb
    return out.reshape(shape)


def blend_lut(rgb: np.ndarray, lut: Lut3D, blend: float) -> np.ndarray:
    """out = rgb + blend * (LUT(rgb) - rgb); blend=0.5 is the '50% with the original' of the spec."""
    if blend <= 0:
        return rgb
    return np.clip(rgb + blend * (apply_lut(rgb, lut) - rgb), 0.0, 1.0)


# ---------------------------------------------------------------- placeholder
# Global Lab model fitted to ONE reference pair (original vs the phone's NAT filter at 100), 1280x943 photo.
#   L' = L + curve(L) + kc*C/30 ;  a' = sa*a + da ;  b' = sb*b + db        (C = chroma)
# Findings that were stable across spatial hold-out splits: chroma x0.79 on both axes (desaturating, cooler),
# blacks lifted, lower mid-tones darker. Hold-out dE76 vs the phone result: 5.6-6.4 (untouched) -> 2.8-3.1.
# It is still ONE scene (skin, hair, foliage): colours not in it (saturated reds/blues, night shots) are
# extrapolated. Replace with make-hald -> phone -> lut-from-hald.
_NAT100_KNOTS = np.array([0, 15, 30, 50, 70, 85, 100.0])
_NAT100_LCURVE = np.array([2.33, -0.48, -3.71, -0.75, -1.16, -2.30, 0.56])
_NAT100_SA, _NAT100_SB, _NAT100_DA, _NAT100_DB, _NAT100_KC = 0.788, 0.786, 0.39, -0.20, -0.77


def nat100_estimate_lab(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    return np.stack([L + np.interp(L, _NAT100_KNOTS, _NAT100_LCURVE) + _NAT100_KC * C / 30.0,
                     a * _NAT100_SA + _NAT100_DA, b * _NAT100_SB + _NAT100_DB], axis=-1)


def placeholder_lut(size: int = 33) -> Lut3D:
    """Rough single-image estimate of NAT at strength 100 (see block comment above). NOT the real filter."""
    from .colorspace import lab_to_rgb, rgb_to_lab_precise
    t = identity_table(size)
    lab = nat100_estimate_lab(rgb_to_lab_precise(t.reshape(-1, 3)))
    out = lab_to_rgb(lab.astype(np.float32)).reshape(t.shape)
    return Lut3D(np.clip(out, 0, 1).astype(np.float32), title="placeholder-nat100-single-image-estimate", is_placeholder=True)


_CACHE: dict[tuple, Lut3D] = {}


def load_lut(path: Path | None) -> Lut3D:
    if path is None:
        key = ("placeholder",)
        if key not in _CACHE:
            _CACHE[key] = placeholder_lut()
        return _CACHE[key]
    p = Path(path)
    key = (str(p.resolve()), p.stat().st_mtime_ns)
    if key not in _CACHE:
        if p.suffix.lower() == ".cube":
            _CACHE[key] = read_cube(p)
        elif p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            _CACHE[key] = read_hald(p)
        else:
            raise ValueError(f"Unsupported LUT type: {p.suffix} (use .cube or Hald .png)")
    return _CACHE[key]
