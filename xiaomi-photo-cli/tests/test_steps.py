import cv2
import numpy as np

from photoflow.config import DEFAULTS
from photoflow.eyes import apply_eyes
from photoflow.face import Face
from photoflow.lut import (Lut3D, apply_lut, blend_lut, identity_table, make_identity_hald, read_cube, read_hald,
                           write_cube)
from photoflow.masks import compute_face_masks
from photoflow.pipeline import lut_blend_factor
from photoflow.skin import apply_skin_texture
from photoflow.synthetic import draw_synthetic_portrait, synthetic_landmarks
from photoflow.tone import ToneParams, apply_tone, tone_params

W, H = 768, 1024


def _setup():
    rgb = draw_synthetic_portrait(W, H).astype(np.float32) / 255
    f = Face(synthetic_landmarks(W, H))
    return rgb, compute_face_masks(f, (H, W), DEFAULTS["steps"]["mask"])


def _hf_std(a, sel):
    g = a.mean(2)
    return float((g - cv2.GaussianBlur(g, (0, 0), 1.5))[sel].std())


def test_skin_texture_adds_detail_only_inside_mask():
    rgb, m = _setup()
    out = apply_skin_texture(rgb, m, DEFAULTS["steps"]["skin_texture"], 100, 42)
    y0, y1, x0, x1 = m.roi
    full = np.zeros((H, W), np.float32)
    full[y0:y1, x0:x1] = m.skin
    assert np.abs(out - rgb)[full < 1e-4].max() < 1e-6          # untouched outside
    inside = full > 0.9
    assert _hf_std(out, inside) > 3 * _hf_std(rgb, inside) + 1e-4  # smooth skin gets real texture
    assert np.abs(out - rgb).max() < 0.1                          # but not wild
    assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1


def test_skin_texture_deterministic_and_zero_strength_noop():
    rgb, m = _setup()
    cfg = DEFAULTS["steps"]["skin_texture"]
    a, b = apply_skin_texture(rgb, m, cfg, 100, 7), apply_skin_texture(rgb, m, cfg, 100, 7)
    assert np.array_equal(a, b)
    assert np.array_equal(apply_skin_texture(rgb, m, cfg, 0, 7), rgb)


def test_eyes_confined_and_no_overshoot():
    rgb, m = _setup()
    out = apply_eyes(rgb, m, DEFAULTS["steps"]["eyes"], 100)
    y0, y1, x0, x1 = m.roi
    union = np.zeros((H, W), np.float32)
    union[y0:y1, x0:x1] = np.maximum(m.iris, m.sclera)
    d = np.abs(out - rgb).max(2)
    assert d[union < 1e-3].max() < 1e-6
    assert d.max() > 0.002                                         # something happened
    # anti-halo clamp: result stays inside local min/max of the original luminance
    g0, g1 = rgb.mean(2), out.mean(2)
    k = max(1, int(round(0.004 * m.face_w)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    assert (g1 - cv2.dilate(g0, ker)).max() < 0.02 and (cv2.erode(g0, ker) - g1).max() < 0.02


def test_tone_scaling_and_direction():
    cfg = DEFAULTS["steps"]["tone"]
    x = np.linspace(0.05, 0.95, 50, dtype=np.float32).reshape(-1, 1, 1).repeat(3, 2)
    p66 = tone_params(cfg)
    p0 = tone_params({**cfg, "strength": 0})
    assert (p0.ev, p0.offset, p0.gain) == (0, 0, 1)
    assert np.array_equal(apply_tone(x, p0), x)
    assert abs(p66.ev - 0.02 * 3 * 0.66) < 1e-9 and abs(p66.offset - 0.0025 * 15 * 0.66) < 1e-9
    y = apply_tone(x, p66)
    assert y.mean() > x.mean()
    assert (y[-1] - y[0]).mean() > (x[-1] - x[0]).mean()           # contrast up


def test_identity_lut_is_identity_and_blend_midpoint():
    ident = Lut3D(identity_table(17))
    x = np.random.default_rng(0).random((64, 64, 3), dtype=np.float32)
    assert np.abs(apply_lut(x, ident) - x).max() < 1e-5
    inv = Lut3D(1.0 - identity_table(17))
    half = blend_lut(x, inv, 0.5)
    assert np.abs(half - (0.5 * x + 0.5 * (1 - x))).max() < 1e-4


def test_hald_and_cube_roundtrip(tmp_path):
    im = make_identity_hald(8)
    assert im.size == (512, 512)
    p = tmp_path / "h.png"
    im.save(p)
    lut = read_hald(p)
    assert lut.size == 64
    x = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    assert np.abs(apply_lut(x, lut) - x).max() < 1.2 / 255            # 8-bit node quantisation only
    small = Lut3D(identity_table(9) ** 1.3)
    write_cube(tmp_path / "a.cube", small)
    back = read_cube(tmp_path / "a.cube")
    assert np.abs(back.table - small.table).max() < 1e-5


def test_lut_baked_strength_is_not_double_applied():
    cfg = {"strength": 50, "lut_baked_strength": 50, "allow_extrapolation": False}
    assert lut_blend_factor(cfg, False) == 1.0                      # Hald made at NAT50 -> use fully for 50%
    assert lut_blend_factor({**cfg, "lut_baked_strength": 100}, False) == 0.5
    assert lut_blend_factor(cfg, True) == 0.5                       # placeholder is 'full strength'
    assert lut_blend_factor({**cfg, "strength": 100}, False) == 1.0  # clipped without extrapolation


def test_placeholder_desaturates_like_the_nat_reference():
    """Direction check from the real NAT-100 reference pair: chroma drops ~20%, never rises."""
    from photoflow.colorspace import rgb_to_lab_precise
    from photoflow.lut import placeholder_lut
    ph = placeholder_lut()
    assert ph.is_placeholder
    rng = np.random.default_rng(0)
    x = rng.random((4000, 3))
    x = x[(x.max(1) - x.min(1)) > 0.15]                                # clearly coloured samples
    y = apply_lut(x.astype(np.float32), ph)
    c = lambda v: np.hypot(*rgb_to_lab_precise(v)[:, 1:].T).mean()
    assert 0.65 < c(y) / c(x) < 0.9
