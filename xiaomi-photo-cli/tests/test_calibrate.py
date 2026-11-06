import numpy as np
from PIL import Image

from photoflow.calibrate import calibrate
from photoflow.config import load_preset
from photoflow.lut import Lut3D, apply_lut, identity_table, write_cube
from photoflow.tone import ToneParams, apply_tone


def test_calibration_recovers_known_recipe(tmp_path):
    n = 17
    t = identity_table(n)
    nat = Lut3D(np.clip(0.5 + (t - 0.5) * 1.15 + np.array([0.02, 0, -0.03], np.float32), 0, 1).astype(np.float32))
    write_cube(tmp_path / "nat.cube", nat)
    rng = np.random.default_rng(0)
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    for i in range(6):
        base = rng.random((6, 8, 3), dtype=np.float32)
        x = np.clip(np.kron(base, np.ones((16, 16, 1), np.float32)), 0, 1)       # smooth patches, full tonal range
        y = apply_tone(x, ToneParams(0.10, 0.02, 1.10, 0.5))
        y = np.clip(y + 0.5 * (apply_lut(y, nat) - y), 0, 1)
        Image.fromarray((x * 255 + .5).astype(np.uint8)).save(tmp_path / "b" / f"p{i}.png")
        Image.fromarray((y * 255 + .5).astype(np.uint8)).save(tmp_path / "a" / f"p{i}.png")
    p = load_preset(None)
    p["steps"]["lut_nat"].update(lut_path=str(tmp_path / "nat.cube"), lut_baked_strength=100, strength=50)
    r = calibrate(tmp_path / "b", tmp_path / "a", p)
    assert r.n_pairs == 6
    assert r.identity["mean"] > 5 and r.before["mean"] > r.after["mean"]
    assert r.after["mean"] < 0.8                                                   # imperceptible residual
    assert abs(r.beta - 0.5) < 0.08 and abs(r.lut_strength - 50) < 8
    # fitted model reproduces the phone result even if EV/offset/gain trade off against each other
    assert set(r.per_pt) == {"exposure_ev_per_pt", "brightness_per_pt", "contrast_per_pt"}


def test_unmatched_aspect_pairs_are_skipped(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    Image.new("RGB", (100, 100), (90, 90, 90)).save(tmp_path / "b" / "x.png")
    Image.new("RGB", (160, 100), (100, 100, 100)).save(tmp_path / "a" / "x.png")
    Image.new("RGB", (100, 100), (90, 120, 90)).save(tmp_path / "b" / "y.png")
    Image.new("RGB", (100, 100), (100, 130, 100)).save(tmp_path / "a" / "y.png")
    r = calibrate(tmp_path / "b", tmp_path / "a", load_preset(None))
    assert r.n_pairs == 1 and len(r.skipped) == 1
