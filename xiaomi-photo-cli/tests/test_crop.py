import numpy as np

from photoflow.crop import crop_box
from photoflow.pipeline import process_array


def test_crop_box_equal_proportional():
    y0, y1, x0, x1 = crop_box(800, 1000, 4.0)
    assert (y0, 800 - y1, x0, 1000 - x1) == (32, 32, 40, 40)
    assert (y1 - y0, x1 - x0) == (736, 920)
    assert abs((x1 - x0) / (y1 - y0) - 1000 / 800) < 0.01        # aspect preserved


def test_crop_strength_scales_and_zero_is_noop():
    assert crop_box(1000, 1000, 4.0, 50)[0] == 20
    assert crop_box(1000, 1000, 4.0, 0) == (0, 1000, 0, 1000)
    assert crop_box(1000, 1000, 0.0) == (0, 1000, 0, 1000)


def test_pipeline_crop_dimensions(preset):
    rgb = np.random.default_rng(0).random((400, 300, 3), dtype=np.float32)
    for k in ("skin_texture", "eyes", "tone", "lut_nat"):
        preset["steps"][k]["enabled"] = False
    res = process_array(rgb, None, preset, None, None, 1)
    assert res.rgb.shape == (400 - 2 * 16, 300 - 2 * 12, 3)
    assert res.applied == ["crop"]
