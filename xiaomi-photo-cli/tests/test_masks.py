import numpy as np

from photoflow import landmark_tables as T
from photoflow.config import DEFAULTS
from photoflow.face import Face
from photoflow.masks import compute_face_masks
from photoflow.synthetic import synthetic_landmarks

W, H = 768, 1024


def _masks():
    f = Face(synthetic_landmarks(W, H))
    return f, compute_face_masks(f, (H, W), DEFAULTS["steps"]["mask"])


def _at(mask, roi, pt):
    y0, _, x0, _ = roi
    return float(mask[int(round(pt[1])) - y0, int(round(pt[0])) - x0])


def test_masks_non_empty_and_bounded():
    f, m = _masks()
    for a in (m.skin, m.iris, m.sclera):
        assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1.0001 and a.sum() > 100


def test_skin_excludes_eyes_brows_lips_but_covers_cheeks():
    f, m = _masks()
    assert _at(m.skin, m.roi, f.lm[T.LEFT_IRIS_CENTER]) < 0.05
    assert _at(m.skin, m.roi, f.lm[T.RIGHT_IRIS_CENTER]) < 0.05
    assert _at(m.skin, m.roi, f.lm[T.LIPS].mean(0)) < 0.05
    assert _at(m.skin, m.roi, f.lm[T.LEFT_EYEBROW].mean(0)) < 0.05
    cheek = f.lm[T.FACE_OVAL].mean(0) + np.array([0.12 * f.width, 0.10 * f.height])
    assert _at(m.skin, m.roi, cheek) > 0.95


def test_iris_and_sclera_do_not_overlap_much():
    _, m = _masks()
    assert (np.minimum(m.iris, m.sclera)).sum() < 0.1 * m.iris.sum()


def test_mask_radii_scale_with_face_size():
    small = compute_face_masks(Face(synthetic_landmarks(384, 512)), (512, 384), DEFAULTS["steps"]["mask"])
    big = compute_face_masks(Face(synthetic_landmarks(W, H)), (H, W), DEFAULTS["steps"]["mask"])
    assert abs(small.skin.sum() * 4 - big.skin.sum()) / big.skin.sum() < 0.08
