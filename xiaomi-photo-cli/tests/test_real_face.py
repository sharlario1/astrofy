"""Optional: runs only if ./samples has an image and the MediaPipe model is present (photoflow fetch-models)."""
import os
from pathlib import Path

import numpy as np
import pytest

from photoflow.config import PROJECT_ROOT, load_preset
from photoflow.face import DEFAULT_MODEL, get_detector
from photoflow.imageio import list_images, load_image
from photoflow.pipeline import process_array

pytestmark = pytest.mark.real_face
SAMPLES = PROJECT_ROOT / "samples"


@pytest.mark.skipif(not SAMPLES.exists() or not DEFAULT_MODEL.exists(), reason="needs ./samples and models/face_landmarker.task")
def test_real_portrait_gets_detected_and_textured(monkeypatch):
    monkeypatch.delenv("PHOTOFLOW_DETECTOR", raising=False)
    imgs = list_images(SAMPLES)
    if not imgs:
        pytest.skip("no images in ./samples")
    p = load_preset("heisenberg")
    det = get_detector(p)
    try:
        li = load_image(imgs[0])
        res = process_array(li.rgb, li.alpha, p, det, None, 1)
    finally:
        det.close()
    assert res.n_faces >= 1 and "skin_texture" in res.applied
    assert np.isfinite(res.rgb).all()
