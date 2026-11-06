import numpy as np
import pytest
from PIL import Image

from photoflow.config import load_preset
from photoflow.synthetic import draw_synthetic_portrait


@pytest.fixture(autouse=True)
def synthetic_detector(monkeypatch):
    """All tests use the fixed-layout detector: no MediaPipe model needed, deterministic, spawn-safe."""
    monkeypatch.setenv("PHOTOFLOW_DETECTOR", "synthetic")


@pytest.fixture
def preset():
    return load_preset(None)          # built-in defaults (placeholder LUT)


@pytest.fixture
def small_set(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(3):
        Image.fromarray(draw_synthetic_portrait(240, 320, seed=i)).save(d / f"img{i}.png")
    return d
