"""Face landmark detection (MediaPipe Face Landmarker) behind a tiny interface."""
from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import landmark_tables as T
from .config import PROJECT_ROOT, resolve_path

log = logging.getLogger("photoflow")

# Verified in this build: HTTP 200 from Google's public bucket; sha256 pinned from the file fetched then.
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"


@dataclass
class Face:
    lm: np.ndarray                      # (478, 2) float32, pixel coords in the processed image

    @property
    def oval(self) -> np.ndarray:
        return self.lm[T.FACE_OVAL]

    @property
    def width(self) -> float:
        o = self.oval
        return float(o[:, 0].max() - o[:, 0].min())

    @property
    def height(self) -> float:
        o = self.oval
        return float(o[:, 1].max() - o[:, 1].min())

    def bbox(self, pad_frac: float = 0.0) -> tuple[int, int, int, int]:
        o = self.oval
        w, h = self.width, self.height
        x0, x1 = o[:, 0].min() - pad_frac * w, o[:, 0].max() + pad_frac * w
        y0, y1 = o[:, 1].min() - pad_frac * h, o[:, 1].max() + pad_frac * h
        return int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))


def fetch_model(dest: Path = DEFAULT_MODEL, force: bool = False) -> Path:
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    log.info("downloading %s", LANDMARKER_URL)
    urllib.request.urlretrieve(LANDMARKER_URL, tmp)
    data = Path(tmp).read_bytes()
    if not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is not a .task bundle (zip)")
    digest = hashlib.sha256(data).hexdigest()
    if digest != LANDMARKER_SHA256:
        log.warning("model sha256 differs from the pinned value (%s); Google may have updated it. Using it anyway.", digest)
    os.replace(tmp, dest)
    return dest


class MediaPipeDetector:
    def __init__(self, model_path: Path, max_faces: int = 4, min_conf: float = 0.5, detect_max_side: int = 1600):
        os.environ.setdefault("GLOG_minloglevel", "2")        # hide MediaPipe's C++ INFO/WARNING spam
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.detect_max_side = detect_max_side
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=max_faces,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            output_face_blendshapes=False,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[Face]:
        h, w = rgb.shape[:2]
        u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        s = self.detect_max_side / max(h, w)
        if s < 1.0:
            u8 = cv2.resize(u8, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(u8))
        res = self._lm.detect(img)
        faces = []
        for pts in res.face_landmarks:
            a = np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)
            if a.shape[0] >= T.N_LANDMARKS:
                faces.append(Face(a[: T.N_LANDMARKS]))
        faces.sort(key=lambda f: -f.width * f.height)
        return faces

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass


class FixedLayoutDetector:
    """Test double: always returns the synthetic portrait layout (see synthetic.py).
    Enabled with env PHOTOFLOW_DETECTOR=synthetic so that spawned workers pick it up too."""

    def detect(self, rgb: np.ndarray) -> list[Face]:
        from .synthetic import synthetic_landmarks
        h, w = rgb.shape[:2]
        return [Face(synthetic_landmarks(w, h))]

    def close(self) -> None:
        pass


def get_detector(preset: dict):
    if os.environ.get("PHOTOFLOW_DETECTOR") == "synthetic":
        return FixedLayoutDetector()
    fc = preset["face"]
    model = resolve_path(fc.get("model_path"), preset) or Path(os.environ.get("PHOTOFLOW_MODEL", DEFAULT_MODEL))
    if not Path(model).exists():
        raise FileNotFoundError(
            f"Face Landmarker model not found at {model}. Run:  photoflow fetch-models")
    return MediaPipeDetector(model, fc["max_faces"], fc["min_confidence"], fc["detect_max_side"])
