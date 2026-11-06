"""Soft masks (skin, iris, sclera) built from face landmarks. Pure NumPy/OpenCV, no MediaPipe import."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import landmark_tables as T
from .face import Face


@dataclass
class FaceMasks:
    roi: tuple[int, int, int, int]       # y0, y1, x0, x1 in image coords
    skin: np.ndarray                     # float32 (roi) 0..1
    iris: np.ndarray                     # float32 (roi) 0..1
    sclera: np.ndarray                   # float32 (roi) 0..1
    face_w: float
    iris_r: float                        # mean iris radius px


def _hull_mask(pts: np.ndarray, shape: tuple[int, int], dilate_px: float = 0.0) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 255)
    if dilate_px >= 1:
        k = int(round(dilate_px)) * 2 + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return m.astype(np.float32) / 255.0


def _blur(m: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.3:
        return m
    return cv2.GaussianBlur(m, (0, 0), sigma)


def iris_geometry(face: Face) -> list[tuple[np.ndarray, float]]:
    out = []
    for c, ring in ((T.RIGHT_IRIS_CENTER, T.RIGHT_IRIS_RING), (T.LEFT_IRIS_CENTER, T.LEFT_IRIS_RING)):
        ctr = face.lm[c]
        r = float(np.linalg.norm(face.lm[ring] - ctr, axis=1).mean())
        out.append((ctr, r))
    return out


def compute_face_masks(face: Face, img_shape: tuple[int, int], cfg: dict) -> FaceMasks:
    """cfg = preset['steps']['mask']; all sizes are fractions of face width."""
    H, W = img_shape
    fw = max(face.width, 8.0)
    feather = max(0.8, cfg["feather"] * fw)
    pad = int(np.ceil(4 * feather + cfg["eye_dilate"] * fw + 0.05 * fw)) + 4
    x0, y0, x1, y1 = face.bbox()
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(W, x1 + pad), min(H, y1 + pad)
    shape = (ry1 - ry0, rx1 - rx0)
    off = np.array([rx0, ry0], np.float32)
    P = lambda idx: face.lm[idx] - off

    # --- skin: face oval (eroded) minus eyes, brows, lips (each softly feathered)
    oval = _hull_mask(P(T.FACE_OVAL), shape)
    er = cfg["oval_erode"] * fw
    if er >= 1:
        k = int(round(er)) * 2 + 1
        oval = cv2.erode(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    skin = _blur(oval, feather)
    for idxs, d in ((T.LEFT_EYE, cfg["eye_dilate"]), (T.RIGHT_EYE, cfg["eye_dilate"]),
                    (T.LEFT_EYEBROW, cfg["brow_dilate"]), (T.RIGHT_EYEBROW, cfg["brow_dilate"]),
                    (T.LIPS, cfg["lip_dilate"])):
        excl = _blur(_hull_mask(P(idxs), shape, d * fw), feather * 0.8)
        skin = skin * (1.0 - excl)

    # --- eyes: iris disks clipped by eyelid hull; sclera = eyelid hull minus iris, eroded
    iris = np.zeros(shape, np.float32)
    sclera = np.zeros(shape, np.float32)
    geo = iris_geometry(face)
    iris_r = float(np.mean([r for _, r in geo]))
    for (ctr, r), eye_idx in zip(geo, (T.RIGHT_EYE, T.LEFT_EYE)):
        lid = _hull_mask(P(eye_idx), shape)
        disk = np.zeros(shape, np.uint8)
        c = (int(round(ctr[0] - rx0)), int(round(ctr[1] - ry0)))
        cv2.circle(disk, c, max(1, int(round(r))), 255, -1, lineType=cv2.LINE_AA)
        disk_f = disk.astype(np.float32) / 255.0
        iris_i = np.minimum(disk_f, lid)
        e = max(1, int(round(cfg_eye_erode(r))))
        lid_er = cv2.erode(lid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1)))
        scl = np.clip(lid_er - cv2.dilate(disk_f, np.ones((3, 3), np.uint8)), 0, 1)
        iris = np.maximum(iris, iris_i)
        sclera = np.maximum(sclera, scl)
    f = max(0.6, 0.12 * iris_r)
    return FaceMasks((ry0, ry1, rx0, rx1), skin.astype(np.float32), _blur(iris, f), _blur(sclera, f), fw, iris_r)


def cfg_eye_erode(iris_r: float, frac: float = 0.12) -> float:
    return frac * iris_r
