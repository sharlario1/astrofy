"""Synthetic 'AI-smooth' portraits + matching landmark layout (for tests and the no-samples demo).

These are cartoons. (MediaPipe may or may not detect a face in them - on 1.2.2 it did.) The tests do
not depend on that: FixedLayoutDetector returns the landmark layout used to draw them, which
exercises the full face pipeline deterministically without a real photo or the model file.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import landmark_tables as T

CX, CY, RX, RY = 0.5, 0.46, 0.22, 0.29     # relative to (w, w, w, h) respectively


def _ell(cx, cy, rx, ry, n, start=0.0):
    a = start + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def synthetic_landmarks(w: int, h: int) -> np.ndarray:
    cx, cy, rx, ry = CX * w, CY * h, RX * w, RY * h
    lm = np.zeros((T.N_LANDMARKS, 2), np.float32)
    rng = np.random.default_rng(1)
    lm[:] = np.stack([cx + rx * 0.8 * rng.uniform(-1, 1, T.N_LANDMARKS),
                      cy + ry * 0.8 * rng.uniform(-1, 1, T.N_LANDMARKS)], 1)
    lm[T.FACE_OVAL] = _ell(cx, cy, rx, ry, len(T.FACE_OVAL))
    ey = cy - 0.12 * ry
    for eye_idx, iris_c, iris_ring, brow, sgn in (
            (T.RIGHT_EYE, T.RIGHT_IRIS_CENTER, T.RIGHT_IRIS_RING, T.RIGHT_EYEBROW, -1),
            (T.LEFT_EYE, T.LEFT_IRIS_CENTER, T.LEFT_IRIS_RING, T.LEFT_EYEBROW, +1)):
        ex = cx + sgn * 0.38 * rx
        lm[eye_idx] = _ell(ex, ey, 0.20 * rx, 0.075 * rx, len(eye_idx))
        lm[iris_c] = (ex, ey)
        ir = 0.085 * rx
        lm[iris_ring] = _ell(ex, ey, ir, ir, 4)
        lm[brow] = _ell(ex, ey - 0.22 * rx, 0.24 * rx, 0.05 * rx, len(brow))
    lm[T.LIPS] = _ell(cx, cy + 0.5 * ry, 0.30 * rx, 0.09 * rx, len(T.LIPS))
    return lm


def draw_synthetic_portrait(w: int = 768, h: int = 1024, seed: int = 0, noisy_background: bool = False) -> np.ndarray:
    """Return uint8 RGB. Smooth skin on purpose (that is the problem the pipeline fixes)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.stack([90 + 40 * yy / h, 100 + 30 * xx / w, 120 + 20 * (1 - yy / h)], 2)
    img = bg.astype(np.float32)
    lm = synthetic_landmarks(w, h)
    cx, cy, rx, ry = CX * w, CY * h, RX * w, RY * h
    layer = np.zeros((h, w, 3), np.float32)
    mask = np.zeros((h, w), np.float32)
    # hair, neck, face
    cv2.ellipse(mask, (int(cx), int(cy - 0.12 * ry)), (int(rx * 1.25), int(ry * 1.2)), 0, 0, 360, 1, -1)
    img = img * (1 - mask[..., None]) + np.array([40, 30, 25], np.float32) * mask[..., None]
    mask[:] = 0
    cv2.rectangle(mask, (int(cx - 0.35 * rx), int(cy + 0.7 * ry)), (int(cx + 0.35 * rx), h), 1, -1)
    skin_c = np.array([205, 160, 135], np.float32)
    shade = 1 - 0.18 * ((xx - cx) / (rx * 1.2)) ** 2
    img = img * (1 - mask[..., None]) + skin_c * 0.9 * mask[..., None]
    mask[:] = 0
    cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    img = img * (1 - mask[..., None]) + skin_c * shade[..., None] * mask[..., None]
    img = img.astype(np.float32)
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    for sgn, eye_idx, ic in ((-1, T.RIGHT_EYE, T.RIGHT_IRIS_CENTER), (1, T.LEFT_EYE, T.LEFT_IRIS_CENTER)):
        pts = np.round(lm[eye_idx]).astype(np.int32)
        cv2.fillConvexPoly(u8, cv2.convexHull(pts), (235, 235, 235), lineType=cv2.LINE_AA)
        c = (int(lm[ic][0]), int(lm[ic][1]))
        r = int(0.085 * rx)
        cv2.circle(u8, c, r, (90, 70, 50), -1, cv2.LINE_AA)
        cv2.circle(u8, c, max(1, r // 2), (15, 15, 15), -1, cv2.LINE_AA)
        # lids occlude iris top/bottom
        cv2.polylines(u8, [cv2.convexHull(pts)], True, (120, 80, 65), 2, cv2.LINE_AA)
    for br in (T.RIGHT_EYEBROW, T.LEFT_EYEBROW):
        cv2.fillConvexPoly(u8, cv2.convexHull(np.round(lm[br]).astype(np.int32)), (55, 40, 32), cv2.LINE_AA)
    cv2.fillConvexPoly(u8, cv2.convexHull(np.round(lm[T.LIPS]).astype(np.int32)), (170, 80, 85), cv2.LINE_AA)
    out = cv2.GaussianBlur(u8, (0, 0), 1.2).astype(np.float32)   # airbrushed look
    if noisy_background:
        bgmask = 1 - cv2.GaussianBlur(
            cv2.ellipse(np.zeros((h, w), np.float32), (int(cx), int(cy)), (int(rx * 1.4), int(ry * 1.3)), 0, 0, 360, 1, -1), (0, 0), 8)
        out += rng.normal(0, 14, out.shape).astype(np.float32) * bgmask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def write_synthetic_set(out_dir: Path, n: int = 6, w: int = 768, h: int = 1024) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        arr = draw_synthetic_portrait(w, h, seed=i, noisy_background=(i % 3 == 2))
        p = out_dir / f"synthetic_{i:02d}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths
