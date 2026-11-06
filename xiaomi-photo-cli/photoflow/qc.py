"""QC report: flags no face / multiple faces / blur / excessive texture (noise in hair & background).

All metrics are measured on a copy downscaled to 1024 px long side so thresholds do not depend on
resolution. Thresholds are heuristics; the CSV keeps the raw numbers so you can re-threshold.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .face import Face, get_detector
from .imageio import list_images, load_image
from .masks import compute_face_masks

COLUMNS = ["file", "width", "height", "n_faces", "face_width_frac", "sharp_face", "bg_noise",
           "skin_noise", "bg_skin_ratio", "flags"]
_ST: dict = {}


def _robust_noise(gray: np.ndarray, region: np.ndarray | None) -> float:
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    v = hf[region] if region is not None else hf.ravel()
    if v.size < 200:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def analyze(path: Path, detector, preset: dict, th: dict) -> dict:
    row = {"file": str(path)}
    li = load_image(path)
    rgb = li.rgb
    h, w = rgb.shape[:2]
    row.update(width=w, height=h)
    s = min(1.0, 1024 / max(h, w))
    small = rgb if s >= 1 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    gray = (small @ np.array([0.299, 0.587, 0.114], np.float32)) * 255.0
    faces = detector.detect(small)
    row["n_faces"] = len(faces)
    flags = []
    if not faces:
        flags.append("no_face")
        c = gray[gray.shape[0] // 4: 3 * gray.shape[0] // 4, gray.shape[1] // 4: 3 * gray.shape[1] // 4]
        row["sharp_face"] = round(float(cv2.Laplacian(c, cv2.CV_32F).var()), 2)
        row["bg_noise"] = round(_robust_noise(gray, None), 3)
        if row["sharp_face"] < th["blur_min"]:
            flags.append("blur")
        if row["bg_noise"] > th["bg_noise_max"]:
            flags.append("excess_texture")
    else:
        if len(faces) > th["max_faces_ok"]:
            flags.append("multiple_faces")
        f: Face = faces[0]
        gh, gw = gray.shape
        row["face_width_frac"] = round(f.width / gw, 3)
        x0, y0, x1, y1 = f.bbox(0.1)
        crop = gray[max(0, y0):min(gh, y1), max(0, x0):min(gw, x1)]
        if crop.size:
            tw = 256
            th_ = max(8, round(crop.shape[0] * tw / crop.shape[1]))
            crop = cv2.resize(crop, (tw, th_), interpolation=cv2.INTER_AREA if crop.shape[1] > tw else cv2.INTER_CUBIC)
            row["sharp_face"] = round(float(cv2.Laplacian(crop, cv2.CV_32F).var()), 2)
        m = compute_face_masks(f, (gh, gw), preset["steps"]["mask"])
        y0r, y1r, x0r, x1r = m.roi
        skin = np.zeros((gh, gw), bool)
        skin[y0r:y1r, x0r:x1r] = m.skin > 0.5
        oval = np.zeros((gh, gw), np.uint8)
        cv2.fillConvexPoly(oval, cv2.convexHull(np.round(f.oval).astype(np.int32)), 1)
        k = int(0.15 * f.width) * 2 + 1
        near_face = cv2.dilate(oval, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        bg = ~near_face
        row["bg_noise"] = round(_robust_noise(gray, bg), 3)
        row["skin_noise"] = round(_robust_noise(gray, skin), 3)
        if row["skin_noise"] and row["skin_noise"] == row["skin_noise"] and row["skin_noise"] > 0:
            row["bg_skin_ratio"] = round(row["bg_noise"] / row["skin_noise"], 3)
        if row.get("sharp_face", 1e9) < th["blur_min"]:
            flags.append("blur")
        bn = row["bg_noise"]
        if bn == bn and (bn > th["bg_noise_max"] or
                         (bn > th["bg_noise_floor"] and row.get("bg_skin_ratio", 0) > th["bg_skin_ratio_max"])):
            flags.append("excess_texture")
    row["flags"] = ";".join(flags)
    return row


def _init(preset):
    cv2.setNumThreads(1)
    _ST["preset"] = preset
    _ST["det"] = get_detector(preset)


def _work(args):
    path, th = args
    try:
        return analyze(Path(path), _ST["det"], _ST["preset"], th)
    except Exception as e:  # noqa: BLE001
        return {"file": str(path), "flags": f"error:{type(e).__name__}:{e}"}


def run_qc(image_dir: Path, out_csv: Path, preset: dict, thresholds: dict, jobs: int = 1, echo=print) -> dict:
    files = list_images(Path(image_dir))
    items = [(str(p), thresholds) for p in files]
    if jobs <= 1:
        _init(preset)
        try:
            rows = [_work(i) for i in items]
        finally:
            _ST["det"].close()
    else:
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(preset,)) as ex:
            rows = list(ex.map(_work, items, chunksize=4))
    for r in rows:
        r["file"] = str(Path(r["file"]).relative_to(image_dir)) if Path(r["file"]).is_relative_to(image_dir) else r["file"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        wr.writeheader()
        wr.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        for f in filter(None, r.get("flags", "").split(";")):
            key = f.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
    return {"n": len(rows), "flagged": sum(1 for r in rows if r.get("flags")), "counts": counts}
