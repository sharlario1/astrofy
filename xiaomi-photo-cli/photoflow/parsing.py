"""Optional face-parsing ONNX mask (user-supplied model).

I could NOT verify any hosted face-parsing ONNX from this build environment (huggingface.co was
blocked), so no download URL is shipped. Any model with the standard CelebAMask-HQ 19-class layout
(BiSeNet, 512x512 input, ImageNet mean/std, output [1,19,h,w]) works:
  skin = label 1, nose = 10  (the other labels: 2/3 brows, 4/5 eyes, 11-13 mouth, 17 hair, ...)
Set face.parsing_onnx in the preset. Providers are tried in order CUDA -> DirectML -> CPU.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("photoflow")
SKIN_LABELS = (1, 10)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def pick_providers(ort) -> list[str]:
    avail = ort.get_available_providers()
    order = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail]
    return order + ["CPUExecutionProvider"]


class FaceParser:
    def __init__(self, model_path: Path, size: int = 512):
        import onnxruntime as ort
        self.size = size
        provs = pick_providers(ort)
        self.sess = ort.InferenceSession(str(model_path), providers=provs)
        self.inp = self.sess.get_inputs()[0].name
        log.info("face parsing providers: %s", self.sess.get_providers())

    def skin_mask(self, rgb_crop: np.ndarray) -> np.ndarray:
        """rgb_crop float32 [0,1] HxWx3 -> float32 HxW skin probability (hard labels, lightly blurred)."""
        h, w = rgb_crop.shape[:2]
        x = cv2.resize(rgb_crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0]
        lab = out[0].argmax(0)
        m = np.isin(lab, SKIN_LABELS).astype(np.float32)
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (0, 0), max(1.0, 0.004 * max(h, w)))


def load_parser(preset: dict):
    from .config import resolve_path
    p = resolve_path(preset["face"].get("parsing_onnx"), preset)
    if not p:
        return None
    if not Path(p).exists():
        log.warning("face.parsing_onnx=%s not found; using landmark mask only", p)
        return None
    try:
        return FaceParser(p)
    except Exception as e:  # noqa: BLE001
        log.warning("could not load face-parsing model (%s); using landmark mask only", e)
        return None
