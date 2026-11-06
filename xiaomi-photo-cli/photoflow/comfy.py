"""Optional low-denoise img2img pass through the ComfyUI HTTP API (default http://127.0.0.1:8188).

Interface only + one example workflow (workflows/img2img_lowdenoise_example.json). Off by default.
Endpoints used (standard ComfyUI server API): POST /upload/image, POST /prompt,
GET /history/{prompt_id}, GET /view. Only the stdlib is used (no `requests`).

Workflow templating: any string value in the workflow JSON equal to one of
  __CKPT__ __IMAGE__ __POSITIVE__ __NEGATIVE__ __DENOISE__ __SEED__
is replaced with the real value (numbers keep their numeric type).
"""
from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class ComfyError(RuntimeError):
    pass


def _fill(node, values: dict):
    if isinstance(node, dict):
        return {k: _fill(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, values) for v in node]
    if isinstance(node, str) and node in values:
        return values[node]
    return node


class ComfyClient:
    def __init__(self, server: str = "127.0.0.1:8188", timeout_s: float = 600.0):
        self.base = server if server.startswith("http") else f"http://{server}"
        self.timeout_s = timeout_s
        self.client_id = uuid.uuid4().hex

    def _req(self, path: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 30.0) -> bytes:
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"{path}: HTTP {e.code} {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise ComfyError(f"cannot reach ComfyUI at {self.base} ({e.reason}). Is it running with --listen/--port?") from e

    def ping(self) -> bool:
        try:
            self._req("/system_stats", timeout=5)
            return True
        except ComfyError:
            return False

    def upload_png(self, png: bytes, name: str) -> str:
        boundary = uuid.uuid4().hex
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            png,
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n--{boundary}--\r\n'.encode(),
        ]
        out = self._req("/upload/image", b"".join(parts),
                        {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        info = json.loads(out)
        return (info.get("subfolder") + "/" if info.get("subfolder") else "") + info["name"]

    def run(self, workflow: dict, poll_s: float = 0.5) -> bytes:
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        resp = json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))
        pid = resp["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < self.timeout_s:
            hist = json.loads(self._req(f"/history/{pid}"))
            if pid in hist:
                outs = hist[pid].get("outputs", {})
                for node in outs.values():
                    for img in node.get("images", []):
                        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                                    "type": img.get("type", "output")})
                        return self._req(f"/view?{q}", timeout=120)
                raise ComfyError(f"prompt {pid} finished without image outputs: {hist[pid].get('status')}")
            time.sleep(poll_s)
        raise ComfyError(f"timeout after {self.timeout_s}s waiting for prompt {pid}")


def img2img_step(rgb: np.ndarray, cfg: dict, strength: float, seed: int, workflow_path: Path) -> np.ndarray:
    """Send the image (downscaled to cfg.max_side if larger), receive the low-denoise result,
    resize back and blend by strength. Fine detail is then re-added by the later skin/eye steps."""
    h, w = rgb.shape[:2]
    s = min(1.0, cfg["max_side"] / max(h, w))
    small = rgb if s >= 1.0 else cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(np.clip(small * 255 + 0.5, 0, 255).astype(np.uint8)).save(buf, "PNG")
    client = ComfyClient(cfg["server"], cfg["timeout_s"])
    name = client.upload_png(buf.getvalue(), f"photoflow_{seed:08x}.png")
    wf = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    wf = _fill(wf, {"__CKPT__": cfg["checkpoint"], "__IMAGE__": name, "__POSITIVE__": cfg["positive"],
                    "__NEGATIVE__": cfg["negative"], "__DENOISE__": float(cfg["denoise"]), "__SEED__": int(seed)})
    out_png = client.run(wf)
    res = np.asarray(Image.open(io.BytesIO(out_png)).convert("RGB"), dtype=np.float32) / 255.0
    if res.shape[:2] != (h, w):
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC)
    a = strength / 100.0
    return np.clip(rgb + a * (res - rgb), 0, 1).astype(np.float32)
