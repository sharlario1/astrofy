import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import yaml
from PIL import Image
from typer.testing import CliRunner

from photoflow.cli import app
from photoflow.comfy import img2img_step
from photoflow.config import PROJECT_ROOT, apply_jitter, load_preset
from photoflow.qc import analyze
from photoflow.synthetic import draw_synthetic_portrait
from photoflow.yamledit import set_scalars
from photoflow.face import FixedLayoutDetector, Face
from photoflow.synthetic import synthetic_landmarks


def test_yamledit_preserves_comments_and_types(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("a:\n  b: 1   # keep me\n  c:\n    d: null\nz: 3\n")
    set_scalars(p, {("a", "b"): 2.5, ("a", "c", "d"): "2026-11-06"})
    txt = p.read_text()
    assert "# keep me" in txt
    d = yaml.safe_load(txt)
    assert d["a"]["b"] == 2.5 and d["a"]["c"]["d"] == "2026-11-06" and d["z"] == 3


def test_shipped_preset_loads_with_user_values():
    p = load_preset("heisenberg")
    s = p["steps"]
    assert s["crop"]["per_side_pct"] == 4.0
    assert (s["skin_texture"]["strength"], s["eyes"]["strength"], s["tone"]["strength"], s["lut_nat"]["strength"]) == (100, 100, 66, 50)
    assert s["tone"]["ui_points"] == {"exposure": 3, "brightness": 15, "contrast": 15}
    assert s["grain"]["enabled"] is False and s["img2img"]["enabled"] is False


def test_jitter_is_deterministic_and_bounded():
    p = load_preset(None)
    p["jitter"]["enabled"] = True
    a = apply_jitter(p, "ab" * 32)
    assert a == apply_jitter(p, "ab" * 32)
    assert a != apply_jitter(p, "cd" * 32)
    assert 66 * 0.95 - 1e-6 <= a["steps"]["tone"]["strength"] <= 66 * 1.05 + 1e-6
    assert load_preset(None)["steps"]["tone"]["strength"] == 66           # original untouched


class _Det:
    def __init__(self, n): self.n = n
    def detect(self, rgb):
        h, w = rgb.shape[:2]
        return [Face(synthetic_landmarks(w, h)) for _ in range(self.n)]


def test_qc_flags(tmp_path):
    p = load_preset(None)
    th = p["qc"]
    clean, noisy = tmp_path / "c.png", tmp_path / "n.png"
    Image.fromarray(draw_synthetic_portrait(768, 1024, 0)).save(clean)
    Image.fromarray(draw_synthetic_portrait(768, 1024, 0, noisy_background=True)).save(noisy)
    rc = analyze(clean, FixedLayoutDetector(), p, th)
    rn = analyze(noisy, FixedLayoutDetector(), p, th)
    assert rc["n_faces"] == 1 and "excess_texture" not in rc["flags"] and "no_face" not in rc["flags"]
    assert "excess_texture" in rn["flags"] and rn["bg_noise"] > 5 * rc["bg_noise"]
    assert "no_face" in analyze(clean, _Det(0), p, th)["flags"]
    assert "multiple_faces" in analyze(clean, _Det(2), p, th)["flags"]
    blur = tmp_path / "b.png"
    import cv2
    Image.fromarray(cv2.GaussianBlur(draw_synthetic_portrait(768, 1024, 0), (0, 0), 6)).save(blur)
    assert "blur" in analyze(blur, FixedLayoutDetector(), p, th)["flags"]


def test_cli_hald_roundtrip_and_synthetic(tmp_path):
    r = CliRunner()
    h = tmp_path / "h.png"
    assert r.invoke(app, ["make-hald", str(h)]).exit_code == 0
    assert Image.open(h).size == (512, 512)
    cube = tmp_path / "n.cube"
    res = r.invoke(app, ["lut-from-hald", str(h), str(cube), "--baked-strength", "50"])
    assert res.exit_code == 0 and cube.exists()
    from photoflow.lut import apply_lut, read_cube
    x = np.random.default_rng(0).random((16, 16, 3), dtype=np.float32)
    assert np.abs(apply_lut(x, read_cube(cube)) - x).max() < 2 / 255
    assert r.invoke(app, ["make-synthetic", str(tmp_path / "s"), "--n", "2"]).exit_code == 0
    assert len(list((tmp_path / "s").glob("*.png"))) == 2


class _Comfy(BaseHTTPRequestHandler):
    posted = {}

    def log_message(self, *a): pass

    def _send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = self.rfile.read(n)
        if self.path == "/upload/image":
            assert b"PNG" in body[:2000] + body[-2000:] or b"\x89PNG" in body
            self._send(json.dumps({"name": "up.png", "subfolder": "", "type": "input"}).encode())
        elif self.path == "/prompt":
            _Comfy.posted = json.loads(body); self._send(json.dumps({"prompt_id": "abc"}).encode())

    def do_GET(self):
        if self.path.startswith("/history/abc"):
            self._send(json.dumps({"abc": {"outputs": {"8": {"images": [{"filename": "o.png", "subfolder": "", "type": "output"}]}}}}).encode())
        elif self.path.startswith("/view"):
            buf = io.BytesIO(); Image.new("RGB", (64, 64), (200, 100, 50)).save(buf, "PNG"); self._send(buf.getvalue(), "image/png")


def test_comfy_client_against_mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Comfy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = dict(load_preset(None)["steps"]["img2img"], server=f"127.0.0.1:{srv.server_port}", checkpoint="my.safetensors",
                   denoise=0.2, max_side=32, timeout_s=10)
        rgb = np.full((48, 64, 3), 0.1, np.float32)
        out = img2img_step(rgb, cfg, 100, 1234, PROJECT_ROOT / "workflows" / "img2img_lowdenoise_example.json")
        assert out.shape == rgb.shape and abs(out[..., 0].mean() - 200 / 255) < 0.02
        wf = _Comfy.posted["prompt"]
        assert wf["1"]["inputs"]["ckpt_name"] == "my.safetensors" and wf["6"]["inputs"]["denoise"] == 0.2
        assert wf["6"]["inputs"]["seed"] == 1234 and wf["2"]["inputs"]["image"] == "up.png"
        half = img2img_step(rgb, cfg, 50, 1, PROJECT_ROOT / "workflows" / "img2img_lowdenoise_example.json")
        assert abs(half[..., 0].mean() - (0.1 + 0.5 * (200 / 255 - 0.1))) < 0.02
    finally:
        srv.shutdown()
