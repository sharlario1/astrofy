import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin

from photoflow.batch import manifest_path, read_manifest, run_batch
from photoflow.synthetic import draw_synthetic_portrait


def _snapshot(d: Path):
    return {p.relative_to(d).as_posix(): (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(d.rglob("*.*")) if ".photoflow" not in p.parts}


def test_batch_is_idempotent_and_resumable(small_set, tmp_path, preset):
    out = tmp_path / "out"
    s1 = run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    assert (s1.processed, s1.errors, s1.up_to_date) == (3, 0, 0)
    snap = _snapshot(out)
    assert len(snap) == 3
    time.sleep(0.05)
    s2 = run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    assert (s2.processed, s2.up_to_date) == (0, 3)
    assert _snapshot(out) == snap                                    # nothing touched, byte-identical

    # input content changes -> only that file is redone
    Image.fromarray(draw_synthetic_portrait(240, 320, seed=99, noisy_background=True)).save(small_set / "img1.png")
    s3 = run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    assert (s3.processed, s3.up_to_date) == (1, 2)

    # output deleted -> regenerated
    (out / "img0.png").unlink()
    assert run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None).processed == 1

    # preset changes -> everything redone
    preset["steps"]["tone"]["strength"] = 30
    assert run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None).processed == 3


def test_interrupted_manifest_line_is_tolerated(small_set, tmp_path, preset):
    out = tmp_path / "out"
    run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    with open(manifest_path(out), "a") as fh:
        fh.write('{"rel": "broken", "sta')                            # simulated crash mid-write
    assert len(read_manifest(out)) == 3
    assert run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None).up_to_date == 3


def test_parallel_matches_serial_and_is_deterministic(small_set, tmp_path, preset):
    a, b = tmp_path / "a", tmp_path / "b"
    run_batch(small_set, a, preset, jobs=1, echo=lambda *_: None)
    s = run_batch(small_set, b, preset, jobs=2, echo=lambda *_: None)    # spawn workers, like Windows
    assert s.processed == 3 and s.errors == 0
    for p in a.glob("*.png"):
        assert np.array_equal(np.asarray(Image.open(p)), np.asarray(Image.open(b / p.name)))


def test_bad_file_does_not_kill_batch_and_is_retried(small_set, tmp_path, preset):
    (small_set / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "out"
    s = run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    assert (s.processed, s.errors) == (3, 1)
    s2 = run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    assert (s2.up_to_date, s2.errors) == (3, 1)                      # error entries are retried


def test_never_overwrites_input_dir(small_set, preset):
    import pytest
    with pytest.raises(ValueError):
        run_batch(small_set, small_set, preset, echo=lambda *_: None)


def test_jpeg_444_q95_and_metadata_untouched(tmp_path, preset):
    d = tmp_path / "in"
    d.mkdir()
    im = Image.fromarray(draw_synthetic_portrait(240, 320))
    exif = Image.Exif()
    exif[0x010F] = "TESTMAKE"
    exif[0x0110] = "TESTMODEL"
    im.save(d / "cam.jpg", quality=90, exif=exif.tobytes(), icc_profile=b"")
    out = tmp_path / "out"
    assert run_batch(d, out, preset, jobs=1, echo=lambda *_: None).errors == 0
    res = Image.open(out / "cam.jpg")
    assert res.format == "JPEG" and JpegImagePlugin.get_sampling(res) == 0     # 4:4:4
    ex = res.getexif()
    assert ex[0x010F] == "TESTMAKE" and ex[0x0110] == "TESTMODEL"
    assert 0x0131 not in ex and 0x0112 not in ex                              # no Software tag, nothing invented
    assert Image.open(d / "cam.jpg").info["exif"] == res.info["exif"]          # EXIF bytes identical
    ref = io.BytesIO()
    Image.new("RGB", (16, 16)).save(ref, "JPEG", quality=95)
    assert res.quantization[0] == Image.open(io.BytesIO(ref.getvalue())).quantization[0]     # quality 95 tables


def test_exif_orientation_is_normalised_only_tag_changed(tmp_path, preset):
    d = tmp_path / "in"
    d.mkdir()
    exif = Image.Exif()
    exif[0x010F] = "TESTMAKE"
    exif[0x0112] = 6
    Image.fromarray(draw_synthetic_portrait(240, 320)).save(d / "rot.jpg", exif=exif.tobytes())
    out = tmp_path / "out"
    run_batch(d, out, preset, jobs=1, echo=lambda *_: None)
    res = Image.open(out / "rot.jpg")
    assert res.getexif()[0x0112] == 1 and res.getexif()[0x010F] == "TESTMAKE"
    assert res.size == (320 - 2 * 13, 240 - 2 * 10)                 # rotated upright (240x320 stored -> 320x240), then cropped 4%


def test_output_inside_input_is_not_reingested(small_set, preset):
    out = small_set / "processed"
    assert run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None).processed == 3
    s = run_batch(small_set, out, preset, jobs=1, echo=lambda *_: None)
    assert (s.total, s.up_to_date) == (3, 3)
