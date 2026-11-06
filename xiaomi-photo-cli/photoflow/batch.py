"""Resumable multiprocessing batch runner.

Manifest: <output>/.photoflow/manifest.jsonl (append-only, last entry per input wins, written by the
parent process only, flushed per image -> Ctrl+C / crash safe). An input is skipped when its sha256,
the config hash (effective preset + LUT bytes + PIPELINE_VERSION) and the output file all match.
"""
from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .config import apply_jitter, config_hash, file_sha256, resolve_path
from .imageio import IMAGE_EXTS, list_images, load_image, resolve_format, save_image

log = logging.getLogger("photoflow")
MANIFEST_NAME = "manifest.jsonl"


@dataclass
class Job:
    rel: str
    src: str
    dst: str
    sha: str


@dataclass
class Summary:
    total: int = 0
    processed: int = 0
    up_to_date: int = 0
    errors: int = 0
    no_face: int = 0
    skipped_no_face: int = 0
    elapsed: float = 0.0
    error_details: list = field(default_factory=list)


# ------------------------------------------------------------------ manifest
def manifest_path(out_dir: Path) -> Path:
    return Path(out_dir) / ".photoflow" / MANIFEST_NAME


def read_manifest(out_dir: Path) -> dict[str, dict]:
    mp_ = manifest_path(out_dir)
    entries: dict[str, dict] = {}
    if mp_.exists():
        with open(mp_, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue            # truncated last line after a crash
                entries[e["rel"]] = e
    return entries


def _append(fh, entry: dict) -> None:
    fh.write(json.dumps(entry) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


# ------------------------------------------------------------------ planning
def out_suffix(src: Path, fmt: str) -> str:
    if fmt == "png" or (fmt == "auto" and src.suffix.lower() == ".png"):
        return ".png"
    return ".jpg"


def plan(input_dir: Path, output_dir: Path, preset: dict, limit: int | None = None) -> list[Job]:
    jobs, used = [], set()
    fmt = preset["output"]["format"]
    out_resolved = output_dir.resolve()
    files = [f for f in list_images(input_dir) if out_resolved not in f.resolve().parents]   # OUTPUT inside INPUT
    if limit:
        files = files[:limit]
    for src in files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.parent / (rel.stem + out_suffix(src, fmt))
        if dst in used:                              # a.png + a.jpg in the same folder
            dst = output_dir / rel.parent / f"{rel.stem}_{src.suffix.lstrip('.').lower()}{dst.suffix}"
        used.add(dst)
        jobs.append(Job(rel.as_posix(), str(src), str(dst), file_sha256(src)))
    return jobs


# ------------------------------------------------------------------ worker side
_STATE: dict = {}


def _close_state() -> None:
    d = _STATE.pop("detector", None)
    if d is not None:
        d.close()


def _init_worker(preset: dict) -> None:
    import cv2
    cv2.setNumThreads(1)
    _STATE["preset"] = preset
    atexit.register(_close_state)


def _ensure_models():
    from .face import get_detector
    from .parsing import load_parser
    preset = _STATE["preset"]
    if "detector" not in _STATE:
        _STATE["detector"] = get_detector(preset)
        _STATE["parser"] = load_parser(preset)
    return _STATE["detector"], _STATE["parser"]


def process_one(job: dict, cfg_hash: str) -> dict:
    from .pipeline import process_array
    t0 = time.perf_counter()
    entry = {"rel": job["rel"], "src_sha256": job["sha"], "config_hash": cfg_hash}
    try:
        preset = _STATE["preset"]
        det, parser = _ensure_models()
        eff = apply_jitter(preset, job["sha"])
        li = load_image(Path(job["src"]))
        res = process_array(li.rgb, li.alpha, eff, det, parser, int(job["sha"][:8], 16))
        entry.update(n_faces=res.n_faces, applied=res.applied, notes=res.notes)
        if res.skipped:
            entry.update(status="skipped_no_face", out=None)
        else:
            dst = Path(job["dst"])
            fmt = resolve_format(preset["output"]["format"], li.src_format, res.alpha is not None)
            if fmt == "png" and dst.suffix.lower() != ".png":
                dst = dst.with_suffix(".png")
            save_image(dst, res.rgb, li.meta, fmt, preset["output"], res.alpha)
            entry.update(status="ok", out=str(dst))
        if li.orientation_fixed:
            entry["notes"] = entry.get("notes", []) + ["orientation_normalised"]
    except Exception as e:  # noqa: BLE001  (one bad file must not kill the batch)
        entry.update(status="error", out=None, error=f"{type(e).__name__}: {e}",
                     trace=traceback.format_exc(limit=4))
    entry["seconds"] = round(time.perf_counter() - t0, 3)
    return entry


def _worker_entry(job: dict, cfg_hash: str) -> dict:
    return process_one(job, cfg_hash)


# ------------------------------------------------------------------ driver
def run_batch(input_dir: Path, output_dir: Path, preset: dict, jobs: int = 1, force: bool = False,
              limit: int | None = None, dry_run: bool = False, echo=print) -> Summary:
    from .pipeline import resolve_lut

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("OUTPUT_DIR must differ from INPUT_DIR (originals are never overwritten)")
    ln = preset["steps"]["lut_nat"]
    lut_path = None
    if ln.get("enabled", True) and float(ln["strength"]) > 0:
        lut, lut_path = resolve_lut(preset)           # fail fast on a bad LUT; warns once if placeholder
        if lut.is_placeholder:
            echo("WARNING: lut_nat uses the PLACEHOLDER approximation curve (no LUT configured).")
    cfg_hash = config_hash(preset, [lut_path])
    plan_ = plan(input_dir, output_dir, preset, limit)
    summ = Summary(total=len(plan_))
    manifest = {} if force else read_manifest(output_dir)

    todo: list[Job] = []
    for j in plan_:
        e = manifest.get(j.rel)
        if (e and e.get("status") in ("ok", "skipped_no_face") and e.get("src_sha256") == j.sha
                and e.get("config_hash") == cfg_hash and (e["status"] != "ok" or Path(e["out"]).exists())):
            summ.up_to_date += 1
        else:
            todo.append(j)
    echo(f"{summ.total} images found, {summ.up_to_date} up to date, {len(todo)} to process "
         f"(jobs={jobs}, config {cfg_hash[:8]})")
    if dry_run or not todo:
        return summ

    (output_dir / ".photoflow").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fh = open(manifest_path(output_dir), "a", encoding="utf-8")

    def record(e: dict, n: int) -> None:
        _append(fh, e)
        st = e["status"]
        if st == "ok":
            summ.processed += 1
            if "no_face" in e.get("notes", []):
                summ.no_face += 1
        elif st == "skipped_no_face":
            summ.skipped_no_face += 1
        else:
            summ.errors += 1
            summ.error_details.append((e["rel"], e.get("error")))
        flag = " [no face]" if "no_face" in e.get("notes", []) else ""
        echo(f"[{n}/{len(todo)}] {e['rel']} -> {st}{flag} ({e['seconds']}s)" + (f"  {e.get('error')}" if st == "error" else ""))

    try:
        if jobs <= 1:
            _init_worker(preset)
            try:
                for n, j in enumerate(todo, 1):
                    record(process_one(j.__dict__, cfg_hash), n)
            finally:
                _close_state()
        else:
            ctx = mp.get_context("spawn")             # Windows-compatible; also avoids forking MediaPipe state
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx, initializer=_init_worker, initargs=(preset,)) as ex:
                futs = [ex.submit(_worker_entry, j.__dict__, cfg_hash) for j in todo]
                try:
                    for n, f in enumerate(as_completed(futs), 1):
                        record(f.result(), n)
                except KeyboardInterrupt:
                    echo("Interrupted: finishing nothing further; rerun the same command to resume.")
                    for f in futs:
                        f.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
    finally:
        fh.close()
    summ.elapsed = time.perf_counter() - t0
    return summ
