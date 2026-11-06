from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from PIL import Image

from .config import PROJECT_ROOT, load_preset, resolve_preset_path

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Batch replica of a Xiaomi 14T Pro photo-editor workflow for AI portraits.")


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s: %(message)s", stream=sys.stderr)


def _rel_or_abs(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(Path(p).resolve())


@app.command()
def run(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder with originals (searched recursively)"),
    output_dir: Path = typer.Argument(..., help="Destination (subfolders are mirrored)"),
    preset: str = typer.Option("heisenberg", help="Preset name in presets/ or path to a .yaml"),
    jobs: int = typer.Option(0, help="Worker processes (0 = half of the CPU cores)"),
    force: bool = typer.Option(False, help="Ignore the manifest and reprocess everything"),
    limit: Optional[int] = typer.Option(None, help="Only the first N images (testing)"),
    dry_run: bool = typer.Option(False, help="Show what would be processed"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Batch-process a folder. Resumable: rerunning skips images whose input hash and preset are unchanged."""
    from .batch import run_batch
    _setup_logging(verbose)
    import os
    p = load_preset(preset)
    if jobs <= 0:
        jobs = max(1, (os.cpu_count() or 2) // 2)
    s = run_batch(input_dir, output_dir, p, jobs=jobs, force=force, limit=limit, dry_run=dry_run, echo=typer.echo)
    typer.echo(f"done: {s.processed} processed, {s.up_to_date} up to date, {s.no_face} without face, "
               f"{s.skipped_no_face} skipped (no face), {s.errors} errors, {s.elapsed:.1f}s")
    for rel, err in s.error_details:
        typer.echo(f"  ERROR {rel}: {err}", err=True)
    raise typer.Exit(1 if s.errors else 0)


@app.command("make-hald")
def make_hald(
    out: Path = typer.Argument(Path("hald_identity_level8.png")),
    level: int = typer.Option(8, help="Hald level; 8 -> 512x512 px, 64^3 cube"),
) -> None:
    """Write an identity Hald CLUT (PNG). Put it through the phone's NAT filter, then use lut-from-hald."""
    from .lut import make_identity_hald
    im = make_identity_hald(level)
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, format="PNG", compress_level=6)           # no metadata, lossless
    typer.echo(f"wrote {out} ({im.width}x{im.height}, level {level})")
    typer.echo("On the phone: open it in the Xiaomi editor, apply ONLY the NAT filter (no other edits, no crop),\n"
               "export at full size. PNG output preferred; JPEG works but use --smooth in lut-from-hald.\n"
               "Tip: NAT at 100 gives the most flexible LUT (blend freely on the PC). If you use NAT 50, pass\n"
               "--baked-strength 50 to lut-from-hald so the 50% is not applied twice.")


@app.command("lut-from-hald")
def lut_from_hald(
    hald: Path = typer.Argument(..., exists=True, dir_okay=False, help="Hald processed on the phone"),
    out: Path = typer.Argument(Path("luts/nat.cube")),
    baked_strength: int = typer.Option(50, help="NAT strength used on the phone when processing the Hald"),
    smooth: float = typer.Option(0.0, help="Gaussian smoothing of the cube in node units (0.6-1.0 for JPEG Halds)"),
    preset: Optional[str] = typer.Option(None, help="Also point this preset's lut_path/lut_baked_strength at the .cube"),
) -> None:
    """Convert a phone-processed Hald PNG into a .cube LUT (optionally wiring it into a preset)."""
    from .lut import identity_table, make_identity_hald, read_hald, smooth_lut, write_cube
    _setup_logging()
    lut = read_hald(hald)
    ident = identity_table(lut.size)
    mad = float(np.abs(lut.table - ident).mean() * 255)
    typer.echo(f"Hald level {round(lut.size ** 0.5)}, cube {lut.size}^3, mean |processed - identity| = {mad:.2f}/255")
    if mad < 0.75:
        typer.echo("WARNING: almost identical to the identity Hald - was the filter really applied/exported?", err=True)
    if hald.suffix.lower() in (".jpg", ".jpeg") and smooth <= 0:
        typer.echo("NOTE: JPEG Hald: compression noise ends up in the LUT. Consider --smooth 0.8", err=True)
    lut = smooth_lut(lut, smooth)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_cube(out, lut, title=f"NAT baked {baked_strength}", comments=[
        f"photoflow: from {hald.name}", f"baked strength on phone: {baked_strength}"])
    typer.echo(f"wrote {out}")
    if preset:
        from .yamledit import set_scalars
        path = resolve_preset_path(preset)
        set_scalars(path, {("steps", "lut_nat", "lut_path"): _rel_or_abs(out).replace("\\", "/"),
                           ("steps", "lut_nat", "lut_baked_strength"): int(baked_strength)})
        typer.echo(f"updated {path}: lut_path, lut_baked_strength={baked_strength}")


@app.command()
def calibrate(
    before_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    after_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    preset: str = typer.Option("heisenberg"),
    write: bool = typer.Option(True, help="Write fitted values into the preset (a .bak copy is kept)"),
    report: Optional[Path] = typer.Option(None, help="Also save the result as JSON"),
    max_side: int = typer.Option(384, help="Pairs are downscaled to this size for fitting"),
    seed: int = typer.Option(0),
) -> None:
    """Fit tone mapping + LUT blend from original/edited phone pairs (matched by file name) in Lab space."""
    from .calibrate import calibrate as do_cal
    _setup_logging()
    p = load_preset(preset)
    r = do_cal(before_dir, after_dir, p, max_side=max_side, seed=seed)
    typer.echo(f"pairs used: {r.n_pairs}  pixels: {r.n_pixels}  skipped: {len(r.skipped)}")
    for name, why in r.skipped:
        typer.echo(f"  skipped {name}: {why}")
    f = lambda d: f"mean {d['mean']:.2f}  median {d['median']:.2f}  p95 {d['p95']:.2f}  |dL| {d['mean_abs_dL']:.2f}"
    typer.echo("dE76 (Lab) vs the phone result:")
    typer.echo(f"  no processing   : {f(r.identity)}")
    typer.echo(f"  preset (before) : {f(r.before)}")
    typer.echo(f"  fitted (after)  : {f(r.after)}")
    typer.echo(f"fitted: EV {r.ev:+.3f}  brightness offset {r.offset:+.4f}  contrast gain {r.gain:.4f}"
               + (f"  LUT blend vs baked LUT {r.beta:.3f} -> strength {r.lut_strength:.1f}" if r.beta is not None else ""))
    for n in r.notes:
        typer.echo(f"NOTE: {n}")
    if write:
        from shutil import copy2
        from .yamledit import set_scalars
        path = resolve_preset_path(preset)
        copy2(path, str(path) + ".bak")
        upd = {("steps", "tone", "mapping", k): v for k, v in r.per_pt.items()}
        if r.lut_strength is not None:
            upd[("steps", "lut_nat", "strength")] = round(r.lut_strength, 2)
        upd[("calibration", "date")] = dt.date.today().isoformat()
        upd[("calibration", "n_pairs")] = r.n_pairs
        upd[("calibration", "de76_mean_before")] = round(r.before["mean"], 3)
        upd[("calibration", "de76_mean_after")] = round(r.after["mean"], 3)
        set_scalars(path, upd)
        typer.echo(f"wrote fitted values to {path} (backup: {path}.bak)")
    if report:
        report.write_text(json.dumps(r.__dict__, indent=2, default=float), encoding="utf-8")


@app.command("contact-sheet")
def contact_sheet(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    output_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    out: Path = typer.Option(Path("contact_sheet.jpg")),
    n: int = typer.Option(8, help="Random sample size"),
    seed: int = typer.Option(0),
    pairs_per_row: int = typer.Option(2),
) -> None:
    """Before/after sheet from a random sample of processed images."""
    from .contact import make_contact_sheet
    names = make_contact_sheet(input_dir, output_dir, out, n, seed, pairs_per_row)
    typer.echo(f"wrote {out} with {len(names)} pairs: {', '.join(names)}")


@app.command()
def qc(
    image_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Usually the OUTPUT dir (also works on originals)"),
    out: Path = typer.Option(Path("qc_report.csv")),
    preset: str = typer.Option("heisenberg"),
    jobs: int = typer.Option(1),
    blur_min: Optional[float] = typer.Option(None),
    bg_noise_max: Optional[float] = typer.Option(None),
    bg_skin_ratio_max: Optional[float] = typer.Option(None),
) -> None:
    """CSV report flagging: no face, multiple faces, blur, excessive texture (noise in hair/background)."""
    from .qc import run_qc
    _setup_logging()
    p = load_preset(preset)
    th = dict(p["qc"])
    for k, v in (("blur_min", blur_min), ("bg_noise_max", bg_noise_max), ("bg_skin_ratio_max", bg_skin_ratio_max)):
        if v is not None:
            th[k] = v
    s = run_qc(image_dir, out, p, th, jobs)
    typer.echo(f"wrote {out}: {s['n']} images, {s['flagged']} flagged {s['counts']}")


@app.command("fetch-models")
def fetch_models(force: bool = typer.Option(False)) -> None:
    """Download the MediaPipe Face Landmarker bundle (official Google URL, sha256-checked)."""
    from .face import DEFAULT_MODEL, fetch_model
    _setup_logging()
    p = fetch_model(DEFAULT_MODEL, force)
    typer.echo(f"model ready: {p} ({p.stat().st_size / 1e6:.1f} MB)")


@app.command("make-synthetic")
def make_synthetic(out_dir: Path = typer.Argument(Path("synthetic_samples")), n: int = 6) -> None:
    """Write synthetic cartoon portraits (smooth, for demos/tests; NOT a substitute for real photos)."""
    from .synthetic import write_synthetic_set
    ps = write_synthetic_set(out_dir, n)
    typer.echo(f"wrote {len(ps)} images to {out_dir}")


if __name__ == "__main__":
    app()
