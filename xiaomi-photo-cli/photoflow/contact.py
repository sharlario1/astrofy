from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .batch import read_manifest
from .imageio import list_images, load_image, to_uint8


def _fit(arr, w, h):
    im = Image.fromarray(to_uint8(arr))
    im.thumbnail((w, h), Image.LANCZOS)
    cell = Image.new("RGB", (w, h), (24, 24, 24))
    cell.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return cell


def make_contact_sheet(input_dir: Path, output_dir: Path, out_path: Path, n: int = 8, seed: int = 0,
                       pairs_per_row: int = 2, cell_w: int = 360, cell_h: int = 480) -> list[str]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    manifest = read_manifest(output_dir)
    avail: list[tuple[Path, Path]] = []
    for src in list_images(input_dir):
        rel = src.relative_to(input_dir).as_posix()
        e = manifest.get(rel)
        if e and e.get("status") == "ok" and e.get("out") and Path(e["out"]).exists():
            avail.append((src, Path(e["out"])))
    if not avail:
        raise ValueError("no processed images found (run `photoflow run` first)")
    rng = random.Random(seed)
    pick = rng.sample(avail, min(n, len(avail)))
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    pad, lab = 8, 22
    cols = pairs_per_row
    rows = (len(pick) + cols - 1) // cols
    pair_w = 2 * cell_w + pad
    W = cols * pair_w + (cols + 1) * pad
    H = rows * (cell_h + lab + pad) + pad
    sheet = Image.new("RGB", (W, H), (12, 12, 12))
    d = ImageDraw.Draw(sheet)
    for i, (src, out) in enumerate(pick):
        r, c = divmod(i, cols)
        x = pad + c * (pair_w + pad)
        y = pad + r * (cell_h + lab + pad)
        sheet.paste(_fit(load_image(src).rgb, cell_w, cell_h), (x, y + lab))
        sheet.paste(_fit(load_image(out).rgb, cell_w, cell_h), (x + cell_w + pad, y + lab))
        d.text((x, y + 2), f"{src.name}   before | after", fill=(220, 220, 220), font=font)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return [s.name for s, _ in pick]
