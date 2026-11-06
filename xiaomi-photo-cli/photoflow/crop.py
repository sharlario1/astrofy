from __future__ import annotations


def crop_box(h: int, w: int, per_side_pct: float, strength: float = 100.0) -> tuple[int, int, int, int]:
    """Equal *proportional* crop on all four sides: p% of the width off left and right, p% of the
    height off top and bottom (aspect ratio preserved). Returns y0, y1, x0, x1."""
    f = per_side_pct / 100.0 * strength / 100.0
    dy, dx = int(round(h * f)), int(round(w * f))
    dy, dx = min(dy, (h - 2) // 2), min(dx, (w - 2) // 2)
    return dy, h - dy, dx, w - dx
