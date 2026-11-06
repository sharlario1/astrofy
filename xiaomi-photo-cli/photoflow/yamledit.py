"""Comment-preserving in-place edit of scalar values in block-style YAML.

PyYAML would drop every comment in the preset; calibration only needs to change a
handful of scalars, so we edit those lines directly and verify by re-parsing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

_KEY = re.compile(r"^(\s*)([A-Za-z0-9_]+):(\s*)(.*?)\s*$")


def _fmt(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(round(v, 6))
    return json.dumps(str(v))          # double-quoted YAML string (also stops 2026-11-06 becoming a date)


def set_scalars(path: Path, updates: dict[tuple[str, ...], object]) -> None:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    for keypath, value in updates.items():
        stack: list[tuple[int, str]] = []
        done = False
        for i, line in enumerate(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _KEY.match(line)
            if not m:
                continue
            indent, key, _, rest = len(m.group(1)), m.group(2), m.group(3), m.group(4)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, key))
            if tuple(k for _, k in stack) == tuple(keypath):
                comment = ""
                if "#" in rest:
                    comment = "  " + rest[rest.index("#"):]
                lines[i] = f"{' ' * indent}{key}: {_fmt(value)}{comment}"
                done = True
                break
        if not done:
            raise KeyError(f"key path {'.'.join(keypath)} not found in {path}")
    new = "\n".join(lines) + "\n"
    parsed = yaml.safe_load(new)
    for keypath, value in updates.items():
        node = parsed
        for k in keypath:
            node = node[k]
        if node != value and not (isinstance(value, float) and abs(node - value) < 1e-6):
            raise ValueError(f"YAML edit verification failed for {'.'.join(keypath)}")
    Path(path).write_text(new, encoding="utf-8")
