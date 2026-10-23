# SPDX-License-Identifier: Apache-2.0
"""Frame lists for the CARLA, KITTI and Cityscapes roots used by the feature extractors.

Moved unchanged from the original extractor so that the DINOv3, DINOv2 and CLIP extractors do not depend on any
other model's code.
"""
from __future__ import annotations

import json
from pathlib import Path


def gather_kitti(root: Path) -> tuple[list[Path], list[str]]:
    paths = sorted(root.glob("*.png"))
    return paths, [p.name for p in paths]


def gather_cityscapes(root: Path) -> tuple[list[Path], list[str]]:
    paths = sorted(root.rglob("*_leftImg8bit.png"))
    rels = [str(p.relative_to(root)) for p in paths]
    return paths, rels


def gather_carla(root: Path) -> tuple[list[Path], list[str], list[str]]:
    paths: list[Path] = []
    rels: list[str] = []
    cells: list[str] = []
    for cell_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = cell_dir / "manifest.jsonl"
        if manifest.exists():
            with manifest.open() as f:
                for line in f:
                    rec = json.loads(line)
                    fpath = cell_dir / rec["frame"]
                    if fpath.exists():
                        paths.append(fpath)
                        rels.append(f"{cell_dir.name}/{rec['frame']}")
                        cells.append(cell_dir.name)
        else:
            for fpath in sorted(cell_dir.glob("frame_*.jpg")):
                paths.append(fpath)
                rels.append(f"{cell_dir.name}/{fpath.name}")
                cells.append(cell_dir.name)
    return paths, rels, cells


