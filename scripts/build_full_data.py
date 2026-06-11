#!/usr/bin/env python3
"""One-click builder for the full P3D-Bench split (thin wrapper around the package).

Downloads the redistributable UID lists + P3D annotations from HuggingFace and
materializes ``data/full/`` + ``data/manifests/*_full.jsonl`` from a local
``--source-root`` (the Fusion 360 + Text2CAD working trees). Equivalent to:

    p3dbench download --split full [--source-root PATH] [--tasks ...] [--limit N]

Run from the repo root, e.g.:

    python scripts/build_full_data.py                       # all tasks, full set
    python scripts/build_full_data.py --tasks image-to-3d --limit 20
"""

from __future__ import annotations

import argparse
import logging

from p3dbench.data.full_builder import ALL_TASKS, DEFAULT_SOURCE_ROOT
from p3dbench.scripts.download_data import download


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT),
                    help=f"local upstream working tree (default: {DEFAULT_SOURCE_ROOT})")
    ap.add_argument("--tasks", nargs="*", choices=list(ALL_TASKS), help="subset of tasks (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N cases per task")
    ap.add_argument("--max-edge", type=int, default=0,
                    help="downscale GT render longest edge to N px (0 = keep full resolution)")
    ap.add_argument("--overwrite", action="store_true", help="re-materialize existing cases")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    download("full", source_root=a.source_root, tasks=tuple(a.tasks) if a.tasks else None,
             limit=a.limit, max_edge=a.max_edge, overwrite=a.overwrite)


if __name__ == "__main__":
    main()
