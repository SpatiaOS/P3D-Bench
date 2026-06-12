"""Data download / materialization helper backing ``p3dbench download``.

- ``demo`` split: ships in-repo (``data/demo/`` + ``data/manifests/``); this only
  reports presence.
- ``full`` split: downloads the redistributable UID lists + P3D annotations from
  HuggingFace (``SpatiaOS/P3D-Bench``) and materializes an evaluator-ready
  ``data/full/`` tree + ``data/manifests/*_full.jsonl`` from a local
  ``--source-root`` (the Fusion 360 + Text2CAD working trees). The Hub does not
  redistribute upstream raw geometry, so the source root must hold it (obtain it
  from the upstream datasets under their licenses — see ``docs/DATA.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

HF_REPO_ID = "SpatiaOS/P3D-Bench"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"


def download(
    split: str = "demo",
    *,
    source_root: Optional[str] = None,
    tasks: Optional[tuple[str, ...]] = None,
    limit: Optional[int] = None,
    max_edge: int = 0,
    overwrite: bool = False,
    token: Optional[str] = None,
) -> None:
    if split == "demo":
        demo = Path("data/demo")
        manifests = Path("data/manifests")
        if demo.exists() and any(manifests.glob("*_demo.jsonl")):
            print("Demo split already present (ships in-repo under data/demo/).")
        else:
            print("Demo split missing — re-checkout the repo; data/demo/ is version-controlled.")
        return

    # ---- full split ----
    from ..data.full_builder import DEFAULT_SOURCE_ROOT, ALL_TASKS, build_full

    src = Path(source_root) if source_root else DEFAULT_SOURCE_ROOT
    sel_tasks = tasks or ALL_TASKS

    print(f"Full split: UID lists + annotations from {HF_URL}")
    if not src.exists():
        print(
            f"\nLocal source root not found: {src}\n"
            "The Hub publishes only UID lists + annotations, not upstream raw geometry.\n"
            "Obtain the upstream assets (Fusion 360 Gallery + Text2CAD) under their\n"
            "licenses and pass their location with `--source-root PATH` (see docs/DATA.md).\n"
            "Layout expected under <source-root>:\n"
            "  fusion360/assembly/{assembly,_shared_cache}/<uid>/...   (image- & assembly-3d)\n"
            "  text2cad/minimal_json/<bucket>/<id>/minimal_json/<id>.json   (text-to-3d)"
        )
        return

    print(f"Materializing into data/full/ from source-root={src}  tasks={','.join(sel_tasks)}"
          + (f"  limit={limit}" if limit else "") + (f"  max_edge={max_edge}" if max_edge else ""))
    report = build_full(
        source_root=src, tasks=tuple(sel_tasks), limit=limit, max_edge=max_edge,
        overwrite=overwrite, token=token,
    )
    print("\nfull split materialized:")
    for task, tr in report["tasks"].items():
        print(f"  {task:12s} built={tr['built']:4d}/{tr['requested']:<4d} "
              f"skipped={tr['skipped']:<4d} -> {tr['manifest']}")
        for uid, why in tr["skipped_detail"]:
            print(f"      - skip {uid}: {why}")
    print("\nNext: p3dbench validate --split full")


def prepare(
    *,
    source_root: Optional[str] = None,
    tasks: Optional[tuple[str, ...]] = None,
    limit: Optional[int] = None,
    max_edge: int = 0,
    overwrite: bool = False,
    token: Optional[str] = None,
    cache_only: bool = False,
) -> int:
    """Stage 2 (PREPARE): build ``_shared_cache`` from raw upstream, then materialize.

    Reads the Hub UID lists/annotations, builds the per-case ``_shared_cache``
    from the local raw tree at ``--source-root`` (Fusion 360 Gallery + Text2CAD)
    via the ported SharedDataCache pipeline, then runs the existing ``build_full``
    materialize (unless ``cache_only``). The raw upstream must be obtained under its
    own license and placed at ``--source-root`` (see docs/DATA.md). Returns an exit code.
    """
    from ..data.full_builder import (ALL_TASKS, DEFAULT_SOURCE_ROOT, build_full,
                                     load_hf_annotations, load_hf_uids)
    from ..data.prepare import prepare_task, render_modes_for_task

    src = Path(source_root) if source_root else DEFAULT_SOURCE_ROOT
    sel_tasks = tasks or ALL_TASKS

    if not src.exists():
        print(
            f"\nLocal source root not found: {src}\n"
            "PREPARE needs the upstream raw geometry (the Hub ships only UID lists +\n"
            "annotations). Obtain Fusion 360 Gallery + Text2CAD under their licenses,\n"
            "place them at --source-root PATH (see docs/DATA.md), and re-run."
        )
        return 1

    # Hard-dependency preflight: image/assembly judge renders need Blender; OCC
    # single-view needs Xvfb. No pyrender fallback (input=OCC, judge=Blender).
    needs_blender = any(
        any(m.startswith("multiview_") for m in render_modes_for_task(t)) for t in sel_tasks
    )
    if needs_blender:
        from ..render import blender

        if not blender.is_blender_available():
            print(
                "\nBlender is required for the judge multiview (image-/assembly-3d) but was\n"
                "not found. Set $P3DBENCH_BLENDER to a Blender binary, or restrict to\n"
                "--tasks text-to-3d. (Input renders also require Xvfb + OCP.)"
            )
            return 1

    print(f"PREPARE: building _shared_cache from raw under {src}  tasks={','.join(sel_tasks)}"
          + (f"  limit={limit}" if limit else ""))
    for task in sel_tasks:
        uids = load_hf_uids(task, token)
        anns = load_hf_annotations(task, token)
        rep = prepare_task(task, uids, src, anns, overwrite=overwrite, limit=limit)
        print(f"  cache/{task:12s} built={rep['built']:4d} skipped={rep['skipped']:<4d} "
              f"-> {rep['cache_root']}")
        for uid, why in rep["skipped_detail"]:
            print(f"      - skip {uid}: {why}")

    if cache_only:
        print("\n--cache-only: skipping data/full materialize.")
        return 0

    print("\nMaterializing data/full from the prepared cache ...")
    report = build_full(source_root=src, tasks=tuple(sel_tasks), limit=limit,
                        max_edge=max_edge, overwrite=overwrite, token=token)
    for task, tr in report["tasks"].items():
        print(f"  {task:12s} built={tr['built']:4d}/{tr['requested']:<4d} "
              f"skipped={tr['skipped']:<4d} -> {tr['manifest']}")
    print("\nNext: p3dbench validate --split full")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download / materialize a P3D-Bench data split")
    ap.add_argument("--split", default="demo", choices=["demo", "full"])
    ap.add_argument("--source-root")
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-edge", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    download(a.split, source_root=a.source_root, tasks=tuple(a.tasks) if a.tasks else None,
             limit=a.limit, max_edge=a.max_edge, overwrite=a.overwrite)
