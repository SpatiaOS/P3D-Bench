"""Data download helper.

The demo split ships in-repo (``data/demo/`` + ``data/manifests/``). Full
P3D-Dataset metadata is hosted on HuggingFace, but evaluator-ready full-split
geometry/render/QA assets are not currently published in the local manifest
layout consumed by this package.
"""

from __future__ import annotations

from pathlib import Path

HF_REPO_ID = "SpatiaOS/P3D-Bench"
HF_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"


def download(split: str = "demo") -> None:
    if split == "demo":
        demo = Path("data/demo")
        manifests = Path("data/manifests")
        if demo.exists() and any(manifests.glob("*.jsonl")):
            print("Demo split already present (ships in-repo under data/demo/).")
        else:
            print("Demo split missing — re-checkout the repo; data/demo/ is version-controlled.")
        return

    print(
        "Full-split evaluation is not enabled in this release.\n"
        f"Metadata is available on HuggingFace: {HF_URL}\n"
        "Use `p3dbench validate --split demo` and `examples/run_smoke.sh` for the "
        "local demo cases."
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download a P3D-Bench data split")
    ap.add_argument("--split", default="demo", choices=["demo", "full"])
    download(ap.parse_args().split)
