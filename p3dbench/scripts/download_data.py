"""Data download helper.

The demo split ships in-repo (``data/demo/`` + ``data/manifests/``). The full
P3D-Dataset (400 Text-to-3D / 400 Image-to-3D / 203 Assembly-3D) is hosted on
HuggingFace: https://huggingface.co/datasets/SpatiaOS/P3D-Bench . This pulls it
via ``huggingface_hub.snapshot_download`` into ``data/full/``.
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

    # Full split: fetch from HuggingFace into data/full/.
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            f"The full P3D-Dataset is on HuggingFace: {HF_URL}\n"
            "Install the client to download it automatically:\n"
            "  pip install huggingface_hub\n"
            "  p3dbench download --split full\n"
            "or fetch manually:\n"
            f"  huggingface-cli download {HF_REPO_ID} --repo-type dataset --local-dir data/full"
        )
        return

    print(f"Downloading the full P3D-Dataset from {HF_URL} into data/full/ ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir="data/full",
    )
    print("Done. Run `p3dbench validate --split full` to verify.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download a P3D-Bench data split")
    ap.add_argument("--split", default="demo", choices=["demo", "full"])
    download(ap.parse_args().split)
