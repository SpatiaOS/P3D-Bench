"""Data download helper.

The demo split ships in-repo (``data/demo/`` + ``data/manifests/``). The full
P3D-Dataset (400 Text-to-3D / 400 Image-to-3D / 203 Assembly-3D) is hosted on
HuggingFace — *coming soon*. Once published this will pull it via the
``datasets`` / ``huggingface_hub`` API and verify checksums.
"""

from __future__ import annotations

from pathlib import Path

HF_REPO_ID = "SpatiaOS/P3D-Dataset"  # placeholder — coming soon


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
        "The full P3D-Dataset is hosted on HuggingFace and is coming soon.\n"
        f"  Repo (placeholder): https://huggingface.co/datasets/{HF_REPO_ID}\n"
        "Once published, this command will download geometry/renders/annotations\n"
        "into data/full/ and verify checksums. For now, use --split demo."
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download a P3D-Bench data split")
    ap.add_argument("--split", default="demo", choices=["demo", "full"])
    download(ap.parse_args().split)
