# Data

## Manifest record schema

One JSONL row per case under `data/manifests/<task>_<split>.jsonl`
([`p3dbench/data/schema.py`](../p3dbench/data/schema.py)):

```json
{
  "id": "p3d_image-to-3d_000001",
  "task": "image-to-3d",
  "split": "demo",
  "input": {
    "text": "",
    "image_paths": ["inputs/p3d_image-to-3d_000001/view_000.png"],
    "part_annotations": []
  },
  "target": {
    "format": "step",
    "code_path": null,
    "step_path": "targets/step/p3d_image-to-3d_000001.step",
    "mesh_path": "targets/mesh/p3d_image-to-3d_000001.stl",
    "render_paths": ["targets/renders/p3d_image-to-3d_000001/view_000.png"],
    "part_paths": [],
    "qa_bank_path": null
  },
  "metadata": {"source": "fusion360-gallery", "source_id": "132535_a374e751",
               "license_group": "fusion360-gallery", "difficulty": "easy"}
}
```

- `input.text` is empty for Image-to-3D; `input.image_paths` is empty for
  Text-to-3D; Assembly-3D populates both plus `part_annotations`.
- `target.code_path` is the GT program for Text-to-3D (`minimal-json`); it is
  `null` when the GT is a direct STEP (`format: "step"`, Image-/Assembly-3D).
- All `input`/`target` paths are relative to the split's data root
  (`data/demo/` in-repo; `data/full/` after download).
- `metadata.difficulty` is the easy/medium/hard tier; `difficulty_raw` keeps the
  source complexity value, and `source_id` keeps the upstream case id.

## Splits

- **demo** — 3 cases per task in-repo. `p3dbench validate --split demo` checks
  manifest integrity and that every referenced file exists.
- **full** — 400 / 400 / 203 cases on
  [HuggingFace](https://huggingface.co/datasets/SpatiaOS/P3D-Bench)
  (`p3dbench download --split full`).

## Difficulty

Cases span **easy → medium → hard** complexity tiers assigned during dataset
construction by a review MLLM; the full split is complexity-balanced across
semantic categories. The demo split deliberately picks low-complexity cases.

## Construction (full split)

- **Text-to-3D** derives from **Text2CAD v1.1**: unevaluable records are dropped,
  survivors ranked by a geometric-complexity heuristic, top candidates kept. The
  descriptive + parametric specifications are written by an annotation MLLM from a
  structured geometric record parsed from the source program, then validated
  number-by-number against that record.
- **Image-/Assembly-3D** derive from the **Fusion 360 Gallery Dataset**.
  Assemblies are kept only with ≤ 20 deduplicated parts; per-part labels and an
  assembly-level caption are written by an annotation MLLM and cross-checked by a
  verification MLLM. Near-duplicates are removed by DINOv2-embedding cosine
  similarity.
- **QA banks** (Text-to-3D Judge) are generated per case and verified, then ship
  with the data — the eval harness loads them, it does not regenerate them.

These steps are documented for provenance; the curation/annotation tooling is
**not** part of this release (it lived in the research repos).

## Licensing & removal policy

The dataset is **non-commercial research use only, with attribution**. Each row's
`metadata.license_group` flags its upstream:

| `license_group` | Upstream | License |
|---|---|---|
| `cc-by-nc-sa-4.0` | Text2CAD v1.1 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| `fusion360-gallery` | Fusion 360 Gallery Dataset | [Autodesk, non-commercial](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md) |

If an upstream rights-holder requests removal of any case, open an issue citing
the `source_id`; the case will be removed from the published split.
