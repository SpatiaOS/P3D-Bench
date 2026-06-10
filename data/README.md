# P3D-Dataset

Current data support:

- **`demo`** — 3 cases per task, shipped in-repo under [`demo/`](demo/) with
  manifests in [`manifests/`](manifests/). Enough to smoke-test the whole
  pipeline (`p3dbench validate --split demo`).
- **`full`** — metadata for the complete **P3D-Dataset** is hosted on
  [🤗 HuggingFace](https://huggingface.co/datasets/SpatiaOS/P3D-Bench), but the full
  geometry/render/QA assets are not currently published in the manifest layout consumed by
  this evaluator. Full-split evaluation is therefore not enabled in this release.

## Layout (per split, rooted at `data/<split>/`)

```
inputs/<id>/view_000.png             # input image(s)  (Image-/Assembly-3D)
targets/step/<id>.step               # GT STEP
targets/mesh/<id>.stl                # GT mesh (tessellated)
targets/minimal-json/<id>.json       # GT program           (Text-to-3D only)
targets/renders/<id>/view_00N.png    # GT multiview renders (Judge bucket)
targets/parts/<id>/part_NN.stl       # GT per-part meshes   (Assembly-3D only)
targets/qa/<id>.json                 # prebuilt QA bank      (Text-to-3D Judge)
```

Manifests (`manifests/<task>_<split>.jsonl`, one JSON row per case) carry the
record schema in [`p3dbench/data/schema.py`](../p3dbench/data/schema.py); all
`input`/`target` paths are relative to the split's data root. See
[`docs/DATA.md`](../docs/DATA.md) for the field-by-field description, the
difficulty buckets, and the licensing / removal policy.

## Sources & licensing

The P3D-Dataset is derived from two upstream datasets; **non-commercial research
use only, with attribution**. Each manifest row records its origin in
`metadata.source` / `metadata.license_group`.

| Task split | Derived from | License |
|---|---|---|
| Text-to-3D | Text2CAD v1.1 (Khan et al., NeurIPS 2024) | CC BY-NC-SA 4.0 |
| Image-to-3D, Assembly-3D | Fusion 360 Gallery Dataset (Willis et al., 2021) | Fusion 360 Gallery Dataset License (Autodesk, non-commercial) |

By using the P3D-Dataset you agree to the upstream license terms.

## Regenerating the demo split

The demo split was assembled by [`scripts/build_demo_data.py`](../scripts/build_demo_data.py)
from the local research datasets. It is a one-off helper (not part of the
package) and is only runnable where those source paths exist; the produced
`demo/` + `manifests/` are version-controlled, so end users never need it.
