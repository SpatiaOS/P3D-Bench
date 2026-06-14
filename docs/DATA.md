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
- All `input`/`target` paths are relative to the split's data root. The current
  release includes the local `data/demo/` root.
- `metadata.difficulty` is the easy/medium/hard tier; `difficulty_raw` keeps the
  source complexity value, and `source_id` keeps the upstream case id.

## Splits

- **demo** — 3 cases per task in-repo (`data/demo/` + `data/manifests/*_demo.jsonl`).
  `p3dbench validate --split demo` checks manifest integrity and that every
  referenced file exists. Use it for a zero-setup smoke test.
- **full** — Text-to-3D 400 / Image-to-3D 400 / Assembly-3D 203 cases.
  [HuggingFace](https://huggingface.co/datasets/SpatiaOS/P3D-Bench) publishes the
  *redistributable* part — the final benchmark **UID lists**, the P3D-derived
  **text / assembly annotations** (`text_param`, `text_desc`, `summary`;
  per-part + assembly-level captions), and the Text-to-3D **QA banks**
  (`data/text_to_3d/qa.jsonl`). It does **not** redistribute upstream raw
  geometry. `p3dbench download --split full` downloads those UID lists +
  annotations and **materializes** an evaluator-ready `data/full/` tree +
  `data/manifests/*_full.jsonl` (identical layout to the demo) from a local
  `--source-root` holding the upstream working trees:

  ```bash
  # A) prebuilt research _shared_cache present -> one-click materialize:
  p3dbench download --split full --source-root /path/to/cad_dataset
  # B) only raw upstream present -> build the cache (PREPARE), then materialize:
  p3dbench prepare  --split full --source-root /path/to/cad_dataset
  p3dbench validate --split full
  # subset / quick check:  --tasks image-to-3d --limit 20
  ```

  Expected `--source-root` layout (obtain the upstream assets under their
  licenses — see below):

  ```
  <source-root>/fusion360/assembly/assembly/<uid>/assembly.step          # Image-/Assembly-3D GT STEP (union)
  <source-root>/fusion360/assembly/assembly/<uid>/<body_id>.step         # Assembly-3D per-body STEP (gt parts)
  <source-root>/fusion360/assembly/_shared_cache/<uid>/                  # built by PREPARE: GT STL, renders, gt_parts, manifest.json
  <source-root>/text2cad/minimal_json/<bucket>/<id>/minimal_json/<id>.json   # Text-to-3D GT program
  ```

  Image- and Assembly-3D copy GT STEP/STL/renders/parts straight from
  `_shared_cache`; Text-to-3D copies the GT minimal-JSON and **generates** the GT
  STEP + STL from it via the same interpreter used to compile predictions
  (Text2CAD STEP/STL are not cached for most cases). The build is idempotent and
  reports any UID whose upstream assets are missing. Text-to-3D **QA banks** come
  from the Hub `qa.jsonl` (so the Judge bucket works even on the from-raw PREPARE
  path, which has no local banks), falling back to a local prebuilt
  `text2cad/qa_bank/<bucket>/<id>/qa_bank.json` when the Hub file is absent. The
  Hub bank carries every text_mode×format variant; the judge selects the active
  run's at eval time. GT renders are materialized only where present locally;
  Geometry/Topology score for all cases.

  **Stage 2 — PREPARE (raw → `_shared_cache`).** When you only have the raw
  upstream (not the research-prepared cache), `p3dbench prepare` reproduces the
  per-case `_shared_cache/<uid>/` and then runs the same materialize step. It ports
  the research data-processing pipeline ([`p3dbench/data/prepare.py`](../p3dbench/data/prepare.py)):
  the **input** image is an **OCC single-view** render and the **judge** images are a
  **Blender clay multiview** set (no pyrender) — both are hard requirements, so
  PREPARE needs Xvfb + OCP (the `cadquery` extra) and a Blender binary on
  `$P3DBENCH_BLENDER`. Assembly per-part geometry is the raw `<uid>/<body_id>.step`
  files joined to the HF `part_level_annotations` by `part_id` (role / `description_short`
  become the per-part `semantic` label). The raw upstream is obtained under its own
  license and placed at `--source-root` (both Text2CAD v1.1 and the Fusion 360 Gallery,
  ~160 GB; there is no auto-download). On this from-raw path `difficulty` degrades to
  `"unknown"` (the review-MLLM
  `_filter/decision.json` is research-only). A prebuilt `_shared_cache` is auto-detected
  (`is_prepared`) and reused, so path A is byte-for-byte unchanged.

  **Footprint.** A complete materialization is large — the full set is ≈31 GB,
  dominated by upstream GT meshes (a few dozen Fusion 360 `gt_model.stl` files are
  heavily over-tessellated, up to ~1.4 GB / ~30 M triangles each; the median GT
  mesh is ~1.5 MB). They load and score correctly (the geometry metric samples the
  surface) but cost time and RAM. To keep it light use `--limit N` / `--tasks ...`
  for a subset, and `--max-edge 768` to downscale GT renders. A typical build
  reports a handful of skips (UIDs whose upstream assets are missing, or GT
  minimal-JSON that does not build valid geometry); these are listed, not fatal.

## Difficulty

Cases span **easy → medium → hard** complexity tiers assigned during dataset
construction by a review MLLM; the full dataset is complexity-balanced across
semantic categories. The demo split deliberately picks low-complexity cases.

## Construction (full dataset)

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
  with evaluator-ready data; the eval harness loads them, it does not regenerate
  them.

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
