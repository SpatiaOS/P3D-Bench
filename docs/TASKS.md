# Tasks

P3D-Bench covers three task families under one protocol. Each is selected with
`--task`, and each accepts a subset of the four output formats.

| Task | Slug | Condition | Formats | Cases (full) |
|------|------|-----------|---------|--------------|
| **Text-to-3D**  | `text-to-3d`  | text | `minimal-json`, `openscad` | 400 |
| **Image-to-3D** | `image-to-3d` | one rendered image | `openscad`, `cadquery`, `threejs` | 400 |
| **Assembly-3D** | `assembly-3d` | image + assembly/part text | `openscad`, `cadquery` | 203 |

The CLI enforces these format sets ([`tasks/*`](../p3dbench/tasks/) declare
`supported_formats`); `minimal-json` is excluded from assemblies (too limited for
multi-part) and `threejs` from Assembly-3D (triangle meshes don't decompose into
per-part solids).

## Text-to-3D

Two text conditions per case, picked with `--text-mode`:

- **`parametric`** (default) — full specification with dimensions, counts,
  offsets. Scored on Geometry (incl. IoU_C), Topology, Judge (QA-S + QA-P), Valid.
  Geometry alignment **preserves absolute scale** (center-only; no scale refine).
- **`descriptive`** — shape/features/function, *no exact dimensions*. Scored on a
  single semantic Judge axis (QA-S + J-Sem).

The model receives the text plus the format's system guidelines; no image.

## Image-to-3D

The model receives **one** rendered image (no text) and reproduces the object.
Scored on Geometry (IoU_V), Topology, Judge (J-Sem / J-Geo / J-Aes), Valid.

## Assembly-3D

The model receives one render **and** a structured text blob (overall caption +
part inventory, possibly with an "Annotation Caveats" section) and outputs a
single unified program. Adds the **Part** bucket on top of Geometry / Topology /
Judge / Valid.

### Decomposition step (single call, no refine)

Part metrics need per-part geometry, so a fixed decomposition model
(Claude Opus 4.6 in the paper, set in [`configs/judge.yaml`](../configs/judge.yaml))
converts the stage-1 unified program into a parts-structured form via a single
call — `Assembly3DTask.build_decompose_prompt`. **Anti-leak invariant:** the
decomposition model sees only the stage-1 code (and optionally one render of its
own union), never the GT part inventory, so part-name alignment stays honest. A
decomposition that redesigns the geometry (fidelity CD > 5e-4 **and** IoU_V <
0.95 vs the stage-1 union) excludes the case from Part means rather than scoring
it wrongly.

The research code wrapped this in a retry loop; the release keeps the single
frozen call only.
