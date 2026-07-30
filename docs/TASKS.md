# Tasks

P3D-Bench covers three task families under one protocol. Each is selected with
`--task`, and each accepts a subset of the four output formats.

| Task | Slug | Condition | Formats | Full dataset size |
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
Generation and the descriptive J-Sem judge resolve the condition through the same
helper ([`p3dbench/text_condition.py`](../p3dbench/text_condition.py)), so the judge
scores against the exact text the model was shown (`metadata.text_desc` in
descriptive mode, with the documented parametric fallback when it is absent).

J-Sem needs 4 GT views. Unlike the other two tasks, Text-to-3D has no GT renders
to ship — its GT mesh is built locally from the GT program — so the Judge bucket
renders them from that mesh at eval time, through the same backend it uses for the
prediction views, and caches them next to the mesh. `download` therefore stays a
pure download and needs no render backend.

> **Demo split note.** The in-repo **demo** split currently carries only the
> parametric text. On the demo split `--text-mode descriptive` therefore selects
> the descriptive *metric panel* but feeds the parametric text. Use the parametric
> mode for demo smoke tests.

## Image-to-3D

The model receives **one** rendered image (no text) and reproduces the object.
Scored on Geometry (IoU_V), Topology, Judge (J-Sem / J-Geo / J-Aes), Valid.

> **Refinement.** Image-to-3D and Assembly-3D generation runs a compile-check-retry
> loop (`--refine-attempts`, default 3): an invalid compile is fed back to the model
> with the error for up to N attempts. Text-to-3D stays single-shot. See
> [API.md](API.md#error-feedback-refinement-image-assembly-3d).

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
