# P3D-Bench — Repository Structure

This document specifies the full layout of the P3D-Bench public release, the
responsibility of each module, the modular dispatch design, and the map from this clean
repo back to the original research code (`cadbenchmark/`, `text2cad-workbench/`). It is
the blueprint for the code-porting pass.

Paper: *P3D: Benchmarking MLLMs for Parametric 3D Generation and Structural Reasoning*
(arXiv coming soon) · Project page: https://lucasqaq.github.io/p3d/

---

## 1. Directory tree

```
P3D-Bench/
├── README.md                 # Project overview, quick start, tables
├── STRUCTURE.md              # This file
├── LICENSE                   # MIT
├── CITATION.cff              # Machine-readable citation
├── pyproject.toml            # Package + optional extras ([geometry], [render], [cadquery], [all])
├── .env.example              # API-key NAMES only — never values
├── .gitignore
│
├── p3dbench/                 # The Python package
│   ├── __init__.py
│   ├── cli.py                # Entry point. Subcommands: infer | compile | score | summarize | run | download | validate
│   ├── config.py             # Load configs/models.yaml; resolve API keys from env (.env)
│   ├── registry.py           # TASK / FORMAT / METRIC registries → modular dispatch
│   │
│   ├── data/
│   │   ├── schema.py         # Manifest record dataclass + JSON schema
│   │   ├── loader.py         # Load demo manifests (local) or full split (HuggingFace)
│   │   └── validate.py       # Manifest integrity + checksum verification
│   │
│   ├── models/               # Provider adapters (BYOK, single-shot, no retry loops)
│   │   ├── base.py           # ModelClient ABC: generate(prompt, images) -> raw_text + usage
│   │   ├── openai_compatible.py
│   │   ├── anthropic.py
│   │   ├── gemini.py
│   │   └── registry.py       # provider-name → client class
│   │
│   ├── tasks/                # One module per task
│   │   ├── base.py           # Task ABC: build_prompt, supported_formats, condition_inputs
│   │   ├── text_to_3d.py
│   │   ├── image_to_3d.py
│   │   └── assembly_3d.py    # + fixed decomposition step (single call, no refine wrapper)
│   │
│   ├── formats/              # One module per executable CAD format
│   │   ├── base.py           # Format ABC: extract_code, compile -> STEP/STL, system_guidelines
│   │   ├── minimal_json.py
│   │   ├── openscad.py
│   │   ├── cadquery.py
│   │   └── threejs.py
│   │
│   ├── compile/              # Code → geometry
│   │   ├── exporter.py       # Dispatch raw code to the right format compiler; emit STEP/STL + valid flag
│   │   ├── step_mesh.py      # STEP → mesh (OCC/OCP); tessellation
│   │   └── sandbox.py        # Subprocess isolation + timeout for untrusted code
│   │
│   ├── render/               # Optional (extra: render)
│   │   ├── occ.py            # Fast single-view / pyrender multiview
│   │   └── blender.py        # Studio-quality multiview for Judge
│   │
│   ├── metrics/              # One module per bucket
│   │   ├── base.py           # Metric ABC: requires{mesh,render,judge,parts}, compute(gt,pred), bucket
│   │   ├── valid.py          # Executable-validity gate
│   │   ├── geometry.py       # CD, F@.05, F@.01, NC, IoU (+ mesh alignment)
│   │   ├── topology.py       # NoOE, InvN, NM
│   │   ├── judge.py          # QA-S, QA-P, J-Sem, J-Geo, J-Aes (LLM-as-judge + QA)
│   │   └── part.py           # PartMatchF1, PartFS (Assembly-3D only)
│   │
│   └── utils/                # JSONL I/O, logging, image encoding, paths
│
├── configs/
│   ├── models.yaml           # Model metadata (provider/model/base_url/api_key_env/limits)
│   ├── judge.yaml            # Judge model + scoring rubric
│   └── tasks/
│       ├── text_to_3d_demo.yaml
│       ├── image_to_3d_demo.yaml
│       └── assembly_3d_demo.yaml
│
├── data/
│   ├── README.md             # HuggingFace "coming soon" + checksum note
│   ├── demo/                 # 3–5 cases per task (placeholder until data publish)
│   └── manifests/
│       ├── text_to_3d_demo.jsonl
│       ├── image_to_3d_demo.jsonl
│       └── assembly_3d_demo.jsonl
│
├── docs/
│   ├── DATA.md               # Manifest schema, splits, licensing & removal policy
│   ├── TASKS.md              # Task definitions, conditions, decomposition step
│   ├── FORMATS.md            # Format specs, compilers, runtime requirements
│   ├── METRICS.md            # Metric definitions, alignment, judge rubric
│   └── API.md                # Adding models / providers / OpenAI-compatible endpoints
│
├── scripts/
│   └── download_data.py      # HuggingFace download (coming-soon stub) + checksum verify
│
└── examples/
    └── run_smoke.sh          # Minimal demo run across all three tasks
```

---

## 2. Design principles

1. **Modular registries.** Tasks, formats, and metrics are independent plug-ins resolved
   at runtime. Adding a task/format/metric is adding one module + one registry entry —
   nothing else changes.
2. **Staged & cacheable.** The pipeline is four stages (`infer → compile → score →
   summarize`), each reading and writing a plain JSONL artifact. Re-score the same
   predictions under a different metric without re-inferring.
3. **No production logic.** No resume/checkpoint, no refine/retry loops, no agent loops,
   no multi-provider relay matrix, no curation/annotation, no dashboards. Single-shot
   evaluation only.
4. **BYOK.** Keys come from the environment; YAML holds only metadata.
5. **Data on HuggingFace.** GitHub holds code + demo; full geometry/renders/annotations
   live on the Hub.
6. **Heavy deps are optional.** Geometry/render stacks are pip extras; a metric whose
   dependency is missing reports that clearly instead of crashing.

---

## 3. Modular dispatch

Three registries in `p3dbench/registry.py` make the three axes orthogonal:

### Task (`tasks/base.py::Task`)
```
build_prompt(case) -> str | messages      # task- and format-aware prompt
supported_formats  -> list[str]           # which formats this task accepts
condition_inputs   -> {"text"|"image"|"image+part_text"}
```
`assembly_3d.py` additionally implements `decompose(prediction) -> per-part programs` — a
single frozen-MLLM call (no refine wrapper) needed before Part metrics.

### Format (`formats/base.py::Format`)
```
extract_code(raw_text) -> str             # pull the program out of the model response
compile(code) -> CompileResult            # STEP + STL + valid flag (delegates to compile/)
system_guidelines -> str                  # format-specific instructions injected into prompt
```

### Metric (`metrics/base.py::Metric`)
```
bucket   -> "valid"|"geometry"|"topology"|"judge"|"part"
requires -> set of {"mesh","render","judge_model","parts"}
compute(gt, pred) -> dict[str, float]
```

**Resolving a `--task T --format F --metric M` triple:**
1. Look up `T` in TASK registry; assert `F ∈ T.supported_formats`.
2. Inference builds the prompt with `T.build_prompt` + `F.system_guidelines`, calls the
   model client, stores raw text → `predictions.jsonl`.
3. Compile runs `F.compile` per prediction → `compiled.jsonl` (with `valid`, STEP/STL paths).
4. Score loads the metrics for bucket `M`; for each, `requires` triggers any needed
   render/decomposition; `compute(gt, pred)` → `metrics.jsonl`.
5. Summarize aggregates per (task, format, model, metric) → `summary.json`.

Each axis is selected independently, so "one task × one format × one metric" is the
natural unit of evaluation.

---

## 4. CLI reference

```
p3dbench infer      --task <slug> --format <slug> --model <name> [--split demo|full]
                    [--limit N] [--dry-run] [--out predictions.jsonl]

p3dbench compile    --pred predictions.jsonl [--out compiled.jsonl]

p3dbench score      --compiled compiled.jsonl --metric <bucket|all>
                    [--judge-model <name>] [--out metrics.jsonl]

p3dbench summarize  --metrics metrics.jsonl [--out summary.json]

p3dbench run        --task <slug> --format <slug> --metric <bucket|all> --model <name>
                    [--split] [--limit] [--dry-run]      # chains the four stages

p3dbench download   --split demo|full                    # full = HuggingFace (coming soon)
p3dbench validate   --split demo|full                    # manifest integrity + checksums
```

Artifacts flow `predictions.jsonl → compiled.jsonl → metrics.jsonl → summary.json`, all
under `results/<run-id>/`. No stage carries hidden state; any stage can be re-run against
an existing upstream artifact.

---

## 5. Source → target extraction map

What to port from the research repos, with production logic removed. (Ported in a later
pass — this release is design docs only.)

| Target module                  | Source (cleaned)                                                             |
|--------------------------------|------------------------------------------------------------------------------|
| `metrics/geometry.py`          | `cadbenchmark/metrics/geometry_metrics.py`                                   |
| `metrics/topology.py`          | `cadbenchmark/metrics/geometry_metrics.py` (validity/topology) + `sequence_metrics.py` |
| `metrics/judge.py`             | `cadbenchmark/metrics/llm_judge.py` + `text2cad_qa.py` + `qa_metrics.py`     |
| `metrics/part.py`              | `cadbenchmark/metrics/assembly_part_eval.py`                                 |
| `metrics/valid.py`             | validity gate inside `cadbenchmark/export/cad_exporter.py`                   |
| `compile/exporter.py`          | `cadbenchmark/export/cad_exporter.py`                                        |
| `compile/step_mesh.py`         | `cadbenchmark/export/step_mesh.py` (+ `text2cad_exporter.py` for minimal-json) |
| `render/occ.py`, `blender.py`  | `cadbenchmark/render/{render_single_view,render_multiview,render_blender_clay}.py` |
| `tasks/*`                      | `cadbenchmark/model/tasks.py` (3 tasks; drop reverse_engineering / json2cad / QA-understanding) |
| `tasks/assembly_3d.py` decomp  | stage-2 split in `cadbenchmark/runner/batch_runner.py::run_stage2_split` (loop stripped) |
| `formats/*`                    | format defs in `cadbenchmark/model/cad_formats.py`                           |
| `tasks/text_to_3d.py` data     | `text2cad-workbench/src/text2cad_bench/text2cad_step.py` (minimal-json → STEP) |
| `models/*`                     | thin rewrite of `cadbenchmark/model/llm/llm_client.py` (single-shot; drop relay matrix) |
| `data/loader.py`               | rewrite — HuggingFace `datasets` loader + local manifests (replaces bespoke loaders) |

### Explicitly NOT ported (production cruft)

- `cadbenchmark/runner/refine.py`, `rerender.py`, `rejudge_failed.py`
- `cadbenchmark/runner/batch_runner.py` (replaced by the staged CLI)
- `cadbenchmark/runner/eval_pool.py`, `eval_worker.py` (subprocess pool / forkserver)
- `cadbenchmark/model/llm/retry.py` (error-feedback retry loop)
- `cadbenchmark/model/third_party/aetherion_runner.py` and all other `third_party/*`
  local-model runners (cadrille, cadcoder, text2cad generator)
- checkpoints / resume / multi-machine sync
- `cadbenchmark/{temp,visual,aggregate_metric,shell}/`
- fair-aggregation / worst-fill penalty (`metrics/fair_aggregation.py`)
- multi-provider ppapi/aionly relay configs (`config_ppapi.yaml`, `config_aionly.yaml`)
- text2cad-workbench: `relabel.py`, `screen*.py`, `assemble.py`, `anchors.py`,
  `cv_dedup.py`, `web.py`, `indexer.py`, `geometry_profile.py`, `queue_policy.py`,
  and all `scripts/` curation/visualization helpers

---

## 6. Naming map (paper ↔ old code)

| Paper / release term       | CLI slug       | Old code term                        |
|----------------------------|----------------|--------------------------------------|
| P3D-Bench (benchmark)      | —              | cadbenchmark / text2cad-workbench    |
| Text-to-3D                 | `text-to-3d`   | text2cad                             |
| Image-to-3D                | `image-to-3d`  | image2cad                            |
| Assembly-3D                | `assembly-3d`  | textimage2cad / text_image2cad       |
| minimal JSON               | `minimal-json` | json                                 |
| OpenSCAD                   | `openscad`     | openscad                             |
| CadQuery                   | `cadquery`     | cadquery                             |
| Three.js                   | `threejs`      | threejs                              |
| Valid / Geometry / Topology / Judge / Part | `valid` `geometry` `topology` `judge` `part` | scattered metric modules |
| P3D-Dataset                | —              | Text2CAD v1.1 + Fusion 360 Gallery loaders |

---

## 7. Data manifest record schema

One JSONL row per case (`data/manifests/<task>_<split>.jsonl`):

```json
{
  "id": "p3d_image-to-3d_000001",
  "task": "image-to-3d",
  "split": "demo",
  "input": {
    "text": "Create a stepped cylindrical part ...",
    "image_paths": ["inputs/p3d_image-to-3d_000001/view_000.png"],
    "part_annotations": []
  },
  "target": {
    "format": "minimal-json",
    "code_path": "targets/minimal-json/p3d_image-to-3d_000001.json",
    "step_path": "targets/step/p3d_image-to-3d_000001.step",
    "render_paths": ["targets/renders/p3d_image-to-3d_000001/view_000.png"]
  },
  "metadata": {
    "source": "fusion360-gallery",
    "license_group": "check-upstream",
    "difficulty": "medium"
  }
}
```

- `input.text` is empty for Image-to-3D; `input.image_paths` is empty for Text-to-3D;
  Assembly-3D populates `image_paths` and `part_annotations`.
- `target.*` paths are relative to the data root (demo: in-repo; full: HuggingFace).
- `metadata.license_group` flags upstream licensing for the removal policy in
  [docs/DATA.md](docs/DATA.md).
