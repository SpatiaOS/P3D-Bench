<div align="center">

# P3D-Bench

**Benchmarking MLLMs for Parametric 3D Generation and Structural Reasoning**

[![Project Page](https://img.shields.io/badge/🌐%20Project-Page-blue.svg)](https://lucasqaq.github.io/p3d/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](#citation)
[![Data](https://img.shields.io/badge/🤗%20P3D--Dataset-coming%20soon-yellow.svg)](#data)

Yikang Yang<sup>1,†</sup> · Zhanpeng Hu<sup>1,†</sup> · Youtian Lin<sup>1</sup> · Mengqi Zhou<sup>1</sup> · Jingxi Xu<sup>2</sup> · Feihu Zhang<sup>2</sup> · Jiaheng Liu<sup>1</sup> · Yao Yao<sup>1,‡</sup>

<sup>1</sup>Nanjing University &nbsp;&nbsp; <sup>2</sup>Envision

<sub>† Equal contribution &nbsp;·&nbsp; ‡ Corresponding author</sub>

<img src="assets/teaser.png" width="100%" alt="Per-task model scores across Text-to-3D, Image-to-3D, and Assembly-3D."/>

<sub>Aggregate model scores on the three P3D-Bench tasks: Text-to-3D, Image-to-3D, Assembly-3D.</sub>

</div>

P3D-Bench is a benchmark for evaluating multimodal LLMs on **parametric 3D CAD
generation**. A model is given a condition (text, image, or image + part-level text),
writes a program in an **executable CAD format**, and the program is compiled, rendered,
and scored against ground-truth geometry across five metric buckets.

This repository is a **lightweight, pure-evaluation demo**. It is deliberately modular:
you can evaluate **one task × one output format × one metric bucket** in isolation. It
contains no training, data-curation, refinement, or dashboard code — just the evaluation
core plus a small demo split. The full **P3D-Dataset** is hosted on HuggingFace.

<div align="center">
<img src="assets/overview.png" width="100%" alt="P3D-Bench overview: three tasks, evaluated models and output formats, and the evaluation metric buckets (Geometry, Topology, Judge, Part)."/>
</div>

---

## Quick Start

```bash
# 1. Install (Python 3.10+). uv recommended.
uv sync                       # or: pip install -e .

# 2. Configure API keys (key NAMES are in the file; fill in values locally)
cp .env.example .env          # then edit .env

# 3. Get the demo data (full split: see "Data" below)
uv run p3dbench download --split demo

# 4. Run one task × one format × one metric, end-to-end
uv run p3dbench run \
  --task image-to-3d --format openscad --metric geometry \
  --model gpt-4o --split demo
```

The `run` command chains the four stages and writes results under `results/<run-id>/`.

---

## Tasks

| Task            | CLI slug       | Input                              | Output             | Main metrics                                  |
|-----------------|----------------|------------------------------------|--------------------|-----------------------------------------------|
| **Text-to-3D**  | `text-to-3d`   | Text specification                 | CAD program        | Valid, Geometry, Topology, Judge (QA-S/QA-P)  |
| **Image-to-3D** | `image-to-3d`  | Rendered image                     | CAD program        | Valid, Geometry, Topology, Judge (J-Geo/J-Sem)|
| **Assembly-3D** | `assembly-3d`  | Image + part-level text annotations| Multi-part program | Part (PartMatchF1, PartFS) + all of the above |

Assembly-3D adds a fixed **decomposition** step: a frozen MLLM splits the predicted
assembly into per-part programs so part-level matching can be scored. See
[docs/TASKS.md](docs/TASKS.md).

---

## Output formats

Each task emits one of four executable formats. A format module knows how to extract code
from the raw model response and compile it to STEP/STL.

| Format        | CLI slug       | Compiler / runtime          | Install extra            |
|---------------|----------------|-----------------------------|--------------------------|
| minimal JSON  | `minimal-json` | sketch-extrude → OCC (STEP) | `p3dbench[geometry]`     |
| OpenSCAD      | `openscad`     | `openscad` CLI              | system `openscad` binary |
| CadQuery      | `cadquery`     | CadQuery / OCP              | `p3dbench[cadquery]`     |
| Three.js      | `threejs`      | Node.js + vendored three.js | Node.js runtime          |

Not every task supports every format; the CLI validates `--format` against the task's
`supported_formats`. See [docs/FORMATS.md](docs/FORMATS.md).

---

## Metrics

Metrics are grouped into buckets. `Valid` is a gate (a prediction that does not compile
and render scores zero on geometry/topology). Select a bucket with `--metric <bucket>` or
`--metric all`.

| Bucket       | CLI slug   | Metrics                               | Requires                  |
|--------------|------------|---------------------------------------|---------------------------|
| **Valid**    | `valid`    | executable validity (gate)            | compile                   |
| **Geometry** | `geometry` | CD, F@.05, F@.01, NC, IoU             | mesh (STL)                |
| **Topology** | `topology` | NoOE, InvN, NM                        | mesh (STL)                |
| **Judge**    | `judge`    | QA-S, QA-P, J-Sem, J-Geo, J-Aes       | multiview render + judge model |
| **Part**     | `part`     | PartMatchF1, PartFS                   | per-part meshes (Assembly-3D) |

See [docs/METRICS.md](docs/METRICS.md) for exact definitions and alignment details.

---

## Modular evaluation

The defining feature of P3D-Bench is that **task, format, and metric are independent**.
Run a full pipeline in one command, or run each stage on its own and reuse cached
intermediates.

One combination, end-to-end:

```bash
p3dbench run --task image-to-3d --format openscad --metric geometry --model gpt-4o
```

Or stage-by-stage (each stage reads/writes a JSONL artifact — no hidden state):

```bash
p3dbench infer     --task text-to-3d --format cadquery --model gpt-4o   # → predictions.jsonl
p3dbench compile   --pred predictions.jsonl                              # → compiled.jsonl
p3dbench score     --compiled compiled.jsonl --metric topology           # → metrics.jsonl
p3dbench summarize --metrics metrics.jsonl                               # → summary.json
```

Useful flags:

- `--metric {valid,geometry,topology,judge,part,all}` — score a single bucket or all.
- `--limit N` — only the first N cases (smoke testing).
- `--dry-run` — build prompts / validate config without calling any model.
- `--split {demo,full}` — demo ships in-repo; full pulls from HuggingFace.

Because scoring reads a `compiled.jsonl`, you can re-score the *same* predictions under a
different metric bucket without re-running inference.

---

## API configuration

Bring your own keys. **Secrets never go in YAML** — `configs/models.yaml` holds only
metadata (provider, model id, base URL, the *name* of the env var to read the key from),
and `.env.example` lists the key names.

`.env.example` (copy to `.env` and fill in):

```bash
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=        # any OpenAI-compatible router
HF_TOKEN=                  # for downloading the full split
P3DBENCH_CACHE_DIR=.cache/p3dbench
```

`configs/models.yaml` (metadata only):

```yaml
models:
  gpt-4o:
    provider: openai
    model: gpt-4o
    api_key_env: OPENAI_API_KEY
    base_url: https://api.openai.com/v1
    max_output_tokens: 4096
    temperature: 0.0

  my-local-model:                       # any OpenAI-compatible endpoint
    provider: openai_compatible
    model: qwen2.5-vl-instruct
    api_key_env: OPENROUTER_API_KEY
    base_url: https://your-router.example.com/v1
```

Add a new model by adding a block here and the matching key in `.env`. See
[docs/API.md](docs/API.md).

---

## Data

- **Demo split** (3–5 cases per task) ships in [`data/demo/`](data/demo/) with manifests
  under [`data/manifests/`](data/manifests/). Enough to smoke-test the full pipeline.
- **Full P3D-Dataset** — 400 Text-to-3D / 400 Image-to-3D / 203 Assembly-3D cases,
  spanning easy → hard difficulty:

<div align="center">
<img src="assets/dataset_gallery.png" width="92%" alt="P3D-Dataset gallery spanning easy to hard difficulty for Text-to-3D and Image-to-3D."/>
</div>

> 🤗 **P3D-Dataset on HuggingFace: 🚧 Coming soon.**
> `p3dbench download --split full` will fetch and checksum-verify it once published.

GitHub holds code, configs, docs, the demo split, and data manifests. Ground-truth
geometry, renders, and annotations live on HuggingFace. See [docs/DATA.md](docs/DATA.md)
for the manifest schema and licensing/removal policy.

---

## Installation extras

The core install is light. Heavy geometry/render dependencies are optional and only
needed for the buckets that use them; a metric that needs a missing dependency degrades
gracefully with a clear message rather than crashing.

```bash
pip install p3dbench                       # core: CLI, model adapters, config
pip install p3dbench[geometry]             # OCC/OCP + trimesh → Geometry/Topology/Part
pip install p3dbench[render]               # pyrender / Blender → Judge renders
pip install p3dbench[cadquery]             # CadQuery format
pip install p3dbench[all]                  # everything
```

OpenSCAD and Three.js need external runtimes (`openscad` binary, Node.js) rather than pip
extras.

---

## Citation

If you use P3D-Bench, please cite our paper (arXiv link 🚧 coming soon; see also the
[project page](https://lucasqaq.github.io/p3d/) and [CITATION.cff](CITATION.cff)):

```bibtex
@article{p3d,
  title   = {P3D: Benchmarking MLLMs for Parametric 3D Generation and Structural Reasoning},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Code and data are licensed separately, and the data follows the terms of its upstream
sources (**non-commercial use only, with attribution**):

| Component | Source | License |
|-----------|--------|---------|
| **Benchmark code** (this repo) | — | MIT (see [LICENSE](LICENSE)) |
| **P3D-Dataset — Text-to-3D split** | derived from Text2CAD v1.1 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| **P3D-Dataset — Image-to-3D & Assembly-3D splits** | derived from [Fusion 360 Gallery Dataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset) | [Fusion 360 Gallery Dataset License](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md) (Autodesk, non-commercial) |

Both dataset sources permit **non-commercial research use only** and require
**attribution**; redistributed portions and modifications must carry the same
restrictions. By using the P3D-Dataset you agree to the upstream license terms. See
[docs/DATA.md](docs/DATA.md) for attribution text and the data-removal policy.

## Documentation

[DATA](docs/DATA.md) · [TASKS](docs/TASKS.md) · [FORMATS](docs/FORMATS.md) ·
[METRICS](docs/METRICS.md) · [API](docs/API.md) · [STRUCTURE](STRUCTURE.md)
