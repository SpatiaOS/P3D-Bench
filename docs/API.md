# Adding models, providers, and endpoints

## Register a model

Add a block to [`configs/models.yaml`](../configs/models.yaml) and the matching
key name to your `.env`. YAML holds only metadata — the `api_key_env` field names
the environment variable that holds the secret (BYOK; keys never go in YAML).

```yaml
models:
  my-model:
    provider: openai_compatible      # openai_compatible | anthropic | gemini
    model: provider-model-id
    api_key_env: MY_API_KEY
    base_url: https://api.example.com/v1
    temperature: 0.0
    max_tokens: 16384
```

Then `p3dbench run --model my-model ...`.

## Providers

Three adapters, one logical request per `generate` (one prompt → one response;
no relay routing):

| `provider` | Adapter | Endpoint | Auth header | Notes |
|---|---|---|---|---|
| `openai_compatible` (alias `openai`, `openrouter`) | `models/openai_compatible.py` | `{base_url}/chat/completions` | `Authorization: Bearer` | Covers OpenAI, OpenRouter, vLLM, LM Studio, … |
| `anthropic` | `models/anthropic.py` | `{base_url}/messages` | `x-api-key` | Native Messages API |
| `gemini` | `models/gemini.py` | `{base_url}/models/{model}:generateContent` | `x-goog-api-key` | Generative Language API |

Any OpenAI-compatible router works through `openai_compatible` — just point
`base_url` at it.

### Transport retry

Every adapter retries transient transport failures with exponential backoff
(5·2ⁿ s) via [`models/_retry.py`](../p3dbench/models/_retry.py): HTTP **429 / 5xx**,
`Timeout` / `ConnectionError` / `ChunkedEncodingError`, and HTTP-**200**
provider-error bodies (e.g. OpenRouter `{"error": {"code": 522}}` with no
`choices`). The budget is **2 attempts** (1 retry) by default, override with
`P3DBENCH_API_MAX_RETRIES`. This is purely transport robustness — the *content*
is still one model response.

### Error-feedback refinement (image-/assembly-3d)

For **image-to-3d** and **assembly-3d**, `infer` runs a compile-check-retry loop:
if a generated program fails to compile, the error (plus any line/traceback
diagnostics, and an escalation note when the same error repeats) is fed back and
the model regenerates, up to `--refine-attempts` times (**default 3**; `1`
disables). An LLM-side failure (call error or empty extraction) or an
export-timeout-only failure stops the loop early. **Text-to-3D is always
single-shot** (no refine). The loop's intermediate compiles only drive the
feedback; the authoritative STEP/STL come from the separate `compile` stage.

## Judge & decomposition models

The Judge and Part buckets use the models named in
[`configs/judge.yaml`](../configs/judge.yaml) (`judge_model`, `decompose_model`);
both must resolve to blocks in `models.yaml`. The paper used Gemini 3.1 Pro as the
judge and Claude Opus 4.6 for Assembly-3D decomposition.

## Adding a task / format / metric

Each axis is a registry plug-in — adding one is adding one module plus one
registry entry, nothing else:

- **Task**: subclass `tasks.base.Task` (set `slug`, `supported_formats`,
  `condition_inputs`, implement `build_prompt`); register in `tasks/__init__.py`.
- **Format**: instantiate `formats.base.Format` (system guidelines + fence
  languages) and add a compile branch in `compile/exporter.py`; register in
  `formats/__init__.py`.
- **Metric**: subclass `metrics.base.MetricBucket` (set `bucket`, `requires`,
  implement `score`), expose `BUCKET`, and add raw sub-metric keys + their
  normalization to `metrics/base.py::METRIC_SPECS`.

## Images

All adapters encode images identically: RGB JPEG, longest edge capped at 1536 px,
quality 85, base64. Pass local file paths; the task builder attaches the right
ones per condition.
