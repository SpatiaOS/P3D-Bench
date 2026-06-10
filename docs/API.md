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

Three adapters, all single-shot (one prompt → one response; no retry/relay):

| `provider` | Adapter | Endpoint | Auth header | Notes |
|---|---|---|---|---|
| `openai_compatible` (alias `openai`, `openrouter`) | `models/openai_compatible.py` | `{base_url}/chat/completions` | `Authorization: Bearer` | Covers OpenAI, OpenRouter, vLLM, LM Studio, … |
| `anthropic` | `models/anthropic.py` | `{base_url}/messages` | `x-api-key` | Native Messages API |
| `gemini` | `models/gemini.py` | `{base_url}/models/{model}:generateContent` | `x-goog-api-key` | Generative Language API |

Any OpenAI-compatible router works through `openai_compatible` — just point
`base_url` at it.

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
