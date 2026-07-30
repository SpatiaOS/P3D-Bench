# Metrics

Five buckets, each a plug-in under [`p3dbench/metrics/`](../p3dbench/metrics/),
selected with `--metric` (`valid` · `geometry` · `topology` · `judge` · `part` ·
`all`). A bucket's `score` returns raw sub-metric values; normalization to
[0,1] and aggregation happen at summarize time
([`metrics/base.py`](../p3dbench/metrics/base.py)). A bucket whose optional
dependency is missing reports that clearly instead of crashing the run.

## Which buckets apply per task

| Bucket | Text-to-3D (param) | Text-to-3D (desc) | Image-to-3D | Assembly-3D |
|--------|:---:|:---:|:---:|:---:|
| Valid | ✓ | ✓ | ✓ | ✓ |
| Geometry | ✓ | – | ✓ | ✓ |
| Topology | ✓ | – | ✓ | ✓ |
| Judge | QA-S, QA-P | QA-S, J-Sem | J-Sem/J-Geo/J-Aes | J-Sem/J-Geo/J-Aes |
| Part | – | – | – | ✓ |

## Valid

`valid == a non-empty STL was produced`. A non-empty error list does not by
itself invalidate a case. Reported as a rate; predictions that fail it are
**worst-filled** (contribute the worst value, i.e. normalized 0) in the four
scored buckets.

## Geometry

Computed on the **aligned** prediction vs GT. Alignment: `pca+discrete` — 48
signed-permutation candidate rotations plus a PCA-canonicalized path (identity
bias 0.05, PCA bias 0.10), then a bounded L-BFGS-B refine of scale ∈ [0.7, 1.3]
and translation ≤ 0.2, minimizing bidirectional Chamfer. Text-to-3D parametric
**locks scale** (center-only). Both clouds are sampled at 8192 points (seed 42);
assemblies use exterior-only sampling so fused outputs compare fairly to unfused GT.

| Metric | Dir | Definition |
|--------|:---:|------------|
| **CD** | ↓ | squared bidirectional Chamfer distance |
| **F@.05** | ↑ | F-score at τ = 0.05·scale |
| **F@.01** | ↑ | F-score at τ = 0.01·scale |
| **NC** | ↑ | normal consistency (mean of precision/recall) |
| **IoU** | ↑ | IoU_C (manifold boolean) for Text-param; IoU_V (128³ voxel) otherwise |

IoU is computed **only when both meshes have zero open edges**; otherwise it is
reported as `None` and dropped from the bucket mean for that case.

## Topology (predicted mesh only)

Over unique undirected edges:

| Metric | Dir | Definition |
|--------|:---:|------------|
| **NoOE** | ↑ | per-case closedness: 1 if no open edges else 0 (reported as a rate) |
| **InvN** | ↓ | inverted-normal edge ratio (inconsistent winding) |
| **NM** | ↓ | non-manifold edge ratio (edges shared by ≥3 faces) |

## Judge (Gemini 3.1 Pro)

The model is set in [`configs/judge.yaml`](../configs/judge.yaml). Three families:

- **Visual judge** (Image-/Assembly-3D): 4 canonical pred views + 4 GT views in
  one call, scored 1–10 on **J-Geo** (shape similarity), **J-Sem** (semantic
  identity), **J-Aes** (aesthetics/detail, semantic-gated: capped to [1,3] when
  J-Sem < 4). Pairing is strict — if either side lacks 4 views the case is skipped.
- **TextDesc J-Sem** (descriptive Text-to-3D) uses its own semantic-only prompt:
  the model generated from text alone and never saw the GT, so the judge is told
  to ignore geometric differences and score only whether PRED is the right
  category of object. It receives the same description the generator saw, 4
  aligned pred views and the 4 same-view GT renders. Text-to-3D ships no GT
  renders, so they are rendered from the GT mesh at eval time (same backend as
  the pred views) and cached next to the mesh.
- **QA** (Text-to-3D): the prediction is probed with a prebuilt multiple-choice
  bank that ships with the data (`targets/qa/<id>.json`). **QA-S** = accuracy over
  the 4 semantic questions, **QA-P** = accuracy over the 8 parametric questions.
  The answerer sees the source artifact text, a measured bbox summary, and the
  same 4 canonical pred views the visual judge uses. Banks are dataset artifacts;
  the release does **not** regenerate them.

## Part (Assembly-3D only)

After the decomposition step (see [TASKS.md](TASKS.md)), GT and predicted parts
are deduplicated by a rotation/translation-invariant geometric fingerprint,
placed in a shared frame (pred reuses the Geometry stage's alignment transform),
matched per-pair over the 24 proper cube rotations by a coverage F-score
(τ = 0.05·GT-part-diagonal, 1024 points/part), and assigned via the Hungarian
algorithm.

| Metric | Dir | Definition |
|--------|:---:|------------|
| **PartFS** | ↑ | mean per-part F-score over all Hungarian pairs |
| **PartMatchF1** | ↑ | F1 of successful matches (F_part ≥ 0.7): P = M/m, R = M/n |

A case whose decomposition fails the fidelity gate (CD > 5e-4 **and** IoU_V <
0.95 vs the stage-1 union) is excluded from Part means.

## Aggregation → headline Score

1. Normalize each sub-metric to [0,1], 1 = best (judge 1–10 → (v−1)/9; bounded
   lower-better → 1−v; CD → max(0, 1−CD/0.01)).
2. **Bucket score** = equal-weight mean of the applicable normalized sub-metrics.
   One exception: parametric Text-to-3D's Judge is the question-level micro-average
   over the bank's 4 QA-S + 8 QA-P questions, i.e. `(QA-S + 2·QA-P)/3`, so the
   4-question split does not outweigh the 8-question one.
3. Invalid predictions are worst-filled (normalized 0) for every member sub-metric.
   A case whose **API call never returned** (`llm_failed`) is a different thing: it
   was never tested, so it is dropped from every metric *and* from the Valid
   denominator, and reported on its own as `llm_fail_cases` / `llm_fail_rate`.
   A model that answered with unusable or uncompilable code is a model failure and
   stays worst-filled. Re-run the failed cases before quoting a Score — a high
   `llm_fail_rate` means the Score covers a smaller, self-selected subset.
4. **Score** (the headline figure) = mean of the non-Valid buckets, ×100, averaged
   over a task's supported formats. Valid is reported alongside, never folded in.

Diagnostics emitted but not bucket members: Hausdorff distance, PartMatchP/R,
visible-view and sequence metrics. They do not affect the Score.
