#!/usr/bin/env bash
# Minimal smoke test across all three tasks on the in-repo demo split.
#
#   --dry-run builds prompts and validates config WITHOUT calling a model,
#   so this runs with no API keys and no heavy geometry/render extras.
#
# Drop --dry-run (and set the keys named in .env) to actually call a model and
# score the predictions. Pick the model with --model <name-from-models.yaml>.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-gpt-4o}"

echo "== validate demo split =="
p3dbench validate --split demo

echo
echo "== Text-to-3D / minimal-json (dry-run) =="
p3dbench run --task text-to-3d --format minimal-json --metric all --model "$MODEL" --split demo --dry-run

echo
echo "== Image-to-3D / openscad (dry-run) =="
p3dbench run --task image-to-3d --format openscad --metric all --model "$MODEL" --split demo --dry-run

echo
echo "== Assembly-3D / cadquery (dry-run) =="
p3dbench run --task assembly-3d --format cadquery --metric all --model "$MODEL" --split demo --dry-run

echo
echo "Dry-run OK. To run for real:  MODEL=<your-model> examples/run_smoke.sh  (remove --dry-run)"
