#!/usr/bin/env bash
# Run from anywhere; never overwrite the original dataset/checkpoints.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ACTION="${1:-all}"
DATASET="${DATASET:-data/function_dataset.jsonl}"
GRAPHS="${GRAPHS:-data/graphs/primevul_cfg.jsonl}"
RUN_DIR="${RUN_DIR:-results/cfg_abc_seed42}"
MODEL_PATH="${MODEL_PATH:-/home/phy/models/Qwen2.5-Coder-7B-Instruct}"
JOERN_DIR="${JOERN_DIR:-/home/phy/joern}"
JAVA_HOME="${JAVA_HOME:-/home/phy/jdk21}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"

build_graphs() {
  "$PYTHON" -m vulnmechanism.cfg_experiment build \
    --dataset "$DATASET" --source-dataset primevul --output "$GRAPHS" \
    --joern-dir "$JOERN_DIR" --java-home "$JAVA_HOME" --batch-size 8 --timeout 300
}
train_models() {
  "$PYTHON" -m vulnmechanism.cfg_experiment run \
    --dataset "$DATASET" --graphs "$GRAPHS" --source-dataset primevul \
    --model-path "$MODEL_PATH" --output-dir "$RUN_DIR" \
    --source-max-length 2048 --batch-size 1 --gradient-accumulation 8 --epochs 3 \
    --learning-rate 2e-4 --graph-learning-rate 1e-3 --seed 42 --device "$DEVICE" --resume
}
case "$ACTION" in
  all) build_graphs; train_models ;;
  build) build_graphs ;;
  train) train_models ;;
  test) "$PYTHON" -m vulnmechanism.cfg_experiment eval --run-dir "$RUN_DIR" --split test --device "$DEVICE" ;;
  compare) "$PYTHON" -m vulnmechanism.cfg_experiment compare --run-dir "$RUN_DIR" --split valid ;;
  *) printf 'Usage: bash scripts/run_cfg_abc.sh {all|build|train|test|compare}\n' >&2; exit 2 ;;
esac
