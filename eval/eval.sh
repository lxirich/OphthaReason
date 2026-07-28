#!/bin/bash
set -euo pipefail

declare -A models_by_dir

models_by_dir["path/to/the/model/"]="\
OphthaReason-Intern \
OphthaReason-Qwen" \

for dir in "${!models_by_dir[@]}"; do
  echo "=== Using MODEL_DIR: $dir ==="
  export MODEL_DIR="$dir"

  read -r -a models <<< "${models_by_dir[$dir]}"

  for model in "${models[@]}"; do
    echo "Evaluating $model with MODEL_DIR=$MODEL_DIR ..."
    python eval.py "$model"
  done
done