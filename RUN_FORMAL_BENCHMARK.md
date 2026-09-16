# Formal benchmark run commands

The formal benchmark is built from `data/benchmark/manifest.jsonl`. The resulting CPG dataset contains PrimeVul, CleanVul, and SVEN together, but training/evaluation must select one source explicitly.

## 1. Build once

```bash
python -m vulnmechanism.cli build \
  --samples data/benchmark/manifest.jsonl \
  --output data/function_dataset.jsonl \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --timeout 300
```

Build failures are written to `data/function_dataset.errors.jsonl`.

At train/eval time the benchmark view is derived only from successful build records:

- PrimeVul keeps every build-success vulnerable record and deterministically selects the same number of build-success benign records within each official train/valid/test split.
- CleanVul and SVEN keep only complete vulnerable/fixed pairs; if either side failed to build, the complete pair is excluded.
- A mixed formal dataset cannot be trained without `--source-dataset`.

The CLI prints `benchmark_dataset_view=...` before loading the model. Save this line with every experiment because it is the exact build-success cohort used by that run.

## 2. PrimeVul source-only baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_baseline.pt \
  --device cuda
```

Evaluate the frozen PrimeVul test split:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_baseline.pt \
  --split test \
  --device cuda
```

## 3. CleanVul source-only baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset cleanvul \
  --dataset data/function_dataset.jsonl \
  --output results/cleanvul_baseline.pt \
  --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset cleanvul \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/cleanvul_baseline.pt \
  --split test \
  --device cuda
```

## 4. SVEN external evaluation

Use an already trained PrimeVul or CleanVul checkpoint. SVEN never selects an epoch or threshold; evaluation uses the decision threshold stored in the checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset sven \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_baseline.pt \
  --split external_test \
  --device cuda
```

Do not train with `--source-dataset sven`.
