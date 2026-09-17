# Formal benchmark run commands

## 1. Prepare the fixed manifest

```bash
python -m vulnmechanism.prepare_benchmark \
  --source-dir /home/PublicData/PHY-data/vul_detect/data \
  --output-dir data/benchmark \
  --final-score 4
```

## 2. Build CPG + mechanism dataset

```bash
python -m vulnmechanism.cli build \
  --samples data/benchmark/manifest.jsonl \
  --output data/function_dataset.jsonl \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --timeout 300 \
  --batch-size 8
```

Current dataset schema is **9**. Outputs:

```text
data/function_dataset.jsonl
data/function_dataset.errors.jsonl
data/function_dataset.audit.json
```

Before training inspect:

- overall and per-dataset/split/label build success rates;
- failure-stage imbalance between vulnerable and benign samples;
- `success_without_cdg` / `success_without_ddg`;
- `success_without_mechanism_candidate`;
- `mechanism_candidate_count_distribution` and `mechanism_candidate_kind_counts`.

Main baseline-vs-method comparisons must use exactly the same build-success cohort. PrimeVul is rebalanced after build; CleanVul/SVEN retain only complete successful pairs.

## 3. Validate mechanism fidelity first

Before training mechanism variants, run pair diagnostics:

```bash
python -m vulnmechanism.audit_semantics \
  --dataset data/function_dataset.jsonl \
  --source-dataset cleanvul \
  --output results/cleanvul_mechanism_audit.json

python -m vulnmechanism.audit_semantics \
  --dataset data/function_dataset.jsonl \
  --source-dataset sven \
  --output results/sven_mechanism_audit.json
```

The audit compares concrete mechanism keys and states across vulnerable/fixed pairs, including candidate removal/addition and state transitions such as a related bound/null condition changing from `not_observed` to `present`.

These diagnostics are not causal ground truth. Review representative pairs manually before claiming fidelity.

## 4. PrimeVul experiments

Baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_baseline.pt \
  --device cuda
```

Raw CPG:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant raw_cpg \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_raw_cpg.pt \
  --device cuda
```

Mechanism concat:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant mechanism_concat \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_mechanism_concat.pt \
  --device cuda
```

Mechanism fusion:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant mechanism_fusion \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_mechanism_fusion.pt \
  --device cuda
```

Evaluate on PrimeVul test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_mechanism_fusion.pt \
  --split test \
  --device cuda
```

## 5. CleanVul training / SVEN external test

CleanVul can be trained with the same four variants using `--source-dataset cleanvul`.

SVEN is external-test-only:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset sven \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_mechanism_fusion.pt \
  --split external_test \
  --device cuda
```

The validation-selected threshold from the training dataset is reused unchanged.
