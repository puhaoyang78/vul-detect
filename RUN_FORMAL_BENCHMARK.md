# Formal benchmark run commands

## 1. Prepare the fixed manifest

```bash
python -m vulnmechanism.prepare_benchmark \
  --source-dir /home/PublicData/PHY-data/vul_detect/data \
  --output-dir data/benchmark \
  --final-score 4
```

## 2. Build once with Joern

```bash
python -m vulnmechanism.cli build \
  --samples data/benchmark/manifest.jsonl \
  --output data/function_dataset.jsonl \
  --joern-dir /home/phy/joern \
  --java-home /home/phy/jdk21 \
  --timeout 300
```

The schema-v7 build produces:

```text
data/function_dataset.jsonl
  successful samples only

data/function_dataset.errors.jsonl
  current failures with dataset/split/label/stage

data/function_dataset.audit.json
  success coverage and CPG quality statistics
```

Before training, inspect `function_dataset.audit.json`. In particular check:

- overall and per-dataset/split/label build success rates;
- whether failure rate differs materially between vulnerable and benign samples;
- `success_without_cdg` / `success_without_ddg`;
- `success_without_potential_pattern`;
- the potential-pattern count distribution.

Main baseline-vs-method comparisons use exactly the same build-success cohort. PrimeVul is rebalanced after build by keeping every successful vulnerable and selecting the same number of successful benign functions per official split. CleanVul/SVEN keep only complete successful pairs.

## 3. PrimeVul baseline

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli train \
  --variant baseline \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --output results/primevul_baseline.pt \
  --device cuda
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset primevul \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_baseline.pt \
  --split test \
  --device cuda
```

## 4. CleanVul baseline

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

## 5. SVEN external evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m vulnmechanism.cli eval \
  --source-dataset sven \
  --dataset data/function_dataset.jsonl \
  --checkpoint results/primevul_baseline.pt \
  --split external_test \
  --device cuda
```

SVEN is test-only. The checkpoint's validation-selected threshold is reused unchanged.

## 6. Audit semantic fidelity before trusting the CPG mechanism

After the full build, run the pair diagnostics:

```bash
python -m vulnmechanism.audit_semantics \
  --dataset data/function_dataset.jsonl \
  --source-dataset cleanvul \
  --output results/cleanvul_semantic_audit.json

python -m vulnmechanism.audit_semantics \
  --dataset data/function_dataset.jsonl \
  --source-dataset sven \
  --output results/sven_semantic_audit.json
```

This reports, among other things:

- vulnerable-side potential-pattern coverage;
- fixed-side potential-pattern coverage;
- percentage of complete pairs in which at least one vulnerable-side pattern disappears after the fix;
- patterns that persist after the fix;
- patterns newly introduced in the fixed version.

These are diagnostics, not causal ground truth. A Joern-successful CPG can still be semantically incomplete or path-insensitive, and a real patch can change more than the vulnerability mechanism. Use these reports together with manual stratified review before claiming that a pattern represents the true root cause.
