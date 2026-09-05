# Staged analysis workflow

Use the staged workflow for experiments:

```bash
python -m semantic_demo.workflow preflight
python -m semantic_demo.workflow normalize
python -m semantic_demo.workflow run
```

## Stage contracts

### 1. `preflight`

`preflight` is the only stage that discovers candidate helper functions.

For each sample it:

1. verifies the requested Git revision and source paths;
2. builds or reuses the Joern CPG and repository index;
3. resolves the exact entry function;
4. discovers summary-relevant helper functions;
5. checks whether candidate source bodies are recoverable;
6. writes a versioned candidate manifest under `data/joern_cpg/`;
7. checkpoints the completed sample immediately.

A completed sample is skipped on the next invocation when the sample, Joern index, discovery policy, and candidate manifest are unchanged.

Candidate discovery does **not** recursively traverse the entire reachable call graph. Every summary-capable direct static callee of the entry is retained. Deeper traversal follows only calls that can contribute caller-visible `ALLOC`, `READ`, `WRITE`, or `VALUE` semantics: the child return flows to the caller return, or a child argument depends on a caller parameter.

The old whole-scope unresolved-static-call count is intentionally not part of this workflow. Only unresolved calls encountered on summary-relevant traversal are reported.

### 2. `normalize`

`normalize` requires a valid candidate manifest from `preflight`.

It never performs candidate discovery. It:

1. loads candidate manifests;
2. checks candidate source fingerprints;
3. reuses candidate-level normalization records when schema, source, backend, and model match;
4. starts the LLM only when at least one candidate is pending;
5. checkpoints every generated candidate immediately;
6. preserves records belonging to samples outside the selected `--samples` file.

A subset run therefore cannot erase normalization records for other samples.

### 3. `run`

`run` requires both a valid candidate manifest and complete normalization coverage for every selected candidate.

It never performs candidate discovery. It validates the normalized summaries with Joern, composes accepted wrapper summaries, runs the baseline and proposed analyzer, and checkpoints completed detection samples.

A subset run preserves detection and semantic-validation records belonging to samples outside the selected `--samples` file. Evaluation is performed only when detections cover the complete oracle sample set.

## Refresh behavior

Use `--refresh` only when intentionally invalidating a stage.

```bash
python -m semantic_demo.workflow preflight --refresh
python -m semantic_demo.workflow normalize --refresh
python -m semantic_demo.workflow run --refresh
```

Changing the candidate-discovery policy invalidates candidate manifests through its policy version. Changing source/index inputs invalidates preflight through the index/sample fingerprint. Changing the normalization model invalidates only affected normalization records.
