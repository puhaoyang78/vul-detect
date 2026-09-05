# Staged analysis workflow

Public commands:

```bash
python -m semantic_demo.cli preflight
python -m semantic_demo.cli normalize
python -m semantic_demo.cli run
```

`semantic_demo.legacy_cli` is an internal execution engine only.

## Stage contract

### Preflight

Preflight is the only stage that may discover candidates. It resolves the fixed repository revision, builds/reuses Joern CPG/index, resolves the exact entry, performs memory-relevance candidate slicing, writes a versioned candidate manifest, and checkpoints each completed sample.

Candidate manifests include selection depth, caller, and selection reason. Deeper traversal follows only calls connected to memory-relevant values/returns/pointer flows; opaque typedefs are retained conservatively.

`--refresh` refreshes selected samples only. Other checkpoints survive.

### Normalize

Normalize requires a valid preflight manifest and never performs discovery.

It combines a shared standard-API semantic registry with localized LLM normalization. Large custom functions use a statically generated relevance slice consisting of endpoint context, reaching assignments, related control conditions, and the function signature.

Every candidate is checkpointed immediately. Cache reuse checks source fingerprint, schema, normalization implementation, backend, and model. A subset refresh preserves all unselected samples and removes stale records that no longer belong to the current selected manifests.

### Run

Run requires complete normalization coverage for every selected manifest candidate. It creates selected temporary replay/result files, validates summaries with Joern, composes wrappers, runs target analysis and Z3, then upserts completed sample results back into global outputs.

A later failure cannot erase older results for samples that were not completed in the current run.

Run does not require the normalization model/backend to be repeated by the user; that metadata is read from the normalization records themselves.

## Semantic boundaries

The cross-procedure schema is limited to:

```text
ALLOC(return, size)
READ(buffer, length)
WRITE(buffer, length)
VALUE(return, expression)
```

Standard API semantics are centralized in `semantic_demo/standard_semantics.py` and shared by discovery, normalization, validation, and target analysis.

Unresolved custom/indirect calls are dependency-local barriers: they only force an access to UNKNOWN when they share values relevant to that access. Parser errors are similarly no longer an unconditional whole-function abort.

Symbolic arithmetic remains conservative. Fixed-width arithmetic is accepted only when path/range constraints prove that mathematical integer evaluation matches C arithmetic without overflow.

## Cache invalidation

- candidate manifests: manifest/discovery policy version + Joern index fingerprint;
- normalization: schema + implementation version + source fingerprint + model/backend;
- detection: automatic hash of analyzer/Z3/validation/Joern-v2/standard-semantics implementation.

Changing core analysis code therefore invalidates the appropriate stage without requiring a manual version bump everywhere.
