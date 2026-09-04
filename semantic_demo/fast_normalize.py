from __future__ import annotations

import functools
import sys

from . import cli
from . import semantics


_ORIGINAL_PARSE_FUNCTIONS = semantics.parse_functions


@functools.lru_cache(maxsize=128)
def _cached_parse_functions(
    path: str,
    source_text: str,
    language_hint: str | None,
):
    return _ORIGINAL_PARSE_FUNCTIONS(
        path,
        source_text,
        language_hint=language_hint,
    )


def _install_parse_cache() -> None:
    def cached(
        path: str,
        source_text: str,
        *,
        language_hint: str | None = None,
    ):
        return _cached_parse_functions(path, source_text, language_hint)

    semantics.parse_functions = cached


def _fast_preflight_samples(
    samples: list[dict[str, object]],
    *,
    joern_dir: str,
    java_home: str,
    cpg_cache_dir: str,
    resume: bool = False,
):
    prepared = []
    failures: list[str] = []

    for sample in samples:
        key = str(sample["sample_key"])
        print(f"normalize_prepare_start={key}", flush=True)
        try:
            repository, index = cli._load_repository_index(
                sample,
                joern_dir=joern_dir,
                java_home=java_home,
                cpg_cache_dir=cpg_cache_dir,
            )
            # methods() is the point that ensures the cached CPG/index exists and
            # parses the TSV repository index into memory.
            method_count = len(index.methods())
            print(
                f"normalize_prepare_index_ready={key} methods={method_count}",
                flush=True,
            )

            entry_method, entry = cli._entry_from_index(sample, repository, index)
            print(
                f"normalize_prepare_entry_ready={key} "
                f"entry={entry_method.name}@{entry_method.start_line}",
                flush=True,
            )

            candidates = semantics.discover_candidates(
                key,
                index,
                entry_method,
                entry.language,
            )
            skipped = sum(
                semantics.candidate_validation_error(candidate.function) is not None
                for candidate in candidates
            )
            print(
                f"normalize_prepare_candidates_ready={key} "
                f"candidates={len(candidates)} unrecoverable={skipped}",
                flush=True,
            )
            prepared.append(
                (
                    sample,
                    repository,
                    index,
                    entry_method,
                    entry,
                    candidates,
                )
            )
        except Exception as error:
            failures.append(f"{key}: {error}")
            print(f"normalize_prepare_failed={key} error={error}", flush=True)

    if failures:
        raise RuntimeError(
            "normalize preparation failed for "
            f"{len(failures)} sample(s):\n" + "\n".join(failures)
        )
    return prepared


def main() -> None:
    _install_parse_cache()
    # discover_candidates was imported into cli, but its globals still live in
    # semantic_demo.semantics, so replacing semantics.parse_functions above is
    # enough to make same-file variant discovery use the cache.
    cli._preflight_samples = _fast_preflight_samples
    parser = cli.build_parser()
    args = parser.parse_args(["normalize", *sys.argv[1:]])
    cli.normalize_command(args)


if __name__ == "__main__":
    main()
