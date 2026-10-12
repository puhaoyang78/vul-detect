#!/usr/bin/env bash
# Run after an experiment: publish source code and metric reports, not checkpoints or predictions.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  printf 'Usage: bash scripts/publish_code_metrics.sh [--dry-run | "commit message"]\n' >&2
  printf '  --dry-run checks selected changes without staging, committing or pushing.\n' >&2
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ "${1:-}" == -* ]]; then
  usage
  exit 2
fi

branch=$(git symbolic-ref --quiet --short HEAD) || {
  printf 'Cannot publish from a detached HEAD.\n' >&2
  exit 1
}
git remote get-url --push origin >/dev/null
if [[ -n "$(git ls-files -u)" ]]; then
  printf 'Resolve merge conflicts before publishing.\n' >&2
  exit 1
fi

# Code locations in this repository. In particular, workspace/ and data/ are excluded.
code_paths=(.gitattributes .gitignore .github README.md RUN_FORMAL_BENCHMARK.md
            RUN_GRAPH_ABC.md requirements.txt docs scripts tests vulnmechanism)

is_code_path() {
  local path=$1
  case "$path" in
    *.pt|*.pth|*.ckpt|*.safetensors|*.bin|*.onnx|*.gguf) return 1 ;;
    .gitattributes|.gitignore|README.md|RUN_FORMAL_BENCHMARK.md|RUN_GRAPH_ABC.md|\
    requirements.txt|.github/*|docs/*|scripts/*|tests/*|vulnmechanism/*) return 0 ;;
  esac
  return 1
}

is_metric_result() {
  local path=$1
  [[ $path == results/* ]] || return 1
  case "${path##*/}" in
    *metrics*.json|*metrics*.csv|comparison*.json|comparison*.csv|history.jsonl|\
    *.training.jsonl|alignment_coverage.json|ddg_audit.json|probe_results.json|results.json)
      return 0 ;;
  esac
  case "$path" in
    results/cfg_ddg_audit_seed42.json|results/cpg_debug/*.audit.json|\
    results/cpg_debug/batch_equivalence.json|results/cpg_debug/export_format_equivalence.json|\
    results/cpg_debug/historical_statistics.json|results/cpg_debug/integrity_checks.json)
      return 0 ;;
  esac
  return 1
}

# A previous run may have been interrupted after staging. Keep only approved paths.
while IFS= read -r -d '' path; do
  if ! is_code_path "$path" && ! is_metric_result "$path"; then
    printf 'Staged file is outside code/metric scope: %q\n' "$path" >&2
    exit 1
  fi
done < <(git diff --cached --name-only -z)

files=()
while IFS= read -r -d '' path; do
  if is_code_path "$path"; then
    files+=("$path")
  fi
done < <(git diff --name-only -z HEAD -- "${code_paths[@]}")
while IFS= read -r -d '' path; do
  if is_code_path "$path"; then
    files+=("$path")
  fi
done < <(git ls-files --others --exclude-standard -z -- "${code_paths[@]}")
while IFS= read -r -d '' path; do
  if is_metric_result "$path"; then
    files+=("$path")
  fi
done < <(git diff --name-only -z HEAD -- results)
while IFS= read -r -d '' path; do
  if is_metric_result "$path"; then
    files+=("$path")
  fi
done < <(git ls-files --others --exclude-standard -z -- results)

if [[ ${#files[@]} -eq 0 ]]; then
  printf 'No new code or metric reports to commit.\n'
else
  printf 'Selected %d code/metric files:\n' "${#files[@]}"
  printf '  %q\n' "${files[@]}"
fi

# Check the exact proposed commit without changing the real index or object store.
# In particular, a failed whitespace check must not leave files staged.
if [[ ${#files[@]} -gt 0 ]]; then
  check_dir=$(mktemp -d)
  trap 'rm -rf -- "$check_dir"' EXIT
  mkdir "$check_dir/objects"
  cp -- "$(git rev-parse --git-path index)" "$check_dir/index"
  object_dir=$(cd "$(git rev-parse --git-path objects)" && pwd)
  (
    export GIT_INDEX_FILE="$check_dir/index"
    export GIT_OBJECT_DIRECTORY="$check_dir/objects"
    export GIT_ALTERNATE_OBJECT_DIRECTORIES="$object_dir${GIT_ALTERNATE_OBJECT_DIRECTORIES:+:$GIT_ALTERNATE_OBJECT_DIRECTORIES}"
    git add -A -- "${files[@]}"
    if ! git diff --cached --check; then
      printf 'Whitespace check failed (e.g. trailing whitespace). Fix the reported file/line and rerun with --dry-run.\n' >&2
      printf 'Your staging area is unchanged; nothing was committed or pushed.\n' >&2
      exit 1
    fi
  )
  rm -rf -- "$check_dir"
  trap - EXIT
fi
if "$dry_run"; then
  exit 0
fi

if [[ ${#files[@]} -gt 0 ]]; then
  git add -A -- "${files[@]}"
  git diff --cached --check
  git --no-pager diff --cached --stat
  if ! git diff --cached --quiet; then
    message=${1:-"Update code and metrics $(date -u '+%Y-%m-%d %H:%M UTC')"}
    git commit -m "$message"
  fi
fi

printf 'Pushing to origin/%s...\n' "$branch"
git push origin "HEAD:$branch"
