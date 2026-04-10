#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set -euo pipefail

BASE_SHA="${1:-}"
BEFORE_SHA="${2:-}"
CURRENT_SHA="${3:-$(git rev-parse HEAD)}"
REF_TYPE="${4:-branch}"

log() {
  echo "[ci-changes] $*"
}

is_valid_commit() {
  local sha="${1:-}"
  [[ -n "$sha" ]] && git rev-parse --verify -q "${sha}^{commit}" >/dev/null
}

is_zero_sha() {
  local sha="${1:-}"
  [[ -n "$sha" && "$sha" =~ ^0+$ ]]
}

run_all=false
docs_only=false
run_python=false
run_go=false
run_node=false
run_wasm=false
run_rust=false
run_check=false
run_packages=false
run_coverage_aggregate=false

if [[ "$REF_TYPE" == "tag" ]]; then
  log "Tag ref detected; forcing the full pipeline."
  run_all=true
else
  diff_range=""
  if is_valid_commit "$BASE_SHA"; then
    diff_range="${BASE_SHA}...${CURRENT_SHA}"
  elif [[ -n "$BEFORE_SHA" ]] && ! is_zero_sha "$BEFORE_SHA" && is_valid_commit "$BEFORE_SHA"; then
    diff_range="${BEFORE_SHA}..${CURRENT_SHA}"
  else
    log "No reliable diff base found; forcing the full pipeline."
    run_all=true
  fi

  if [[ "$run_all" == false ]]; then
    mapfile -t changed_files < <(git diff --name-only "$diff_range")
    if ((${#changed_files[@]} == 0)); then
      log "No changed files detected."
      docs_only=true
    else
      log "Diff range: $diff_range"
      printf '%s\n' "${changed_files[@]}" | sed 's/^/[ci-changes] changed: /'
      docs_only=true
    fi

    for file in "${changed_files[@]}"; do
      if [[ "$file" == *.md || "$file" == docs/* || "$file" == LICENSE || "$file" == SECURITY.md ]]; then
        continue
      fi

      docs_only=false

      case "$file" in
        third_party/langgraph_tests/*)
          run_python=true
          ;;
        crates/core/*|crates/adaptive/*|crates/otel/*|crates/openinference/*|Cargo.toml|Cargo.lock|rust-toolchain.toml|deny.toml|.github/workflows/*|scripts/*|patches/*|third_party/*)
          run_all=true
          break
          ;;
        python/*|crates/python/*|pyproject.toml|uv.lock|examples/python/*|examples/agent_with_logging.py|examples/pyproject.toml|examples/uv.lock)
          run_python=true
          ;;
        go/*|crates/ffi/*|examples/go/*)
          run_go=true
          ;;
        crates/node/*|examples/node/*)
          run_node=true
          ;;
        crates/wasm/*|examples/wasm/*)
          run_wasm=true
          ;;
        *)
          log "Unclassified path '${file}'; forcing the full pipeline."
          run_all=true
          break
          ;;
      esac
    done
  fi
fi

if [[ "$run_all" == true ]]; then
  docs_only=false
  run_python=true
  run_go=true
  run_node=true
  run_wasm=true
  run_rust=true
  run_check=true
  run_packages=true
  run_coverage_aggregate=true
elif [[ "$docs_only" != true && ( "$run_python" == true || "$run_go" == true || "$run_node" == true || "$run_wasm" == true ) ]]; then
  run_rust=true
  run_check=true
fi

{
  echo "run_all=${run_all}"
  echo "docs_only=${docs_only}"
  echo "run_python=${run_python}"
  echo "run_go=${run_go}"
  echo "run_node=${run_node}"
  echo "run_wasm=${run_wasm}"
  echo "run_rust=${run_rust}"
  echo "run_check=${run_check}"
  echo "run_packages=${run_packages}"
  echo "run_coverage_aggregate=${run_coverage_aggregate}"
} >> "${GITHUB_OUTPUT}"
