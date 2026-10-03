#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

bundle="${1:-}"
output="${2:-}"
if [[ -z "$bundle" || ! -f "$bundle/README.md" ]]; then
  echo "usage: $0 /absolute/report-bundle [/absolute/report.pdf]" >&2
  exit 2
fi
if [[ "$bundle" != /* ]]; then
  echo "report bundle must be absolute" >&2
  exit 2
fi
output="${output:-$bundle/README.pdf}"
if [[ "$output" != /* ]]; then
  echo "report PDF path must be absolute" >&2
  exit 2
fi
for dependency in pandoc xelatex rsvg-convert python3 pdfinfo pdftotext; do
  command -v "$dependency" >/dev/null || {
    echo "missing PDF dependency: $dependency" >&2
    exit 1
  }
done
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

(
  cd "$bundle"
  pandoc README.md \
    --defaults pandoc-defaults.yaml \
    --metadata title="$(sed -n '1s/^# //p' README.md)" \
    --output "$output"
)
python3 "$script_root/finalize_pdf.py" "$bundle" "$output"
echo "$output"
