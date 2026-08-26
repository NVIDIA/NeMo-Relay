#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a rendered report PDF and refresh its bundle manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: finalize_pdf.py /absolute/report-bundle /absolute/report.pdf", file=sys.stderr)
        return 2
    bundle = Path(sys.argv[1]).resolve()
    pdf = Path(sys.argv[2]).resolve()
    try:
        relative_pdf = pdf.relative_to(bundle)
    except ValueError:
        print("rendered PDF must remain inside its report bundle", file=sys.stderr)
        return 2
    if not pdf.is_file() or pdf.stat().st_size == 0:
        print("rendered PDF is missing or empty", file=sys.stderr)
        return 2

    info_text = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
    info = {}
    for line in info_text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    page_count = int(info.get("Pages", "0"))
    letter_page = info.get("Page size", "").startswith("612 x 792 pts")

    with tempfile.TemporaryDirectory(prefix="terminal-bench-report-pdf-") as temporary:
        text_path = Path(temporary) / "report.txt"
        subprocess.run(["pdftotext", "-layout", str(pdf), str(text_path)], check=True)
        rendered_text = text_path.read_text(encoding="utf-8", errors="replace")
    forbidden = ["Bearer ", "SWITCHYARD_PROVIDER_AUTHORIZATION=", "/localhome/", "/home/"]
    findings = [pattern for pattern in forbidden if pattern in rendered_text]
    validation = {
        "schema_version": "terminal-bench-report.pdf-validation.v1",
        "status": "passed" if page_count > 0 and letter_page and not findings else "failed",
        "pdf": str(relative_pdf),
        "sha256": sha256_file(pdf),
        "size_bytes": pdf.stat().st_size,
        "page_count": page_count,
        "letter_page_size": letter_page,
        "text_scan_findings": findings,
        "visual_inspection_required": True,
    }
    validation_path = bundle / "evidence" / "pdf-validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if validation["status"] != "passed":
        print(json.dumps(validation, indent=2, sort_keys=True), file=sys.stderr)
        return 2

    manifest_path = bundle / "evidence" / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [path for path in sorted(bundle.rglob("*")) if path.is_file() and path != manifest_path]
    manifest["files"] = {str(path.relative_to(bundle)): sha256_file(path) for path in files}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
