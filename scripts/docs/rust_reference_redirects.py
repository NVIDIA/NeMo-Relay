# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep generated Fern redirects aligned with each version's Rust reference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

METADATA_NAME = "_redirects.json"
METADATA_PATH = Path("reference/api/rust-library-reference") / METADATA_NAME
REDIRECT_PATH = Path("redirects/rust-reference.yml")
HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
    "# Generated from Rust reference grouping; do not edit.\n"
)


def write_metadata(path: Path, redirects: dict[str, str]) -> None:
    # Store the versionless mapping with the content so a release keeps its own
    # destinations even after dev grouping changes. Pre-migration snapshots have
    # no metadata and continue serving their original item pages.
    path.write_text(json.dumps({"redirects": redirects}, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _rules(redirects: dict[str, str], version: str) -> list[dict[str, Any]]:
    prefix = "/nemo/relay" + (f"/{version}" if version else "")
    return [
        {"source": prefix + source, "destination": prefix + destination, "permanent": True}
        for source, destination in sorted(redirects.items())
    ]


def _write(fern: Path, rules: list[dict[str, Any]]) -> None:
    path = fern / REDIRECT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + yaml.safe_dump({"redirects": rules}, sort_keys=False), encoding="utf-8")


def write_local_redirects(repo_root: Path, redirects: dict[str, str]) -> None:
    # The source checkout exposes dev as both the default and explicit version.
    _write(repo_root / "fern", _rules(redirects, "") + _rules(redirects, "dev"))


def write_published_redirects(fern: Path, product: dict[str, Any]) -> None:
    rules = []
    versions = [("", product["path"]), *((entry["slug"], entry["path"]) for entry in product.get("versions", []))]
    for slug, navigation in versions:
        snapshot = Path(navigation).stem
        metadata = fern / f"pages-{snapshot}" / METADATA_PATH
        if metadata.is_file():
            redirects = json.loads(metadata.read_text(encoding="utf-8"))["redirects"]
            rules.extend(_rules(redirects, slug))
    _write(fern, rules)
