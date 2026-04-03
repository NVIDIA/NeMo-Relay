#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Aggregate project coverage from all CI coverage artifacts."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageInput:
    label: str
    path: str
    kind: str


INPUTS = (
    CoverageInput("Go wrapper", "go_coverage_report.xml", "cobertura"),
    CoverageInput("Node wrapper", "coverage-summary.json", "node_json"),
    CoverageInput("Node Rust binding", "node-rust.xml", "cobertura"),
    CoverageInput("Python wrapper", "pytest_coverage_report.xml", "cobertura"),
    CoverageInput("Python Rust binding", "python-rust.xml", "cobertura"),
    CoverageInput("Rust workspace", "rust-workspace.xml", "cobertura"),
    CoverageInput("WASM JS wrapper", "wasm-js.xml", "cobertura"),
    CoverageInput("WASM Rust crate", "wasm-rust.xml", "cobertura"),
)


def read_cobertura(path: Path) -> tuple[int, int]:
    root = ET.parse(path).getroot()
    covered = root.attrib.get("lines-covered")
    valid = root.attrib.get("lines-valid")
    if covered is None or valid is None:
        raise ValueError(f"{path} is missing Cobertura line totals")
    return int(covered), int(valid)


def read_node_summary(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    total = summary["total"]["lines"]
    return int(total["covered"]), int(total["total"])


def main(argv: list[str]) -> int:
    coverage_dir = Path(argv[1]) if len(argv) > 1 else Path("target/coverage")
    summary_path = coverage_dir / "project-coverage-summary.md"

    rows: list[tuple[str, int, int, float]] = []
    total_covered = 0
    total_valid = 0

    for item in INPUTS:
        path = coverage_dir / item.path
        if not path.exists():
            raise FileNotFoundError(f"missing coverage artifact: {path}")
        if item.kind == "cobertura":
            covered, valid = read_cobertura(path)
        elif item.kind == "node_json":
            covered, valid = read_node_summary(path)
        else:
            raise ValueError(f"unsupported coverage kind: {item.kind}")
        rate = covered / valid if valid else 0.0
        rows.append((item.label, covered, valid, rate))
        total_covered += covered
        total_valid += valid

    total_rate = total_covered / total_valid if total_valid else 0.0

    lines = [
        "# Project Coverage Summary",
        "",
        "| Surface | Covered | Total | Rate |",
        "|---|---:|---:|---:|",
    ]
    for label, covered, valid, rate in rows:
        lines.append(f"| {label} | {covered} | {valid} | {rate * 100:.2f}% |")
    lines.extend(
        [
            "",
            f"| TOTAL | {total_covered} | {total_valid} | {total_rate * 100:.2f}% |",
            "",
        ]
    )
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Coverage summary written to {summary_path}")
    for label, covered, valid, rate in rows:
        print(f"{label}: {covered}/{valid} ({rate * 100:.2f}%)")
    print(f"TOTAL {total_rate * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
