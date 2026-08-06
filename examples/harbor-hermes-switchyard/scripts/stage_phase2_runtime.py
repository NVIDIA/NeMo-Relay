#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage and verify the immutable Phase 2 runtime harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

RUNTIME_SUFFIXES = {".py", ".sh", ".toml", ".yaml"}
RUNTIME_TOP_LEVEL = (
    "run_terminal_bench.sh",
    "run_phase2_cohort.sh",
    "supervise_phase2_cohort.sh",
)
RUNTIME_DIRECTORIES = ("agents", "config", "scripts")


def runtime_files(root: Path) -> list[Path]:
    files = [root / name for name in RUNTIME_TOP_LEVEL]
    for relative in RUNTIME_DIRECTORIES:
        files.extend(
            path
            for path in (root / relative).rglob("*")
            if path.is_file() and path.suffix in RUNTIME_SUFFIXES
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime source is missing: {missing[0]}")
    return sorted(files)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_digest(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def expected_digest(plan_path: Path) -> str:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    value = plan.get("inputs", {}).get("runtime_sources_sha256")
    if not isinstance(value, str) or not value:
        raise ValueError("plan does not contain inputs.runtime_sources_sha256")
    return value


def verify_runtime(root: Path, expected: str) -> tuple[str, int]:
    files = runtime_files(root)
    observed = runtime_digest(root, files)
    if observed != expected:
        raise ValueError(f"runtime harness hash mismatch: expected {expected}, observed {observed}")
    return observed, len(files)


def stage_runtime(source: Path, destination: Path, plan_path: Path) -> dict[str, object]:
    source = source.resolve()
    destination = destination.resolve()
    expected = expected_digest(plan_path)
    if destination.is_dir():
        observed, file_count = verify_runtime(destination, expected)
        return {"status": "verified", "runtime_sources_sha256": observed, "file_count": file_count}
    if destination.exists():
        raise ValueError(f"runtime harness destination is not a directory: {destination}")

    files = runtime_files(source)
    observed = runtime_digest(source, files)
    if observed != expected:
        raise ValueError(f"live runtime differs from immutable plan: expected {expected}, observed {observed}")

    temporary = destination.with_name(f"{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(mode=0o700, parents=False)
    try:
        for path in files:
            target = temporary / path.relative_to(source)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(path, target)
        (temporary / "snapshot.json").write_text(
            json.dumps(
                {
                    "schema_version": "harbor-hermes-switchyard.phase2-runtime-snapshot.v1",
                    "runtime_sources_sha256": observed,
                    "file_count": len(files),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"status": "staged", "runtime_sources_sha256": observed, "file_count": len(files)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = stage_runtime(args.source, args.destination, args.plan)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
