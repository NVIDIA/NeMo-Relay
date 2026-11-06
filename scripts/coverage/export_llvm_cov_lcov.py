#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export an lcov report from an instrumented nextest run.

llvm-cov keeps one record per function name, so unmangled functions lose the
coverage of their second build. This script exports unit test binaries and
other objects separately and merges the reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

ROOT = Path(__file__).resolve().parents[2]
UNIT_TEST_KINDS = frozenset({"bin", "example", "lib", "proc-macro"})
INTEGRATION_TEST_KINDS = frozenset({"bench", "test"})
SKIPPED_DIRECTORIES = frozenset({".fingerprint", "build", "incremental"})
SKIPPED_SUFFIXES = frozenset({".d", ".rlib", ".rmeta"})
WINDOWS_OBJECT_SUFFIXES = frozenset({".dll", ".exe"})
ERE_SPECIAL_CHARACTERS = frozenset("\\.^$*+?{}[]|()")


class CoverageExportError(Exception):
    """Report could not be produced from the inputs."""


@dataclass(frozen=True)
class TestBuild:
    """Build output locations and unit test binaries from a nextest listing."""

    target_directory: Path
    build_directories: tuple[Path, ...]
    unit_test_binaries: frozenset[Path]


@dataclass(frozen=True)
class Workspace:
    """Workspace root and artifact names of selected packages."""

    root: Path
    artifact_names: frozenset[str]


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CoverageExportError(f"{context} is not an object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise CoverageExportError(f"{context} is not an array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise CoverageExportError(f"{context} is not a string")
    return value


def parse_nextest_list(document: object) -> TestBuild:
    """Read build directories and unit test binaries from nextest list JSON.

    Raises:
        CoverageExportError: If the document is not nextest list JSON, a suite
            kind is unknown, or no build directory is listed.
    """
    root = _mapping(document, "nextest list output")
    meta = _mapping(root.get("rust-build-meta"), "rust-build-meta")
    target_directory = Path(_string(meta.get("target-directory"), "target-directory")).resolve()
    build_root = target_directory
    if "build-directory" in meta:
        build_root = Path(_string(meta["build-directory"], "build-directory")).resolve()
    base_directories = [
        _string(entry, "base-output-directories entry")
        for entry in _sequence(meta.get("base-output-directories"), "base-output-directories")
    ]
    if not base_directories:
        raise CoverageExportError("nextest list output names no build directories")
    roots = [build_root] if build_root == target_directory else [build_root, target_directory]
    build_directories = tuple(root / entry for root in roots for entry in base_directories)
    unit_test_binaries: set[Path] = set()
    for suite_id, suite in _mapping(root.get("rust-suites"), "rust-suites").items():
        suite_fields = _mapping(suite, f"suite {suite_id}")
        kind = _string(suite_fields.get("kind"), f"suite {suite_id} kind")
        binary = Path(_string(suite_fields.get("binary-path"), f"suite {suite_id} binary-path")).resolve()
        if kind in UNIT_TEST_KINDS:
            unit_test_binaries.add(binary)
        elif kind not in INTEGRATION_TEST_KINDS:
            raise CoverageExportError(f"suite {suite_id} has unsupported kind {kind!r}")
    return TestBuild(target_directory, build_directories, frozenset(unit_test_binaries))


def load_workspace(packages: Sequence[str]) -> Workspace:
    """Query cargo metadata for workspace root and selected artifact names."""
    completed = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return select_workspace(json.loads(completed.stdout), packages)


def select_workspace(metadata: object, packages: Sequence[str]) -> Workspace:
    """Collect package and target names for selected workspace packages.

    Raises:
        CoverageExportError: If a package is not a workspace member.
    """
    document = _mapping(metadata, "cargo metadata output")
    root = Path(_string(document.get("workspace_root"), "workspace_root")).resolve()
    members = [_mapping(package, "package") for package in _sequence(document.get("packages"), "packages")]
    member_names = {_string(member.get("name"), "package name"): member for member in members}
    missing = sorted(set(packages) - member_names.keys())
    if missing:
        raise CoverageExportError(f"packages are not workspace members: {', '.join(missing)}")
    selected = member_names.values() if not packages else [member_names[name] for name in packages]
    names: set[str] = set()
    for member in selected:
        names.add(_string(member.get("name"), "package name"))
        for target in _sequence(member.get("targets"), "package targets"):
            names.add(_string(_mapping(target, "target").get("name"), "target name"))
    return Workspace(root, frozenset(names))


def artifact_pattern(artifact_names: frozenset[str]) -> re.Pattern[str]:
    """Return a regex matching artifact stems of the named packages and targets."""
    alternatives = sorted("(-|_)".join(re.escape(part) for part in name.split("-")) for name in artifact_names)
    return re.compile(f"^(lib)?({'|'.join(alternatives)})(-[0-9a-f]+)?$")


def file_stem_recursive(name: str) -> str:
    """Strip every suffix from a file name."""
    while True:
        stem = Path(name).stem
        if stem == name:
            return name
        name = stem


def is_object(path: Path) -> bool:
    """Return whether a build output can carry coverage mappings."""
    if path.suffix in SKIPPED_SUFFIXES or path.name.endswith(".cargo-lock"):
        return False
    try:
        metadata = path.stat()
    except OSError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        return False
    if sys.platform == "win32":
        return path.suffix.lower() in WINDOWS_OBJECT_SUFFIXES
    return stat.S_IMODE(metadata.st_mode) & 0o111 != 0


def _walk(directory: Path) -> Iterator[Path]:
    for current, directories, files in os.walk(directory):
        directories[:] = sorted(name for name in directories if name not in SKIPPED_DIRECTORIES)
        for name in files:
            yield Path(current, name)


def collect_objects(build_directories: Sequence[Path], pattern: re.Pattern[str]) -> list[Path]:
    """List instrumented executables and shared libraries in sorted order.

    Raises:
        CoverageExportError: If no object matches under the build directories.
    """
    objects: set[Path] = set()
    for directory in build_directories:
        for path in _walk(directory):
            if pattern.match(file_stem_recursive(path.name)) and is_object(path):
                objects.add(path.resolve())
    if not objects:
        raise CoverageExportError(f"no coverage objects under {', '.join(str(d) for d in build_directories)}")
    return sorted(objects)


def partition_objects(objects: Sequence[Path], unit_test_binaries: frozenset[Path]) -> tuple[list[Path], list[Path]]:
    """Split objects into shared build outputs and unit test binaries.

    Raises:
        CoverageExportError: If a listed unit test binary is not among the objects.
    """
    missing = sorted(unit_test_binaries - set(objects))
    if missing:
        raise CoverageExportError(f"unit test binaries missing from objects: {', '.join(str(p) for p in missing)}")
    shared = [path for path in objects if path not in unit_test_binaries]
    unit_tests = [path for path in objects if path in unit_test_binaries]
    return shared, unit_tests


def profile_files(target_directory: Path) -> list[Path]:
    """List raw profiles written by instrumented processes.

    Raises:
        CoverageExportError: If the target directory has no .profraw files.
    """
    profiles = sorted(target_directory.glob("*.profraw"))
    if not profiles:
        raise CoverageExportError(f"no .profraw files in {target_directory}")
    return profiles


def ere_escape(text: str) -> str:
    """Escape text for an LLVM POSIX extended regular expression."""
    return "".join(f"\\{character}" if character in ERE_SPECIAL_CHARACTERS else character for character in text)


def ignore_filename_regex(workspace_root: Path, ignored_roots: Sequence[Path]) -> str:
    """Return a regex for test sources, generated outputs, and non-workspace sources."""
    sep = ere_escape(os.sep)
    root = ere_escape(str(workspace_root))
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo")).resolve()
    rustup_home = Path(os.environ.get("RUSTUP_HOME", Path.home() / ".rustup")).resolve()
    parts = [
        f"{sep}rustc{sep}([0-9a-f]+|[0-9]+\\.[0-9]+\\.[0-9]+){sep}",
        f"^{root}({sep}.*)?{sep}(tests|examples|benches){sep}",
        f"^{root}({sep}.*)?{sep}(tests\\.rs|[0-9a-zA-Z_-]+[_-]tests\\.rs)$",
        f"^{ere_escape(str(cargo_home))}{sep}(registry|git){sep}",
        f"^{ere_escape(str(rustup_home / 'toolchains'))}($|{sep})",
    ]
    parts.extend(f"^{ere_escape(str(path))}($|{sep})" for path in ignored_roots)
    return "|".join(parts)


def llvm_tool(name: str) -> Path:
    """Locate an llvm-tools binary in the active rustup toolchain.

    Raises:
        CoverageExportError: If the toolchain does not have the tool.
    """
    completed = subprocess.run(
        ["rustc", "--print", "target-libdir"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    tool = Path(completed.stdout.strip()).parent / "bin" / f"{name}{suffix}"
    if not tool.is_file():
        raise CoverageExportError(f"{tool} is missing; run `rustup component add llvm-tools`")
    return tool


LineCounts: TypeAlias = dict[str, dict[int, int]]


def line_counts(lcov: str, workspace_root: Path) -> LineCounts:
    """Read per-line execution counts keyed by workspace-relative POSIX path.

    Raises:
        CoverageExportError: If a count precedes its source record, a source is
            outside the workspace, or the export has no sources.
    """
    counts: LineCounts = {}
    current: dict[int, int] | None = None
    for line in lcov.splitlines():
        if line.startswith("SF:"):
            source = Path(line[3:]).resolve()
            try:
                relative = source.relative_to(workspace_root).as_posix()
            except ValueError as error:
                raise CoverageExportError(f"{source} is outside {workspace_root}") from error
            current = counts.setdefault(relative, {})
        elif line.startswith("DA:"):
            if current is None:
                raise CoverageExportError("lcov DA record precedes its SF record")
            number, count = line[3:].split(",")[:2]
            current[int(number)] = int(count)
    if not counts:
        raise CoverageExportError("llvm-cov exported no source files")
    return counts


def merge_line_counts(first: LineCounts, second: LineCounts) -> LineCounts:
    """Combine two exports, keeping the larger count per line."""
    merged: LineCounts = {path: dict(lines) for path, lines in first.items()}
    for path, lines in second.items():
        target = merged.setdefault(path, {})
        for number, count in lines.items():
            target[number] = max(target.get(number, 0), count)
    return merged


def render_lcov(counts: LineCounts) -> str:
    """Serialize line counts as an lcov tracefile."""
    records: list[str] = []
    for path in sorted(counts):
        lines = counts[path]
        records.append(f"SF:{path}")
        records.extend(f"DA:{number},{lines[number]}" for number in sorted(lines))
        records.append(f"LF:{len(lines)}")
        records.append(f"LH:{sum(1 for count in lines.values() if count > 0)}")
        records.append("end_of_record")
    return "\n".join(records) + "\n"


def export_group(llvm_cov: Path, profile: Path, ignore_regex: str, objects: Sequence[Path]) -> str:
    """Run llvm-cov export for one group of objects."""
    command = [
        str(llvm_cov),
        "export",
        "-format=lcov",
        "-skip-functions",
        f"-instr-profile={profile}",
        f"-ignore-filename-regex={ignore_regex}",
    ]
    for path in objects:
        command.extend(["-object", str(path)])
    return subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout


def export_lcov(build: TestBuild, workspace: Workspace, shared: Sequence[Path], unit_tests: Sequence[Path]) -> str:
    """Merge raw profiles and export the union of both object groups."""
    profdata = llvm_tool("llvm-profdata")
    llvm_cov = llvm_tool("llvm-cov")
    ignored_roots = [build.target_directory, *(d for d in build.build_directories if d != build.target_directory)]
    ignore_regex = ignore_filename_regex(workspace.root, ignored_roots)
    with tempfile.TemporaryDirectory() as directory:
        profile_list = Path(directory, "profraw-list")
        profile_list.write_text("".join(f"{path}\n" for path in profile_files(build.target_directory)))
        merged = Path(directory, "coverage.profdata")
        subprocess.run(
            [str(profdata), "merge", "-sparse", "-f", str(profile_list), "-o", str(merged)],
            check=True,
        )
        counts: LineCounts = {}
        for group in (shared, unit_tests):
            if group:
                counts = merge_line_counts(
                    counts, line_counts(export_group(llvm_cov, merged, ignore_regex, group), workspace.root)
                )
    return render_lcov(counts)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the export from the command line.

    Returns:
        Exit code, 1 when the export fails.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nextest-list", required=True, type=Path, help="nextest list --message-format json output")
    parser.add_argument("--output", required=True, type=Path, help="lcov file to write")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="limit objects to this package's artifacts, repeatable, default is the whole workspace",
    )
    arguments = parser.parse_args(argv)
    nextest_list: Path = arguments.nextest_list
    output: Path = arguments.output
    packages: list[str] = arguments.package
    try:
        build = parse_nextest_list(json.loads(nextest_list.read_text()))
        workspace = load_workspace(packages)
        objects = collect_objects(build.build_directories, artifact_pattern(workspace.artifact_names))
        shared, unit_tests = partition_objects(objects, build.unit_test_binaries)
        report = export_lcov(build, workspace, shared, unit_tests)
    except CoverageExportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
