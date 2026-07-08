#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect read-only Git evidence for NeMo Relay documentation release notes."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

RELEASE_NOTES_DIR = "docs/about-nemo-relay/release-notes"
VERSION_PATTERN = re.compile(r"(?:NVIDIA )?NeMo Relay (\d+\.\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Commit:
    """One non-merge commit and its changed paths."""

    sha: str
    subject: str
    paths: tuple[str, ...]


def run_git(repo: Path, *args: str) -> str:
    """Run Git without a shell and return stdout or raise a readable error."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def resolve_ref(repo: Path, ref: str) -> str:
    """Return the commit ID for a user-supplied ref."""
    return run_git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()


def note_files(repo: Path, ref: str) -> tuple[str, ...]:
    """List release-note files that exist at a ref."""
    output = run_git(repo, "ls-tree", "-r", "--name-only", ref, "--", RELEASE_NOTES_DIR)
    return tuple(path for path in output.splitlines() if path)


def note_versions(repo: Path, ref: str, paths: tuple[str, ...]) -> tuple[str, ...]:
    """Return the NeMo Relay versions mentioned in release-note files."""
    versions: set[str] = set()
    for path in paths:
        content = run_git(repo, "show", f"{ref}:{path}")
        versions.update(VERSION_PATTERN.findall(content))
    return tuple(sorted(versions))


def commits(repo: Path, previous: str, current: str) -> tuple[Commit, ...]:
    """Read non-merge commits and their changed paths, newest first."""
    output = run_git(repo, "log", "--no-merges", "--format=%H%x09%s", f"{previous}..{current}")
    collected: list[Commit] = []
    for line in output.splitlines():
        sha, subject = line.split("\t", maxsplit=1)
        names = run_git(repo, "diff-tree", "--no-commit-id", "--name-status", "-r", sha)
        paths = tuple(entry.rsplit("\t", maxsplit=1)[-1] for entry in names.splitlines() if entry)
        collected.append(Commit(sha=sha, subject=subject, paths=paths))
    return tuple(collected)


def category(commit: Commit) -> str:
    """Classify a commit for review without treating the result as a release claim."""
    subject = commit.subject.lower()
    if "!" in commit.subject.split(":", maxsplit=1)[0] or re.search(
        r"\b(remove|drop|breaking|deprecat|migration)\b", subject
    ):
        return "Breaking and migration candidates"
    if subject.startswith("feat"):
        return "Feature candidates"
    if subject.startswith(("fix", "perf")):
        return "Fix candidates"
    return "Documentation and tooling candidates"


def public_paths(commits_to_report: tuple[Commit, ...]) -> tuple[str, ...]:
    """Return changed paths likely to support public release-note claims."""
    package_paths = {
        "Cargo.toml",
        "crates/node/package.json",
        "integrations/openclaw/package.json",
        "python/plugin/pyproject.toml",
    }
    return tuple(
        sorted(
            {
                path
                for commit in commits_to_report
                for path in commit.paths
                if path == "README.md"
                or path.startswith("docs/")
                or path.endswith("/README.md")
                or path in package_paths
            }
        )
    )


def print_note_tree(repo: Path, label: str, ref: str) -> None:
    """Print structure and version mentions for one release ref."""
    paths = note_files(repo, ref)
    print(f"## {label} Release-Notes Tree")
    if not paths:
        print(f"No files exist under `{RELEASE_NOTES_DIR}` at `{ref}`.\n")
        return
    print("Files:")
    for path in paths:
        print(f"- `{path}`")
    versions = note_versions(repo, ref, paths)
    description = ", ".join(versions) if versions else "no NeMo Relay version text detected"
    print(f"Version text: {description}\n")


def main() -> int:
    """Parse arguments and print the evidence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, help="Previous release branch or commit")
    parser.add_argument("--current", required=True, help="Current release branch or commit")
    parser.add_argument("--version", required=True, help="Target major.minor release version")
    parser.add_argument("--repo", default=".", help="Repository root (default: current directory)")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    try:
        previous_sha = resolve_ref(repo, args.previous)
        current_sha = resolve_ref(repo, args.current)
        run_git(repo, "merge-base", "--is-ancestor", previous_sha, current_sha)
    except RuntimeError as error:
        parser.error(str(error))

    print(f"# Release-Note Evidence: {args.version}")
    print()
    print(f"- Previous: `{args.previous}` ({previous_sha[:12]})")
    print(f"- Current: `{args.current}` ({current_sha[:12]})")
    print(f"- Merge base: `{run_git(repo, 'merge-base', previous_sha, current_sha).strip()[:12]}`")
    print()
    print_note_tree(repo, "Previous", args.previous)
    print_note_tree(repo, "Current", args.current)

    changes = commits(repo, previous_sha, current_sha)
    grouped: dict[str, list[Commit]] = {
        "Breaking and migration candidates": [],
        "Feature candidates": [],
        "Fix candidates": [],
        "Documentation and tooling candidates": [],
    }
    for commit in changes:
        grouped[category(commit)].append(commit)

    print("## Commit Candidates")
    for heading, entries in grouped.items():
        print(f"### {heading}")
        if not entries:
            print("- None")
        for commit in entries:
            paths = ", ".join(f"`{path}`" for path in commit.paths[:4])
            suffix = f" — {paths}" if paths else ""
            print(f"- `{commit.sha[:12]}` {commit.subject}{suffix}")
        print()

    print("## Changed Public Documentation and Package Surfaces")
    paths = public_paths(changes)
    if paths:
        for path in paths:
            print(f"- `{path}`")
    else:
        print("- None")
    print()
    print("Review every candidate against current public docs and APIs before drafting prose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
