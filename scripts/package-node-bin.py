#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package the NeMo Relay Node.js binding as a metapackage and native package."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "nemo-relay-node"


@dataclass(frozen=True)
class Platform:
    """Describe one supported Node.js native package."""

    key: str
    package_suffix: str
    npm_os: str
    npm_cpu: str
    binary: str
    libc: str | None = None

    @property
    def package_name(self) -> str:
        """Return the public npm package name."""
        return f"{PACKAGE_NAME}-{self.package_suffix}"


PLATFORMS = {
    platform.key: platform
    for platform in (
        Platform("linux-amd64", "linux-x64-gnu", "linux", "x64", "nemo-relay.linux-x64-gnu.node", "glibc"),
        Platform("linux-arm64", "linux-arm64-gnu", "linux", "arm64", "nemo-relay.linux-arm64-gnu.node", "glibc"),
        Platform("linux-musl-amd64", "linux-x64-musl", "linux", "x64", "nemo-relay.linux-x64-musl.node", "musl"),
        Platform("linux-musl-arm64", "linux-arm64-musl", "linux", "arm64", "nemo-relay.linux-arm64-musl.node", "musl"),
        Platform("macos-arm64", "darwin-arm64", "darwin", "arm64", "nemo-relay.darwin-arm64.node"),
        Platform("windows-amd64", "win32-x64-msvc", "win32", "x64", "nemo-relay.win32-x64-msvc.node"),
        Platform("windows-arm64", "win32-arm64-msvc", "win32", "arm64", "nemo-relay.win32-arm64-msvc.node"),
    )
}


def add_tar_bytes(archive: tarfile.TarFile, path: str, content: bytes, mode: int = 0o644) -> None:
    """Add one regular file to an npm tarball."""
    info = tarfile.TarInfo(path)
    info.size = len(content)
    info.mode = mode
    archive.addfile(info, io.BytesIO(content))


def repository_metadata(source: dict[str, object]) -> dict[str, object]:
    """Return metadata shared by the metapackage and native packages."""
    fields = ("description", "keywords", "homepage", "bugs", "repository", "author", "engines", "license")
    return {field: source[field] for field in fields if field in source}


def metapackage_files(manifest: dict[str, object]) -> list[str]:
    """Return the JavaScript and declaration files exported by the package."""
    files = {str(manifest["main"]), str(manifest["types"]), "README.md"}
    exports = manifest.get("exports")
    if not isinstance(exports, dict):
        raise ValueError("Node package manifest is missing exports")
    for entry in exports.values():
        if not isinstance(entry, dict):
            raise ValueError("Node package export must contain types and default paths")
        files.update(str(path).removeprefix("./") for path in entry.values())
    return sorted(files)


def build_native_package(node_dir: Path, platform: Platform, version: str, output: Path) -> Path:
    """Build one OS- and CPU-constrained native npm package."""
    source_manifest = json.loads((node_dir / "package.json").read_text())
    manifest = {
        "name": platform.package_name,
        "version": version,
        **repository_metadata(source_manifest),
        "main": platform.binary,
        "files": [platform.binary],
        "os": [platform.npm_os],
        "cpu": [platform.npm_cpu],
    }
    if platform.libc is not None:
        manifest["libc"] = [platform.libc]
    binary = node_dir / platform.binary
    if not binary.is_file():
        raise FileNotFoundError(f"Node native binary does not exist: {binary}")
    artifact_suffix = f"{platform.npm_os}-{platform.npm_cpu}"
    if platform.libc == "musl":
        artifact_suffix += "-musl"
    destination = output / f"nemo-relay-node-npm-{artifact_suffix}-{version}.tgz"
    with tarfile.open(destination, "w:gz") as archive:
        add_tar_bytes(archive, "package/package.json", json.dumps(manifest, indent=2).encode() + b"\n")
        add_tar_bytes(archive, f"package/{platform.binary}", binary.read_bytes(), mode=0o755)
        add_tar_bytes(archive, "package/LICENSE", (ROOT / "LICENSE").read_bytes())
    return destination


def build_metapackage(node_dir: Path, version: str, output: Path) -> Path:
    """Build the portable Node.js metapackage."""
    source_manifest = json.loads((node_dir / "package.json").read_text())
    manifest = {
        "name": PACKAGE_NAME,
        "version": version,
        **repository_metadata(source_manifest),
        "main": source_manifest["main"],
        "types": source_manifest["types"],
        "exports": source_manifest["exports"],
        "files": metapackage_files(source_manifest),
        "optionalDependencies": {platform.package_name: version for platform in PLATFORMS.values()},
    }
    destination = output / f"nemo-relay-node-npm-{version}.tgz"
    with tarfile.open(destination, "w:gz") as archive:
        add_tar_bytes(archive, "package/package.json", json.dumps(manifest, indent=2).encode() + b"\n")
        for filename in manifest["files"]:
            path = node_dir / filename
            add_tar_bytes(archive, f"package/{filename}", path.read_bytes())
        add_tar_bytes(archive, "package/LICENSE", (ROOT / "LICENSE").read_bytes())
    return destination


def parse_args() -> argparse.Namespace:
    """Parse Node package assembly arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--node-dir", type=Path, default=ROOT / "crates" / "node")
    parser.add_argument("--platform", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metapackage", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the requested native package and optional metapackage."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        build_native_package(args.node_dir, PLATFORMS[args.platform], args.version, args.output_dir),
    ]
    if args.metapackage:
        artifacts.append(build_metapackage(args.node_dir, args.version, args.output_dir))
    for artifact in artifacts:
        print(artifact)


if __name__ == "__main__":
    main()
