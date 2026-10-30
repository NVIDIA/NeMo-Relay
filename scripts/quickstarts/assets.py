# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve explicit Relay assets; never substitute another version or platform."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import tarfile
import tomllib
import urllib.parse
import urllib.request
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Any

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version


def host_platform() -> str:
    """Use the platform identifiers already used by Relay's package workflows."""
    hosts = {
        ("Linux", "x86_64"): "linux-amd64",
        ("Linux", "aarch64"): "linux-arm64",
        ("Darwin", "arm64"): "macos-arm64",
        ("Windows", "AMD64"): "windows-amd64",
        ("Windows", "x86_64"): "windows-amd64",
    }
    key = (platform.system(), platform.machine())
    if key not in hosts or (key[0] == "Linux" and platform.libc_ver()[0] != "glibc"):
        raise ValueError(f"The execution runner does not yet support {key}")
    return hosts[key]


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_json(*command: str) -> Any:
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    return json.loads(result.stdout)


def safe_url(value: str) -> str:
    """Asset URLs are public locations; credentials belong in gh's environment."""
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Asset locations must be HTTPS URLs without embedded credentials, queries, or fragments")
    return value


def checksum_index(content: str) -> dict[str, str]:
    result = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?([^/\\\s]+)", line)
        if match is None or match[2] in {".", ".."} or match[2] in result:
            raise ValueError("SHA256SUMS must contain unique, plain filenames and SHA-256 digests")
        result[match[2]] = match[1].lower()
    if not result:
        raise ValueError("SHA256SUMS is empty")
    return result


class Assets:
    """Discover assets from release metadata, CI artifacts, or a local directory."""

    def __init__(self, version: str, directory: Path, location: str, repository: str, run_id: str = ""):
        if not re.fullmatch(r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?(?:\+[0-9A-Za-z.-]+)?", version):
            raise ValueError("Use an explicit Relay SemVer version, such as 0.11.0-rc.2")
        self.version = version
        self.directory = directory
        self.directory.mkdir(parents=True)
        self.platform = host_platform()
        self.records: dict[str, Any] = {}
        self.source: dict[str, Any] = {"repository": repository, "version": version}
        self.entries: list[dict[str, str]] = []
        if location.startswith("https://"):
            safe_url(location)
        if run_id:
            if location:
                raise ValueError("Choose an asset location or a build run ID")
            if not run_id.isdigit():
                raise ValueError("A build run ID must contain only digits")
            info = run_json(
                "gh", "run", "view", run_id, "--repo", repository, "--json", "headSha,url,status,conclusion"
            )
            if info["status"] != "completed":
                raise ValueError("The source build must have completed before its assets can be tested")
            self.source.update(build_run=run_id, **info)
            build = self.directory / "build"
            subprocess.run(
                ["gh", "run", "download", run_id, "--repo", repository, "--dir", str(build)], check=True, timeout=600
            )
            self._local(build)
        elif location and not location.startswith("https://"):
            self._local(Path(location).expanduser().resolve())
            self.source["directory"] = str(Path(location).expanduser().resolve())
        else:
            # Release metadata is the inventory; filenames are not synthesized from a version.
            release = re.fullmatch(r"https://github.com/([^/]+/[^/]+)/releases/tag/([^/]+)", location)
            if not location or release:
                repo, tag = (release[1], release[2]) if release else (repository, version)
                info = run_json("gh", "release", "view", tag, "--repo", repo, "--json", "tagName,url,assets")
                self.source.update(release=info["url"], tag=info["tagName"])
                self.entries = [{"name": item["name"], "url": item["url"]} for item in info["assets"]]
                checksums = next((item for item in self.entries if item["name"] == "SHA256SUMS"), None)
                if checksums is None:
                    raise ValueError("Release assets must include SHA256SUMS")
                sums = checksum_index(self._read_url(checksums["url"]).decode())
                for entry in self.entries:
                    entry["sha256"] = sums.get(entry["name"], "")
            else:
                base = safe_url(location).rstrip("/")
                sums = checksum_index(self._read_url(base + "/SHA256SUMS").decode())
                self.entries = [
                    {"name": name, "url": base + "/" + name, "sha256": digest} for name, digest in sums.items()
                ]
                self.source["base_url"] = base

    @staticmethod
    def _read_url(url: str) -> bytes:
        with urllib.request.urlopen(safe_url(url), timeout=120) as response:
            return response.read()

    def _local(self, folder: Path) -> None:
        if not folder.is_dir():
            raise ValueError(f"Asset directory does not exist: {folder}")
        # Reusing a generated source build retains the original commit and integrity
        # evidence. This file is written by the runner, never maintained per release.
        build_manifest = folder.parent / "build.json" if folder.name == "packages" else folder / "build.json"
        build = json.loads(build_manifest.read_text()) if build_manifest.is_file() else None
        if build is not None:
            if build.get("status") != "built" or not re.fullmatch(r"[0-9a-f]{40}", build.get("commit", "")):
                raise ValueError("Source build manifest is incomplete or has no immutable commit")
            self.source["build"] = build
        for path in sorted(folder.rglob("*")):
            if path.is_file() and (
                path.suffix in {".whl", ".tgz", ".crate"} or path.name.startswith("nemo-relay-cli-")
            ):
                sums_path = path.parent / "SHA256SUMS"
                sums = checksum_index(sums_path.read_text()) if sums_path.exists() else {}
                if build is not None:
                    expected = build["files"].get(path.name)
                    if expected is None:
                        raise ValueError(f"Source build manifest does not identify {path.name}")
                    sums[path.name] = expected
                self.entries.append({"name": path.name, "path": str(path), "sha256": sums.get(path.name, "")})

    def materialize(self, entry: dict[str, str]) -> Path:
        if Path(entry["name"]).name != entry["name"] or entry["name"] in {".", ".."}:
            raise ValueError("Asset filenames must not contain directory traversal")
        expected = entry.get("sha256", "")
        if "path" in entry:
            path = Path(entry["path"])
        else:
            if not expected:
                raise ValueError(f"No trusted checksum supplied for {entry['name']}")
            path = self.directory / entry["name"]
            path.write_bytes(self._read_url(entry["url"]))
        digest = sha256(path)
        if expected and digest != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")
        self.records[path.name] = {"location": entry.get("url", str(path)), "sha256": digest}
        return path

    def python(self) -> Path:
        compatible = set(sys_tags())
        candidates = []
        for entry in self.entries:
            if not entry["name"].endswith(".whl"):
                continue
            name, version, _, tags = parse_wheel_filename(entry["name"])
            if name == "nemo-relay" and version == Version(self.version) and compatible.intersection(tags):
                candidates.append(entry)
        if len(candidates) != 1:
            raise ValueError(f"Expected one compatible nemo-relay {self.version} wheel, found {len(candidates)}")
        path = self.materialize(candidates[0])
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("Wheel must contain one package metadata file")
            info = Parser().parsestr(archive.read(metadata[0]).decode())
            if info["Name"] != "nemo-relay" or Version(info["Version"]) != Version(self.version):
                raise ValueError("Wheel metadata does not match the requested package/version")
        return path

    def cli(self) -> Path:
        """Resolve the same raw host binary used by the Relay installer."""
        triples = {
            "macos-arm64": "aarch64-apple-darwin",
            "linux-amd64": "x86_64-unknown-linux-gnu",
            "linux-arm64": "aarch64-unknown-linux-gnu",
            "windows-amd64": "x86_64-pc-windows-msvc",
        }
        filename = f"nemo-relay-cli-{triples[self.platform]}-{self.version}" + (
            ".exe" if self.platform.startswith("windows") else ""
        )
        candidates = [entry for entry in self.entries if entry["name"] == filename]
        if len(candidates) != 1:
            raise ValueError(f"Expected one CLI asset {filename}, found {len(candidates)}")
        path = self.materialize(candidates[0])
        path.chmod(0o755)
        installed = subprocess.check_output([str(path), "--version"], text=True, timeout=30).strip().split()[-1]
        if installed != self.version:
            raise ValueError(f"CLI reports {installed}, expected {self.version}")
        return path

    def nodejs(self) -> list[Path]:
        suffixes = {
            "linux-amd64": "linux-x64-gnu",
            "linux-arm64": "linux-arm64-gnu",
            "macos-arm64": "darwin-arm64",
            "windows-amd64": "win32-x64-msvc",
        }
        expected = {"nemo-relay-node", "nemo-relay-node-" + suffixes[self.platform]}
        selected: dict[str, Path] = {}
        for entry in self.entries:
            if not entry["name"].endswith(".tgz"):
                continue
            # Filter the advertised version before reading native tarballs from other releases.
            if "path" not in entry and not entry["name"].endswith(f"-{self.version}.tgz"):
                continue
            # Remote inventories can contain seven native variants; only download this host's files.
            native = {
                "linux-amd64": "linux-x64",
                "linux-arm64": "linux-arm64",
                "macos-arm64": "darwin-arm64",
                "windows-amd64": "win32-x64",
            }[self.platform]
            if "path" not in entry and entry["name"] not in {
                f"nemo-relay-node-npm-{self.version}.tgz",
                f"nemo-relay-node-npm-{native}-{self.version}.tgz",
            }:
                continue
            path = self.materialize(entry)
            with tarfile.open(path) as archive:
                member = archive.extractfile("package/package.json")
                if member is None:
                    raise ValueError("npm tarball has no package.json")
                info = json.load(member)
            if info["name"] in expected and info["version"] == self.version:
                if info["name"] in selected:
                    raise ValueError(f"Ambiguous npm asset for {info['name']}")
                selected[info["name"]] = path
        if set(selected) != expected:
            raise ValueError(f"Missing npm metapackage or {self.platform} native package for {self.version}")
        # Install the native dependency explicitly; do not retrieve optional binaries from another registry.
        return [selected[name] for name in sorted(expected, reverse=True)]

    def rust(self, source: str = "") -> dict[str, str]:
        if source:
            path = Path(source).expanduser().resolve()
            info = tomllib.loads((path / "Cargo.toml").read_text())
            if info["package"]["name"] != "nemo-relay":
                raise ValueError("Rust source override must point to the nemo-relay crate")
            # Cargo metadata resolves inherited workspace versions and path dependencies.
            metadata = run_json(
                "cargo", "metadata", "--no-deps", "--format-version", "1", "--manifest-path", str(path / "Cargo.toml")
            )
            package = next(item for item in metadata["packages"] if item["name"] == "nemo-relay")
            if package["version"] != self.version:
                raise ValueError("Rust source version does not match the requested version")
            commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
            root = Path(metadata["workspace_root"])
            files = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"], text=True
            ).split("\0")
            digest = hashlib.sha256()
            for filename in sorted(set(files) - {""}):
                source_file = root / filename
                if source_file.is_file():
                    digest.update(filename.encode() + b"\0" + bytes.fromhex(sha256(source_file)))
            self.source["rust_source"] = {
                "path": str(path),
                "commit": commit,
                "dirty": bool(dirty),
                "tree_sha256": digest.hexdigest(),
            }
            return {"nemo-relay": str(path)}
        patches = {}
        for entry in self.entries:
            if not entry["name"].startswith("nemo-relay") or not entry["name"].endswith(f"-{self.version}.crate"):
                continue
            archive_path = self.materialize(entry)
            destination = self.directory / "crates" / archive_path.stem
            destination.mkdir(parents=True)
            with tarfile.open(archive_path) as archive:
                # Refuse links and traversal even on Python versions without the newer data filter.
                for member in archive.getmembers():
                    if not (member.isfile() or member.isdir()) or not (
                        destination / member.name
                    ).resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Unsafe crate archive member")
                archive.extractall(destination, filter="data")
            manifests = list(destination.glob("*/Cargo.toml"))
            if len(manifests) != 1:
                raise ValueError("Crate archive must contain one root Cargo.toml")
            info = tomllib.loads(manifests[0].read_text())["package"]
            if info["version"] != self.version or not info["name"].startswith("nemo-relay"):
                raise ValueError("Crate metadata does not match the requested version")
            if info["name"] in patches:
                raise ValueError(f"Ambiguous crate asset for {info['name']}")
            patches[info["name"]] = str(manifests[0].parent)
        if patches and "nemo-relay" not in patches:
            raise ValueError("Rust assets are missing the nemo-relay crate")
        # Released Rust crates live on crates.io, rather than in the GitHub binary assets.
        self.source["rust_registry"] = None if patches else "crates-io"
        return patches

    def override(self, overrides: dict[str, Any]) -> str:
        """Optional local asset exceptions; regular runs only need version and location."""
        if not isinstance(overrides, dict) or set(overrides) - {"python", "nodejs", "rust_source"}:
            raise ValueError("Overrides accept only python, nodejs, and rust_source")
        for surface, suffix in [("python", ".whl"), ("nodejs", ".tgz")]:
            if surface not in overrides:
                continue
            values = overrides[surface]
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
                raise ValueError(f"{surface} override must be a path or a nonempty list of paths")
            self.entries = [entry for entry in self.entries if not entry["name"].endswith(suffix)]
            for value in values:
                path = Path(value).expanduser().resolve()
                if not path.is_file() or path.suffix != suffix:
                    raise ValueError(f"Invalid {surface} asset override")
                self.entries.append({"name": path.name, "path": str(path)})
        source = overrides.get("rust_source", "")
        if not isinstance(source, str):
            raise ValueError("rust_source must be a local crate directory")
        return source
