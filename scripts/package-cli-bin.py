#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a prebuilt NeMo Relay CLI binary for PyPI and npm."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

PACKAGE_NAME = "nemo-relay-cli-bin"
SUMMARY = "Prebuilt NeMo Relay command-line interface."
LICENSE = "Apache-2.0"
REPOSITORY = "https://github.com/NVIDIA/NeMo-Relay"
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Platform:
    """Describe one supported CLI distribution platform."""

    target: str
    npm_suffix: str
    npm_os: str
    npm_cpu: str
    wheel_platforms: tuple[str, ...]
    executable: str

    @property
    def npm_package(self) -> str:
        """Return the npm package name for this platform."""
        return f"{PACKAGE_NAME}-{self.npm_suffix}"


PLATFORMS = {
    platform.target: platform
    for platform in (
        Platform(
            "x86_64-unknown-linux-musl",
            "linux-x64",
            "linux",
            "x64",
            ("manylinux_2_17_x86_64", "musllinux_1_2_x86_64"),
            "nemo-relay",
        ),
        Platform(
            "aarch64-unknown-linux-musl",
            "linux-arm64",
            "linux",
            "arm64",
            ("manylinux_2_17_aarch64", "musllinux_1_2_aarch64"),
            "nemo-relay",
        ),
        Platform(
            "aarch64-apple-darwin",
            "darwin-arm64",
            "darwin",
            "arm64",
            ("macosx_11_0_arm64",),
            "nemo-relay",
        ),
        Platform(
            "x86_64-pc-windows-msvc",
            "win32-x64",
            "win32",
            "x64",
            ("win_amd64",),
            "nemo-relay.exe",
        ),
        Platform(
            "aarch64-pc-windows-msvc",
            "win32-arm64",
            "win32",
            "arm64",
            ("win_arm64",),
            "nemo-relay.exe",
        ),
    )
}


def wheel_version(version: str) -> str:
    """Translate the repository SemVer spelling to PEP 440."""
    import re

    match = re.fullmatch(
        r"(?P<release>\d+\.\d+\.\d+)"
        r"(?:-(?P<label>alpha|beta|rc)\.(?P<number>\d+))?"
        r"(?:\+(?P<local>[0-9A-Za-z._-]+))?",
        version,
    )
    if match is None:
        raise ValueError(f"unsupported package version: {version}")
    translated = match.group("release")
    if label := match.group("label"):
        translated += {"alpha": "a", "beta": "b", "rc": "rc"}[label]
        translated += match.group("number")
    if local := match.group("local"):
        translated += "+" + ".".join(part.lower() for part in re.split(r"[._-]+", local))
    return translated


def record_entry(path: str, content: bytes) -> str:
    """Return one wheel RECORD entry for the provided file."""
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
    return f"{path},sha256={digest},{len(content)}"


def add_zip_file(archive: zipfile.ZipFile, path: str, content: bytes, executable: bool = False) -> None:
    """Add one regular file to a wheel archive."""
    info = zipfile.ZipInfo(path)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | (0o755 if executable else 0o644)) << 16
    archive.writestr(info, content)


def build_wheel(binary: Path, platform: Platform, version: str, output: Path) -> Path:
    """Build a platform-tagged wheel containing the CLI binary."""
    pep440_version = wheel_version(version)
    normalized_name = PACKAGE_NAME.replace("-", "_")
    platform_tag = ".".join(platform.wheel_platforms)
    filename = f"{normalized_name}-{pep440_version}-py3-none-{platform_tag}.whl"
    destination = output / filename
    dist_info = f"{normalized_name}-{pep440_version}.dist-info"
    script_path = f"{normalized_name}-{pep440_version}.data/scripts/{platform.executable}"
    metadata = (
        "Metadata-Version: 2.4\n"
        f"Name: {PACKAGE_NAME}\n"
        f"Version: {pep440_version}\n"
        f"Summary: {SUMMARY}\n"
        f"License-Expression: {LICENSE}\n"
        "Requires-Python: >=3.11\n"
        f"Project-URL: Repository, {REPOSITORY}\n"
        "Description-Content-Type: text/markdown\n"
        "\n"
        "This platform wheel installs the prebuilt `nemo-relay` command-line interface.\n"
    ).encode()
    wheel = (
        "Wheel-Version: 1.0\n"
        "Generator: NeMo Relay package-cli-bin.py\n"
        "Root-Is-Purelib: false\n" + "".join(f"Tag: py3-none-{tag}\n" for tag in platform.wheel_platforms) + "\n"
    ).encode()
    license_text = (ROOT / "LICENSE").read_bytes()
    binary_content = binary.read_bytes()
    files = {
        script_path: binary_content,
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": wheel,
        f"{dist_info}/licenses/LICENSE": license_text,
    }
    record_path = f"{dist_info}/RECORD"
    record = "\n".join(record_entry(path, content) for path, content in files.items())
    record += f"\n{record_path},,\n"
    with zipfile.ZipFile(destination, "w") as archive:
        for path, content in files.items():
            add_zip_file(archive, path, content, executable=path == script_path)
        add_zip_file(archive, record_path, record.encode())
    return destination


def add_tar_bytes(archive: tarfile.TarFile, path: str, content: bytes, mode: int = 0o644) -> None:
    """Add one regular file to an npm tarball."""
    info = tarfile.TarInfo(path)
    info.size = len(content)
    info.mode = mode
    archive.addfile(info, io.BytesIO(content))


def build_npm_platform(binary: Path, platform: Platform, version: str, output: Path) -> Path:
    """Build an OS- and CPU-constrained npm package containing the CLI binary."""
    filename = f"nemo-relay-bin-npm-{platform.npm_os}-{platform.npm_cpu}-{version}.tgz"
    destination = output / filename
    manifest = {
        "name": platform.npm_package,
        "version": version,
        "description": f"{SUMMARY} ({platform.npm_os} {platform.npm_cpu})",
        "os": [platform.npm_os],
        "cpu": [platform.npm_cpu],
        "files": [f"bin/{platform.executable}"],
        "license": LICENSE,
        "repository": {"type": "git", "url": f"git+{REPOSITORY}.git"},
    }
    with tarfile.open(destination, "w:gz") as archive:
        add_tar_bytes(archive, "package/package.json", json.dumps(manifest, indent=2).encode() + b"\n")
        add_tar_bytes(
            archive,
            f"package/bin/{platform.executable}",
            binary.read_bytes(),
            mode=0o755,
        )
        add_tar_bytes(archive, "package/LICENSE", (ROOT / "LICENSE").read_bytes())
    return destination


def launcher_source() -> bytes:
    """Return the Node.js launcher that selects the installed native package."""
    mapping = {
        f"{platform.npm_os}-{platform.npm_cpu}": {
            "package": platform.npm_package,
            "executable": platform.executable,
        }
        for platform in PLATFORMS.values()
    }
    return (
        "#!/usr/bin/env node\n"
        "// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "// SPDX-License-Identifier: Apache-2.0\n\n"
        "const { spawnSync } = require('node:child_process');\n"
        "const path = require('node:path');\n\n"
        f"const platforms = {json.dumps(mapping, indent=2)};\n"
        "const key = `${process.platform}-${process.arch}`;\n"
        "const selected = platforms[key];\n"
        "if (!selected) {\n"
        "  console.error(`nemo-relay-cli-bin does not support ${process.platform}/${process.arch}`);\n"
        "  process.exit(1);\n"
        "}\n"
        "let manifest;\n"
        "try {\n"
        "  manifest = require.resolve(`${selected.package}/package.json`);\n"
        "} catch (error) {\n"
        "  console.error(\n"
        "    `The native package ${selected.package} is missing. ` +\n"
        "      'Reinstall nemo-relay-cli-bin without omitting optional dependencies.',\n"
        "  );\n"
        "  process.exit(1);\n"
        "}\n"
        "const executable = path.join(path.dirname(manifest), 'bin', selected.executable);\n"
        "const result = spawnSync(executable, process.argv.slice(2), { stdio: 'inherit' });\n"
        "if (result.error) {\n"
        "  console.error(`Failed to start ${executable}: ${result.error.message}`);\n"
        "  process.exit(1);\n"
        "}\n"
        "process.exit(result.status === null ? 1 : result.status);\n"
    ).encode()


def build_npm_launcher(version: str, output: Path) -> Path:
    """Build the portable npm launcher package."""
    destination = output / f"nemo-relay-bin-npm-{version}.tgz"
    manifest = {
        "name": PACKAGE_NAME,
        "version": version,
        "description": SUMMARY,
        "bin": {"nemo-relay": "bin/nemo-relay.js"},
        "files": ["bin/nemo-relay.js"],
        "engines": {"node": ">=24.0.0"},
        "optionalDependencies": {platform.npm_package: version for platform in PLATFORMS.values()},
        "license": LICENSE,
        "repository": {"type": "git", "url": f"git+{REPOSITORY}.git"},
    }
    with tarfile.open(destination, "w:gz") as archive:
        add_tar_bytes(archive, "package/package.json", json.dumps(manifest, indent=2).encode() + b"\n")
        add_tar_bytes(archive, "package/bin/nemo-relay.js", launcher_source(), mode=0o755)
        add_tar_bytes(archive, "package/LICENSE", (ROOT / "LICENSE").read_bytes())
    return destination


def parse_args() -> argparse.Namespace:
    """Parse CLI package assembly arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--npm-launcher", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the wheel and npm artifacts requested on the command line."""
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"CLI binary does not exist: {args.binary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    platform = PLATFORMS[args.target]
    artifacts = [
        build_wheel(args.binary, platform, args.version, args.output_dir),
        build_npm_platform(args.binary, platform, args.version, args.output_dir),
    ]
    if args.npm_launcher:
        artifacts.append(build_npm_launcher(args.version, args.output_dir))
    for artifact in artifacts:
        print(artifact)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
