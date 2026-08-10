#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package a prebuilt NeMo Relay CLI binary for PyPI."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import stat
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
    wheel_platforms: tuple[str, ...]
    executable: str


PLATFORMS = {
    platform.target: platform
    for platform in (
        Platform(
            "x86_64-unknown-linux-gnu",
            ("manylinux_2_17_x86_64",),
            "nemo-relay",
        ),
        Platform(
            "aarch64-unknown-linux-gnu",
            ("manylinux_2_17_aarch64",),
            "nemo-relay",
        ),
        Platform(
            "x86_64-unknown-linux-musl",
            ("musllinux_1_2_x86_64",),
            "nemo-relay",
        ),
        Platform(
            "aarch64-unknown-linux-musl",
            ("musllinux_1_2_aarch64",),
            "nemo-relay",
        ),
        Platform(
            "aarch64-apple-darwin",
            ("macosx_11_0_arm64",),
            "nemo-relay",
        ),
        Platform(
            "x86_64-pc-windows-msvc",
            ("win_amd64",),
            "nemo-relay.exe",
        ),
        Platform(
            "aarch64-pc-windows-msvc",
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


def parse_args() -> argparse.Namespace:
    """Parse CLI package assembly arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=sorted(PLATFORMS), required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Build the wheel requested on the command line."""
    args = parse_args()
    if not args.binary.is_file():
        raise SystemExit(f"CLI binary does not exist: {args.binary}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    platform = PLATFORMS[args.target]
    print(build_wheel(args.binary, platform, args.version, args.output_dir))


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
