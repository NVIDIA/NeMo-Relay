# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and verify platform archives for the Rampart native plugin."""

from __future__ import annotations

import argparse
import ctypes
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_ROOT = "nemo-relay-pii-rampart-plugin"
PLUGIN_SYMBOL = "nemo_relay_register_plugin"
TARGET_LIBRARIES = {
    "x86_64-unknown-linux-gnu": "libnemo_relay_pii_rampart_plugin.so",
    "aarch64-unknown-linux-gnu": "libnemo_relay_pii_rampart_plugin.so",
    "x86_64-unknown-linux-musl": "libnemo_relay_pii_rampart_plugin.so",
    "aarch64-unknown-linux-musl": "libnemo_relay_pii_rampart_plugin.so",
    "aarch64-apple-darwin": "libnemo_relay_pii_rampart_plugin.dylib",
    "x86_64-pc-windows-msvc": "nemo_relay_pii_rampart_plugin.dll",
    "aarch64-pc-windows-msvc": "nemo_relay_pii_rampart_plugin.dll",
}
PACKAGE_FILES = {
    "ATTRIBUTIONS-Rust.md": Path("ATTRIBUTIONS-Rust.md"),
    "LICENSE": Path("LICENSE"),
    "README.md": Path("plugins/pii-rampart/README.md"),
    "config.schema.json": Path("plugins/pii-rampart/config.schema.json"),
}
VERSION_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-(?:alpha|beta|rc)\.(?:0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"|dev-[0-9A-Fa-f]{8}"
)


def sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def library_name(target: str) -> str:
    """Return the native plugin filename for a supported Rust target."""
    try:
        return TARGET_LIBRARIES[target]
    except KeyError as error:
        supported = ", ".join(sorted(TARGET_LIBRARIES))
        raise ValueError(f"unsupported target {target!r}; expected one of: {supported}") from error


def render_manifest(template: Path, library: str, digest: str) -> bytes:
    """Materialize the authored manifest for a packaged library."""
    text = template.read_text()
    if text.count("<platform-library-file>") != 2:
        raise ValueError("expected exactly two platform library placeholders")
    if text.count("<artifact-sha256>") != 1:
        raise ValueError("expected exactly one artifact digest placeholder")
    text = text.replace("target/release/<platform-library-file>", f"lib/{library}")
    text = text.replace("<artifact-sha256>", digest)
    return text.encode()


def archive_entries(repository: Path, library: Path, target: str) -> dict[str, bytes]:
    """Collect the complete, normalized package contents."""
    expected_library = library_name(target)
    if library.name != expected_library:
        raise ValueError(f"library filename {library.name!r} does not match {expected_library!r} for {target}")
    if not library.is_file():
        raise FileNotFoundError(f"plugin library does not exist: {library}")

    entries = {name: (repository / source).read_bytes() for name, source in PACKAGE_FILES.items()}
    digest = sha256(library)
    entries["relay-plugin.toml"] = render_manifest(
        repository / "plugins/pii-rampart/relay-plugin.toml", expected_library, digest
    )
    entries[f"lib/{expected_library}"] = library.read_bytes()
    return entries


def add_tar_entry(archive: tarfile.TarFile, name: str, content: bytes, executable: bool) -> None:
    """Add one reproducible regular-file entry to a tar archive."""
    info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{name}")
    info.size = len(content)
    info.mode = 0o755 if executable else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(content))


def write_tar_gz(destination: Path, entries: dict[str, bytes]) -> None:
    """Write a deterministic gzip-compressed tar archive."""
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in sorted(entries):
                    add_tar_entry(archive, name, entries[name], name.startswith("lib/"))


def write_zip(destination: Path, entries: dict[str, bytes]) -> None:
    """Write a deterministic ZIP archive."""
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if name.startswith("lib/") else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, entries[name])


def build_archive(args: argparse.Namespace) -> Path:
    """Build one platform archive and return its path."""
    if VERSION_PATTERN.fullmatch(args.version) is None:
        raise ValueError(f"unsupported plugin archive version: {args.version!r}")
    repository = args.repository.resolve()
    library = args.library.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = archive_entries(repository, library, args.target)
    extension = ".zip" if args.target.endswith("windows-msvc") else ".tar.gz"
    destination = output_dir / (f"nemo-relay-pii-rampart-plugin-{args.target}-{args.version}{extension}")
    if extension == ".zip":
        write_zip(destination, entries)
    else:
        write_tar_gz(destination, entries)
    print(destination)
    return destination


def safe_member_path(name: str) -> PurePosixPath:
    """Validate and normalize one archive member path."""
    raw_parts = name.split("/")
    if any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in raw_parts):
        raise ValueError(f"unsafe archive member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract an archive after rejecting links and traversal paths."""
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[PurePosixPath] = set()
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                path = safe_member_path(member.filename)
                if member.is_dir():
                    continue
                if path in seen:
                    raise ValueError(f"duplicate archive member path: {member.filename!r}")
                seen.add(path)
                output = destination.joinpath(*path.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(source.read(member))
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                path = safe_member_path(member.name)
                if not member.isfile():
                    raise ValueError(f"archive member is not a regular file: {member.name!r}")
                if path in seen:
                    raise ValueError(f"duplicate archive member path: {member.name!r}")
                seen.add(path)
                stream = source.extractfile(member)
                if stream is None:
                    raise ValueError(f"could not read archive member: {member.name!r}")
                output = destination.joinpath(*path.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(stream.read())
                os.chmod(output, member.mode)
    else:
        raise ValueError(f"unsupported archive extension: {archive.name}")


def verify_package(root: Path, target: str, load_library: bool) -> None:
    """Validate a materialized package and optionally load its native symbol."""
    expected_library = library_name(target)
    expected_files = {
        "ATTRIBUTIONS-Rust.md",
        "LICENSE",
        "README.md",
        "config.schema.json",
        "relay-plugin.toml",
        f"lib/{expected_library}",
    }
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError(f"unexpected package contents; expected {sorted(expected_files)}, got {sorted(actual_files)}")

    manifest = tomllib.loads((root / "relay-plugin.toml").read_text())
    json.loads((root / "config.schema.json").read_text())
    library_path = f"lib/{expected_library}"
    if manifest["plugin"] != {"id": "pii_rampart", "kind": "rust_dynamic"}:
        raise ValueError("unexpected plugin identity in packaged manifest")
    if manifest["source"]["artifact"] != library_path:
        raise ValueError("source.artifact does not reference the packaged library")
    if manifest["load"]["library"] != library_path:
        raise ValueError("load.library does not reference the packaged library")
    if manifest["load"]["symbol"] != PLUGIN_SYMBOL:
        raise ValueError("unexpected native registration symbol")
    actual_digest = f"sha256:{sha256(root / library_path)}"
    if manifest["integrity"]["sha256"] != actual_digest:
        raise ValueError("packaged library digest does not match the manifest")

    if load_library:
        loaded = ctypes.CDLL(str(root / library_path))
        getattr(loaded, PLUGIN_SYMBOL)


def verify_archive(args: argparse.Namespace) -> None:
    """Extract and verify one platform archive."""
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"plugin archive does not exist: {archive}")
    if args.extract_dir is None:
        with tempfile.TemporaryDirectory(prefix="nemo-relay-pii-rampart-") as temporary:
            destination = Path(temporary)
            extract_archive(archive, destination)
            verify_package(destination / ARCHIVE_ROOT, args.target, args.load_library)
    else:
        destination = args.extract_dir.resolve()
        if destination.exists():
            raise FileExistsError(f"extraction directory already exists: {destination}")
        extract_archive(archive, destination)
        verify_package(destination / ARCHIVE_ROOT, args.target, args.load_library)


def parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    result = argparse.ArgumentParser(description=__doc__)
    subcommands = result.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build", help="build a plugin distribution archive")
    build.add_argument("--repository", type=Path, default=Path.cwd())
    build.add_argument("--library", type=Path, required=True)
    build.add_argument("--target", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.set_defaults(handler=build_archive)

    verify = subcommands.add_parser("verify", help="verify a plugin distribution archive")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument("--extract-dir", type=Path)
    verify.add_argument("--load-library", action="store_true")
    verify.set_defaults(handler=verify_archive)
    return result


def main() -> None:
    """Run the requested package operation."""
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
