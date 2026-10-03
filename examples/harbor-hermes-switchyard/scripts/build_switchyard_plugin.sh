#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

switchyard_plugin_repository="${SWITCHYARD_PLUGIN_REPOSITORY:-NVIDIA/NeMo-Relay-Plugins}"
switchyard_plugin_version="${SWITCHYARD_PLUGIN_VERSION:-0.3.0}"
output_dir="${1:-}"

if [[ -z "$output_dir" ]]; then
  echo "usage: $0 /absolute/output-directory" >&2
  exit 2
fi
if [[ "$output_dir" != /* ]]; then
  echo "output directory must be absolute" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite existing output directory: $output_dir" >&2
  exit 2
fi

for dependency in gh sha256sum tar python3; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done

host_architecture="$(uname -m)"
case "$host_architecture" in
  x86_64|amd64) host_architecture="x86_64" ;;
  aarch64|arm64) host_architecture="aarch64" ;;
  *)
    echo "unsupported host architecture: $host_architecture" >&2
    exit 2
    ;;
esac
target_architecture="${SWITCHYARD_TARGET_ARCHITECTURE:-$host_architecture}"
case "$target_architecture" in
  x86_64) platform="linux-x86_64" ;;
  aarch64) platform="linux-arm64" ;;
  *)
    echo "SWITCHYARD_TARGET_ARCHITECTURE must be x86_64 or aarch64" >&2
    exit 2
    ;;
esac

release_tag="switchyard-plugin-${switchyard_plugin_version}"
asset_stem="switchyard-plugin-${switchyard_plugin_version}-${platform}"

# Stage beside the requested output so the download and result use the same
# filesystem for the final atomic rename.
output_parent="$(dirname "$output_dir")"
if [[ ! -d "$output_parent" ]]; then
  echo "output parent must already exist: $output_parent" >&2
  exit 2
fi
download_dir="$(mktemp -d "$output_parent/.switchyard-plugin-download.XXXXXX")"
staging_dir="${output_dir}.partial.$$"
if [[ -e "$staging_dir" ]]; then
  echo "refusing to overwrite existing staging directory: $staging_dir" >&2
  exit 2
fi
cleanup() {
  rm -rf "$download_dir" "$staging_dir"
}
trap cleanup EXIT

gh release download "$release_tag" \
  --repo "$switchyard_plugin_repository" \
  --pattern "${asset_stem}.tar.gz*" \
  --dir "$download_dir"

archive="$download_dir/${asset_stem}.tar.gz"
checksum_file="${archive}.sha256"
manifest_file="${archive}.json"
for required in "$archive" "$checksum_file" "$manifest_file"; do
  test -f "$required" || { echo "missing expected release asset: $required" >&2; exit 1; }
done
(cd "$download_dir" && sha256sum -c "$(basename "$checksum_file")")

mkdir -m 0700 "$staging_dir"
# The archive contains one top-level directory (e.g. switchyard-plugin/);
# flatten it so the bundle layout matches what this example's scripts expect.
tar -xzf "$archive" -C "$staging_dir" --strip-components=1

python3 - "$staging_dir" "$manifest_file" "$switchyard_plugin_repository" "$release_tag" "$target_architecture" <<'PY'
import hashlib
import json
import pathlib
import re
import sys


def plugin_id(path: pathlib.Path) -> str | None:
    """Read the one manifest field this validator needs without a TOML dependency."""
    section = ""
    section_pattern = re.compile(r"^\[([A-Za-z0-9_.-]+)\]$")
    id_pattern = re.compile(r'^id\s*=\s*"([^"\\]+)"\s*(?:#.*)?$')
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if match := section_pattern.fullmatch(line):
            section = match.group(1)
            continue
        if section == "plugin" and (match := id_pattern.fullmatch(line)):
            return match.group(1)
    raise ValueError("bundle manifest does not define plugin.id")


output = pathlib.Path(sys.argv[1])
manifest_json_path = pathlib.Path(sys.argv[2])
plugin_repository, release_tag, target_architecture = sys.argv[3:]
manifest_path = output / "relay-plugin.toml"
if not manifest_path.is_file():
    raise SystemExit("bundle did not contain relay-plugin.toml")
if plugin_id(manifest_path) != "nvidia.switchyard":
    raise SystemExit("bundle manifest has the wrong plugin id")
libraries = [
    path for path in output.iterdir()
    if path.is_file() and path.suffix in {".so", ".dylib", ".dll"}
]
if len(libraries) != 1:
    raise SystemExit(f"expected one native library, found {len(libraries)}")
with libraries[0].open("rb") as stream:
    elf_header = stream.read(20)
if elf_header[:4] != b"\x7fELF" or elf_header[5] != 1:
    raise SystemExit("native library must be a little-endian ELF artifact")
machine = int.from_bytes(elf_header[18:20], "little")
expected_machine = {"x86_64": 62, "aarch64": 183}[target_architecture]
if machine != expected_machine:
    raise SystemExit(
        f"native library architecture mismatch: expected {target_architecture}, ELF e_machine={machine}"
    )


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


release_manifest = json.loads(manifest_json_path.read_text(encoding="utf-8"))
provenance = {
    "schema_version": "harbor-hermes-switchyard.bundle.v1",
    "repository": f"https://github.com/{plugin_repository}.git",
    "release_tag": release_tag,
    "switchyard_source_repository": "https://github.com/NVIDIA-NeMo/Switchyard.git",
    "switchyard_source_commit": release_manifest["source_commit"],
    "relay_tag": release_manifest["relay"]["tag"],
    "target_architecture": target_architecture,
    "plugin_id": "nvidia.switchyard",
    "manifest_sha256": digest(manifest_path),
    "library": libraries[0].name,
    "library_sha256": digest(libraries[0]),
}
(output / "bundle-provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(provenance, indent=2))
PY

mv "$staging_dir" "$output_dir"
