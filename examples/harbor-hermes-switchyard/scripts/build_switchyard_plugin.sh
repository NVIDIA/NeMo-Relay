#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

switchyard_repository="${SWITCHYARD_REPOSITORY:-https://github.com/bbednarski9/Switchyard.git}"
switchyard_commit="${SWITCHYARD_COMMIT:-8daac03edf8544144833af1fd009b3da737715bc}"
target_architecture="${SWITCHYARD_TARGET_ARCHITECTURE:-x86_64}"
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
if [[ ! "$switchyard_commit" =~ ^[0-9a-f]{40}$ ]]; then
  echo "SWITCHYARD_COMMIT must be a full commit SHA" >&2
  exit 2
fi

for dependency in docker git python3; do
  command -v "$dependency" >/dev/null || {
    echo "missing required command: $dependency" >&2
    exit 1
  }
done
docker info >/dev/null

docker_architecture="$(docker info --format '{{.Architecture}}')"
if [[ "$target_architecture" != "x86_64" && "$target_architecture" != "aarch64" ]]; then
  echo "SWITCHYARD_TARGET_ARCHITECTURE must be x86_64 or aarch64" >&2
  exit 2
fi
if [[ "$docker_architecture" == "aarch64" || "$docker_architecture" == "arm64" ]]; then
  builder_image="${SWITCHYARD_BUILDER_IMAGE:-rust:1.96.1-bullseye@sha256:69e444ec65a82386d041a4a3d15e47a797967b90ae24aa342bd8a3600dd9e244}"
  builder_platform="linux/arm64"
  if [[ "$target_architecture" == "x86_64" ]]; then
    cargo_target="x86_64-unknown-linux-gnu"
    library_path="/tmp/target/x86_64-unknown-linux-gnu/release/libswitchyard_nemo_relay_plugin.so"
  else
    cargo_target=""
    library_path="/tmp/target/release/libswitchyard_nemo_relay_plugin.so"
  fi
else
  if [[ "$target_architecture" != "x86_64" ]]; then
    echo "aarch64 cross-builds from an x86_64 Docker host are not supported" >&2
    exit 2
  fi
  builder_image="${SWITCHYARD_BUILDER_IMAGE:-rust:1.96.1-bullseye@sha256:65136b30fc6b10112cbae63a868da085a878679a80d562272e485ecaaad3276a}"
  builder_platform="linux/amd64"
  cargo_target=""
  library_path="/tmp/target/release/libswitchyard_nemo_relay_plugin.so"
fi

# Stage beside the requested output so source and result use the same
# Docker-shared filesystem. Colima installations often do not share $TMPDIR or
# host /private/tmp even though those paths also exist inside the VM.
output_parent="$(dirname "$output_dir")"
if [[ ! -d "$output_parent" ]]; then
  echo "output parent must already exist: $output_parent" >&2
  exit 2
fi
build_root="$(mktemp -d "$output_parent/.switchyard-relay-plugin.XXXXXX")"
source_dir="$build_root/source"
staging_dir="${output_dir}.partial.$$"
if [[ -e "$staging_dir" ]]; then
  echo "refusing to overwrite existing staging directory: $staging_dir" >&2
  exit 2
fi
mkdir -m 0700 "$staging_dir"
cleanup() {
  rm -rf "$build_root" "$staging_dir"
}
trap cleanup EXIT

git clone --filter=blob:none --no-checkout "$switchyard_repository" "$source_dir"
git -C "$source_dir" fetch --depth 1 origin "$switchyard_commit"
git -C "$source_dir" checkout --detach "$switchyard_commit"
actual_commit="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$switchyard_commit" ]]; then
  echo "Switchyard checkout mismatch: expected $switchyard_commit, got $actual_commit" >&2
  exit 1
fi

docker run --rm \
  --platform "$builder_platform" \
  --env PHASE1_CARGO_TARGET="$cargo_target" \
  --env PHASE1_LIBRARY_PATH="$library_path" \
  --volume "$source_dir:/src:ro" \
  --volume "$staging_dir:/out" \
  "$builder_image" \
  bash -lc '
    set -euo pipefail
    test -f /src/Cargo.toml
    export DEBIAN_FRONTEND=noninteractive
    export PATH="/usr/local/cargo/bin:$PATH"
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates clang cmake pkg-config python3
    if [[ -n "$PHASE1_CARGO_TARGET" ]]; then
      apt-get install -y --no-install-recommends crossbuild-essential-amd64
      rustup target add "$PHASE1_CARGO_TARGET"
      export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc
      export CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc
      export CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++
      export AR_x86_64_unknown_linux_gnu=x86_64-linux-gnu-ar
    fi
    mkdir -p /tmp/switchyard
    cp -a /src/. /tmp/switchyard/
    cd /tmp/switchyard
    export CARGO_TARGET_DIR=/tmp/target
    cargo_args=(build --locked --release -p switchyard-nemo-relay-plugin)
    if [[ -n "$PHASE1_CARGO_TARGET" ]]; then
      cargo_args+=(--target "$PHASE1_CARGO_TARGET")
    fi
    cargo "${cargo_args[@]}"
    python3 crates/switchyard-nemo-relay-plugin/scripts/package_bundle.py \
      --library "$PHASE1_LIBRARY_PATH" \
      --output /out
  '

python3 - "$staging_dir" "$switchyard_repository" "$switchyard_commit" "$builder_image" "$builder_platform" "$cargo_target" "$target_architecture" <<'PY'
import hashlib
import json
import pathlib
import sys
import tomllib

output = pathlib.Path(sys.argv[1])
repository, commit, builder, builder_platform, cargo_target, target_architecture = sys.argv[2:]
manifest_path = output / "relay-plugin.toml"
if not manifest_path.is_file():
    raise SystemExit("bundle did not contain relay-plugin.toml")
with manifest_path.open("rb") as stream:
    manifest = tomllib.load(stream)
if manifest.get("plugin", {}).get("id") != "nvidia.switchyard":
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

provenance = {
    "schema_version": "harbor-hermes-switchyard.bundle.v1",
    "repository": repository,
    "commit": commit,
    "builder_image": builder,
    "builder_platform": builder_platform,
    "cargo_target": cargo_target or f"native-{target_architecture}",
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
