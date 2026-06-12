# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

export REPO_ROOT := justfile_directory()
export NEMO_RELAY_REPO_ROOT := REPO_ROOT

# Shared knobs used by the CI-oriented build, test, and package targets below.
ci := "false"
output_dir := ""
# When set, package targets use this exact version instead of synthesizing one.
ref_name := ""
# Linux package artifacts target this minimum glibc version for compatibility.
linux_glibc_version := "2.17"

bash_helpers := '''
set -euo pipefail

export_uv_python_runtime() {
    local python_runtime_exports=""
    local python_executable=""
    python_executable="$(uv_python_executable)"
    python_runtime_exports="$(
        cd "$NEMO_RELAY_REPO_ROOT"
        "$python_executable" - <<'PY'
import shlex
from pathlib import Path
import os
import sys
import sysconfig

def emit(name, value):
    print(f"export {name}={shlex.quote('' if value is None else str(value))}")

emit("PYO3_PYTHON", sys.executable)
emit("PYTHON_EXE_DIR", Path(sys.executable).parent)
emit("PYTHONHOME", sys.base_prefix)
emit("PYTHON_PATHSEP", os.pathsep)
emit("PYTHON_STDLIB", sysconfig.get_path("stdlib"))
emit("PYTHON_PLATSTDLIB", sysconfig.get_path("platstdlib"))
emit("PYTHON_LIBDIR", sysconfig.get_config_var("LIBDIR"))
PY
    )"

    eval "$python_runtime_exports"
    if command -v cygpath >/dev/null 2>&1; then
        export PATH="$(cygpath -u "$PYTHON_EXE_DIR"):$PATH"
    else
        export PATH="$PYTHON_EXE_DIR:$PATH"
    fi
    export PYTHONPATH="${PYTHON_STDLIB}${PYTHON_PATHSEP}${PYTHON_PLATSTDLIB}"
    export LD_LIBRARY_PATH="${PYTHON_LIBDIR:+${PYTHON_LIBDIR}}${PYTHON_LIBDIR:+${LD_LIBRARY_PATH:+:}}${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="${PYTHON_LIBDIR:+${PYTHON_LIBDIR}}${PYTHON_LIBDIR:+${DYLD_LIBRARY_PATH:+:}}${DYLD_LIBRARY_PATH:-}"
}

uv_python_executable() {
    (
        cd "$NEMO_RELAY_REPO_ROOT"
        uv python find
    )
}

activate_project_venv() {
    # Ensure PATH-based tool lookups (for example, `zig`) resolve from the
    # synced project environment without asking uv to resync the project.
    if [[ -f "$NEMO_RELAY_REPO_ROOT/.venv/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$NEMO_RELAY_REPO_ROOT/.venv/bin/activate"
    elif [[ -f "$NEMO_RELAY_REPO_ROOT/.venv/Scripts/activate" ]]; then
        # shellcheck disable=SC1091
        source "$NEMO_RELAY_REPO_ROOT/.venv/Scripts/activate"
    else
        echo "ERROR: expected project virtualenv activation script under .venv" >&2
        exit 1
    fi
}

project_python_executable() {
    local python_executable=""
    if [[ -x "$NEMO_RELAY_REPO_ROOT/.venv/bin/python" ]]; then
        python_executable="$NEMO_RELAY_REPO_ROOT/.venv/bin/python"
    elif [[ -x "$NEMO_RELAY_REPO_ROOT/.venv/Scripts/python.exe" ]]; then
        python_executable="$NEMO_RELAY_REPO_ROOT/.venv/Scripts/python.exe"
    else
        echo "ERROR: expected project virtualenv Python executable under .venv" >&2
        exit 1
    fi

    if command -v cygpath >/dev/null 2>&1; then
        cygpath -u "$python_executable"
    else
        printf '%s\n' "$python_executable"
    fi
}

prepend_ziglang_to_path() {
    local python_executable="$1"
    local zig_dir=""

    zig_dir="$("$python_executable" - <<'PY'
from pathlib import Path
import importlib.util
import sys

spec = importlib.util.find_spec("ziglang")
if spec is None or spec.origin is None:
    raise SystemExit("ERROR: expected ziglang from the locked uv environment")

zig = Path(spec.origin).resolve().parent / ("zig.exe" if sys.platform == "win32" else "zig")
if not zig.exists():
    raise SystemExit(f"ERROR: expected zig binary at {zig}")

print(zig.parent)
PY
    )"

    export PATH="$zig_dir:$PATH"
}

use_project_python_source() {
    local python_executable="$1"
    local python_pathsep=""
    local python_source_path="$NEMO_RELAY_REPO_ROOT/python"

    python_pathsep="$("$python_executable" - <<'PY'
import os
print(os.pathsep)
PY
    )"
    if command -v cygpath >/dev/null 2>&1; then
        python_source_path="$(cygpath -w "$python_source_path")"
    fi
    export PYTHONPATH="${python_source_path}${PYTHONPATH:+${python_pathsep}${PYTHONPATH}}"
}

docs_dependencies_ready() {
    case "${NEMO_RELAY_DOCS_DEPS_READY:-}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

ensure_docs_dependencies() {
    if docs_dependencies_ready; then
        return 0
    fi

    cd "$NEMO_RELAY_REPO_ROOT"
    uv sync --inexact --no-default-groups --group dev --no-install-project
    npm install --ignore-scripts
}

ensure_node_docs_declarations() {
    cd "$NEMO_RELAY_REPO_ROOT"
    local types_file="crates/node/index.d.ts"
    local newest_source=""

    if [[ -f "$types_file" ]]; then
        newest_source="$(
            find crates/node/src crates/node/Cargo.toml crates/node/build.rs \
                -type f -newer "$types_file" -print -quit
        )"
    fi

    if [[ ! -f "$types_file" || -n "$newest_source" ]]; then
        npm run build-debug --workspace=nemo-relay-node
    fi
}

generate_docs_api_references() {
    cd "$NEMO_RELAY_REPO_ROOT"
    uv run --no-sync python scripts/docs/generate_python_library_reference.py
    ensure_node_docs_declarations
    uv run --no-sync python scripts/docs/generate_node_library_reference.py
    uv run --no-sync python scripts/docs/generate_rust_library_reference.py
}

prepare_parent_dir() {
    mkdir -p "$(dirname "$1")"
}

artifact_path() {
    local filename="$1"
    local base_dir="${output_dir:-$NEMO_RELAY_REPO_ROOT/target/coverage}"
    printf '%s/%s\n' "$base_dir" "$filename"
}

prepare_artifact() {
    local path
    path="$(artifact_path "$1")"
    prepare_parent_dir "$path"
    printf '%s\n' "$path"
}

# Package artifacts are grouped by ecosystem so local runs mirror CI layout.
package_output_dir() {
    local channel="$1"
    local base_dir="${output_dir:-$NEMO_RELAY_REPO_ROOT/target/packages}"
    printf '%s/%s\n' "$base_dir" "$channel"
}

prepare_package_dir() {
    local path
    path="$(package_output_dir "$1")"
    mkdir -p "$path"
    printf '%s\n' "$path"
}

head_git_sha() {
    git -C "$NEMO_RELAY_REPO_ROOT" rev-parse --short=8 HEAD
}

# Version helpers intentionally mutate package metadata in-place to match how CI
# sets release and non-release artifact versions before building them.
read_npm_package_version() {
    local pkg_path="$1"

    node - "$pkg_path" <<'NODE'
const fs = require('fs');
const [pkgPath] = process.argv.slice(2);
const manifest = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

if (!manifest.version) {
  throw new Error(`${pkgPath} missing version field`);
}

console.log(manifest.version);
NODE
}

set_npm_package_version() {
    local pkg_path="$1"
    local lock_path="${2:-}"
    local version="$3"
    local lock_package_path="${4:-}"

    node - "$pkg_path" "$lock_path" "$version" "$lock_package_path" <<'NODE'
const fs = require('fs');
const [pkgPath, lockPath, version, lockPackagePath] = process.argv.slice(2);

function readJson(path) {
  return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function writeJson(path, value) {
  fs.writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

function requireVersion(value, label) {
  if (
    !Object.prototype.hasOwnProperty.call(value, 'version') ||
    typeof value.version !== 'string' ||
    value.version.length === 0
  ) {
    throw new Error(`${label} missing version field`);
  }
}

try {
  const manifest = readJson(pkgPath);
  requireVersion(manifest, pkgPath);
  const manifestChanged = manifest.version !== version;
  let lock = null;
  let lockChanged = false;
  let lockPackageChanged = false;

  if (lockPath) {
    lock = readJson(lockPath);
    if (!lock.packages) {
      throw new Error(`${lockPath} missing packages`);
    }

    if (typeof lock.version === 'string' && lock.version.length > 0) {
      lockChanged = lock.version !== version;
    }

    const packageEntryKey = lockPackagePath || '';
    const packageEntry = lock.packages[packageEntryKey];
    if (!packageEntry) {
      throw new Error(`${lockPath} missing packages["${packageEntryKey}"]`);
    }
    requireVersion(packageEntry, `${lockPath} packages["${packageEntryKey}"]`);
    lockPackageChanged = packageEntry.version !== version;
  }

  manifest.version = version;
  if (lock) {
    if (typeof lock.version === 'string' && lock.version.length > 0) {
      lock.version = version;
    }
    const packageEntryKey = lockPackagePath || '';
    lock.packages[packageEntryKey].version = version;
  }

  if (manifestChanged) {
    writeJson(pkgPath, manifest);
    console.log(`${pkgPath} version updated to ${version}`);
  } else {
    console.log(`${pkgPath} already set to ${version}`);
  }

  if (lockPath) {
    if (lockChanged || lockPackageChanged) {
      writeJson(lockPath, lock);
      console.log(`${lockPath} version updated to ${version}`);
    } else {
      console.log(`${lockPath} already set to ${version}`);
    }
  }
} catch (error) {
  console.error(`Error updating package version: ${error.message}`);
  process.exit(1);
}
NODE
}

set_npm_package_dependency_version() {
    local pkg_path="$1"
    local lock_path="${2:-}"
    local lock_package_path="$3"
    local dependency_name="$4"
    local version="$5"

    node - "$pkg_path" "$lock_path" "$lock_package_path" "$dependency_name" "$version" <<'NODE'
const fs = require('fs');
const [pkgPath, lockPath, lockPackagePath, dependencyName, version] = process.argv.slice(2);

function readJson(path) {
  return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function writeJson(path, value) {
  fs.writeFileSync(path, JSON.stringify(value, null, 2) + '\n');
}

function requireDependency(container, label) {
  if (
    !container.dependencies ||
    !Object.prototype.hasOwnProperty.call(container.dependencies, dependencyName)
  ) {
    throw new Error(`${label} missing dependencies["${dependencyName}"]`);
  }
}

try {
  const manifest = readJson(pkgPath);
  requireDependency(manifest, pkgPath);
  const manifestChanged = manifest.dependencies[dependencyName] !== version;

  let lock = null;
  let lockChanged = false;
  if (lockPath) {
    lock = readJson(lockPath);
    if (!lock.packages) {
      throw new Error(`${lockPath} missing packages`);
    }
    const packageEntry = lock.packages[lockPackagePath];
    if (!packageEntry) {
      throw new Error(`${lockPath} missing packages["${lockPackagePath}"]`);
    }
    requireDependency(packageEntry, `${lockPath} packages["${lockPackagePath}"]`);
    lockChanged = packageEntry.dependencies[dependencyName] !== version;
  }

  manifest.dependencies[dependencyName] = version;
  if (lock) {
    lock.packages[lockPackagePath].dependencies[dependencyName] = version;
  }

  if (manifestChanged) {
    writeJson(pkgPath, manifest);
  }

  if (lockPath) {
    if (lockChanged) {
      writeJson(lockPath, lock);
    }
  }
} catch (error) {
  console.error(`Error updating package dependency: ${error.message}`);
  process.exit(1);
}
NODE
}

set_coding_agent_plugin_versions() {
    local version="$1"

    node - "$version" <<'NODE'
const fs = require('fs');
const version = process.argv[2];
const manifests = [
  'integrations/coding-agents/claude-code/.claude-plugin/plugin.json',
  'integrations/coding-agents/codex/.codex-plugin/plugin.json',
];

for (const manifestPath of manifests) {
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  if (!manifest.version) {
    throw new Error(`${manifestPath} missing version field`);
  }
  if (manifest.version === version) {
    console.log(`${manifestPath} already set to ${version}`);
    continue;
  }
  manifest.version = version;
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
  console.log(`${manifestPath} version updated to ${version}`);
}
NODE
}

read_workspace_version() {
    local python_executable=""
    python_executable="$(uv_python_executable)"
    "$python_executable" - <<'PY'
from pathlib import Path
import re

text = Path("Cargo.toml").read_text()
match = re.search(r'^version = "(.*)"$', text, flags=re.MULTILINE)
if not match:
    raise SystemExit("Failed to read version from Cargo.toml")
print(match.group(1))
PY
}

set_cargo_workspace_version() {
    local version="$1"
    local python_executable=""
    python_executable="$(uv_python_executable)"

    "$python_executable" - "$version" <<'PY'
from pathlib import Path
import re
import sys

version = sys.argv[1]
if version.startswith("v"):
    raise SystemExit("Release tags must not start with 'v'; use raw SemVer such as 0.1.0")
if not re.fullmatch(r"\d+\.\d+\.\d+(?:-(?:alpha|beta|rc)\.\d+)?", version):
    raise SystemExit(f"Unsupported release tag '{version}'; use 0.1.0 or prereleases like 0.1.0-rc.1")

path = Path("Cargo.toml")
text = path.read_text()
section = ""
output = []
changed = []
found_workspace_version = False
local_dependencies = (
    "nemo-relay",
    "nemo-relay-adaptive",
    "nemo-relay-pii-redaction",
    "nemo-relay-ffi",
    "nemo-relay-cli",
)
found_dependencies = set()

for line in text.splitlines(keepends=True):
    section_match = re.match(r"^\s*\[([^\]]+)\]\s*(?:#.*)?$", line)
    if section_match:
        section = section_match.group(1)

    updated = line
    if section == "workspace.package":
        updated, count = re.subn(
            r'^(version\s*=\s*")([^"]+)(".*)$',
            rf"\g<1>{version}\g<3>",
            line,
        )
        if count == 1:
            found_workspace_version = True
            if updated != line:
                changed.append("workspace.package.version")
    elif section == "workspace.dependencies":
        for dependency in local_dependencies:
            updated, count = re.subn(
                rf"^({re.escape(dependency)}\s*=\s*\{{[^}}]*\bversion\s*=\s*\")([^\"]+)(\".*)$",
                rf"\g<1>{version}\g<3>",
                updated,
            )
            if count == 1:
                found_dependencies.add(dependency)
                if updated != line:
                    changed.append(f"workspace.dependencies.{dependency}.version")
                break

    output.append(updated)

missing = []
if not found_workspace_version:
    missing.append("workspace.package.version")
for dependency in local_dependencies:
    if dependency not in found_dependencies:
        missing.append(f"workspace.dependencies.{dependency}.version")
if missing:
    raise SystemExit(f"Failed to find expected Cargo version fields: {', '.join(missing)}")

path.write_text("".join(output))
if changed:
    print(f"Cargo.toml version set to {version}: {', '.join(changed)}")
else:
    print(f"Cargo.toml already set to {version}")
PY

    local metadata_file=""
    metadata_file="$(mktemp)"
    if ! cargo metadata --no-deps --format-version 1 > "$metadata_file"; then
        rm -f "$metadata_file"
        return 1
    fi
    if ! "$python_executable" - "$version" "$metadata_file" <<'PY'
import json
import sys
from pathlib import Path

version = sys.argv[1]
metadata = json.loads(Path(sys.argv[2]).read_text())
workspace_members = set(metadata["workspace_members"])
mismatched = []
checked = 0

for package in metadata["packages"]:
    if package["id"] not in workspace_members or not package["name"].startswith("nemo-relay"):
        continue
    checked += 1
    if package["version"] != version:
        mismatched.append(f"{package['name']}={package['version']}")

if checked == 0:
    raise SystemExit("Cargo metadata did not include any nemo-relay workspace packages")
if mismatched:
    raise SystemExit(f"Cargo workspace packages do not all resolve to {version}: {', '.join(mismatched)}")
print(f"Cargo metadata resolves {checked} nemo-relay workspace packages to {version}")
PY
    then
        rm -f "$metadata_file"
        return 1
    fi
    rm -f "$metadata_file"
}

set_node_package_versions() {
    local version="$1"
    set_npm_package_version crates/node/package.json package-lock.json "$version" crates/node
    set_npm_package_version integrations/openclaw/package.json package-lock.json "$version" integrations/openclaw
    set_npm_package_dependency_version integrations/openclaw/package.json package-lock.json integrations/openclaw nemo-relay-node "$version"
}

set_node_package_version() {
    set_node_package_versions "$1"
}

set_project_version() {
    local version="$1"
    set_cargo_workspace_version "$version"
    set_node_package_versions "$version"
    set_coding_agent_plugin_versions "$version"
}

set_python_package_version() {
    local version="$1"
    local python_executable=""
    python_executable="$(uv_python_executable)"

    "$python_executable" - "$version" <<'PY'
from pathlib import Path
import re
import sys

def semver_to_pep440(version: str) -> str:
    pattern = re.compile(
        r"^(?P<release>\d+\.\d+\.\d+)"
        r"(?:-(?P<pre_label>alpha|beta|rc)(?:\.(?P<pre_num>\d+))?)?"
        r"(?:\+(?P<local>[0-9A-Za-z.-]+))?$"
    )
    match = pattern.fullmatch(version)
    if not match:
        raise SystemExit(
            "Unsupported Python package version format. Expected SemVer with optional "
            "alpha/beta/rc prerelease and optional build metadata."
        )

    pep440 = match.group("release")
    pre_label = match.group("pre_label")
    if pre_label:
        pre_map = {"alpha": "a", "beta": "b", "rc": "rc"}
        pre_num = match.group("pre_num") or "0"
        pep440 += f"{pre_map[pre_label]}{pre_num}"

    local = match.group("local")
    if local:
        normalized_local = ".".join(part.lower() for part in re.split(r"[._-]+", local) if part)
        if not normalized_local:
            raise SystemExit("Python package local version metadata cannot be empty")
        pep440 += f"+{normalized_local}"

    return pep440


version = semver_to_pep440(sys.argv[1])
cargo_version = sys.argv[1]
path = Path("pyproject.toml")
text = path.read_text()

if 'dynamic = ["version"]' in text:
    updated = text.replace('dynamic = ["version"]', f'version = "{version}"', 1)
elif re.search(r'^version = "(.*)"$', text, flags=re.MULTILINE):
    updated, count = re.subn(
        r'^version = "(.*)"$',
        f'version = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Failed to update version in pyproject.toml")
else:
    raise SystemExit("Failed to find dynamic or static version field in pyproject.toml")

path.write_text(updated)
print(f"pyproject.toml version updated to {version}")

path = Path("crates/python/Cargo.toml")
text = path.read_text()

if "version.workspace = true" in text:
    updated = text.replace("version.workspace = true", f'version = "{cargo_version}"', 1)
elif re.search(r'^version = "(.*)"$', text, flags=re.MULTILINE):
    updated, count = re.subn(
        r'^version = "(.*)"$',
        f'version = "{cargo_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("Failed to update version in crates/python/Cargo.toml")
else:
    raise SystemExit("Failed to find workspace or static version field in crates/python/Cargo.toml")

path.write_text(updated)
print(f"crates/python/Cargo.toml version updated to {cargo_version}")
PY
}

# Keep local wheel packaging aligned with the CI matrix without requiring raw
# maturin flags to be passed through `just --set`.
linux_manylinux_compatibility() {
    local glibc_version="${linux_glibc_version:-2.17}"
    printf 'manylinux_%s\n' "${glibc_version//./_}"
}

python_wheel_build_args() {
    local os_name
    os_name="$(uname -s)"
    case "$os_name" in
        Linux)
            printf '%s\0' --compatibility "$(linux_manylinux_compatibility)" --zig
            ;;
        Darwin)
            printf '%s\0' --compatibility pypi
            ;;
        CYGWIN*|MINGW*|MSYS*)
            printf '%s\0' --compatibility pypi
            ;;
        *)
            echo "ERROR: unsupported OS for package-python: $os_name" >&2
            exit 1
            ;;
    esac
}

prepend_go_bin_to_path() {
    local go_bin
    go_bin="$(go env GOBIN)"
    if [[ -z "$go_bin" ]]; then
        go_bin="$(go env GOPATH)/bin"
    fi
    export PATH="$go_bin:$PATH"
}

prepare_llvm_cov_workspace() {
    eval "$(cargo llvm-cov show-env --sh)"
    cargo llvm-cov clean --workspace
}

rust_source_coverage_supported() {
    local host
    host="$(rustc -vV | sed -n 's/^host: //p')"
    case "$host" in
        aarch64-pc-windows-msvc)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

is_true() {
    case "$1" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            return 0
            ;;
        0|false|FALSE|False|no|NO|No|off|OFF|Off|"")
            return 1
            ;;
        *)
            echo "ERROR: expected a boolean-like value, got: $1" >&2
            exit 1
            ;;
    esac
}
'''

# validate the documentation site, links, navigation, and generated references
docs:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    ensure_docs_dependencies
    generate_docs_api_references
    cd "$NEMO_RELAY_REPO_ROOT/fern"
    npx fern check --warnings
    npx fern docs broken-links --strict

# validate documentation links and navigation without a full site build
docs-linkcheck:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    ensure_docs_dependencies
    generate_docs_api_references
    cd "$NEMO_RELAY_REPO_ROOT/fern"
    npx fern check --warnings
    npx fern docs broken-links --strict

# regenerate the ignored Fern API reference pages
docs-api-reference:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    ensure_docs_dependencies
    generate_docs_api_references

# --set [ci=true|false]
build-rust:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    export_uv_python_runtime
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        prepare_llvm_cov_workspace
        cargo test --workspace --no-run
    else
        cargo build --workspace
    fi

# --set [ci=true|false]
build-python:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    cd "$NEMO_RELAY_REPO_ROOT"
    uv sync --inexact --no-install-project --no-install-package nemo-relay --extra langchain --extra langgraph --extra deepagents
    activate_project_venv
    if is_true "{{ ci }}"; then
        prepare_llvm_cov_workspace
    fi
    python_executable="$(project_python_executable)"
    "$python_executable" -m maturin develop


# --set [ci=true|false]
build-go:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        cargo build -p nemo-relay-ffi
    else
        cargo build --release -p nemo-relay-ffi
    fi


# --set [ci=true|false]
build-node:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    if is_true "{{ ci }}"; then
        prepare_llvm_cov_workspace
    fi
    cd "$NEMO_RELAY_REPO_ROOT"
    npm install --workspace=nemo-relay-node --ignore-scripts
    if is_true "{{ ci }}"; then
        npm run build-debug --workspace=nemo-relay-node
    else
        npm run build --workspace=nemo-relay-node
    fi

# --set [ci=true|false]
build-wasm:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        npm run build:pkg --workspace=nemo-relay-wasm
    else
        NEMO_RELAY_WASM_RELEASE=1 npm run build:pkg --workspace=nemo-relay-wasm
    fi

build-all: build-rust build-python build-go build-node build-wasm

# remove local build and test artifacts
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob globstar
    rm -rf \
        .coverage \
        .pytest_cache \
        *.profraw \
        crates/**/*.profraw \
        crates/node/*.node \
        crates/node/coverage \
        crates/node/index.d.ts \
        crates/node/index.js \
        crates/node/junit.xml \
        crates/node/node_modules \
        crates/wasm/node_modules \
        crates/wasm/pkg-test/ \
        crates/wasm/pkg/ \
        integrations/openclaw/*.profraw \
        integrations/openclaw/.test-dist \
        integrations/openclaw/dist \
        integrations/openclaw/node_modules \
        node_modules \
        docs/_build/ \
        docs/reference/api/**/_generated/ \
        docs/reference/api/**/_source/ \
        go/nemo_relay/coverage.out \
        python/nemo_relay/*.so \
        python/nemo_relay/__pycache__ \
        python/nemo_relay/_native*.pyd \
        python/tests/__pycache__ \
        target

# --set [output_dir=<path>] [ci=true|false]
test-rust:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    output_dir="{{ output_dir }}"
    junit_out=""
    test_config_root="$(mktemp -d)"
    cleanup_test_config_root() {
        rm -rf "$test_config_root"
    }
    trap cleanup_test_config_root EXIT

    host_os="$(uname -s 2>/dev/null || true)"
    is_windows=false
    case "${RUNNER_OS:-}:${OSTYPE:-}:$host_os" in
        Windows:*|*:msys*:*|*:win32*:*|*:*:MINGW*|*:*:MSYS*|*:*:CYGWIN*)
            is_windows=true
            ;;
    esac
    native_test_config_path() {
        if [[ "$is_windows" == true ]] && command -v cygpath >/dev/null 2>&1; then
            cygpath -w "$1"
        else
            printf '%s\n' "$1"
        fi
    }

    xdg_config_home="$test_config_root/xdg"
    mkdir -p "$xdg_config_home"
    export XDG_CONFIG_HOME="$(native_test_config_path "$xdg_config_home")"
    if [[ "$is_windows" == true ]]; then
        appdata_home="$test_config_root/AppData/Roaming"
        localappdata_home="$test_config_root/AppData/Local"
        mkdir -p "$appdata_home" "$localappdata_home"
        export APPDATA="$(native_test_config_path "$appdata_home")"
        export LOCALAPPDATA="$(native_test_config_path "$localappdata_home")"
    fi

    export_uv_python_runtime
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        coverage_out="$(prepare_artifact rust-workspace.xml)"
        junit_out="$(prepare_artifact rust_junit_report.xml)"
        if rust_source_coverage_supported; then
            prepare_llvm_cov_workspace
        fi
        cargo nextest run --workspace --profile ci --no-fail-fast
        cp "$NEMO_RELAY_REPO_ROOT/target/nextest/ci/rust_junit_report.xml" "$junit_out"
        if rust_source_coverage_supported; then
            cargo llvm-cov report \
                --ignore-filename-regex '.*/tests/.*\.rs$' \
                --cobertura \
                --output-path "$coverage_out"
        fi
    else
        cargo test --workspace
    fi

# --set [output_dir=<path>] [ci=true|false]
test-python:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    output_dir="{{ output_dir }}"
    pytest_cmd=(pytest)
    coverage_out=""
    junit_out=""
    rust_coverage_out=""
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        coverage_out="$(prepare_artifact python-coverage.xml)"
        junit_out="$(prepare_artifact python-junit.xml)"
        pytest_cmd+=(--cov=nemo_relay --cov-report term-missing --cov-report "xml:$coverage_out")
        pytest_cmd+=(--junit-xml "$junit_out")
        export_uv_python_runtime
        if rust_source_coverage_supported; then
            rust_coverage_out="$(prepare_artifact python-rust.xml)"
            prepare_llvm_cov_workspace
        fi
        cargo test -p nemo-relay-python --lib
    fi
    uv sync --inexact --no-install-project --no-install-package nemo-relay
    activate_project_venv
    python_executable="$(project_python_executable)"
    use_project_python_source "$python_executable"
    "$python_executable" -m maturin develop --skip-install
    "$python_executable" -m "${pytest_cmd[@]}" --ignore=python/tests/integrations
    if is_true "{{ ci }}" && [[ -n "$rust_coverage_out" ]]; then
        cargo llvm-cov report \
            -p nemo-relay-python \
            --ignore-filename-regex '.*/tests/.*\.rs$' \
            --cobertura \
            --output-path "$rust_coverage_out"
    fi

test-python-langchain:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    pytest_cmd=(pytest)
    cd "$NEMO_RELAY_REPO_ROOT"
    uv sync --inexact --no-install-project --no-install-package nemo-relay --extra langchain --extra langgraph --extra deepagents
    activate_project_venv
    python_executable="$(project_python_executable)"
    use_project_python_source "$python_executable"
    "$python_executable" -m maturin develop --skip-install
    "$python_executable" -m "${pytest_cmd[@]}" \
        python/tests/integrations/deepagents_tests \
        python/tests/integrations/langchain_tests \
        python/tests/integrations/langgraph_tests

# --set [output_dir=<path>] [ci=true|false]
test-go:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    output_dir="{{ output_dir }}"

    target="release"
    flag="--release"
    go_test_cmd=(go test -v)
    go_ldflags=()
    if is_true "{{ ci }}"; then
        target="debug"
        flag=""
    fi
    coverage_out=""
    junit_out=""
    lib_dir="$NEMO_RELAY_REPO_ROOT/target/$target"
    host_os="$(uname -s 2>/dev/null || true)"
    is_windows=false
    case "${RUNNER_OS:-}:${OSTYPE:-}:$host_os" in
        Windows:*|*:msys*:*|*:win32*:*|*:*:MINGW*|*:*:MSYS*|*:*:CYGWIN*)
            is_windows=true
            ;;
    esac
    cd "$NEMO_RELAY_REPO_ROOT"
    cargo build $flag -p nemo-relay-ffi

    if [[ "$is_windows" == true ]]; then
        export CC=clang
        export CXX=clang++
        # Go's Windows linker checks -extldflags before deciding whether to
        # inject a GNU linker script; CGO_LDFLAGS is too late for that check.
        go_ldflags+=(-extldflags=-fuse-ld=lld)
        if command -v cygpath >/dev/null 2>&1; then
            export PATH="$(cygpath -u "$lib_dir"):$PATH"
        else
            export PATH="$lib_dir:$PATH"
        fi
    fi
    if [[ "$OSTYPE" == "darwin"* ]]; then
        export CGO_LDFLAGS="-Wl,-w ${CGO_LDFLAGS:-}"
    fi
    export CGO_LDFLAGS="-L$lib_dir ${CGO_LDFLAGS:-}"
    export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export DYLD_LIBRARY_PATH="$lib_dir${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
    if is_true "{{ ci }}"; then
        coverage_out="$(prepare_artifact go-coverage.xml)"
        junit_out="$(prepare_artifact go-junit.xml)"
        go_test_cmd+=(-coverprofile=coverage.out)
        if [[ "$is_windows" == true ]]; then
            go_ldflags+=(-w)
        fi
        go install github.com/jstemmer/go-junit-report/v2@latest
        go install github.com/boumenot/gocover-cobertura@latest
        prepend_go_bin_to_path
    fi
    if [[ "$is_windows" == true ]]; then
        echo "Go Windows linker: RUNNER_OS=${RUNNER_OS:-} OSTYPE=${OSTYPE:-} uname=$host_os CC=$CC CXX=$CXX ldflags=${go_ldflags[*]:-}"
    fi
    if [[ ${#go_ldflags[@]} -gt 0 ]]; then
        go_test_cmd+=("-ldflags=${go_ldflags[*]}")
    fi
    go_test_cmd+=(./...)
    cd "$NEMO_RELAY_REPO_ROOT/go/nemo_relay"
    if is_true "{{ ci }}"; then
        # Work-around /dev/stderr not being available on Windows
        "${go_test_cmd[@]}" 2>&1 | tee >(cat >&2) | go-junit-report -set-exit-code > "$junit_out"
        gocover-cobertura < coverage.out > "$coverage_out"
    else
        "${go_test_cmd[@]}"
    fi

# --set [output_dir=<path>] [ci=true|false]
test-node:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    output_dir="{{ output_dir }}"
    coverage_out=""
    junit_out=""
    rust_coverage_out=""
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        coverage_out="$(prepare_artifact node-coverage.xml)"
        junit_out="$(prepare_artifact node-junit.xml)"
        if rust_source_coverage_supported; then
            rust_coverage_out="$(prepare_artifact node-rust.xml)"
            prepare_llvm_cov_workspace
        fi
        cargo test -p nemo-relay-node --lib
    fi
    npm install --workspace=nemo-relay-node --ignore-scripts
    if is_true "{{ ci }}"; then
        npm run coverage --workspace=nemo-relay-node
        cp crates/node/coverage/cobertura-coverage.xml "$coverage_out"
        cp crates/node/junit.xml "$junit_out"
        cd "$NEMO_RELAY_REPO_ROOT"
        if [[ -n "$rust_coverage_out" ]]; then
            cargo llvm-cov report \
                -p nemo-relay-node \
                --ignore-filename-regex '.*/tests/.*\.rs$' \
                --cobertura \
                --output-path "$rust_coverage_out"
        fi
    else
        npm test --workspace=nemo-relay-node
    fi

# --set [ci=true|false]
test-openclaw:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    cd "$NEMO_RELAY_REPO_ROOT"
    if is_true "{{ ci }}"; then
        npm ci --ignore-scripts
        npm run build-debug --workspace=nemo-relay-node
    else
        npm install --ignore-scripts
    fi
    npm run typecheck --workspace=nemo-relay-openclaw
    npm test --workspace=nemo-relay-openclaw
    npm run test:live --workspace=nemo-relay-openclaw
    npm run pack:check --workspace=nemo-relay-openclaw

# --set [output_dir=<path>] [ci=true|false]
test-wasm:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    output_dir="{{ output_dir }}"
    coverage_out=""
    junit_out=""
    cd "$NEMO_RELAY_REPO_ROOT"
    wasm-pack test --node crates/wasm
    npm install --workspace=nemo-relay-wasm --ignore-scripts
    if is_true "{{ ci }}"; then
        coverage_out="$(prepare_artifact wasm-js.xml)"
        junit_out="$(prepare_artifact wasm-junit.xml)"
        npm run coverage:pkg --workspace=nemo-relay-wasm
        cp crates/wasm/coverage/cobertura-coverage.xml "$coverage_out"
        cp crates/wasm/junit.xml "$junit_out"
    else
        npm run test:pkg --workspace=nemo-relay-wasm
    fi

# --set [output_dir=<path>] [ci=true|false]
test-all: test-rust test-python test-python-langchain test-go test-node test-openclaw test-wasm

# [version] or --set ref_name=<version>
set-version version="":
    #!/usr/bin/env bash
    {{ bash_helpers }}
    version="{{ version }}"
    if [[ -z "$version" ]]; then
        version="{{ ref_name }}"
    fi
    if [[ -z "$version" ]]; then
        echo "Error: version is required for set-version" >&2
        exit 1
    fi
    cd "$NEMO_RELAY_REPO_ROOT"
    set_project_version "$version"

# --set [output_dir=<path>] [ref_name=<name>]
package-node:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    # If `ref_name` is empty, append the current short HEAD SHA to the version.
    # If `ref_name` is set, write it as the exact package version before packing.
    linux_glibc_version="{{ linux_glibc_version }}"
    output_dir="{{ output_dir }}"
    cd "$NEMO_RELAY_REPO_ROOT"
    package_dir="$(prepare_package_dir npm)"
    if [[ -z "{{ ref_name }}" ]]; then
        sha="$(head_git_sha)"
        version="$(read_npm_package_version crates/node/package.json)"
        package_version="${version}+${sha}"
        echo "Non-release build: appending commit hash to version"
        set_npm_package_version crates/node/package.json package-lock.json "$package_version" crates/node
        set_npm_package_dependency_version integrations/openclaw/package.json package-lock.json integrations/openclaw nemo-relay-node "$package_version"
    else
        package_version="{{ ref_name }}"
        echo "Using explicit version {{ ref_name }}"
        set_npm_package_version crates/node/package.json package-lock.json "$package_version" crates/node
        set_npm_package_dependency_version integrations/openclaw/package.json package-lock.json integrations/openclaw nemo-relay-node "$package_version"
    fi
    build_args=(build)
    if is_true "{{ ci }}" && [[ "$(uname -s)" == "Linux" ]]; then
        # Zig is provided by the uv.lock `ziglang` entry; keep any explicit CI
        # Zig version pin aligned with that lockfile version.
        uv sync --inexact --no-install-project --no-install-package nemo-relay --no-default-groups --group dev
        activate_project_venv
        prepend_ziglang_to_path "$(project_python_executable)"
        build_args+=(-- --zig --zig-abi-suffix "$linux_glibc_version")
    fi
    npm install --workspace=nemo-relay-node --ignore-scripts
    npm run --workspace=nemo-relay-node "${build_args[@]}"
    npm pack --workspace=nemo-relay-node --pack-destination "$package_dir"
    shopt -s nullglob
    packages=("$package_dir"/*.tgz)
    if ((${#packages[@]} == 0)); then
        echo "Error: No npm packages found in $package_dir"
        exit 1
    fi

# --set [output_dir=<path>] [ref_name=<name>] [ci=true|false]
package-openclaw:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    # If `ref_name` is empty, append the current short HEAD SHA to the version.
    # If `ref_name` is set, write it as the exact package version before packing.
    output_dir="{{ output_dir }}"
    cd "$NEMO_RELAY_REPO_ROOT"
    package_dir="$(prepare_package_dir openclaw)"
    if [[ -z "{{ ref_name }}" ]]; then
        sha="$(head_git_sha)"
        version="$(read_npm_package_version integrations/openclaw/package.json)"
        package_version="${version}+${sha}"
        echo "Non-release build: appending commit hash to version"
        set_npm_package_version crates/node/package.json package-lock.json "$package_version" crates/node
        set_npm_package_version integrations/openclaw/package.json package-lock.json "$package_version" integrations/openclaw
        set_npm_package_dependency_version integrations/openclaw/package.json package-lock.json integrations/openclaw nemo-relay-node "$package_version"
    else
        package_version="{{ ref_name }}"
        echo "Using explicit version {{ ref_name }}"
        set_npm_package_version crates/node/package.json package-lock.json "$package_version" crates/node
        set_npm_package_version integrations/openclaw/package.json package-lock.json "$package_version" integrations/openclaw
        set_npm_package_dependency_version integrations/openclaw/package.json package-lock.json integrations/openclaw nemo-relay-node "$package_version"
    fi
    npm install --workspace=nemo-relay-node --workspace=nemo-relay-openclaw --ignore-scripts
    if is_true "{{ ci }}"; then
        npm run build-debug --workspace=nemo-relay-node
    else
        npm run build --workspace=nemo-relay-node
    fi
    npm pack --workspace=nemo-relay-openclaw --pack-destination "$package_dir"
    shopt -s nullglob
    packages=("$package_dir"/*.tgz)
    if ((${#packages[@]} == 0)); then
        echo "Error: No OpenClaw npm packages found in $package_dir"
        exit 1
    fi

# --set [output_dir=<path>] [ref_name=<name>]
package-python:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    # Uses the workspace version from Cargo.toml, then writes the Python package
    # version into pyproject.toml using PEP 440 before building a platform wheel.
    output_dir="{{ output_dir }}"
    linux_glibc_version="{{ linux_glibc_version }}"
    export_uv_python_runtime
    cd "$NEMO_RELAY_REPO_ROOT"
    package_dir="$(prepare_package_dir wheels)"
    sync_args=(--no-install-project --no-install-package nemo-relay)
    uv sync --inexact "${sync_args[@]}"
    activate_project_venv
    if [[ -z "{{ ref_name }}" ]]; then
        sha="$(head_git_sha)"
        version="$(read_workspace_version)"
        echo "Non-release build: appending commit hash to version"
        set_python_package_version "${version}+${sha}"
    else
        echo "Using explicit version {{ ref_name }}"
        set_python_package_version "{{ ref_name }}"
    fi
    build_args=()
    while IFS= read -r -d '' arg; do
        build_args+=("$arg")
    done < <(python_wheel_build_args)
    maturin build --release "${build_args[@]}" --out "$package_dir"
    shopt -s nullglob
    wheels=("$package_dir"/*.whl)
    if ((${#wheels[@]} == 0)); then
        echo "Error: No wheels found in $package_dir"
        exit 1
    fi

# --set [output_dir=<path>] [ref_name=<name>]
package-wasm:
    #!/usr/bin/env bash
    {{ bash_helpers }}
    # `prepare_pkg.mjs` rewrites the wasm-pack output into the publishable npm
    # layout before this target sets the package version and packs the tarball.
    output_dir="{{ output_dir }}"
    cd "$NEMO_RELAY_REPO_ROOT"
    package_dir="$(prepare_package_dir wasm)"
    wasm-pack build --release crates/wasm
    node crates/wasm/scripts/prepare_pkg.mjs
    if [[ -z "{{ ref_name }}" ]]; then
        sha="$(head_git_sha)"
        version="$(read_npm_package_version crates/wasm/pkg/package.json)"
        echo "Non-release build: appending commit hash to version"
        set_npm_package_version crates/wasm/pkg/package.json "" "${version}-${sha}"
    else
        echo "Using explicit version {{ ref_name }}"
        set_npm_package_version crates/wasm/pkg/package.json "" "{{ ref_name }}"
    fi
    pushd crates/wasm/pkg >/dev/null
    npm pack --pack-destination "$package_dir"
    popd >/dev/null
    shopt -s nullglob
    packages=("$package_dir"/*.tgz)
    if ((${#packages[@]} == 0)); then
        echo "Error: No wasm packages found in $package_dir"
        exit 1
    fi
