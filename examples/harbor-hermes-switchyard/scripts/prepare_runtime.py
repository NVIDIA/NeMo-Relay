# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one immutable Phase 1 run root and render its Relay config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile

HERMES_REPOSITORY = "https://github.com/bbednarski9/hermes-agent.git"
HERMES_REF = "feat/relay-native-plugin-init"
HERMES_COMMIT = "efb63e714abc436af88af9b0d6734751c199aa6d"
SWITCHYARD_REPOSITORY = "https://github.com/bbednarski9/Switchyard.git"
SWITCHYARD_COMMIT = "8293936a0f5758aa1a782639d485b8b8948cf03e"
RELAY_VERSION = "0.7.0"
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_url(value: str, name: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be a credential-free HTTP(S) URL")
    return value.rstrip("/")


def checked_label(value: str, name: str) -> str:
    if not SAFE_LABEL.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def download_relay_wheel(destination: Path, architecture: str) -> Path:
    destination.mkdir(mode=0o700, parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--platform",
            f"manylinux2014_{architecture}",
            "--implementation",
            "cp",
            "--python-version",
            "311",
            "--abi",
            "abi3",
            "--dest",
            str(destination),
            f"nemo-relay=={RELAY_VERSION}",
        ],
        check=True,
    )
    wheels = sorted(destination.glob("nemo_relay-0.7.0-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one Relay wheel, found {len(wheels)}")
    return wheels[0]


def verify_relay_wheel(path: Path, architecture: str) -> None:
    if not path.is_file() or not path.name.startswith("nemo_relay-0.7.0-"):
        raise ValueError("Relay wheel must be a nemo_relay-0.7.0 wheel")
    if "manylinux" not in path.name or architecture not in path.name:
        raise ValueError(f"Relay wheel must target Linux {architecture}")
    with ZipFile(path) as wheel:
        metadata_names = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Relay wheel has an ambiguous METADATA payload")
        metadata = wheel.read(metadata_names[0]).decode("utf-8", errors="strict")
    if "Name: nemo-relay\n" not in metadata or "Version: 0.7.0\n" not in metadata:
        raise ValueError("Relay wheel metadata does not identify nemo-relay==0.7.0")


def verify_native_library(path: Path, architecture: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:4] != b"\x7fELF" or len(header) < 20 or header[5] != 1:
        raise ValueError("Switchyard native library must be a little-endian ELF artifact")
    machine = int.from_bytes(header[18:20], "little")
    expected = {"x86_64": 62, "aarch64": 183}[architecture]
    if machine != expected:
        raise ValueError(f"Switchyard library does not target {architecture}: ELF e_machine={machine}")


def render_config(template: Path, output: Path, replacements: dict[str, str]) -> None:
    rendered = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"replacement {key} contains a newline")
        rendered = rendered.replace(f"@{key}@", value)
    unresolved = sorted(set(re.findall(r"@[A-Z0-9_]+@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved Relay config placeholders: {unresolved}")
    output.write_text(rendered, encoding="utf-8")
    os.chmod(output, 0o600)
    with output.open("rb") as stream:
        tomllib.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--switchyard-bundle", type=Path, required=True)
    parser.add_argument("--relay-wheel", type=Path)
    parser.add_argument("--relay-architecture", choices=("x86_64", "aarch64"), default="x86_64")
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--upstream-auth-env", default="SWITCHYARD_PROVIDER_AUTHORIZATION")
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--openinference-endpoint", required=True)
    parser.add_argument("--phoenix-project", required=True)
    parser.add_argument("--eval-cohort", required=True)
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parents[1]
    run_root = args.run_root.expanduser().resolve()
    if run_root.exists():
        raise FileExistsError(f"run root already exists: {run_root}")
    run_root.mkdir(mode=0o700, parents=True)
    runtime = run_root / "runtime"
    artifacts = run_root / "artifacts"
    jobs = run_root / "jobs"
    for path in (runtime, artifacts, jobs):
        path.mkdir(mode=0o700)

    source_bundle = args.switchyard_bundle.expanduser().resolve()
    if not (source_bundle / "relay-plugin.toml").is_file():
        raise FileNotFoundError(source_bundle / "relay-plugin.toml")
    bundle = runtime / "switchyard-plugin"
    shutil.copytree(source_bundle, bundle)

    if args.relay_wheel:
        source_wheel = args.relay_wheel.expanduser().resolve()
        verify_relay_wheel(source_wheel, args.relay_architecture)
        wheel_dir = runtime / "wheels"
        wheel_dir.mkdir(mode=0o700)
        relay_wheel = wheel_dir / source_wheel.name
        shutil.copy2(source_wheel, relay_wheel)
    else:
        relay_wheel = download_relay_wheel(runtime / "wheels", args.relay_architecture)
        verify_relay_wheel(relay_wheel, args.relay_architecture)

    upstream_base_url = checked_url(args.upstream_base_url, "upstream_base_url")
    openinference_endpoint = checked_url(args.openinference_endpoint, "openinference_endpoint")
    if not ENV_NAME.fullmatch(args.upstream_auth_env):
        raise ValueError("upstream_auth_env must be an uppercase environment variable name")
    target_model = checked_label(args.target_model, "target_model")
    phoenix_project = checked_label(args.phoenix_project, "phoenix_project")
    eval_cohort = checked_label(args.eval_cohort, "eval_cohort")

    config_path = runtime / "plugins.toml"
    render_config(
        example_root / "config" / "relay.toml.in",
        config_path,
        {
            "TARGET_MODEL": target_model,
            "HERMES_COMMIT": HERMES_COMMIT,
            "OPENINFERENCE_ENDPOINT": openinference_endpoint,
            "PHOENIX_PROJECT": phoenix_project,
            "EVAL_COHORT": eval_cohort,
            "UPSTREAM_BASE_URL": upstream_base_url,
            "UPSTREAM_AUTH_ENV": args.upstream_auth_env,
        },
    )

    manifest = bundle / "relay-plugin.toml"
    libraries = sorted(path for path in bundle.iterdir() if path.is_file() and path.suffix in {".so", ".dylib", ".dll"})
    if len(libraries) != 1:
        raise ValueError("Switchyard bundle must contain exactly one native library")
    verify_native_library(libraries[0], args.relay_architecture)
    provenance = {
        "schema_version": "harbor-hermes-switchyard.phase1.v1",
        "nemo_relay": {
            "version": RELAY_VERSION,
            "architecture": args.relay_architecture,
            "wheel": relay_wheel.name,
            "wheel_sha256": sha256(relay_wheel),
        },
        "hermes": {
            "repository": HERMES_REPOSITORY,
            "ref": HERMES_REF,
            "commit": HERMES_COMMIT,
        },
        "switchyard": {
            "repository": SWITCHYARD_REPOSITORY,
            "commit": SWITCHYARD_COMMIT,
            "manifest_sha256": sha256(manifest),
            "library": libraries[0].name,
            "library_sha256": sha256(libraries[0]),
        },
        "relay_config_sha256": sha256(config_path),
        "phoenix_project": phoenix_project,
        "eval_cohort": eval_cohort,
    }
    provenance_path = runtime / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(provenance_path, 0o600)
    print(json.dumps({"run_root": str(run_root), "provenance": provenance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
