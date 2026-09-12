# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare one immutable evaluation run root and render its Relay config."""

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

import tomli_w
from relay_version import RELAY_REQUIREMENT, require_supported_version, wheel_version

HERMES_REPOSITORY = "https://github.com/bbednarski9/hermes-agent.git"
HERMES_REF = "feat/relay-native-plugin-init"
HERMES_COMMIT = "a3d472f0e6bdc376df87b1436a461c4796db6747"
SWITCHYARD_REPOSITORY = "https://github.com/bbednarski9/Switchyard.git"
SWITCHYARD_COMMIT = "8daac03edf8544144833af1fd009b3da737715bc"
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
            RELAY_REQUIREMENT,
        ],
        check=True,
    )
    wheels = sorted(destination.glob("nemo_relay-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one Relay wheel, found {len(wheels)}")
    return wheels[0]


def verify_relay_wheel(path: Path, architecture: str) -> str:
    if not path.is_file() or not path.name.startswith("nemo_relay-"):
        raise ValueError("Relay wheel must be a nemo_relay wheel")
    if "manylinux" not in path.name or architecture not in path.name:
        raise ValueError(f"Relay wheel must target Linux {architecture}")
    return wheel_version(path)


def verify_native_library(path: Path, architecture: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:4] != b"\x7fELF" or len(header) < 20 or header[5] != 1:
        raise ValueError("Switchyard native library must be a little-endian ELF artifact")
    machine = int.from_bytes(header[18:20], "little")
    expected = {"x86_64": 62, "aarch64": 183}[architecture]
    if machine != expected:
        raise ValueError(f"Switchyard library does not target {architecture}: ELF e_machine={machine}")


def plugin_settings(config: dict[str, object]) -> dict[str, str]:
    plugins = config.get("plugins")
    if not isinstance(plugins, dict) or not isinstance(plugins.get("dynamic"), list):
        raise ValueError("plugins.toml.in must define one dynamic plugin")
    dynamic = plugins["dynamic"]
    if len(dynamic) != 1 or not isinstance(dynamic[0], dict):
        raise ValueError("plugins.toml.in must define exactly one dynamic plugin")
    plugin_config = dynamic[0].get("config")
    if not isinstance(plugin_config, dict) or not isinstance(plugin_config.get("targets"), dict):
        raise ValueError("Switchyard dynamic plugin targets are missing")
    targets = plugin_config["targets"]
    if set(targets) != {"strong", "weak", "judge"}:
        raise ValueError("Switchyard must define strong, weak, and judge targets")
    settings: dict[str, str] = {}
    algorithm = plugin_config.get("algorithm")
    classifier_target = algorithm.get("classifier_target") if isinstance(algorithm, dict) else None
    if classifier_target not in targets:
        raise ValueError("Switchyard classifier_target must reference a configured target")
    settings["classifier_target"] = classifier_target
    for name in ("strong", "weak", "judge"):
        target = targets[name]
        if not isinstance(target, dict):
            raise ValueError(f"Switchyard target is invalid: {name}")
        model = checked_label(str(target.get("model", "")), f"{name}_model")
        base_url = checked_url(str(target.get("base_url", "")), f"{name}_base_url")
        header_env = target.get("header_env")
        if header_env != {"authorization": "SWITCHYARD_PROVIDER_AUTHORIZATION"}:
            raise ValueError("plugins.toml.in must reference SWITCHYARD_PROVIDER_AUTHORIZATION")
        settings[f"{name}_model"] = model
        settings[f"{name}_base_url"] = base_url
    if settings["strong_model"] == settings["weak_model"]:
        raise ValueError("strong and weak models must be distinct")
    components = config.get("components")
    if not isinstance(components, list):
        raise ValueError("Relay components are missing")
    observability = next(
        (
            component
            for component in components
            if isinstance(component, dict) and component.get("kind") == "observability"
        ),
        None,
    )
    if not isinstance(observability, dict):
        raise ValueError("Relay observability component is missing")
    observation_config = observability.get("config")
    if not isinstance(observation_config, dict) or not isinstance(observation_config.get("atif"), dict):
        raise ValueError("Relay ATIF configuration is missing")
    settings["hermes_caller_model"] = checked_label(
        str(observation_config["atif"].get("model_name", "")), "hermes_caller_model"
    )
    if settings["hermes_caller_model"] in {
        settings["strong_model"], settings["weak_model"], settings["judge_model"]
    }:
        raise ValueError("Hermes caller model must be distinct from Switchyard targets")
    return settings


def render_config(
    template: Path,
    output: Path,
    replacements: dict[str, str],
    test_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    rendered = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"replacement {key} contains a newline")
        rendered = rendered.replace(f"@{key}@", value)
    unresolved = sorted(set(re.findall(r"@[A-Z0-9_]+@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved Relay config placeholders: {unresolved}")
    config = tomllib.loads(rendered)
    if test_overrides:
        plugin = config["plugins"]["dynamic"][0]["config"]
        old_models = {name: plugin["targets"][name]["model"] for name in ("strong", "weak", "judge")}
        for name in ("strong", "weak", "judge"):
            override = f"{name}_model"
            plugin["targets"][name]["model"] = test_overrides[override]
            plugin["targets"][name]["base_url"] = test_overrides["provider_base_url"]
        pricing = config["components"][0]["config"]["sources"][0]["catalog"]["entries"]
        replacement_models = {
            old_models[name]: test_overrides[f"{name}_model"]
            for name in old_models
        }
        for entry in pricing:
            entry["model_id"] = replacement_models.get(entry["model_id"], entry["model_id"])
    settings = plugin_settings(config)
    output.write_text(tomli_w.dumps(config), encoding="utf-8")
    os.chmod(output, 0o600)
    return settings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--switchyard-bundle", type=Path, required=True)
    parser.add_argument("--relay-wheel", type=Path)
    parser.add_argument("--relay-architecture", choices=("x86_64", "aarch64"), default="x86_64")
    parser.add_argument("--plugin-config-template", type=Path)
    parser.add_argument("--test-provider-base-url")
    parser.add_argument("--test-strong-model")
    parser.add_argument("--test-weak-model")
    parser.add_argument("--test-judge-model")
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
        relay_version = verify_relay_wheel(source_wheel, args.relay_architecture)
        wheel_dir = runtime / "wheels"
        wheel_dir.mkdir(mode=0o700)
        relay_wheel = wheel_dir / source_wheel.name
        shutil.copy2(source_wheel, relay_wheel)
    else:
        relay_wheel = download_relay_wheel(runtime / "wheels", args.relay_architecture)
        relay_version = verify_relay_wheel(relay_wheel, args.relay_architecture)

    openinference_endpoint = checked_url(args.openinference_endpoint, "openinference_endpoint")
    phoenix_project = checked_label(args.phoenix_project, "phoenix_project")
    eval_cohort = checked_label(args.eval_cohort, "eval_cohort")
    plugin_template = (args.plugin_config_template or example_root / "config" / "plugins.toml.in").resolve(strict=True)
    test_values = (args.test_provider_base_url, args.test_strong_model, args.test_weak_model, args.test_judge_model)
    if any(test_values) and not all(test_values):
        raise ValueError("all test provider overrides must be supplied together")
    test_overrides = None
    if all(test_values):
        test_overrides = {
            "provider_base_url": checked_url(args.test_provider_base_url, "test_provider_base_url"),
            "strong_model": checked_label(args.test_strong_model, "test_strong_model"),
            "weak_model": checked_label(args.test_weak_model, "test_weak_model"),
            "judge_model": checked_label(args.test_judge_model, "test_judge_model"),
        }

    config_path = runtime / "plugins.toml"
    routing = render_config(
        plugin_template,
        config_path,
        {
            "HERMES_COMMIT": HERMES_COMMIT,
            "OPENINFERENCE_ENDPOINT": openinference_endpoint,
            "PHOENIX_PROJECT": phoenix_project,
            "EVAL_COHORT": eval_cohort,
        },
        test_overrides,
    )

    manifest = bundle / "relay-plugin.toml"
    libraries = sorted(path for path in bundle.iterdir() if path.is_file() and path.suffix in {".so", ".dylib", ".dll"})
    if len(libraries) != 1:
        raise ValueError("Switchyard bundle must contain exactly one native library")
    verify_native_library(libraries[0], args.relay_architecture)
    provenance = {
        "schema_version": "harbor-hermes-switchyard.phase1.v1",
        "nemo_relay": {
            "version": relay_version,
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
        "plugin_config_template_sha256": sha256(plugin_template),
        "routing": {
            "algorithm": "llm_classifier",
            **routing,
        },
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
