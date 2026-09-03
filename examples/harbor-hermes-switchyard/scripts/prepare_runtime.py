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
from plugin_config_paths import (
    SWITCHYARD_ROUTES_PATH,
    experiment_paths,
    plugin_config_identity_sha256,
)
from relay_version import RELAY_REQUIREMENT, wheel_version

HERMES_REPOSITORY = "https://github.com/NousResearch/hermes-agent.git"
HERMES_REF = "main"
HERMES_COMMIT = "48c0c3a873bc5adaf20c632b5b7630a4fac000b4"
SWITCHYARD_REPOSITORY = "https://github.com/NVIDIA-NeMo/Switchyard.git"
SWITCHYARD_COMMIT = "7a72c0667774244d66a8b631e375c9d6e393bf57"
SAFE_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
DIRECT_PROVIDER_BASE_URL = "https://inference-api.nvidia.com/v1"


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


def plugin_settings(
    config: dict[str, object], switchyard_routes: dict[str, object] | None
) -> dict[str, object]:
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
    caller_model = checked_label(str(observation_config["atif"].get("model_name", "")), "hermes_caller_model")

    plugins = config.get("plugins")
    dynamic_plugins = plugins.get("dynamic") if isinstance(plugins, dict) else None
    if dynamic_plugins is None:
        pricing = next(
            (
                component
                for component in components
                if isinstance(component, dict) and component.get("kind") == "pricing"
            ),
            None,
        )
        entries = (
            pricing.get("config", {}).get("sources", [{}])[0].get("catalog", {}).get("entries", [])
            if isinstance(pricing, dict)
            else []
        )
        if not isinstance(entries, list) or len(entries) != 1 or entries[0].get("model_id") != caller_model:
            raise ValueError("direct baseline must price exactly its Hermes caller model")
        return {
            "algorithm": "direct",
            "direct_model": caller_model,
            "direct_base_url": DIRECT_PROVIDER_BASE_URL,
            "hermes_caller_model": caller_model,
        }
    if not isinstance(dynamic_plugins, list):
        raise ValueError("plugins.dynamic must be a list")
    dynamic = dynamic_plugins
    if len(dynamic) != 1 or not isinstance(dynamic[0], dict):
        raise ValueError("plugins.toml.in must define exactly one dynamic plugin")
    plugin_config = dynamic[0].get("config")
    if not isinstance(plugin_config, dict):
        raise ValueError("Switchyard plugin configuration is missing")
    if plugin_config.get("switchyard_config_path") != SWITCHYARD_ROUTES_PATH:
        raise ValueError("plugins.toml.in must reference the staged switchyard-routes.toml")
    switchyard_config = switchyard_routes
    if not isinstance(switchyard_config, dict) or switchyard_config.get("schema_version") != 1:
        raise ValueError("switchyard-routes.toml must declare schema_version = 1")
    llm_clients = switchyard_config.get("llm_clients")
    if not isinstance(llm_clients, dict) or set(llm_clients) != {"nvidia"}:
        raise ValueError("Switchyard deployment must define exactly one llm_clients.nvidia client")
    client = llm_clients["nvidia"]
    base_url = checked_url(str(client.get("base_url", "")), "nvidia_base_url")
    if client.get("api_key_env") != "SWITCHYARD_PROVIDER_AUTHORIZATION":
        raise ValueError("switchyard-routes.toml must reference SWITCHYARD_PROVIDER_AUTHORIZATION")
    if not isinstance(switchyard_config.get("targets"), dict):
        raise ValueError("Switchyard dynamic plugin targets are missing")
    targets = switchyard_config["targets"]
    routes = switchyard_config.get("routes")
    if not isinstance(routes, dict) or set(routes) != {"default"}:
        raise ValueError("Switchyard deployment must define exactly one routes.default route")
    route = routes["default"]
    if not isinstance(route, dict):
        raise ValueError("routes.default must be a table")
    settings: dict[str, object] = {}
    algorithm_kind = route.get("type")
    if algorithm_kind != "stage_router" and set(targets) != {"strong", "weak", "judge"}:
        raise ValueError("this Switchyard routing mode requires strong, weak, and judge targets")
    if algorithm_kind == "random":
        if route != {"id": route.get("id"), "type": "random", "targets": ["strong", "weak"]}:
            raise ValueError("Switchyard random routing must omit a fixed seed")
        settings["random_weights"] = {"strong": 1.0, "weak": 1.0}
    elif algorithm_kind == "llm_classifier":
        classifier_target = route.get("classifier_target")
        if classifier_target not in targets:
            raise ValueError("Switchyard classifier_target must reference a configured target")
        settings["classifier_target"] = classifier_target
        classifier_mode = route.get("mode", "capability")
        if classifier_mode not in {"capability", "escalation"}:
            raise ValueError("Switchyard LLM-classifier mode is invalid")
        settings["classifier_mode"] = classifier_mode
        if classifier_mode == "escalation":
            escalation = route.get("escalation")
            if escalation != {
                "confirmations": 1,
                "recent_turn_window": 28,
                "window_message_chars": 500,
            }:
                raise ValueError("Switchyard escalation settings do not match the trial contract")
            settings["escalation"] = dict(escalation)
    elif algorithm_kind == "stage_router":
        if route.get("capable_target") != "strong" or route.get("efficient_target") != "weak":
            raise ValueError("Switchyard stage-router tiers must reference strong and weak targets")
        if route.get("picker") not in {"capable_first", "efficient_first"}:
            raise ValueError("Switchyard stage-router picker is invalid")
        confidence_threshold = route.get("confidence_threshold")
        if not isinstance(confidence_threshold, (int, float)) or not 0 <= confidence_threshold <= 1:
            raise ValueError("Switchyard stage-router confidence threshold is invalid")
        classifier = route.get("classifier")
        if classifier is not None and (not isinstance(classifier, dict) or classifier.get("target") != "judge"):
            raise ValueError("Switchyard stage-router classifier must reference the judge target")
        expected_targets = {"strong", "weak", "judge"} if classifier is not None else {"strong", "weak"}
        if set(targets) != expected_targets:
            raise ValueError(f"Switchyard stage-router targets must be {sorted(expected_targets)}")
        settings.update(
            {
                "classifier_target": classifier["target"] if classifier is not None else None,
                "picker": route["picker"],
                "confidence_threshold": float(confidence_threshold),
                "recent_turn_window": route.get("recent_turn_window"),
            }
        )
    else:
        raise ValueError(f"unsupported Switchyard algorithm: {algorithm_kind!r}")
    target_names = tuple(targets)
    for name in target_names:
        target = targets[name]
        if not isinstance(target, dict):
            raise ValueError(f"Switchyard target is invalid: {name}")
        if target.get("llm_client") != "nvidia":
            raise ValueError(f"Switchyard target must use the nvidia llm_client: {name}")
        model = checked_label(str(target.get("id", "")), f"{name}_model")
        settings[f"{name}_model"] = model
        settings[f"{name}_base_url"] = base_url
    if settings["strong_model"] == settings["weak_model"]:
        raise ValueError("strong and weak models must be distinct")
    settings["algorithm"] = algorithm_kind
    settings["hermes_caller_model"] = caller_model
    if route.get("id") != caller_model:
        raise ValueError("Switchyard route id must match the Hermes caller model")
    provider_models = {settings["strong_model"], settings["weak_model"]}
    if "judge_model" in settings:
        provider_models.add(settings["judge_model"])
    if settings["hermes_caller_model"] in provider_models:
        raise ValueError("Hermes caller model must be distinct from Switchyard targets")
    return settings


def render_config(
    template: Path,
    output: Path,
    bundle: Path,
    replacements: dict[str, str],
    switchyard_source: Path | None,
    pricing_source: Path | None,
    test_overrides: dict[str, str] | None = None,
) -> dict[str, object]:
    rendered = template.read_text(encoding="utf-8")
    for key, value in replacements.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"replacement {key} contains a newline")
        rendered = rendered.replace(f"@{key}@", value)
    unresolved = sorted(set(re.findall(r"@[A-Z0-9_]+@", rendered)))
    if unresolved:
        raise ValueError(f"unresolved Relay config placeholders: {unresolved}")
    config = tomllib.loads(rendered)

    plugins = config.get("plugins")
    dynamic_plugins = plugins.get("dynamic") if isinstance(plugins, dict) else None
    if dynamic_plugins is not None and (switchyard_source is None or pricing_source is None):
        raise ValueError("this plugin template requires --switchyard-experiment")
    if dynamic_plugins is None and (switchyard_source is not None or pricing_source is not None):
        raise ValueError("--switchyard-experiment must not be set for a direct baseline template")
    switchyard_routes: dict[str, object] | None = None
    if dynamic_plugins is not None:
        with switchyard_source.open("rb") as stream:
            switchyard_routes = tomllib.load(stream)
        pricing_catalog = json.loads(pricing_source.read_text(encoding="utf-8"))
        if test_overrides:
            target_names = tuple(switchyard_routes["targets"])
            old_models = {name: switchyard_routes["targets"][name]["id"] for name in target_names}
            for name in target_names:
                override = f"{name}_model"
                switchyard_routes["targets"][name]["id"] = test_overrides[override]
            for client in switchyard_routes.get("llm_clients", {}).values():
                client["base_url"] = test_overrides["provider_base_url"]
            replacement_models = {old_models[name]: test_overrides[f"{name}_model"] for name in old_models}
            for entry in pricing_catalog.get("entries", []):
                entry["model_id"] = replacement_models.get(entry["model_id"], entry["model_id"])
        (bundle / "switchyard-routes.toml").write_bytes(tomli_w.dumps(switchyard_routes).encode("utf-8"))
        (bundle / "pricing.json").write_text(
            json.dumps(pricing_catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    settings = plugin_settings(config, switchyard_routes)
    output.write_text(tomli_w.dumps(config), encoding="utf-8")
    os.chmod(output, 0o600)
    return settings


def initialize_run_root(run_root: Path, *, allow_existing_collector_state: bool) -> None:
    if run_root.exists():
        if not allow_existing_collector_state:
            raise FileExistsError(f"run root already exists: {run_root}")
        allowed_entries = {"collector.container-id", "telemetry"}
        unexpected = sorted(path.name for path in run_root.iterdir() if path.name not in allowed_entries)
        collector_id = run_root / "collector.container-id"
        telemetry = run_root / "telemetry"
        if unexpected or not collector_id.is_file() or not telemetry.is_dir():
            raise FileExistsError(
                f"run root contains state other than the active collector bootstrap: {run_root}"
            )
        return
    run_root.mkdir(mode=0o700, parents=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--switchyard-bundle", type=Path, required=True)
    parser.add_argument("--relay-wheel", type=Path)
    parser.add_argument("--relay-architecture", choices=("x86_64", "aarch64"), default="x86_64")
    parser.add_argument("--plugin-config-template", type=Path)
    parser.add_argument("--switchyard-experiment")
    parser.add_argument("--test-provider-base-url")
    parser.add_argument("--test-strong-model")
    parser.add_argument("--test-weak-model")
    parser.add_argument("--test-judge-model")
    parser.add_argument("--openinference-endpoint", required=True)
    parser.add_argument("--phoenix-project", required=True)
    parser.add_argument("--eval-cohort", required=True)
    parser.add_argument("--allow-existing-collector-state", action="store_true")
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parents[1]
    run_root = args.run_root.expanduser().resolve()
    initialize_run_root(run_root, allow_existing_collector_state=args.allow_existing_collector_state)
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
    switchyard_source: Path | None = None
    pricing_source: Path | None = None
    if args.switchyard_experiment is not None:
        switchyard_source, pricing_source = experiment_paths(plugin_template.parent, args.switchyard_experiment)
        switchyard_source = switchyard_source.resolve(strict=True)
        pricing_source = pricing_source.resolve(strict=True)
    test_values = (args.test_provider_base_url, args.test_strong_model, args.test_weak_model)
    if any(test_values) and not all(test_values):
        raise ValueError("test provider, strong model, and weak model overrides must be supplied together")
    test_overrides = None
    if all(test_values):
        test_overrides = {
            "provider_base_url": checked_url(args.test_provider_base_url, "test_provider_base_url"),
            "strong_model": checked_label(args.test_strong_model, "test_strong_model"),
            "weak_model": checked_label(args.test_weak_model, "test_weak_model"),
        }
        if args.test_judge_model:
            test_overrides["judge_model"] = checked_label(args.test_judge_model, "test_judge_model")

    config_path = runtime / "plugins.toml"
    routing = render_config(
        plugin_template,
        config_path,
        bundle,
        {
            "HERMES_COMMIT": HERMES_COMMIT,
            "OPENINFERENCE_ENDPOINT": openinference_endpoint,
            "PHOENIX_PROJECT": phoenix_project,
            "EVAL_COHORT": eval_cohort,
        },
        switchyard_source,
        pricing_source,
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
        "plugin_config_template_sha256": plugin_config_identity_sha256(
            plugin_template, switchyard_source, pricing_source
        ),
        "routing": {
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
