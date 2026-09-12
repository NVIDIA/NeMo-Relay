# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary Harbor Hermes agent for a fork-hosted, commit-pinned Hermes build.

The class deliberately inherits Harbor's built-in Hermes lifecycle.  It only
changes installation, copies an immutable Relay configuration/plugin bundle
into the task environment, and frames the additional Phase 1 artifacts around
``super().run``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import time
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from harbor.agents.installed.hermes import Hermes
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from typing_extensions import override

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DEFAULT_HERMES_REPOSITORY = "https://github.com/bbednarski9/hermes-agent.git"
_DEFAULT_HERMES_REF = "feat/relay-native-plugin-init"
_DEFAULT_HERMES_COMMIT = "a3d472f0e6bdc376df87b1436a461c4796db6747"
_DEFAULT_SWITCHYARD_COMMIT = "8daac03edf8544144833af1fd009b3da737715bc"
_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
_PROVIDER_AUTHORIZATION_FILE = "/run/secrets/switchyard-provider-authorization"
_HERMETIC_RUNTIME_ROOT = "/opt/hermes-runtime"
_HERMETIC_RUNTIME_SCHEMA = "harbor-hermes-switchyard.hermetic-runtime.v1"
_HERMETIC_CA_BUNDLE_RELATIVE = Path("hermes-agent-src/venv/lib/python3.11/site-packages/certifi/cacert.pem")
_HERMETIC_CA_BUNDLE = f"{_HERMETIC_RUNTIME_ROOT}/{_HERMETIC_CA_BUNDLE_RELATIVE.as_posix()}"
_HERMETIC_RUNTIME_READY_ATTEMPTS = 6
_HERMETIC_RUNTIME_READY_DELAY_SECONDS = 2


def _require_full_sha(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if not _FULL_SHA.fullmatch(normalized):
        raise ValueError(f"{name} must be a full 40-character hexadecimal commit")
    return normalized


def _require_sha256(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_hermetic_runtime(
    path: Path,
    *,
    expected_digest: str,
    hermes_commit: str,
    relay_wheel_sha256: str,
    relay_architecture: str,
) -> dict[str, Any]:
    marker = path / "payload.json"
    if not marker.is_file():
        raise FileNotFoundError(marker)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    expected = {
        "schema_version": _HERMETIC_RUNTIME_SCHEMA,
        "status": "passed",
        "content_sha256": expected_digest,
        "hermes_commit": hermes_commit,
        "relay_wheel_sha256": relay_wheel_sha256,
        "relay_architecture": relay_architecture,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"hermetic runtime metadata mismatch: {mismatches}")
    required = (
        path / "bin" / "hermes",
        path / "bin" / "python",
        path / "bin" / "uv",
        path / "hermes-agent-src" / "venv",
        path / _HERMETIC_CA_BUNDLE_RELATIVE,
    )
    missing = [str(candidate) for candidate in required if not candidate.exists()]
    if missing:
        raise FileNotFoundError(f"hermetic runtime is incomplete: {missing}")
    return payload


def _hermetic_runtime_readiness_command(
    runtime_root: str = _HERMETIC_RUNTIME_ROOT,
    *,
    attempts: int = _HERMETIC_RUNTIME_READY_ATTEMPTS,
    delay_seconds: int = _HERMETIC_RUNTIME_READY_DELAY_SECONDS,
) -> str:
    """Return a bounded probe that executes both nested runtime entrypoints."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative")
    runtime = shlex.quote(runtime_root)
    attempt_numbers = " ".join(str(attempt) for attempt in range(1, attempts + 1))
    return (
        "runtime_ready=1; "
        f"for attempt in {attempt_numbers}; do "
        f'if {runtime}/bin/python -c "import importlib.metadata as m; '
        "assert tuple(map(int, m.version('nemo-relay').split('.'))) >= (0, 7, 0)\" "
        f"&& {runtime}/bin/hermes version; then "
        "runtime_ready=0; break; "
        "else runtime_ready=$?; fi; "
        f'if [ "$attempt" -lt {attempts} ]; then sleep {delay_seconds}; fi; '
        "done; "
        '[ "$runtime_ready" -eq 0 ] || exit "$runtime_ready"; '
    )


def _verify_elf_architecture(path: Path, architecture: str) -> None:
    with path.open("rb") as stream:
        header = stream.read(20)
    if header[:4] != b"\x7fELF" or len(header) < 20 or header[5] != 1:
        raise ValueError("Switchyard native library must be a little-endian ELF artifact")
    expected_machine = {"x86_64": 62, "aarch64": 183}[architecture]
    machine = int.from_bytes(header[18:20], "little")
    if machine != expected_machine:
        raise ValueError(f"Switchyard native library does not target {architecture}: ELF e_machine={machine}")


def _require_public_https_git_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith(".git")
    ):
        raise ValueError("repository_url must be a credential-free https://github.com/...git URL")
    return value


def _find_named_component(config: dict[str, Any], kind: str) -> dict[str, Any]:
    matches = [
        component
        for component in config.get("components", [])
        if isinstance(component, dict) and component.get("kind") == kind and component.get("enabled", True)
    ]
    if len(matches) != 1:
        raise ValueError(f"Relay config must contain exactly one enabled {kind!r} component")
    return matches[0]


def _validate_relay_config(path: Path) -> None:
    with path.open("rb") as stream:
        config = tomllib.load(stream)

    if config.get("version") != 1:
        raise ValueError("Relay config must use top-level version = 1")
    if config.get("dynamic_plugins"):
        raise ValueError("Hermes [[dynamic_plugins]] worker records are not allowed in this example")

    plugins = config.get("plugins")
    dynamic = plugins.get("dynamic") if isinstance(plugins, dict) else None
    if not isinstance(dynamic, list) or len(dynamic) != 1:
        raise ValueError("Relay config must contain exactly one [[plugins.dynamic]] record")
    plugin = dynamic[0]
    manifest = plugin.get("manifest") if isinstance(plugin, dict) else None
    if manifest != "/opt/relay-plugins/nvidia.switchyard/relay-plugin.toml":
        raise ValueError("the dynamic plugin must reference the staged Switchyard manifest")

    plugin_config = plugin.get("config")
    if not isinstance(plugin_config, dict) or plugin_config.get("version") != 2:
        raise ValueError("the Switchyard plugin must use config version = 2")
    algorithm = plugin_config.get("algorithm")
    expected_algorithm = {
        "kind": "llm_classifier",
        "classifier_target": "judge",
        "weak_target": "weak",
        "strong_target": "strong",
        "base_threshold": 0.5,
        "recent_turn_window": 0,
        "session_affinity": True,
        "message_hash_fallback": True,
    }
    if algorithm != expected_algorithm:
        raise ValueError("the Switchyard classifier contract does not match the Phase 1 design")
    if plugin_config.get("default_targets") != {"openai_chat": "strong"}:
        raise ValueError("the Switchyard OpenAI default target must be strong")

    targets = plugin_config.get("targets")
    if not isinstance(targets, dict) or set(targets) != {"strong", "weak", "judge"}:
        raise ValueError("Switchyard must define exactly the strong, weak, and judge targets")
    provider_models: set[str] = set()
    for name, target in targets.items():
        if not isinstance(target, dict):
            raise ValueError(f"Switchyard target {name!r} must be a table")
        if target.get("protocol") != "openai_chat" or target.get("endpoint") != "/v1/chat/completions":
            raise ValueError(f"Switchyard target {name!r} must use the OpenAI chat protocol")
        if target.get("drop_caller_extra_body") is not True:
            raise ValueError(f"Switchyard target {name!r} must drop Hermes' caller-specific extra_body wrapper")
        base_url = target.get("base_url")
        parsed_base_url = urlsplit(base_url) if isinstance(base_url, str) else None
        if (
            parsed_base_url is None
            or parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.hostname
            or parsed_base_url.username is not None
            or parsed_base_url.password is not None
        ):
            raise ValueError(f"Switchyard target {name!r} must use a credential-free HTTP(S) base URL")
        model = target.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Switchyard target {name!r} must define a model")
        provider_models.add(model)
        header_env = target.get("header_env")
        authorization_env = header_env.get("authorization") if isinstance(header_env, dict) else None
        if not isinstance(authorization_env, str) or not _ENV_NAME.fullmatch(authorization_env):
            raise ValueError(f"Switchyard target {name!r} must source authorization from an environment variable")
    if len(provider_models) != 3:
        raise ValueError("Switchyard strong, weak, and judge targets must use distinct models")

    pricing = _find_named_component(config, "pricing")
    pricing_config = pricing.get("config")
    sources = pricing_config.get("sources") if isinstance(pricing_config, dict) else None
    if not isinstance(sources, list) or len(sources) != 1 or sources[0].get("type") != "inline":
        raise ValueError("pricing must use exactly one inline catalog")
    catalog = sources[0].get("catalog")
    entries = catalog.get("entries") if isinstance(catalog, dict) and catalog.get("version") == 1 else None
    if not isinstance(entries, list) or {entry.get("model_id") for entry in entries} != provider_models:
        raise ValueError("pricing entries must match the Switchyard provider models")
    for entry in entries:
        rates = entry.get("rates") if isinstance(entry, dict) else None
        if not isinstance(rates, dict) or any(
            not isinstance(rates.get(key), (int, float)) or rates[key] <= 0
            for key in ("input_per_million", "output_per_million", "cache_read_per_million")
        ):
            raise ValueError("pricing entries must contain positive input, output, and cache-read rates")

    observability = _find_named_component(config, "observability")
    observability_config = observability.get("config")
    if not isinstance(observability_config, dict) or observability_config.get("version") != 3:
        raise ValueError("the observability component must use schema version = 3")
    atif = observability_config.get("atif")
    caller_model = atif.get("model_name") if isinstance(atif, dict) else None
    if not isinstance(caller_model, str) or not caller_model or caller_model in provider_models:
        raise ValueError("the fail-closed Hermes caller model must not be a Switchyard provider model")
    opentelemetry = observability_config.get("opentelemetry")
    endpoints = opentelemetry.get("endpoints") if isinstance(opentelemetry, dict) else None
    if not isinstance(opentelemetry, dict) or opentelemetry.get("enabled") is not True:
        raise ValueError("OpenTelemetry export must be enabled")
    if not isinstance(endpoints, list) or len(endpoints) != 1:
        raise ValueError("observability must define exactly one OpenInference endpoint")
    endpoint = endpoints[0]
    if endpoint.get("type") != "openinference" or endpoint.get("transport") != "http_binary":
        raise ValueError("the only telemetry endpoint must be OpenInference over OTLP/HTTP protobuf")
    resource_attributes = endpoint.get("resource_attributes")
    if not isinstance(resource_attributes, dict) or not all(
        isinstance(resource_attributes.get(key), str) and resource_attributes[key]
        for key in ("openinference.project.name", "evaluation.cohort")
    ):
        raise ValueError("OpenInference must carry project and evaluation cohort resource attributes")

    def reject_literal_headers(value: Any, location: str = "config") -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "headers":
                    raise ValueError(f"literal headers are forbidden; use header_env ({location}.headers)")
                reject_literal_headers(nested, f"{location}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                reject_literal_headers(nested, f"{location}[{index}]")

    reject_literal_headers(config)


class HarborHermesAgent(Hermes):
    """Hermes #77915 bridge retaining Harbor's built-in Hermes behavior."""

    def __init__(
        self,
        *args: Any,
        repository_url: str = _DEFAULT_HERMES_REPOSITORY,
        repository_ref: str = _DEFAULT_HERMES_REF,
        commit: str = _DEFAULT_HERMES_COMMIT,
        relay_config_path: str,
        switchyard_bundle_dir: str,
        relay_wheel_path: str,
        relay_wheel_sha256: str,
        relay_architecture: str = "x86_64",
        switchyard_commit: str = _DEFAULT_SWITCHYARD_COMMIT,
        artifact_root: str = "/logs/agent/direct-hermes",
        inject_post_response_failure: bool = False,
        hermetic_runtime_dir: str | None = None,
        hermetic_runtime_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.repository_url = _require_public_https_git_url(repository_url)
        self.repository_ref = repository_ref.strip()
        if not self.repository_ref or self.repository_ref.startswith("-"):
            raise ValueError("repository_ref must be a non-option branch or tag name")
        self.commit = _require_full_sha(commit, "commit")
        self.switchyard_commit = _require_full_sha(switchyard_commit, "switchyard_commit")
        self.relay_wheel_sha256 = _require_sha256(relay_wheel_sha256, "relay_wheel_sha256")
        if relay_architecture not in {"x86_64", "aarch64"}:
            raise ValueError("relay_architecture must be x86_64 or aarch64")
        self.relay_architecture = relay_architecture

        self.relay_config_path = Path(relay_config_path).expanduser().resolve()
        self.switchyard_bundle_dir = Path(switchyard_bundle_dir).expanduser().resolve()
        self.relay_wheel_path = Path(relay_wheel_path).expanduser().resolve()
        self.artifact_root = artifact_root.rstrip("/")
        self.inject_post_response_failure = inject_post_response_failure
        self.hermetic_runtime_dir: Path | None = None
        self.hermetic_runtime_sha256: str | None = None
        self._load_provider_authorization = False
        if not self.artifact_root.startswith("/logs/agent/"):
            raise ValueError("artifact_root must be an absolute child of /logs/agent")
        if not self.relay_config_path.is_file():
            raise FileNotFoundError(self.relay_config_path)
        if not self.relay_wheel_path.is_file():
            raise FileNotFoundError(self.relay_wheel_path)
        if _sha256(self.relay_wheel_path) != self.relay_wheel_sha256:
            raise ValueError("Relay wheel digest does not match relay_wheel_sha256")
        if "manylinux" not in self.relay_wheel_path.name or relay_architecture not in self.relay_wheel_path.name:
            raise ValueError(f"Relay wheel must target Linux {relay_architecture}")
        _validate_relay_config(self.relay_config_path)

        self.switchyard_manifest = self.switchyard_bundle_dir / "relay-plugin.toml"
        if not self.switchyard_manifest.is_file():
            raise FileNotFoundError(self.switchyard_manifest)
        libraries = sorted(
            path
            for path in self.switchyard_bundle_dir.iterdir()
            if path.is_file() and path.suffix in {".so", ".dylib", ".dll"}
        )
        if len(libraries) != 1:
            raise ValueError("Switchyard bundle must contain exactly one native library")
        self.switchyard_library = libraries[0]
        _verify_elf_architecture(self.switchyard_library, relay_architecture)

        if (hermetic_runtime_dir is None) != (hermetic_runtime_sha256 is None):
            raise ValueError("hermetic_runtime_dir and hermetic_runtime_sha256 must be supplied together")
        if hermetic_runtime_dir is not None and hermetic_runtime_sha256 is not None:
            runtime_dir = Path(hermetic_runtime_dir).expanduser().resolve()
            runtime_digest = _require_sha256(hermetic_runtime_sha256, "hermetic_runtime_sha256")
            _load_hermetic_runtime(
                runtime_dir,
                expected_digest=runtime_digest,
                hermes_commit=self.commit,
                relay_wheel_sha256=self.relay_wheel_sha256,
                relay_architecture=self.relay_architecture,
            )
            self.hermetic_runtime_dir = runtime_dir
            self.hermetic_runtime_sha256 = runtime_digest

        self._example_root = Path(__file__).resolve().parents[1]
        self._finalizer_path = self._example_root / "scripts" / "finalize_artifacts.py"
        if not self._finalizer_path.is_file():
            raise FileNotFoundError(self._finalizer_path)
        self._relay_version_path = self._example_root / "scripts" / "relay_version.py"
        if not self._relay_version_path.is_file():
            raise FileNotFoundError(self._relay_version_path)

        extra_env = dict(kwargs.pop("extra_env", None) or {})
        extra_env["HERMES_NEMO_RELAY_PLUGINS_TOML"] = "/tmp/hermes/relay/plugins.toml"
        super().__init__(*args, version=self.commit, extra_env=extra_env, **kwargs)

    @override
    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> Any:
        if self.hermetic_runtime_dir is not None:
            ca_bundle = shlex.quote(_HERMETIC_CA_BUNDLE)
            command = (
                f"test -r {ca_bundle}; "
                f"export SSL_CERT_FILE={ca_bundle}; "
                f"export REQUESTS_CA_BUNDLE={ca_bundle}; "
                f"export CURL_CA_BUNDLE={ca_bundle}; "
                f"{command}"
            )
        if self._load_provider_authorization:
            secret_file = shlex.quote(_PROVIDER_AUTHORIZATION_FILE)
            command = (
                f"test -r {secret_file}; "
                f'export SWITCHYARD_PROVIDER_AUTHORIZATION="$(cat -- {secret_file})"; '
                'test -n "$SWITCHYARD_PROVIDER_AUTHORIZATION"; '
                f"{command}"
            )
        return await super().exec_as_agent(
            environment,
            command,
            env=env,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        if self.hermetic_runtime_dir is not None:
            runtime = shlex.quote(_HERMETIC_RUNTIME_ROOT)
            await self.exec_as_agent(
                environment,
                command=(
                    "set -euo pipefail; "
                    f"test -r {runtime}/payload.json; "
                    f"test -x {runtime}/bin/hermes; "
                    f"test -x {runtime}/bin/python; "
                    f"test -x {runtime}/bin/uv; "
                    f"test -r {shlex.quote(_HERMETIC_CA_BUNDLE)}; "
                    f"{_hermetic_runtime_readiness_command()}"
                    "rm -rf /tmp/hermes-agent-src; "
                    f"ln -s {runtime}/hermes-agent-src /tmp/hermes-agent-src; "
                    'mkdir -p /tmp/hermes/bin "$HOME/.local/bin"; '
                    f'ln -sf {runtime}/bin/hermes "$HOME/.local/bin/hermes"; '
                    f"ln -sf {runtime}/bin/uv /tmp/hermes/bin/uv; "
                    f"if test -x {runtime}/bin/rg; then "
                    f'ln -sf {runtime}/bin/rg "$HOME/.local/bin/rg"; fi; '
                    'export PATH="$HOME/.local/bin:$PATH"'
                ),
                timeout_sec=90,
            )
            return

        await self.exec_as_root(
            environment,
            command=(
                "set -euo pipefail; last_status=1; "
                "for attempt in 1 2 3; do "
                "if apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates build-essential curl git ripgrep xz-utils; then exit 0; "
                "else last_status=$?; fi; "
                'if [ "$attempt" -eq 3 ]; then break; fi; '
                "rm -rf /var/lib/apt/lists/partial; sleep $((attempt * 5)); "
                'done; exit "$last_status"'
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        repository = shlex.quote(self.repository_url)
        repository_ref = shlex.quote(self.repository_ref)
        commit = shlex.quote(self.commit)
        install_dir = "/tmp/hermes-agent-src"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"git clone --no-tags --branch {repository_ref} {repository} {install_dir}; "
                f"git -C {install_dir} fetch --depth 1 origin {commit}; "
                f"git -C {install_dir} checkout --detach {commit}; "
                f'test "$(git -C {install_dir} rev-parse HEAD)" = {commit}; '
                "mkdir -p /tmp/hermes-install-path; "
                "ln -sf /bin/true /tmp/hermes-install-path/ffmpeg; "
                f"HERMES_HOME=/tmp/hermes HERMES_INSTALL_DIR={install_dir} "
                "PATH=/tmp/hermes-install-path:$PATH "
                f"bash {install_dir}/scripts/install.sh --skip-setup --skip-browser "
                f"--no-skills --dir {install_dir} --branch {repository_ref} "
                f"--commit {commit} --force-commit; "
                f'test "$(git -C {install_dir} rev-parse HEAD)" = {commit}; '
                f"cd {install_dir}; "
                f"UV_PROJECT_ENVIRONMENT={install_dir}/venv "
                "/tmp/hermes/bin/uv sync --frozen --extra all; "
                'export PATH="$HOME/.local/bin:$PATH"; '
                "hermes version; "
                f'{install_dir}/venv/bin/python -c "import importlib.metadata as m; '
                "assert tuple(map(int, m.version('nemo-relay').split('.'))) >= (0, 7, 0)\""
            ),
        )

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        await self.exec_as_root(
            environment,
            command=(
                "mkdir -p /tmp/hermes/relay /opt/relay-wheels /opt/relay-plugins/nvidia.switchyard /installed-agent"
            ),
        )
        await environment.upload_file(self.relay_config_path, "/tmp/hermes/relay/plugins.toml")
        relay_wheel = f"/opt/relay-wheels/{self.relay_wheel_path.name}"
        await environment.upload_file(self.relay_wheel_path, relay_wheel)
        if self.hermetic_runtime_dir is None:
            relay_install = (
                "/tmp/hermes/bin/uv pip install "
                "--python /tmp/hermes-agent-src/venv/bin/python "
                f"--force-reinstall --no-deps {shlex.quote(relay_wheel)}; "
                "/tmp/hermes-agent-src/venv/bin/python"
            )
        else:
            relay_install = f"{_HERMETIC_RUNTIME_ROOT}/bin/python"
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"test \"$(sha256sum {shlex.quote(relay_wheel)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(self.relay_wheel_sha256)}; "
                f'{relay_install} -c "import importlib.metadata as m; '
                "assert tuple(map(int, m.version('nemo-relay').split('.'))) >= (0, 7, 0)\""
            ),
            timeout_sec=120,
        )
        await environment.upload_dir(self.switchyard_bundle_dir, "/opt/relay-plugins/nvidia.switchyard")
        await environment.upload_file(self._finalizer_path, "/installed-agent/finalize_artifacts.py")
        await environment.upload_file(self._relay_version_path, "/installed-agent/relay_version.py")
        probe_python = (
            f"{_HERMETIC_RUNTIME_ROOT}/bin/python"
            if self.hermetic_runtime_dir is not None
            else "/tmp/hermes-agent-src/venv/bin/python"
        )
        switchyard_library = f"/opt/relay-plugins/nvidia.switchyard/{self.switchyard_library.name}"
        await self.exec_as_agent(
            environment,
            command=(
                f"{probe_python} -c "
                + shlex.quote(
                    "import ctypes, importlib.metadata as m; "
                    "assert tuple(map(int, m.version('nemo-relay').split('.'))) >= (0, 7, 0); "
                    f"library = ctypes.CDLL({switchyard_library!r}); "
                    "assert getattr(library, 'nemo_relay_register_plugin')"
                )
            ),
            timeout_sec=30,
        )
        await self.exec_as_agent(
            environment,
            command=self._finalizer_command("initialize"),
            env={"HERMES_HOME": "/tmp/hermes"},
            timeout_sec=30,
        )

    def _finalizer_command(
        self,
        mode: str,
        *,
        started_at: float | None = None,
        error_type: str = "",
    ) -> str:
        arguments = [
            (
                f"{_HERMETIC_RUNTIME_ROOT}/bin/python"
                if self.hermetic_runtime_dir is not None
                else "/tmp/hermes-agent-src/venv/bin/python"
            ),
            "/installed-agent/finalize_artifacts.py",
            mode,
            "--artifact-root",
            self.artifact_root,
            "--hermes-repository",
            self.repository_url,
            "--hermes-commit",
            self.commit,
            "--switchyard-commit",
            self.switchyard_commit,
            "--relay-wheel-sha256",
            self.relay_wheel_sha256,
            "--relay-config",
            "/tmp/hermes/relay/plugins.toml",
            "--switchyard-manifest",
            "/opt/relay-plugins/nvidia.switchyard/relay-plugin.toml",
            "--switchyard-library",
            f"/opt/relay-plugins/nvidia.switchyard/{self.switchyard_library.name}",
            "--session-handle",
            self.session_id or "",
        ]
        if started_at is not None:
            arguments.extend(["--started-at", str(started_at)])
        if error_type:
            arguments.extend(["--error-type", error_type])
        return " ".join(shlex.quote(value) for value in arguments)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        started_at = time.time()
        error: BaseException | None = None
        try:
            self._load_provider_authorization = True
            try:
                await super().run(instruction, environment, context)
            finally:
                self._load_provider_authorization = False
        except BaseException as exc:
            error = exc
            raise
        finally:
            try:
                await self.exec_as_agent(
                    environment,
                    command=self._finalizer_command(
                        "complete",
                        started_at=started_at,
                        error_type=(
                            type(error).__name__
                            if error is not None
                            else ("InjectedPostResponseFailure" if self.inject_post_response_failure else "")
                        ),
                    ),
                    env={"HERMES_HOME": "/tmp/hermes"},
                    timeout_sec=30,
                )
            except Exception:
                if error is None:
                    raise
                self.logger.exception("Could not frame direct Hermes artifacts after agent failure")


__all__ = ["HarborHermesAgent"]
