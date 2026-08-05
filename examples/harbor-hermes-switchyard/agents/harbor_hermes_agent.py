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
_DEFAULT_HERMES_COMMIT = "a07830e086b3055e313b74cc0c8fd5326a4c2c00"
_DEFAULT_SWITCHYARD_COMMIT = "8293936a0f5758aa1a782639d485b8b8948cf03e"


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
    manifest = dynamic[0].get("manifest") if isinstance(dynamic[0], dict) else None
    if not isinstance(manifest, str) or not manifest.endswith("/relay-plugin.toml"):
        raise ValueError("the dynamic plugin must reference a Relay plugin manifest")

    _find_named_component(config, "pricing")
    observability = _find_named_component(config, "observability")
    observability_config = observability.get("config")
    if not isinstance(observability_config, dict) or observability_config.get("version") != 3:
        raise ValueError("the observability component must use schema version = 3")

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
        switchyard_commit: str = _DEFAULT_SWITCHYARD_COMMIT,
        artifact_root: str = "/logs/agent/direct-hermes",
        inject_post_response_failure: bool = False,
        **kwargs: Any,
    ) -> None:
        self.repository_url = _require_public_https_git_url(repository_url)
        self.repository_ref = repository_ref.strip()
        if not self.repository_ref or self.repository_ref.startswith("-"):
            raise ValueError("repository_ref must be a non-option branch or tag name")
        self.commit = _require_full_sha(commit, "commit")
        self.switchyard_commit = _require_full_sha(switchyard_commit, "switchyard_commit")
        self.relay_wheel_sha256 = _require_sha256(relay_wheel_sha256, "relay_wheel_sha256")

        self.relay_config_path = Path(relay_config_path).expanduser().resolve()
        self.switchyard_bundle_dir = Path(switchyard_bundle_dir).expanduser().resolve()
        self.relay_wheel_path = Path(relay_wheel_path).expanduser().resolve()
        self.artifact_root = artifact_root.rstrip("/")
        self.inject_post_response_failure = inject_post_response_failure
        if not self.artifact_root.startswith("/logs/agent/"):
            raise ValueError("artifact_root must be an absolute child of /logs/agent")
        if not self.relay_config_path.is_file():
            raise FileNotFoundError(self.relay_config_path)
        if not self.relay_wheel_path.is_file():
            raise FileNotFoundError(self.relay_wheel_path)
        if _sha256(self.relay_wheel_path) != self.relay_wheel_sha256:
            raise ValueError("Relay wheel digest does not match relay_wheel_sha256")
        if "manylinux" not in self.relay_wheel_path.name or "x86_64" not in self.relay_wheel_path.name:
            raise ValueError("Relay wheel must target Linux x86_64")
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

        self._example_root = Path(__file__).resolve().parents[1]
        self._finalizer_path = self._example_root / "scripts" / "finalize_artifacts.py"
        if not self._finalizer_path.is_file():
            raise FileNotFoundError(self._finalizer_path)

        extra_env = dict(kwargs.pop("extra_env", None) or {})
        extra_env["HERMES_NEMO_RELAY_PLUGINS_TOML"] = "/tmp/hermes/relay/plugins.toml"
        super().__init__(*args, version=self.commit, extra_env=extra_env, **kwargs)

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && "
                "apt-get install -y --no-install-recommends "
                "ca-certificates build-essential curl git ripgrep xz-utils"
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
                f"HERMES_HOME=/tmp/hermes HERMES_INSTALL_DIR={install_dir} "
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
                "assert m.version('nemo-relay') == '0.7.0'\""
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
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                f"test \"$(sha256sum {shlex.quote(relay_wheel)} | cut -d' ' -f1)\" = "
                f"{shlex.quote(self.relay_wheel_sha256)}; "
                "/tmp/hermes/bin/uv pip install "
                "--python /tmp/hermes-agent-src/venv/bin/python "
                f"--force-reinstall --no-deps {shlex.quote(relay_wheel)}; "
                "/tmp/hermes-agent-src/venv/bin/python -c "
                "\"import importlib.metadata as m; assert m.version('nemo-relay') == '0.7.0'\""
            ),
            timeout_sec=120,
        )
        await environment.upload_dir(self.switchyard_bundle_dir, "/opt/relay-plugins/nvidia.switchyard")
        await environment.upload_file(self._finalizer_path, "/installed-agent/finalize_artifacts.py")
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
            "/tmp/hermes-agent-src/venv/bin/python",
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
            await super().run(instruction, environment, context)
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
