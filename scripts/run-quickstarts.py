#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the credential-free Quick Starts locally or in CI against explicit assets."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from packaging.version import Version
from quickstarts import agents, build
from quickstarts.assets import Assets, host_platform, sha256
from quickstarts.provider import Fixture

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quickstart_contracts", ROOT / "scripts/check-quickstarts.py")
assert SPEC is not None and SPEC.loader is not None
CONTRACTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACTS)
SURFACES = {
    **dict.fromkeys(agents.PYTHON_CASES | {"hermes"}, "python"),
    **dict.fromkeys({"codex", "claude-code", "pi"}, "cli"),
    "nodejs": "nodejs",
    "rust": "rust",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def inventory() -> dict[str, Any]:
    """The contracts own the case list; adapters explicitly declare executable coverage."""
    return {
        page.stem: CONTRACTS.extract(page, "bash")
        for page in sorted(CONTRACTS.PAGES.glob("*.mdx"))
        if page.stem != "index"
    }


def select_cases(cases: str, scenarios: dict[str, Any]) -> list[str]:
    if cases == "local":
        selected = [name for name, scenario in scenarios.items() if scenario["mode"] == "local"]
    elif cases == "all":
        selected = list(scenarios)
    else:
        selected = list(dict.fromkeys(cases.split(",")))
    unknown = set(selected) - set(scenarios)
    unsupported = set(selected) - set(SURFACES)
    if unknown:
        raise ValueError(f"Unknown Quick Starts: {', '.join(sorted(unknown))}")
    if unsupported:
        raise ValueError(
            f"No unattended adapter yet for: {', '.join(sorted(unsupported))}. "
            "Select 'local' or supported cases explicitly."
        )
    if not selected:
        raise ValueError("No Quick Starts selected")
    return selected


def execute(command: str, cwd: Path, environment: dict[str, str], stdout: Path, stderr: Path, timeout: int) -> None:
    """Own the whole process group so an expired install cannot leave children running."""
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=cwd,
            env=environment,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # A child can outlive its shell after TERM; KILL the remaining group as well.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        finally:
            # Helpers started by these commands belong to the disposable invocation.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    if code:
        raise ValueError(f"Command exited {code}; see {stderr}")


def install_command(block: dict[str, Any], name: str, assets: Assets, resolved: dict[str, Any]) -> str:
    """Only installation inputs change; the displayed programs and checks run verbatim."""
    content = block["content"]
    if name in agents.PYTHON_CASES:
        pattern = r"uv add [\"\']?(nemo-relay(?:\[[^\"\']+\])?)==[^\s\"\']+[\"\']?"
        content, count = re.subn(
            pattern, lambda match: "uv add " + shlex.quote(match[1] + " @ " + resolved["python"].as_uri()), content
        )
    elif name == "nodejs":
        command = "npm install --save-exact --ignore-scripts --omit=optional " + " ".join(
            shlex.quote(str(path)) for path in resolved["nodejs"]
        )
        content, count = re.subn(r"npm install --save-exact nemo-relay-node@\S+", lambda _: command, content)
    else:
        patches = resolved["rust"]
        package = (
            "cargo add nemo-relay --path " + shlex.quote(patches["nemo-relay"])
            if patches
            else "cargo add " + shlex.quote("nemo-relay@=" + assets.version)
        )
        content, count = re.subn(r'cargo add "nemo-relay@=[^\"]+"', lambda _: package, content)
    if count != 1:
        raise ValueError(f"Install block for {name} changed; review its asset substitution")
    return content


def run_case(
    name: str,
    scenario: dict[str, Any],
    assets: Assets,
    resolved: dict[str, Any],
    folder: Path,
    hermes_ref: str = agents.HERMES_COMMIT,
) -> dict[str, Any]:
    folder.mkdir()
    # uv discovers enclosing workspaces, so examples must execute outside this checkout.
    # Copy only evidence back; dependency environments and exported shell state are disposable.
    temporary = tempfile.TemporaryDirectory(prefix="relay-quickstart-")
    work = Path(temporary.name)
    result: dict[str, Any] = {
        "case": name,
        "status": "failed",
        "commands": [],
        "samples": {},
        "version": assets.version,
        "block": "host-setup",
    }
    started = time.monotonic()
    # Keep host credentials/configuration out of example discovery without changing the user's HOME.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            (
                "NEMO_RELAY_",
                "PYTHON",
                "UV_PROJECT",
                "VIRTUAL_ENV",
                "CODEX_",
                "CLAUDE_",
                "ANTHROPIC_",
                "OPENAI_",
                "NVIDIA_",
                "HERMES_",
                "PI_CODING_AGENT_",
            )
        )
    }
    environment.update(
        UV_PYTHON=sys.executable,
        UV_PYTHON_DOWNLOADS="never",
        XDG_CONFIG_HOME=str(work / ".config"),
        XDG_DATA_HOME=str(work / ".data"),
        npm_config_cache=str(work / ".npm"),
        npm_config_userconfig=str(work / ".npmrc"),
    )
    fixture = Fixture(work / scenario["cwd"], folder / "provider.jsonl") if scenario.get("mode") == "live" else None
    try:
        if fixture:
            fixture.start()
            result["host"] = agents.prepare(name, work, folder, environment, resolved, fixture, execute, hermes_ref)
        if name == "rust" and resolved["rust"]:
            config = work / ".cargo/config.toml"
            config.parent.mkdir()
            config.write_text(
                "[patch.crates-io]\n"
                + "".join(
                    f"{json.dumps(package)} = {{ path = {json.dumps(path)} }}\n"
                    for package, path in resolved["rust"].items()
                )
            )
        for path in scenario["fresh"]:
            if CONTRACTS.artifact_path(path, scenario, work, {}).exists():
                raise ValueError(f"Stale artifact: {path}")
        outputs = {}
        for block in scenario["blocks"]:
            result["block"] = block["id"]
            cwd = work / block["cwd"]
            if block["role"] == "file":
                path = cwd / block["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                content = block["content"]
                if fixture and block["id"] == "config":
                    content = agents.gateway_config(content, fixture)
                path.write_text(content, encoding="utf-8")
                result["samples"][str(path.relative_to(work))] = sha256(path)
            elif block["role"] in {"run", "manual", "input"}:
                if "background" in block:
                    raise ValueError("Background blocks require a dedicated unattended adapter")
                content, adaptation = (
                    agents.command(block, name, work, fixture) if fixture else (block["content"], "verbatim")
                )
                if block["role"] != "run" and adaptation == "verbatim":
                    raise ValueError(f"Unattended setup is missing for {block['id']} ({block['role']})")
                if (
                    content is not None
                    and block["id"] == "install"
                    and name in agents.PYTHON_CASES | {"nodejs", "rust"}
                ):
                    content = install_command(block, name, assets, resolved)
                    adaptation = "selected-assets"
                result["commands"].append(
                    {
                        "id": block["id"],
                        "cwd": block["cwd"],
                        "original": block["content"],
                        "executed": content,
                        "adaptation": adaptation,
                    }
                )
                if content is None:
                    continue
                # Shell exports persist across blocks in the same declared session.
                state = work / (".session-" + str(len(outputs)) + ".sh")
                previous = outputs.get("session:" + block["session"])
                prefix = "source " + shlex.quote(str(previous)) + "\n" if previous else ""
                command = (
                    prefix + "cd " + shlex.quote(str(cwd)) + "\n" + content + "\nexport -p > " + shlex.quote(str(state))
                )
                if adaptation == "diagnostic":
                    # Preserve session exports even when optional doctor checks fail.
                    command = (
                        prefix
                        + "cd "
                        + shlex.quote(str(cwd))
                        + "\nset +e\n"
                        + content
                        + "\nquickstart_doctor_status=$?\nexport -p > "
                        + shlex.quote(str(state))
                        + '\nexit "$quickstart_doctor_status"'
                    )
                out, err = folder / (block["id"] + ".stdout.log"), folder / (block["id"] + ".stderr.log")
                try:
                    execute(command, cwd, environment, out, err, block.get("timeout_seconds", 180))
                except ValueError:
                    if adaptation != "diagnostic":
                        raise
                    result["doctor_exit_nonzero"] = True
                outputs[block["id"]] = out
                outputs["session:" + block["session"]] = state
            elif block["role"] not in {"expected"}:
                raise ValueError(f"Unattended setup is missing for {block['id']} ({block['role']})")
        result["block"] = "verify"
        expected = next(block for block in scenario["blocks"] if block["role"] == "expected")
        CONTRACTS.verify(scenario, work, outputs[expected["from"]].read_text(), {})
        if fixture:
            fixture.verify()
            if "RELAY_QUICKSTART_RESPONSE_OK" not in outputs["run"].read_text():
                raise ValueError("The agent did not print the final provider response marker")
        project = work / scenario["cwd"]
        if name in agents.PYTHON_CASES | {"hermes"}:
            python = work / "hermes-venv/bin/python" if name == "hermes" else project / ".venv/bin/python"
            command = [str(python), "-c", "import importlib.metadata as m; print(m.version('nemo-relay'))"]
            installed = subprocess.check_output(command, text=True, timeout=30).strip()
            if Version(installed) != Version(assets.version):
                raise ValueError(f"Installed Python version {installed} differs from requested {assets.version}")
            dependencies = subprocess.check_output(
                ["uv", "pip", "list", "--python", str(python), "--format", "json"], text=True, timeout=60
            )
            write_json(folder / "dependencies.json", json.loads(dependencies))
        elif name == "nodejs":
            installed = json.loads((project / "node_modules/nemo-relay-node/package.json").read_text())["version"]
            if installed != assets.version:
                raise ValueError("Installed Node package version differs from the requested version")
        elif name == "rust":
            lock = project / "Cargo.lock"
            import tomllib

            packages = tomllib.loads(lock.read_text())["package"]
            relay = next(package for package in packages if package["name"] == "nemo-relay")
            if relay["version"] != assets.version:
                raise ValueError("Installed Rust crate version differs from the requested version")
            assets.records["Cargo.lock"] = {
                "location": str(folder / "workspace" / scenario["cwd"] / "Cargo.lock"),
                "sha256": sha256(lock),
            }
        result.update(status="passed", block="complete")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        result["error"] = str(error)
    finally:
        if name == "hermes" and environment.get("HERMES_HOME") == str(work / ".hermes"):
            # Recent Hermes chat commands start a detached, profile-scoped gateway.
            # Stop only this disposable profile before removing its runtime and state.
            try:
                execute(
                    "hermes gateway stop",
                    work,
                    environment,
                    folder / "cleanup.stdout.log",
                    folder / "cleanup.stderr.log",
                    60,
                )
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                result.update(status="failed", error=f"Hermes gateway cleanup failed: {error}")
        if fixture:
            fixture.close()
            result["provider"] = {
                "requests": fixture.requests,
                "tool_calls": fixture.tool_calls,
                "completions": fixture.completions,
                "diagnostics": fixture.diagnostics,
                "errors": fixture.errors,
            }
        evidence = set(result["samples"])
        evidence.update(str(Path(scenario["cwd"]) / check["path"]) for check in scenario["checks"])
        evidence.update(str(Path(scenario["cwd"]) / name) for name in ("uv.lock", "package-lock.json", "Cargo.lock"))
        for relative in evidence:
            path = work / relative
            if path.is_file() and path.resolve().is_relative_to(work.resolve()):
                destination = folder / "workspace" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                if relative in result["samples"]:
                    result.setdefault("final_samples", {})[relative] = sha256(destination)
        try:
            temporary.cleanup()
        except OSError as error:
            result.update(status="failed", error=f"Temporary workspace cleanup failed: {error}")
        result["seconds"] = round(time.monotonic() - started, 3)
        write_json(folder / "result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", help="Exact Relay SemVer, including prerelease/build suffix if applicable")
    parser.add_argument(
        "--source-ref", default="", help="Build assets and read guides from this local Git ref, pinned to its commit"
    )
    parser.add_argument("--hermes-ref", default=agents.HERMES_COMMIT, help="Immutable Hermes source commit")
    parser.add_argument(
        "--asset-location",
        default="",
        help="Local directory, GitHub release URL, or HTTPS directory containing SHA256SUMS",
    )
    parser.add_argument(
        "--repository", default="NVIDIA/NeMo-Relay", help="GitHub repository for release or build assets"
    )
    parser.add_argument("--run-id", default="", help="Completed GitHub Actions build to download instead of a release")
    parser.add_argument(
        "--overrides", type=Path, help="Optional local Python/npm asset paths and rust_source directory"
    )
    parser.add_argument(
        "--cases", default="all", help="all, local (four provider-free cases), or comma-separated case IDs"
    )
    parser.add_argument(
        "--output", type=Path, help="New, empty result directory; defaults below ignored artifacts/quickstarts"
    )
    parser.add_argument("--list", action="store_true", help="Show discovered cases and unattended adapter coverage")
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                {name: {"mode": case["mode"], "supported": name in SURFACES} for name, case in inventory().items()},
                indent=2,
            )
        )
        return 0
    if not args.version and not args.source_ref:
        parser.error("--version or --source-ref is required for execution")
    if args.source_ref and (args.asset_location or args.run_id or args.overrides):
        parser.error("--source-ref builds all selected assets; do not combine it with external assets or overrides")
    output = (args.output or ROOT / "artifacts/quickstarts" / str(uuid4())).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be a new or empty directory; previous artifacts cannot satisfy a new run")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": 1, "status": "failed", "requested_version": args.version, "results": []}
    assets = None
    try:
        scenarios = inventory()
        selected = select_cases(args.cases, scenarios)
        source = None
        if args.source_ref:
            source, version, provenance = build.prepare(
                ROOT, args.source_ref, output, {SURFACES[name] for name in selected}, execute
            )
            if args.version and args.version != version:
                raise ValueError(f"Source version {version} differs from requested {args.version}")
            args.version = version
            args.asset_location = str(output / "packages")
            report["build"] = provenance
            report["requested_version"] = version
            # Run the guides belonging to the tested development commit, not stale harness copies.
            setattr(CONTRACTS, "ROOT", source)
            setattr(CONTRACTS, "PAGES", source / "docs/getting-started/quick-start")
            scenarios = inventory()
            selected = select_cases(args.cases, scenarios)
        report.update(
            platform=host_platform(),
            cases=selected,
            not_selected=[name for name in scenarios if name not in selected],
            not_supported=[name for name in scenarios if name not in SURFACES],
            source_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            checkout_dirty=bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
            checker_sha256=sha256(ROOT / "scripts/check-quickstarts.py"),
            runner_sha256=sha256(Path(__file__)),
            resolver_sha256=sha256(ROOT / "scripts/quickstarts/assets.py"),
            protocol_fixture_sha256=sha256(ROOT / "scripts/test-support/codex_mock_provider.py"),
            adapters={path.name: sha256(path) for path in (ROOT / "scripts/quickstarts").glob("*.py")},
            contracts={name: sha256(CONTRACTS.PAGES / (name + ".mdx")) for name in selected},
        )
        system_plugins = Path("/etc/nemo-relay/plugins.toml")
        if system_plugins.is_file():
            # Operator policy still applies to real assets. Record its identity, not its contents.
            report["system_plugin_config"] = {"path": str(system_plugins), "sha256": sha256(system_plugins)}
        assets = Assets(args.version, output / "downloads", args.asset_location, args.repository, args.run_id)
        if source:
            assets.source["build"] = report["build"]
        rust_source = (
            str(source / "crates/core")
            if source
            else (assets.override(json.loads(args.overrides.read_text())) if args.overrides else "")
        )
        surfaces = {SURFACES[name] for name in selected}
        resolved = {}
        if "python" in surfaces:
            resolved["python"] = assets.python()
        if "nodejs" in surfaces:
            resolved["nodejs"] = assets.nodejs()
        if "rust" in surfaces:
            resolved["rust"] = assets.rust(rust_source)
        if "cli" in surfaces:
            resolved["cli"] = assets.cli()
        for name in selected:
            print(f"Running {name} against Relay {args.version}...", flush=True)
            result = run_case(name, scenarios[name], assets, resolved, output / name, args.hermes_ref)
            report["results"].append(result)
            print(f"{name}: {result['status']}" + (" — " + result["error"] if "error" in result else ""), flush=True)
        report["status"] = "passed" if all(result["status"] == "passed" for result in report["results"]) else "failed"
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(f"Quick Start execution failed: {error}", file=sys.stderr)
    finally:
        if (output / "build.json").is_file():
            report["build"] = json.loads((output / "build.json").read_text())
        if assets is not None:
            write_json(output / "resolved-assets.json", {"source": assets.source, "files": assets.records})
        else:
            write_json(
                output / "resolved-assets.json", {"requested_version": args.version, "error": report.get("error")}
            )
        write_json(output / "results.json", report)
        print(f"Results: {output / 'results.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":

    def interrupted(_signal: int, _frame: Any) -> None:
        raise InterruptedError("Quick Start run interrupted")

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
