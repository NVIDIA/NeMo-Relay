# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare fixed and routed pi coding tasks through Relay's Switchyard plugin."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from summarize import summarize

TASKS = {
    "ticket-id": (
        "Implement normalize_ticket_id(value) in service.py. Strip surrounding whitespace, accept "
        "case-insensitive INC- followed by exactly four ASCII digits, return the uppercase canonical "
        "ID, and raise ValueError for any other input including non-string values. Write and run "
        "your own tests, including full-width INC-１２３４, Arabic-digit INC-١٢٣٤, and non-string "
        "inputs. The prefix letters must be ASCII I/N/C in either case; reject dotless-i ınc-1234 "
        "and dotted-I İNC-1234. Do not transliterate Unicode digits or letters. "
        "Leave retry_delays unchanged."
    ),
    "retry-policy": (
        "Implement retry_delays(attempts, base, cap) in service.py. Return a list with attempts "
        "delays: min(base * 2**i, cap) for i starting at zero. attempts must be a nonnegative "
        "integer (not bool); base and cap must be positive numbers. Raise ValueError for invalid "
        "inputs. Support fractional base and cap. Write and run your own tests. "
        "Leave normalize_ticket_id unchanged."
    ),
}


def stop_task(process):
    """Stop the wrapper and detached descendants created by harness launchers."""
    try:
        listing = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid="], text=True, timeout=5)
        rows = [tuple(map(int, line.split())) for line in listing.splitlines() if line.strip()]
        owned = {process.pid}
        while True:
            descendants = {pid for pid, parent, _ in rows if parent in owned}
            expanded = owned | descendants
            if expanded == owned:
                break
            owned = expanded
        # Relay starts pi in another process group. Kill those groups too, but
        # never signal a group whose snapshot includes an unrelated process.
        groups = {group for pid, _, group in rows if pid in owned}
        for group in groups:
            members = {pid for pid, _, candidate in rows if candidate == group}
            if group != os.getpgrp() and members <= owned:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for pid in owned:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    except (OSError, ValueError, subprocess.SubprocessError):
        # Stop the wrapper, then stop the comparison: starting another task
        # without confirming descendant cleanup would contaminate its timing.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise ValueError(
            "Task cleanup could not enumerate/stop descendants; check ps permissions and task processes"
        ) from None
    process.wait()


def run_task(command, *, cwd, env, log, timeout):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        stop_task(process)
        if isinstance(error, KeyboardInterrupt):
            raise
        return "timeout"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never modifies your pi setup")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds allowed for each task")
    parser.add_argument("--repeat", type=int, default=1, help="Fresh trials of each task in each arm")
    parser.add_argument(
        "--max-output-tokens", type=int, default=16384, help="Shared per-call output budget, including model reasoning"
    )
    args = parser.parse_args()
    if not 1 <= args.max_output_tokens <= 16384:
        parser.error("--max-output-tokens must be between 1 and 16384 for this 32768-token example")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    names = ["NVIDIA_BASE_URL", "EFFICIENT_MODEL", "CAPABLE_MODEL", "SWITCHYARD_MANIFEST"]
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        parser.error("Set " + ", ".join(missing))
    if not args.prepare_only and not os.environ.get("NVIDIA_API_KEY"):
        parser.error("Set NVIDIA_API_KEY in the launching environment")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not args.prepare_only and any(not shutil.which(name) for name in ["nemo-relay", "pi"]):
        parser.error("Install the Relay CLI and pi versions used by the pi quickstart first")
    endpoint = urlsplit(os.environ["NVIDIA_BASE_URL"])
    if (
        endpoint.scheme != "https"
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        parser.error("NVIDIA_BASE_URL must be HTTPS without embedded credentials, query, or fragment")
    if not args.prepare_only and not Path(os.environ["SWITCHYARD_MANIFEST"]).is_file():
        parser.error("SWITCHYARD_MANIFEST must name an installed relay-plugin.toml; follow the installation section")
    source = Path(__file__).resolve().parent
    output = args.output.resolve()
    routes = (source / "routes.toml").read_text()
    for name in ["NVIDIA_BASE_URL", "EFFICIENT_MODEL", "CAPABLE_MODEL", "JUDGE_MODEL"]:
        value = os.environ.get(name) or os.environ["EFFICIENT_MODEL"]
        routes = routes.replace(json.dumps(f"__{name}__"), json.dumps(value))
    tomllib.loads(routes)
    try:
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        parser.error("Output directory exists: choose a new --output; previous results are preserved")
    (output / "routes.toml").write_text(routes)
    environment = os.environ.copy()
    environment["PI_CODING_AGENT_DIR"] = str(output / "pi")
    environment["XDG_CONFIG_HOME"] = str(output / "config")
    (output / "pi").mkdir()
    # A custom provider keeps the virtual route IDs out of pi's NVIDIA model catalog.
    provider = {
        "baseUrl": os.environ["NVIDIA_BASE_URL"],
        "api": "openai-completions",
        "apiKey": "relay-route-placeholder",
        "compat": {"supportsDeveloperRole": False},
        "models": [
            {
                "id": f"switchyard/{arm}",
                "name": arm,
                "reasoning": False,
                "input": ["text"],
                "contextWindow": 32768,
                "maxTokens": args.max_output_tokens,
            }
            for arm in ["baseline", "efficient", "routed"]
        ],
    }
    (output / "pi/models.json").write_text(json.dumps({"providers": {"relay-nvidia": provider}}, indent=2) + "\n")
    results = []
    for arm in ["baseline", "efficient", "routed"]:
        arm_dir = output / arm
        arm_dir.mkdir()
        # --config discovers plugins.toml beside this file; keep the pair together.
        (arm_dir / "config.toml").write_text(
            '[agents.pi]\ncommand = "pi"\n\n[upstream]\nopenai_base_url = '
            + json.dumps(os.environ["NVIDIA_BASE_URL"])
            + "\n"
        )
        plugins = f"""version = 1

[[components]]
kind = "observability"
enabled = true
[components.config]
version = 4
enable_full_payloads = false
[components.config.atof]
enabled = true
[[components.config.atof.sinks]]
type = "file"
output_directory = {json.dumps(str(arm_dir / "logs"))}
filename = "events.jsonl"
mode = "append"
[components.config.opentelemetry]
enabled = false

[[plugins.dynamic]]
manifest = {json.dumps(str(Path(os.environ["SWITCHYARD_MANIFEST"]).resolve()))}
[plugins.dynamic.config]
priority = 0
switchyard_config_path = {json.dumps(str(output / "routes.toml"))}
[plugins.policy.overrides."nvidia.switchyard"]
attestation = "integrity_only"
"""
        tomllib.loads(plugins)
        (arm_dir / "plugins.toml").write_text(plugins)
        for trial in range(1, args.repeat + 1):
            for task in TASKS:
                task_dir = arm_dir / f"{task}-{trial}"
                task_dir.mkdir()
                shutil.copyfile(source / "service.py", task_dir / "service.py")
    print(f"Prepared example: {output}")
    if args.prepare_only:
        return
    subprocess.run(["nemo-relay", "install", "pi"], env=environment, check=True)
    arms = ["baseline", "efficient", "routed"]
    for arm in arms:
        command = ["nemo-relay", "--config", str(output / arm / "config.toml")]
        # Dynamic manifests are added disabled. Explicit lifecycle enablement is required.
        subprocess.run(command + ["plugins", "enable", "nvidia.switchyard"], env=environment, check=True)
        subprocess.run(command + ["plugins", "validate", "nvidia.switchyard"], env=environment, check=True)
    # Manifest validation does not instantiate the native plugin's routes.
    # Start the same gateway once without an inference task to catch route/client
    # conflicts before repeating identical startup failures across every arm.
    with (output / "startup.log").open("w") as log:
        try:
            startup = subprocess.run(
                ["nemo-relay", "--config", str(output / "baseline/config.toml"), "pi", "--", "--version"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            raise ValueError("Routing startup timed out; inspect startup.log before rerunning") from None
    if startup.returncode:
        raise ValueError("Routing startup failed; inspect startup.log before running task comparisons")
    for trial in range(1, args.repeat + 1):
        order = arms[(trial - 1) % 3 :] + arms[: (trial - 1) % 3]
        for task, prompt in TASKS.items():
            for arm in order:
                arm_dir = output / arm
                task_dir = arm_dir / f"{task}-{trial}"
                command = ["nemo-relay", "--config", str(arm_dir / "config.toml")]
                started = time.monotonic()
                with (task_dir / "agent.log").open("w") as log:
                    status = run_task(
                        command
                        + [
                            "pi",
                            "--",
                            "--provider",
                            "relay-nvidia",
                            "--model",
                            f"switchyard/{arm}",
                            "--no-session",
                            "--mode",
                            "json",
                            "--tools",
                            "read,write,bash",
                            "-p",
                            prompt + " Read service.py, implement the change with the available file tools, "
                            "and run your tests. Use only local files and tools in this directory. "
                            "Finish with a concise status.",
                        ],
                        cwd=task_dir,
                        env=environment,
                        log=log,
                        timeout=args.timeout,
                    )
                seconds = round(time.monotonic() - started, 3)
                try:
                    check = subprocess.run(
                        [sys.executable, str(source / "acceptance.py"), str(task_dir), task],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    accepted = check.returncode == 0
                    (task_dir / "acceptance.log").write_text(check.stdout + check.stderr)
                except subprocess.TimeoutExpired:
                    accepted = False
                    (task_dir / "acceptance.log").write_text("Acceptance check timed out\n")
                results.append(
                    {
                        "arm": arm,
                        "task": task,
                        "trial": trial,
                        "agent_status": status,
                        "accepted": status == 0 and accepted,
                        "seconds": seconds,
                    }
                )
                (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    summary = summarize(output)
    if any(not row["accepted"] for row in results):
        raise SystemExit("Some tasks failed; inspect results.json and each task's agent.log and acceptance.log")
    if any(not row["decisions"] for row in summary):
        raise SystemExit("Routing telemetry is missing; inspect plugin validation and each arm's logs/events.jsonl")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Relay setup exited {error.returncode}; inspect plugin/install diagnostics. Retry with a new --output"
        ) from None
    except (OSError, ValueError) as error:
        raise SystemExit(f"Example failed: {error}. Inspect the run directory; use a new --output to retry.") from None
