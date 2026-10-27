# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a real SDK or Hermes support-triage scenario against one OTLP destination."""

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlencode, urlsplit


def langsmith_headers(environment):
    """Choose one addressing mode; LangSmith rejects mixed or partial modes."""
    agent = environment.get("LANGSMITH_AGENT_ID")
    stage = environment.get("LANGSMITH_AGENT_ENVIRONMENT")
    project = environment.get("LANGSMITH_PROJECT")
    if agent or stage:
        if not agent or not stage:
            raise ValueError("Set both LANGSMITH_AGENT_ID and LANGSMITH_AGENT_ENVIRONMENT")
        if project:
            raise ValueError("Choose LangSmith agent/environment or LANGSMITH_PROJECT, not both")
        if not re.fullmatch(r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", agent):
            raise ValueError("LANGSMITH_AGENT_ID must be a 1–63 character lowercase agent identifier")
        if stage not in {"local", "development", "staging", "production"}:
            raise ValueError("LANGSMITH_AGENT_ENVIRONMENT must be local, development, staging, or production")
        return {"Langsmith-Agent-Id": agent, "Langsmith-Agent-Environment": stage}
    return {"Langsmith-Project": project or "relay-support-triage"}


def validate_endpoint(endpoint, allow_insecure=False):
    url = urlsplit(endpoint)
    if not url.hostname or url.username or url.password or url.fragment or url.query:
        raise ValueError("Use a complete OTLP trace URL without embedded credentials, query, or fragment")
    if url.scheme != "https" and not (
        url.scheme == "http" and (url.hostname in {"localhost", "127.0.0.1", "::1"} or allow_insecure)
    ):
        raise ValueError("Use HTTPS, or explicitly allow trusted private-network HTTP with --allow-insecure-endpoint")


def check_events(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    scopes = {event.get("uuid"): event for event in events if event.get("kind") == "scope"}
    starts = {
        event.get("uuid"): event
        for event in events
        if event.get("kind") == "scope" and event.get("scope_category") == "start"
    }
    scopes = {
        key: event
        for key, event in scopes.items()
        if key in starts
        and event.get("scope_category") == "end"
        and all(event.get(field) == starts[key].get(field) for field in ("name", "category", "parent_uuid"))
    }
    categories = {event.get("category") for event in scopes.values()}
    if not {"agent", "llm", "tool"} <= categories:
        raise ValueError("Local capture needs agent, model, and tool scopes; inspect hermes doctor and events.jsonl")
    # Verify ancestry, rather than accepting unrelated model/tool events in the file.
    for category in ("llm", "tool"):
        correlated = False
        for event in scopes.values():
            if event.get("category") != category:
                continue
            parent = event.get("parent_uuid")
            visited = set()
            while parent in scopes and parent not in visited:
                visited.add(parent)
                if scopes[parent].get("category") == "agent":
                    correlated = True
                    break
                parent = scopes[parent].get("parent_uuid")
        if not correlated:
            raise ValueError(f"No {category} scope connects to an agent scope; inspect events.jsonl")
    return events


def run_hermes(command, directory, environment, timeout):
    with (directory / "agent.log").open("w") as log:
        # Linux/macOS process groups ensure timeout also stops harness children.
        process = subprocess.Popen(
            command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            status = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise ValueError("Hermes stopped before completion; inspect agent.log and run hermes doctor") from None
    if status:
        raise ValueError(f"Hermes exited {status}; inspect agent.log and NeMo Relay Plugins in hermes doctor")


def verify_phoenix(events, directory):
    """Reconcile this run's closed scopes with delivered spans, not a fixed count."""
    closed = {e["uuid"]: e for e in events if e.get("kind") == "scope" and e.get("scope_category") == "end"}
    agent = next(e for e in closed.values() if e.get("category") == "agent")
    trace_id = agent["uuid"].replace("-", "")
    endpoint = urlsplit(os.environ.get("RELAY_OTLP_ENDPOINT", "http://127.0.0.1:6006/v1/traces"))
    base = os.environ.get("PHOENIX_API_URL", endpoint.scheme + "://" + endpoint.netloc).rstrip("/")
    validate_endpoint(base)
    url = base + "/v1/projects/relay-support-triage/spans?" + urlencode({"trace_id": trace_id, "limit": 1000})
    headers = {}
    if os.environ.get("PHOENIX_AUTHORIZATION"):
        headers["Authorization"] = os.environ["PHOENIX_AUTHORIZATION"]
    expected = {key for key, e in closed.items() if e.get("category") in {"agent", "llm", "tool"}}
    failure = "No matching trace"
    for _ in range(15):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as response:
                spans = json.load(response)["data"]
            delivered = {s.get("attributes", {}).get("nemo_relay.uuid") for s in spans}
            ids = {s["context"]["span_id"] for s in spans}
            missing_parents = {s["parent_id"] for s in spans if s.get("parent_id") and s["parent_id"] not in ids}
            if expected <= delivered and not missing_parents:
                if "[REDACTED]" in json.dumps(events):
                    payload = json.dumps(spans)
                    if "sample.employee@example.com" in payload or "[REDACTED]" not in payload:
                        raise ValueError("Phoenix payloads did not establish synthetic email redaction")
                evidence = {
                    "trace_id": trace_id,
                    "delivered_spans": len(spans),
                    "matched_local_scopes": len(expected),
                    "missing_parents": [],
                }
                (directory / "delivery.json").write_text(json.dumps(evidence, indent=2) + "\n")
                print("Phoenix delivery verified: " + json.dumps(evidence))
                return
            failure = f"{len(expected - delivered)} local scopes absent; {len(missing_parents)} parents absent"
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise ValueError(f"Phoenix API returned HTTP {error.code}; check authorization and API URL") from None
        time.sleep(1)
    raise ValueError("Phoenix delivery check failed: " + failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", choices=["phoenix", "langsmith", "weave", "datadog"], required=True)
    parser.add_argument("--harness", choices=["sdk", "hermes"], default="sdk")
    parser.add_argument("--verify", action="store_true", help="Query Phoenix for this run's trace")
    parser.add_argument("--redaction-demo", action="store_true", help="Enable synthetic payloads and email redaction")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this run")
    parser.add_argument("--prepare-only", action="store_true", help="Write configuration without starting Hermes")
    parser.add_argument(
        "--allow-insecure-endpoint", action="store_true", help="Allow HTTP to a trusted private collector"
    )
    parser.add_argument("--timeout", type=int, default=600, help="Maximum Hermes run time in seconds")
    args = parser.parse_args()
    if args.verify and args.destination != "phoenix":
        parser.error("--verify is supported only for Phoenix; inspect other destinations in their UI")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    directory = args.output.resolve()
    endpoint = os.environ.get("RELAY_OTLP_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
    if args.destination != "phoenix" and not os.environ.get("RELAY_OTLP_ENDPOINT"):
        parser.error("Set RELAY_OTLP_ENDPOINT to the destination's complete trace URL")
    try:
        validate_endpoint(endpoint, args.allow_insecure_endpoint)
    except ValueError as error:
        parser.error(str(error))
    source = Path(__file__).resolve().parent
    config = (source / "plugins.toml").read_text()
    for placeholder, value in {
        "__LOG_DIRECTORY__": str(directory / "logs"),
        "__PROJECTION__": "gen_ai" if args.destination == "weave" else "openinference",
        "__TRACE_ENDPOINT__": endpoint,
    }.items():
        config = config.replace(json.dumps(placeholder), json.dumps(value))
    required = []
    if args.destination == "phoenix":
        config += "\n[components.config.opentelemetry.endpoints.resource_attributes]\n"
        config += '"openinference.project.name" = "relay-support-triage"\n'
    if args.destination == "phoenix" and os.environ.get("PHOENIX_AUTHORIZATION"):
        config += '\n[components.config.opentelemetry.endpoints.header_env]\nauthorization = "PHOENIX_AUTHORIZATION"\n'
    elif args.destination == "langsmith":
        required = ["LANGSMITH_API_KEY"]
        if not args.prepare_only and not any(
            os.environ.get(name) for name in ("LANGSMITH_PROJECT", "LANGSMITH_AGENT_ID", "LANGSMITH_AGENT_ENVIRONMENT")
        ):
            parser.error("Set LANGSMITH_PROJECT, or LANGSMITH_AGENT_ID and LANGSMITH_AGENT_ENVIRONMENT")
        config += '\n[components.config.opentelemetry.endpoints.header_env]\n"x-api-key" = "LANGSMITH_API_KEY"\n'
        config += "\n[components.config.opentelemetry.endpoints.headers]\n"
        for key, value in langsmith_headers(os.environ).items():
            config += json.dumps(key) + " = " + json.dumps(value) + "\n"
    elif args.destination == "weave":
        required = ["WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"]
        config += '\n[components.config.opentelemetry.endpoints.header_env]\n"wandb-api-key" = "WANDB_API_KEY"\n'
        config += "\n[components.config.opentelemetry.endpoints.resource_attributes]\n"
        config += '"wandb.entity" = ' + json.dumps(os.environ.get("WANDB_ENTITY", "")) + "\n"
        config += '"wandb.project" = ' + json.dumps(os.environ.get("WANDB_PROJECT", "")) + "\n"
    # Datadog credentials belong to the receiving Collector, not this process.
    if not args.prepare_only:
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            parser.error("Set " + ", ".join(missing))
        if args.harness == "hermes" and not shutil.which("hermes"):
            parser.error("Install and authenticate Hermes with native Relay support first")
        if args.harness == "sdk":
            missing = [key for key in ("NVIDIA_API_KEY", "NVIDIA_MODEL") if not os.environ.get(key)]
            if missing:
                parser.error("Set " + ", ".join(missing))
            validate_endpoint(os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"))
    if args.redaction_demo:
        config = config.replace("enable_full_payloads = false", "enable_full_payloads = true")
        config += """
[[components]]
kind = "pii_redaction"
enabled = true
[components.config]
version = 1
[[components.config.profiles]]
mode = "builtin"
priority = 80
[components.config.profiles.builtin]
action = "redact"
detector = "email"
"""
    tomllib.loads(config)
    try:
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    except FileExistsError:
        parser.error("Output directory exists: choose a new --output; previous runs are preserved")
    (directory / "plugins.toml").write_text(config)
    shutil.copyfile(source / "requests.csv", directory / "requests.csv")
    print(f"Example files: {directory}")
    if args.prepare_only:
        return
    environment = os.environ.copy()
    environment["HERMES_NEMO_RELAY_PLUGINS_TOML"] = str(directory / "plugins.toml")
    if args.harness == "sdk":
        from sdk_workload import run

        asyncio.run(run(directory, environment, args.timeout))
    else:
        run_hermes(
            [
                "hermes",
                "chat",
                "--oneshot",
                "-q",
                (
                    "Read requests.csv using a tool. Prioritize the incidents by customer impact and create "
                    "handoff.md with a short action for each request ID. "
                    "Use only the supplied synthetic data and local file tools; do not contact outside services."
                ),
            ],
            directory,
            environment,
            args.timeout,
        )
    handoff = (directory / "handoff.md").read_text()
    ids = ["INC-1042", "INC-1043", "INC-1044"]
    if any(item not in handoff for item in ids) or handoff.index(ids[0]) > handoff.index(ids[1]):
        raise SystemExit("Handoff needs every incident, with checkout outage before report delay; inspect handoff.md")
    events = check_events(directory / "logs/events.jsonl")
    if args.redaction_demo:
        payload = json.dumps(events)
        if "sample.employee@example.com" in payload or "[REDACTED]" not in payload:
            raise ValueError("Synthetic email redaction was not established in local events")
    if args.verify:
        verify_phoenix(events, directory)
    print("Handoff and correlated local agent/model/tool scopes found. Verify delivery in your destination UI.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Example failed: {error}. Inspect the run directory; use a new --output to retry.") from None
