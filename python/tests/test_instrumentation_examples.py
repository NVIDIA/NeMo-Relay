# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the source displayed by the Instrument Applications guides."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

import nemo_relay
from nemo_relay.runtime_registrations import list_runtime_registrations

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GUIDES = REPOSITORY_ROOT / "docs/instrument-applications"
EXAMPLES = GUIDES / "_code/python"
EXAMPLE_NAMES = ("tool_call", "llm_call", "query_policy", "tool_policy", "print_event")
REFERENCE_NAMES = (
    "managed_llm",
    "streaming_llm",
    "manual_tool_lifecycle",
    "partial_middleware",
    "scope_context",
    "global_registration_gate",
)


def example(name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(f"documentation_{name}", EXAMPLES / f"{name}.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def documented_output(page_name: str, title: str, language: str = "text") -> str:
    content = (GUIDES / f"{page_name}.mdx").read_text()
    match = re.search(rf'^```{language} title="{re.escape(title)}"\n(.*?)^```', content, re.MULTILINE | re.DOTALL)
    assert match is not None, (page_name, title)
    return match.group(1).strip()


def assert_pair(events: list[nemo_relay.Event], category: str, name: str) -> dict:
    records = [event.to_dict() for event in events]
    matches = [record for record in records if record.get("category") == category and record["name"] == name]
    starts = [record for record in matches if record.get("scope_category") == "start"]
    ends = [record for record in matches if record.get("scope_category") == "end"]
    assert len(starts) == len(ends) == 1, matches
    assert starts[0]["uuid"] == ends[0]["uuid"]
    assert starts[0]["parent_uuid"] == ends[0]["parent_uuid"]
    return starts[0]


@pytest.fixture(autouse=True)
def clean_example_policies():
    yield
    nemo_relay.guardrails.deregister_tool_conditional_execution("search.require_query")
    nemo_relay.guardrails.deregister_tool_sanitize_request("search.redact_api_key")
    nemo_relay.intercepts.deregister_tool_execution("search.measure")


def test_displayed_sources_match_registered_examples():
    expected = set()
    for language, extension in (("python", "py"), ("nodejs", "js"), ("rust", "rs")):
        for name in EXAMPLE_NAMES:
            filename = name.replace("_", "-") if language == "nodejs" else name
            expected.add((GUIDES / "_code" / language / f"{filename}.{extension}").resolve())
        for name in (*REFERENCE_NAMES, "scopes_and_marks"):
            filename = name.replace("_", "-") if language == "nodejs" else name
            source_extension = "ts" if language == "nodejs" and name in REFERENCE_NAMES else extension
            expected.add((GUIDES / "_code" / language / f"{filename}.{source_extension}").resolve())
    expected.add((EXAMPLES / "tool_result_message.py").resolve())
    displayed = set()
    for page in GUIDES.glob("*.mdx"):
        content = page.read_text()
        assert not re.search(r"^```(?:python|js|javascript|ts|typescript|rust)\b", content, re.MULTILINE), page
        for source in re.findall(r'<Code\s+src="(_code/[^\"]+)"', content):
            displayed.add((GUIDES / source).resolve())
    assert displayed == expected
    actual = {path.resolve() for path in (GUIDES / "_code").rglob("*") if path.suffix in {".py", ".js", ".ts", ".rs"}}
    assert actual == expected


@pytest.mark.parametrize(
    "name,category,call_name", [("tool_call", "tool", "search"), ("llm_call", "llm", "demo-provider")]
)
async def test_minimal_examples_return_results_and_paired_events(name, category, call_name, subscribed_events):
    result = await example(name).main()
    await nemo_relay.subscribers.flush_async()
    if name == "tool_call":
        assert result.result == {"query": "runtime instrumentation", "hits": [{"title": "NeMo Relay"}]}
    else:
        assert result == {"text": "hello", "messages": [{"role": "user", "content": "hello"}]}
    documented = json.loads(documented_output(f"instrument-{name.replace('_', '-')}", "Expected result", "json"))
    assert (result.result if name == "tool_call" else result) == documented
    agent = assert_pair(subscribed_events, "agent", "agent-run")
    call = assert_pair(subscribed_events, category, call_name)
    assert call["parent_uuid"] == agent["uuid"]
    subscriber_name = "instrumentation-check" if name == "tool_call" else "llm-check"
    assert subscriber_name not in {registration.local_name for registration in list_runtime_registrations()}


async def test_reference_provider_and_stream_return_documented_results(subscribed_events):
    response = await example("managed_llm").run_example()
    assert response == json.loads(documented_output("code-examples", "Expected managed LLM result", "json"))
    await nemo_relay.subscribers.flush_async()
    assert_pair(subscribed_events, "llm", "demo-provider")
    subscribed_events.clear()
    assert await example("streaming_llm").run_example() == json.loads(
        documented_output("code-examples", "Expected first stream chunk", "json")
    )
    await nemo_relay.subscribers.flush_async()
    assert_pair(subscribed_events, "llm", "demo-provider")


@pytest.mark.parametrize("query", ["weather", ""])
async def test_reference_manual_call_records_completion_on_both_paths(query, subscribed_events):
    program = example("manual_tool_lifecycle")
    if query:
        assert program.run_example(query) == {"hits": 2}
    else:
        with pytest.raises(ValueError, match="^query is required$"):
            program.run_example(query)
    await nemo_relay.subscribers.flush_async()
    assert_pair(subscribed_events, "tool", "search")

    tool_events = [
        event for event in subscribed_events if event.to_dict().get("category") == "tool" and event.name == "search"
    ]
    assert [f"{event.kind} {event.name}" for event in tool_events] == documented_output(
        "code-examples", "Expected manual tool lifecycle events"
    ).splitlines()


async def test_reference_partial_middleware_does_not_execute_a_call(subscribed_events):
    tool_arguments, request = await example("partial_middleware").run_example()
    assert tool_arguments == {"query": "weather"}
    assert request.content == {"messages": [{"role": "user", "content": "hello"}]}
    await nemo_relay.subscribers.flush_async()
    assert not any(event.to_dict().get("category") in {"tool", "llm"} for event in subscribed_events)


async def test_reference_context_attaches_worker_mark_to_request(subscribed_events):
    example("scope_context").run_example()
    await nemo_relay.subscribers.flush_async()
    parent = assert_pair(subscribed_events, "agent", "request")
    worker = next(event.to_dict() for event in subscribed_events if event.name == "worker-ran")
    assert worker["parent_uuid"] == parent["uuid"]


async def test_reference_gate_reports_missing_required_exporter():
    async def run_application():
        pytest.fail("application must not run without the required subscriber")

    with pytest.raises(RuntimeError, match="OpenTelemetry subscriber was not found"):
        await example("global_registration_gate").run_example(run_application)


def test_scopes_program_prints_documented_checkpoints(capsys):
    example("scopes_and_marks").main()
    lines = capsys.readouterr().out.splitlines()
    assert lines == documented_output("adding-scopes-and-marks", "Expected output").splitlines()
    assert "scope-check" not in {registration.local_name for registration in list_runtime_registrations()}


async def test_tool_message_uses_business_payload_without_annotation():
    async def search(_arguments):
        return nemo_relay.ToolExecutionResult({"hits": 2}, annotation={"internal": "retained"})

    messages = []
    result = await example("tool_result_message").append_tool_result(
        "search", {"query": "weather"}, search, {"id": "call-1"}, messages
    )
    assert messages == [{"role": "tool", "content": '{"hits": 2}', "tool_call_id": "call-1"}]
    assert result.annotation == {"internal": "retained"}


@pytest.mark.parametrize("name", ["tool_call", "llm_call"])
def test_documented_program_entrypoints(name):
    environment = os.environ.copy()
    assert nemo_relay.__file__ is not None
    package_directory = str(Path(nemo_relay.__file__).resolve().parent.parent)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join([package_directory, existing_python_path]) if existing_python_path else package_directory
    )
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / f"{name}.py")],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "scope agent-run" in result.stdout
    assert ("NeMo Relay" if name == "tool_call" else "hello") in result.stdout
    event_lines = [line for line in result.stdout.splitlines() if line.startswith(("scope ", "mark "))]
    assert event_lines == documented_output(f"instrument-{name.replace('_', '-')}", "Expected event lines").splitlines()


@pytest.mark.parametrize("policy", ["query_policy", "tool_policy"])
@pytest.mark.parametrize("query", ["runtime instrumentation", ""])
async def test_documented_policies_allow_or_reject_before_execution(policy, query, subscribed_events, capsys):
    module = example(policy)
    getattr(module, "register_query_policy" if policy == "query_policy" else "register_tool_policy")()
    received = []

    async def search(tool_arguments):
        received.append(tool_arguments)
        return nemo_relay.ToolExecutionResult({"query": tool_arguments["query"]})

    with nemo_relay.scope.scope("agent-run", nemo_relay.ScopeType.Agent) as handle:
        arguments = {"query": query, "api_key": "demo-value"}
        if query:
            result = await nemo_relay.tools.execute("search", arguments, search, handle=handle)
            assert result.result == {"query": query}
            assert received == [arguments]
        else:
            with pytest.raises(RuntimeError, match="search.query is required"):
                await nemo_relay.tools.execute("search", arguments, search, handle=handle)
            assert received == []
    await nemo_relay.subscribers.flush_async()
    output = capsys.readouterr().out
    if policy == "tool_policy" and query:
        start = assert_pair(subscribed_events, "tool", "search")
        assert {
            "name": start["name"],
            "data": {key: start["data"][key] for key in ("query", "api_key")},
        } == json.loads(documented_output("advanced-guide", "Expected tool start fields", "json"))
        timing_lines = [line for line in output.splitlines() if line.startswith("search completed in")]
        assert len(timing_lines) == 1
        timing_pattern = re.escape(documented_output("advanced-guide", "Expected timing pattern")).replace(
            re.escape("<duration>"), r"\d+(?:\.\d+)? ms"
        )
        assert re.fullmatch(timing_pattern, timing_lines[0])
    elif not query:
        assert "search completed in" not in output
        records = [event.to_dict() for event in subscribed_events]
        assert any(
            record["kind"] == "mark"
            and record["name"] == "search"
            and record["data"]["rejected"] is True
            and record["data"]["rejection_reason"] == "search.query is required"
            for record in records
        )
        assert not any(record.get("category") == "tool" and record["name"] == "search" for record in records)
        assert [f"{event.kind} {event.name}" for event in subscribed_events] == documented_output(
            "advanced-guide", "Expected rejection events"
        ).splitlines()


async def test_subscriber_source_serializes_real_events(capsys):
    nemo_relay.subscribers.register("documentation-json", example("print_event").log_event)
    try:
        nemo_relay.scope.event("documentation-check", data={"sample": "synthetic"})
        await nemo_relay.subscribers.flush_async()
    finally:
        nemo_relay.subscribers.deregister("documentation-json")
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(
        record["name"] == "documentation-check" and record["data"] == {"sample": "synthetic"} for record in records
    )


@pytest.mark.parametrize("annotation", [None, {"verification": "preserved"}])
async def test_combined_policy_preserves_annotation(annotation):
    example("tool_policy").register_tool_policy()

    async def search(_tool_arguments):
        return nemo_relay.ToolExecutionResult({"ok": True}, annotation=annotation)

    result = await nemo_relay.tools.execute("search", {"query": "weather"}, search)
    assert result.result == {"ok": True}
    assert result.annotation == annotation


@pytest.mark.parametrize("name,category", [("tool_call", "tool"), ("llm_call", "llm")])
async def test_example_failure_closes_scope_and_removes_subscriber(name, category, subscribed_events):
    register = getattr(nemo_relay.guardrails, f"register_{category}_conditional_execution")
    deregister = getattr(nemo_relay.guardrails, f"deregister_{category}_conditional_execution")
    register("documentation.reject", 1, lambda *_arguments: "documentation rejection")
    try:
        with pytest.raises(RuntimeError, match="documentation rejection"):
            await example(name).main()
    finally:
        deregister("documentation.reject")
    await nemo_relay.subscribers.flush_async()
    assert_pair(subscribed_events, "agent", "agent-run")
    subscriber_name = "instrumentation-check" if name == "tool_call" else "llm-check"
    assert subscriber_name not in {registration.local_name for registration in list_runtime_registrations()}
