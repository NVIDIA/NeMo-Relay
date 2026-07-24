# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the real Relay worker path across size, concurrency, and fanout."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import (
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    TypedDict,
)

import nemo_relay
from nemo_relay import llm, plugin, subscribers, tools

PLUGIN_ID = "nvidia.rampart_pii"
FAILURE_REPLACEMENT = "[REDACTED:PII_DETECTION_FAILURE]"


class BenchmarkProfile(TypedDict):
    sequential: dict[int, int]
    concurrent: tuple[tuple[int, int, int], ...]
    fanout: tuple[tuple[int, int], ...]
    limits: bool
    overload: tuple[tuple[int, int, int], ...]
    soak: tuple[tuple[int, int, int, int], ...]


class LatencySummary(TypedDict):
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


class ScenarioResult(TypedDict, total=False):
    operations: int
    concurrency: int
    wall_ms: float
    throughput_ops_s: float
    latency: LatencySummary
    lifecycle_events: int
    failure_replacements: int
    content_chars: int
    string_fields_per_event: int
    round: int
    workers: list[dict[str, int | str]]
    elapsed_after_overload_ms: float


_PROFILES: dict[str, BenchmarkProfile] = {
    "quick": {
        "sequential": {64: 20, 1_024: 8, 8_192: 2},
        "concurrent": ((64, 4, 8), (1_024, 4, 8), (8_192, 4, 4)),
        "fanout": ((1, 4), (8, 2)),
        "limits": False,
        "overload": (),
        "soak": (),
    },
    "comprehensive": {
        "sequential": {64: 80, 1_024: 30, 8_192: 8},
        "concurrent": (
            (64, 1, 16),
            (64, 4, 32),
            (64, 16, 64),
            (64, 64, 256),
            (1_024, 1, 8),
            (1_024, 4, 16),
            (1_024, 16, 32),
            (1_024, 64, 128),
            (8_192, 1, 4),
            (8_192, 4, 8),
        ),
        "overload": (
            (8_192, 8, 16),
            (8_192, 16, 32),
            (8_192, 64, 64),
            (32_768, 1, 2),
            (32_768, 4, 4),
            (32_768, 16, 16),
            (65_536, 1, 1),
        ),
        "fanout": ((1, 12), (8, 6), (32, 3), (64, 2)),
        "limits": True,
        "soak": (
            (64, 16, 250, 4),
            (1_024, 8, 50, 4),
        ),
    },
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _summary(samples_ms: list[float]) -> LatencySummary:
    return {
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _rss_kib(pid: int) -> int | None:
    ps = shutil.which("ps")
    if ps is None:
        return None
    result = subprocess.run(
        [ps, "-o", "rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def _worker_processes(
    *,
    include_pids: set[int] | None = None,
) -> list[dict[str, int | str]]:
    ps = shutil.which("ps")
    if ps is None:
        return []
    result = subprocess.run(
        [ps, "-axo", "pid=,rss=,command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    workers = []
    for line in result.stdout.splitlines():
        if "nemo_relay_rampart_worker.worker:main" not in line:
            continue
        pid, rss, command = line.strip().split(maxsplit=2)
        parsed_pid = int(pid)
        if include_pids is not None and parsed_pid not in include_pids:
            continue
        workers.append({"pid": parsed_pid, "rss_kib": int(rss), "command": command})
    return workers


def _count_value(value: object, target: str) -> int:
    if isinstance(value, str):
        return int(value == target)
    if isinstance(value, list):
        return sum(_count_value(item, target) for item in value)
    if isinstance(value, dict):
        return sum(_count_value(item, target) for item in value.values())
    return 0


def _failure_count(events: list[nemo_relay.Event]) -> int:
    return sum(_count_value(event.to_dict(), FAILURE_REPLACEMENT) for event in events)


@asynccontextmanager
async def _baseline() -> AsyncIterator[dict[str, object]]:
    state: dict[str, object] = {}
    started = time.perf_counter()
    await plugin.initialize(plugin.PluginConfig())
    state["activation_ms"] = (time.perf_counter() - started) * 1_000
    try:
        yield state
    finally:
        subscribers.flush()
        close_started = time.perf_counter()
        plugin.clear()
        state["close_ms"] = (time.perf_counter() - close_started) * 1_000


@asynccontextmanager
async def _worker(
    manifest: Path,
    environment: Path,
    *,
    allow_network: bool,
    max_concurrency: int,
    max_content_chars: int,
    max_latency_ms: int,
) -> AsyncIterator[dict[str, object]]:
    state: dict[str, object] = {}
    existing_worker_pids = {int(worker["pid"]) for worker in _worker_processes()}
    started = time.perf_counter()
    activation = await plugin.initialize_with_dynamic_plugins(
        plugin.PluginConfig(),
        [
            plugin.DynamicPluginActivationSpec(
                plugin_id=PLUGIN_ID,
                kind="worker",
                manifest_ref=str(manifest),
                environment_ref=str(environment),
                config={
                    "version": 1,
                    "allow_network": allow_network,
                    "max_concurrency": max_concurrency,
                    "max_content_chars": max_content_chars,
                    "max_latency_ms": max_latency_ms,
                },
            )
        ],
    )
    state["activation_ms"] = (time.perf_counter() - started) * 1_000
    worker_pids = {
        int(worker["pid"]) for worker in _worker_processes() if int(worker["pid"]) not in existing_worker_pids
    }
    state["workers"] = _worker_processes(include_pids=worker_pids)
    state["worker_pids"] = sorted(worker_pids)
    try:
        yield state
    finally:
        subscribers.flush()
        state["workers_before_close"] = _worker_processes(include_pids=worker_pids)
        close_started = time.perf_counter()
        await activation.close()
        state["close_ms"] = (time.perf_counter() - close_started) * 1_000
        if _worker_processes(include_pids=worker_pids):
            raise RuntimeError("Rampart worker process survived activation close")
        plugin.clear()


def _payload(length: int) -> str:
    prefix = "Contact Alex Rivera at alex.rivera@example.com. "
    if length <= len(prefix):
        return prefix[:length]
    filler = "project status and deployment notes "
    return (prefix + filler * ((length - len(prefix)) // len(filler) + 1))[:length]


async def _measure_calls(
    call: Callable[[int], Awaitable[object]],
    *,
    operations: int,
    concurrency: int,
) -> tuple[list[float], float]:
    async def timed(index: int) -> float:
        started = time.perf_counter()
        await call(index)
        return (time.perf_counter() - started) * 1_000

    started = time.perf_counter()
    samples = []
    for offset in range(0, operations, concurrency):
        samples.extend(
            await asyncio.gather(*(timed(index) for index in range(offset, min(offset + concurrency, operations))))
        )
    return samples, time.perf_counter() - started


def _llm_call(text: str, name: str) -> Callable[[int], Awaitable[object]]:
    request = nemo_relay.LLMRequest(
        {"x-trace-id": "550e8400-e29b-41d4-a716-446655440000"},
        {
            "model": "benchmark-model",
            "messages": [{"role": "user", "content": text}],
        },
    )
    response = {
        "model": "benchmark-model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }

    async def call(index: int) -> object:
        value = await llm.execute(
            f"{name}-{index}",
            request,
            lambda _request: response,
        )
        if value != response:
            raise RuntimeError("LLM callback response was changed")
        return value

    return call


def _tool_payload_call(
    arguments: dict[str, object],
    name: str,
) -> Callable[[int], Awaitable[object]]:
    result = dict(arguments)

    async def call(index: int) -> object:
        value = await tools.execute(
            f"{name}-{index}",
            arguments,
            lambda _arguments: result,
        )
        if value != result:
            raise RuntimeError("tool callback response was changed")
        return value

    return call


def _tool_call(fields: int, name: str) -> Callable[[int], Awaitable[object]]:
    return _tool_payload_call(
        {f"field_{index}": f"Contact person {index} at person{index}@example.com" for index in range(fields)},
        name,
    )


async def _scenario(
    call: Callable[[int], Awaitable[object]],
    *,
    events: list[nemo_relay.Event],
    operations: int,
    concurrency: int,
) -> ScenarioResult:
    events.clear()
    samples, wall_seconds = await _measure_calls(
        call,
        operations=operations,
        concurrency=concurrency,
    )
    subscribers.flush()
    return {
        "operations": operations,
        "concurrency": concurrency,
        "wall_ms": wall_seconds * 1_000,
        "throughput_ops_s": operations / wall_seconds,
        "latency": _summary(samples),
        "lifecycle_events": len(events),
        "failure_replacements": _failure_count(events),
    }


async def _measure_mode(
    mode: str,
    context_factory: Callable[[], AsyncContextManager[dict[str, object]]],
    profile_name: str,
) -> dict[str, object]:
    profile_config = _PROFILES[profile_name]
    events: list[nemo_relay.Event] = []
    subscriber_name = f"rampart-benchmark-{mode}"
    subscribers.register(subscriber_name, events.append)
    results: dict[str, object] = {
        "sequential": {},
        "concurrent": [],
        "field_fanout": [],
        "limits": {},
        "overload": [],
        "recovery": [],
        "soak": [],
    }
    try:
        async with context_factory() as activation:
            results["activation"] = activation
            results["host_rss_kib"] = _rss_kib(os.getpid())
            recorded_worker_pids = activation.get("worker_pids")
            worker_pids = (
                {pid for pid in recorded_worker_pids if isinstance(pid, int) and not isinstance(pid, bool)}
                if isinstance(recorded_worker_pids, list)
                else set()
            )

            for length, operations in profile_config["sequential"].items():
                text = _payload(length)
                results["sequential"][str(length)] = await _scenario(
                    _llm_call(text, f"{mode}-sequential-{length}"),
                    events=events,
                    operations=operations,
                    concurrency=1,
                )

            for length, concurrency, operations in profile_config["concurrent"]:
                text = _payload(length)
                result = await _scenario(
                    _llm_call(text, f"{mode}-concurrent-{length}-{concurrency}"),
                    events=events,
                    operations=operations,
                    concurrency=concurrency,
                )
                result["content_chars"] = length
                results["concurrent"].append(result)

            for fields, operations in profile_config["fanout"]:
                result = await _scenario(
                    _tool_call(fields, f"{mode}-fanout-{fields}"),
                    events=events,
                    operations=operations,
                    concurrency=1,
                )
                result["string_fields_per_event"] = fields
                results["field_fanout"].append(result)

            if profile_config["limits"]:
                limit_payloads: dict[str, dict[str, object]] = {
                    "oversized_text": {"content": "x" * 65_537},
                    "excessive_nodes": {"values": [0] * 4_096},
                    "excessive_content_fields": {f"field_{index}": "safe" for index in range(65)},
                }
                for limit_name, payload in limit_payloads.items():
                    results["limits"][limit_name] = await _scenario(
                        _tool_payload_call(
                            payload,
                            f"{mode}-limit-{limit_name}",
                        ),
                        events=events,
                        operations=1,
                        concurrency=1,
                    )

            for length, concurrency, operations, rounds in profile_config["soak"]:
                text = _payload(length)
                for round_number in range(1, rounds + 1):
                    result = await _scenario(
                        _llm_call(
                            text,
                            f"{mode}-soak-{length}-{round_number}",
                        ),
                        events=events,
                        operations=operations,
                        concurrency=concurrency,
                    )
                    result["content_chars"] = length
                    result["round"] = round_number
                    result["workers"] = _worker_processes(include_pids=worker_pids)
                    results["soak"].append(result)

            results["workers_after_steady_state"] = _worker_processes(
                include_pids=worker_pids,
            )
            for length, concurrency, operations in profile_config["overload"]:
                text = _payload(length)
                result = await _scenario(
                    _llm_call(text, f"{mode}-overload-{length}-{concurrency}"),
                    events=events,
                    operations=operations,
                    concurrency=concurrency,
                )
                result["content_chars"] = length
                results["overload"].append(result)

            if profile_config["overload"]:
                overload_finished = time.perf_counter()
                for attempt in range(10):
                    if attempt:
                        await asyncio.sleep(0.1)
                    result = await _scenario(
                        _llm_call(_payload(64), f"{mode}-recovery-{attempt}"),
                        events=events,
                        operations=1,
                        concurrency=1,
                    )
                    result["elapsed_after_overload_ms"] = (time.perf_counter() - overload_finished) * 1_000
                    results["recovery"].append(result)
                    if result["failure_replacements"] == 0 and result["latency"]["max_ms"] < 25:
                        break
            results["workers_after_overload"] = _worker_processes(
                include_pids=worker_pids,
            )
            results["host_rss_after_kib"] = _rss_kib(os.getpid())
    finally:
        subscribers.deregister(subscriber_name)
    return results


async def _main(args: argparse.Namespace) -> None:
    modes = (args.mode,) if args.mode != "both" else ("baseline", "worker")
    if "worker" in modes and (args.manifest is None or args.environment is None):
        raise ValueError("--manifest and --environment are required for worker mode")

    results = {}
    for mode in modes:
        if mode == "baseline":
            context_factory = _baseline
        else:
            assert args.manifest is not None
            assert args.environment is not None

            def context_factory() -> AsyncContextManager[dict[str, object]]:
                return _worker(
                    args.manifest.resolve(),
                    args.environment.resolve(),
                    allow_network=args.allow_network,
                    max_concurrency=args.max_concurrency,
                    max_content_chars=args.max_content_chars,
                    max_latency_ms=args.max_latency_ms,
                )

        results[mode] = await _measure_mode(mode, context_factory, args.profile)

    output = {
        "environment": {
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "nemo_relay": importlib.metadata.version("nemo-relay"),
            "cpu_count": os.cpu_count(),
            "process_inspection": shutil.which("ps") is not None,
        },
        "profile": args.profile,
        "worker_config": {
            "allow_network": args.allow_network,
            "max_concurrency": args.max_concurrency,
            "max_content_chars": args.max_content_chars,
            "max_latency_ms": args.max_latency_ms,
        },
        "modes": results,
    }
    serialized = json.dumps(output, indent=2)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument(
        "--mode",
        choices=("baseline", "worker", "both"),
        default="both",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILES),
        default="quick",
    )
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--max-content-chars", type=int, default=8_192)
    parser.add_argument("--max-latency-ms", type=int, default=250)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
