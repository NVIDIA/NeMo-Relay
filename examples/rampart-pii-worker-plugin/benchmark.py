# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare baseline and Rampart worker managed-call overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncContextManager, AsyncIterator, Awaitable, Callable

import nemo_relay
from nemo_relay import llm, plugin, subscribers, tools

PLUGIN_ID = "nvidia.rampart_pii"


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percentile))
    return ordered[index]


def _summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": statistics.median(samples_ms),
        "p95_ms": _percentile(samples_ms, 0.95),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _rss_kib(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return int(value) if value.isdigit() else None


def _worker_processes() -> list[dict[str, int | str]]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,rss=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    workers = []
    for line in result.stdout.splitlines():
        if "nemo_relay_rampart_worker.worker:main" not in line:
            continue
        pid, rss, command = line.strip().split(maxsplit=2)
        workers.append({"pid": int(pid), "rss_kib": int(rss), "command": command})
    return workers


@asynccontextmanager
async def _baseline() -> AsyncIterator[dict[str, object]]:
    started = time.perf_counter()
    await plugin.initialize(plugin.PluginConfig())
    try:
        yield {"activation_ms": (time.perf_counter() - started) * 1_000}
    finally:
        subscribers.flush()
        plugin.clear()


@asynccontextmanager
async def _worker(
    manifest: Path,
    environment: Path,
) -> AsyncIterator[dict[str, object]]:
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
                    "allow_network": False,
                    "max_latency_ms": 5_000,
                },
            )
        ],
    )
    try:
        yield {
            "activation_ms": (time.perf_counter() - started) * 1_000,
            "workers": _worker_processes(),
        }
    finally:
        subscribers.flush()
        close_started = time.perf_counter()
        await activation.close()
        close_ms = (time.perf_counter() - close_started) * 1_000
        if _worker_processes():
            raise RuntimeError("Rampart worker process survived activation close")
        print(json.dumps({"worker_close_ms": close_ms}))


def _payload(length: int) -> str:
    prefix = "Contact Alex Rivera at alex.rivera@example.com. "
    if length <= len(prefix):
        return prefix[:length]
    return (
        prefix + ("project status and deployment notes " * ((length - len(prefix)) // 36 + 1))[: length - len(prefix)]
    )


async def _measure_call(
    call: Callable[[], Awaitable[object]],
    iterations: int,
) -> list[float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        await call()
        samples.append((time.perf_counter() - started) * 1_000)
    return samples


async def _measure_mode(
    mode: str,
    context_factory: Callable[[], AsyncContextManager[dict[str, object]]],
) -> dict[str, object]:
    lengths = {64: 80, 1_024: 30, 8_192: 8}
    results: dict[str, object] = {}
    async with context_factory() as activation:
        results["activation"] = activation
        results["host_rss_kib"] = _rss_kib(os.getpid())
        for length, iterations in lengths.items():
            text = _payload(length)
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

            async def llm_call() -> object:
                value = await llm.execute(
                    f"{mode}-llm",
                    request,
                    lambda _request: response,
                )
                if value != response:
                    raise RuntimeError("LLM callback response was changed")
                return value

            async def tool_call() -> object:
                arguments = {
                    "content": text,
                    "region": "us-west-2",
                    "trace_id": "550e8400-e29b-41d4-a716-446655440000",
                }
                result = {"content": text, "region": "us-west-2"}
                value = await tools.execute(
                    f"{mode}-tool",
                    arguments,
                    lambda _arguments: result,
                )
                if value != result:
                    raise RuntimeError("tool callback response was changed")
                return value

            await llm_call()
            await tool_call()
            results[str(length)] = {
                "iterations": iterations,
                "llm": _summary(await _measure_call(llm_call, iterations)),
                "tool": _summary(await _measure_call(tool_call, iterations)),
            }
    return results


async def _main(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    environment = args.environment.resolve()
    modes = {
        "baseline": lambda: _measure_mode("baseline", _baseline),
        "worker": lambda: _measure_mode(
            "worker",
            lambda: _worker(manifest, environment),
        ),
    }
    output = {
        "platform": os.uname().machine,
        "modes": {args.mode: await modes[args.mode]()},
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("baseline", "worker"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
