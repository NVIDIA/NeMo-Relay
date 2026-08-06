# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gateway, hook, and process-startup benchmark suites."""

from __future__ import annotations

import concurrent.futures
import contextlib
import http.client
import json
import math
import random
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .fixtures import isolated_environment
from .processes import RelayProcess, TransparentRelayProcess
from .protocol import make_request, request_headers, request_path
from .servers import connection_for


def percentile(values: Sequence[int | float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_ns(values: list[int]) -> dict[str, Any]:
    milliseconds = [value / 1_000_000 for value in values]
    return {
        "samples": len(values),
        "p50_ms": round(percentile(milliseconds, 0.50), 6),
        "p95_ms": round(percentile(milliseconds, 0.95), 6),
        "p99_ms": round(percentile(milliseconds, 0.99), 6),
        "min_ms": round(min(milliseconds), 6),
        "max_ms": round(max(milliseconds), 6),
    }


def median_confidence_interval_ns(values: list[int], *, seed: int, resamples: int = 1_000) -> list[float]:
    """Return a deterministic bootstrap 95% confidence interval for the median."""
    randomizer = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [values[randomizer.randrange(len(values))] for _ in values]
        medians.append(percentile(sample, 0.50) / 1_000_000)
    return [
        round(percentile(medians, 0.025), 6),
        round(percentile(medians, 0.975), 6),
    ]


def perform_request(
    connection: http.client.HTTPConnection,
    provider: str,
    body: bytes,
    streaming: bool,
) -> dict[str, int]:
    started = time.perf_counter_ns()
    connection.request("POST", request_path(provider), body=body, headers=request_headers(provider))
    response = connection.getresponse()
    if response.status != 200:
        details = response.read().decode(errors="replace")
        raise RuntimeError(f"benchmark request failed with HTTP {response.status}: {details}")
    if not streaming:
        response.read()
        return {"total_ns": time.perf_counter_ns() - started}
    first_content_ns = 0
    while True:
        line = response.readline()
        if not line:
            break
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        event = json.loads(payload)
        if event.get("type") in {"response.output_text.delta", "content_block_delta"}:
            first_content_ns = first_content_ns or time.perf_counter_ns() - started
    if first_content_ns == 0:
        raise RuntimeError("stream ended before a content delta was received")
    return {
        "first_content_ns": first_content_ns,
        "total_ns": time.perf_counter_ns() - started,
    }


def benchmark_scenario(
    urls: dict[str, str],
    *,
    provider: str,
    model: str,
    request_fill: str,
    streaming: bool,
    payload_bytes: int,
    samples: int,
    warmup: int,
    concurrency: int,
) -> dict[str, Any]:
    variants = tuple(urls)
    body = make_request(
        provider,
        streaming,
        payload_bytes,
        model=model,
        request_fill=request_fill,
    )
    observations: list[dict[str, dict[str, int]]] = []
    observation_lock = threading.Lock()
    barrier = threading.Barrier(concurrency)

    def worker(indices: list[int]) -> None:
        connections = {name: connection_for(url) for name, url in urls.items()}
        try:
            for _ in range(warmup):
                for name in variants:
                    perform_request(connections[name], provider, body, streaming)
            barrier.wait()
            local = []
            for index in indices:
                order = list(variants)
                shift = index % len(order)
                order = order[shift:] + order[:shift]
                local.append({name: perform_request(connections[name], provider, body, streaming) for name in order})
            with observation_lock:
                observations.extend(local)
        except BaseException:
            barrier.abort()
            raise
        finally:
            for connection in connections.values():
                connection.close()

    assignments = [list(range(worker_id, samples, concurrency)) for worker_id in range(concurrency)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, indices) for indices in assignments]
        for future in futures:
            future.result()

    metrics = ("total_ns",) if not streaming else ("first_content_ns", "total_ns")
    absolute = {
        name: {
            metric.removesuffix("_ns"): summarize_ns([cycle[name][metric] for cycle in observations])
            for metric in metrics
        }
        for name in variants
    }
    relay_variants = tuple(name for name in variants if name != "direct")
    comparison_pairs = {f"{name}_vs_direct": (name, "direct") for name in relay_variants}
    for name in relay_variants:
        if name == "relay-minimal":
            continue
        label = name.removeprefix("relay-")
        comparison = {
            "file": "file_exporter_vs_minimal",
            "otlp": "otlp_exporter_vs_minimal",
        }.get(label, f"{label}_vs_minimal")
        comparison_pairs[comparison] = (name, "relay-minimal")
    comparisons = {}
    for comparison, (left, right) in comparison_pairs.items():
        comparisons[comparison] = {}
        for metric_index, metric in enumerate(metrics):
            deltas = [cycle[left][metric] - cycle[right][metric] for cycle in observations]
            summary = summarize_ns(deltas)
            summary["median_ci95_ms"] = median_confidence_interval_ns(
                deltas,
                seed=payload_bytes + concurrency * 101 + metric_index * 10_007,
            )
            comparisons[comparison][metric.removesuffix("_ns")] = summary
    return {
        "provider": provider,
        "mode": "streaming" if streaming else "buffered",
        "payload_bytes": payload_bytes,
        "serialized_request_bytes": len(body),
        "concurrency": concurrency,
        "absolute": absolute,
        "comparisons": comparisons,
    }


def run_subprocess_timed(command: list[str], *, root: Path, input_bytes: bytes | None = None) -> int:
    started = time.perf_counter_ns()
    result = subprocess.run(
        command,
        cwd=root,
        env=isolated_environment(root),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter_ns() - started
    if result.returncode != 0:
        details = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"command exited with status {result.returncode}: {command}\n{details}")
    return elapsed


def benchmark_hooks(
    binary: Path,
    root: Path,
    provider_url: str,
    configs: dict[str, Path],
    *,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    relay_variants = tuple(configs)
    measurements = {"process_baseline": []}
    for agent in ("codex", "claude"):
        for variant in relay_variants:
            measurements[f"{agent}_{variant.removeprefix('relay-')}"] = []

    with contextlib.ExitStack() as stack:
        relay_urls = {
            variant: stack.enter_context(
                TransparentRelayProcess(
                    binary,
                    root,
                    provider_url,
                    configs[variant],
                    f"transparent-{variant}",
                )
            ).url
            for variant in relay_variants
        }

        def hook(agent: str, variant: str, index: int) -> int:
            variant_name = variant.removeprefix("relay-")
            event_name = "sessionStart" if agent == "codex" else "SessionStart"
            payload = json.dumps(
                {
                    "session_id": f"benchmark-{agent}-{variant_name}-{index}",
                    "hook_event_name": event_name,
                }
            ).encode()
            return run_subprocess_timed(
                [
                    str(binary),
                    "hook-forward",
                    agent,
                    "--gateway-url",
                    relay_urls[variant],
                    "--transparent-run",
                    "--fail-closed",
                ],
                root=root,
                input_bytes=payload,
            )

        for index in range(-warmup, samples):
            cycle = {"process_baseline": lambda: run_subprocess_timed([str(binary), "--version"], root=root)}
            for agent in ("codex", "claude"):
                for variant in relay_variants:
                    name = f"{agent}_{variant.removeprefix('relay-')}"
                    cycle[name] = lambda agent=agent, variant=variant: hook(agent, variant, index)
            names = list(cycle)
            random.Random(index).shuffle(names)
            values = {name: cycle[name]() for name in names}
            if index >= 0:
                for name, value in values.items():
                    measurements[name].append(value)

    baseline = measurements["process_baseline"]
    result: dict[str, Any] = {
        "absolute": {name: summarize_ns(values) for name, values in measurements.items()},
        "comparisons": {},
    }
    for agent in ("codex", "claude"):
        for variant in relay_variants:
            name = f"{agent}_{variant.removeprefix('relay-')}"
            deltas = [left - right for left, right in zip(measurements[name], baseline)]
            summary = summarize_ns(deltas)
            summary["median_ci95_ms"] = median_confidence_interval_ns(deltas, seed=len(name) * 1_009)
            result["comparisons"][f"{name}_vs_process_baseline"] = summary
    return result


def benchmark_startup(
    binary: Path,
    root: Path,
    provider_url: str,
    configs: dict[str, Path],
    *,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    relay_variants = tuple(configs)
    measurements = {"process_baseline": [], **{variant: [] for variant in relay_variants}}
    for index in range(-warmup, samples):
        baseline = run_subprocess_timed([str(binary), "--version"], root=root)
        cycle = {}
        for variant in relay_variants:
            with RelayProcess(
                binary,
                root,
                provider_url,
                configs[variant],
                f"startup-{variant}-{index}",
            ) as process:
                cycle[variant] = process.startup_ns
        if index >= 0:
            measurements["process_baseline"].append(baseline)
            for variant, value in cycle.items():
                measurements[variant].append(value)
    result: dict[str, Any] = {
        "absolute": {name: summarize_ns(values) for name, values in measurements.items()},
        "comparisons": {},
    }
    baseline = measurements["process_baseline"]
    for variant in relay_variants:
        deltas = [left - right for left, right in zip(measurements[variant], baseline)]
        summary = summarize_ns(deltas)
        summary["median_ci95_ms"] = median_confidence_interval_ns(deltas, seed=len(variant) * 2_003)
        result["comparisons"][f"{variant}_readiness_vs_process_baseline"] = summary
    return result
