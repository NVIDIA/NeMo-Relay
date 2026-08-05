# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command-line orchestration for the coding-agent latency benchmark."""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .benchmarks import benchmark_hooks, benchmark_scenario, benchmark_startup
from .config import BenchmarkConfig, parse_args
from .fixtures import write_plugin_configs, write_relay_config
from .html_report import write_html_report
from .processes import RelayProcess
from .reporting import environment_record, print_results
from .servers import OtlpHandler, ProviderHandler, local_server


def _benchmark_gateway(
    binary: Path,
    root: Path,
    provider_url: str,
    configs: dict[str, Path],
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    with contextlib.ExitStack() as stack:
        relays = {
            variant: stack.enter_context(RelayProcess(binary, root, provider_url, configs[variant], variant))
            for variant in configs
        }
        urls = {"direct": provider_url} | {variant: relay.url for variant, relay in relays.items()}
        scenarios = []
        for provider in config.providers:
            model = config.openai_model if provider == "openai" else config.anthropic_model
            for mode in config.modes:
                streaming = mode == "streaming"
                for payload_bytes in config.payload_sizes:
                    for concurrency in config.concurrency:
                        print(
                            f"Benchmarking {provider} {mode}, payload={payload_bytes}, concurrency={concurrency}...",
                            flush=True,
                        )
                        scenarios.append(
                            benchmark_scenario(
                                urls,
                                provider=provider,
                                model=model,
                                request_fill=config.request_fill,
                                streaming=streaming,
                                payload_bytes=payload_bytes,
                                samples=config.samples,
                                warmup=config.warmup,
                                concurrency=concurrency,
                            )
                        )
        return scenarios


def run_benchmarks(binary: Path, config: BenchmarkConfig) -> dict[str, Any]:
    """Run selected suites and return the versioned result document."""
    OtlpHandler.reset()
    results: dict[str, Any] = {
        "schema_version": 2,
        "environment": environment_record(binary),
        "parameters": config.parameters(),
    }
    with tempfile.TemporaryDirectory(prefix="nemo-relay-latency-") as temporary:
        root = Path(temporary)
        write_relay_config(root)
        with (
            local_server(
                ProviderHandler,
                response_bytes=config.response_bytes,
                stream_chunks=config.stream_chunks,
                response_fill=config.response_fill,
                openai_model=config.openai_model,
                anthropic_model=config.anthropic_model,
            ) as provider_url,
            local_server(OtlpHandler) as otlp_url,
        ):
            configs = write_plugin_configs(root, otlp_url, config.middleware)
            if "gateway" in config.tests:
                results["gateway"] = _benchmark_gateway(binary, root, provider_url, configs, config)
            if "hooks" in config.tests:
                results["hooks"] = benchmark_hooks(
                    binary,
                    root,
                    provider_url,
                    configs,
                    samples=config.hook_samples,
                    warmup=config.warmup,
                )
            if "startup" in config.tests:
                results["startup"] = benchmark_startup(
                    binary,
                    root,
                    provider_url,
                    configs,
                    samples=config.startup_samples,
                    warmup=config.warmup,
                )

            if {"gateway", "hooks"}.intersection(config.tests):
                atof_path = root / "atof" / "events.jsonl"
                if not atof_path.is_file() or atof_path.stat().st_size == 0:
                    raise RuntimeError("local ATOF exporter did not write benchmark events")
                if OtlpHandler.request_count == 0:
                    raise RuntimeError("local OTLP receiver did not receive benchmark exports")
                results["exporter_delivery"] = {
                    "atof_bytes": atof_path.stat().st_size,
                    "otlp_requests": OtlpHandler.request_count,
                }
    return results


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    results = run_benchmarks(options.relay_bin, options.config)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html_report(results, options.report)
    print_results(results)
    print(f"\nJSON results: {options.output}")
    print(f"HTML report: {options.report}")
