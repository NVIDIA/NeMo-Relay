# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark environment metadata and terminal reporting."""

from __future__ import annotations

import datetime as dt
import platform
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _git_output(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def environment_record(binary: Path) -> dict[str, Any]:
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    git_status = _git_output("status", "--porcelain")
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": git_status not in {"", "unknown"},
        "relay_version": version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _print_gateway_results(gateway: list[dict[str, Any]]) -> None:
    print("\nGateway incremental latency (milliseconds; negative values are measurement noise)")
    headers = ("provider", "mode", "bytes", "c", "comparison", "metric", "p50", "p95", "p99")
    print("  ".join(f"{header:>12}" for header in headers))
    for scenario in gateway:
        for comparison, metrics in scenario["comparisons"].items():
            if not comparison.endswith("_vs_direct"):
                continue
            for metric, summary in metrics.items():
                values = (
                    scenario["provider"],
                    scenario["mode"],
                    str(scenario["payload_bytes"]),
                    str(scenario["concurrency"]),
                    comparison.replace("relay-", "").replace("_vs_direct", ""),
                    metric,
                    f"{summary['p50_ms']:.3f}",
                    f"{summary['p95_ms']:.3f}",
                    f"{summary['p99_ms']:.3f}",
                )
                print("  ".join(f"{value:>12}" for value in values))


def _print_absolute_results(title: str, results: dict[str, Any]) -> None:
    print(title)
    for name, summary in results["absolute"].items():
        print(f"  {name:<24} {summary['p50_ms']:>9.3f}")


def print_results(results: dict[str, Any]) -> None:
    """Print only the result sections produced by selected test suites."""
    if "gateway" in results:
        _print_gateway_results(results["gateway"])

    if "hooks" in results:
        _print_absolute_results("\nHook subprocess wall time (p50 milliseconds)", results["hooks"])

    if "startup" in results:
        _print_absolute_results("\nCold process/readiness time (p50 milliseconds)", results["startup"])

    if "exporter_delivery" in results:
        delivery = results["exporter_delivery"]
        print("\nExporter delivery verification")
        print(f"  ATOF bytes written       {delivery['atof_bytes']:>12}")
        print(f"  OTLP requests received   {delivery['otlp_requests']:>12}")

    if "validation_errors" in results:
        print("\nValidation errors")
        for error in results["validation_errors"]:
            print(f"  {error}")
