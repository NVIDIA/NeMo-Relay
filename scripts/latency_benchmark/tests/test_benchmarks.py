# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency benchmark measurement coordination."""

import threading
import unittest
from unittest import mock

from scripts.latency_benchmark.src import benchmarks


class GatewayBenchmarkTests(unittest.TestCase):
    def test_rotates_variants_by_sample_index(self) -> None:
        calls_by_thread: dict[int, list[str]] = {}
        calls_lock = threading.Lock()

        def connection_for(url: str) -> mock.Mock:
            return mock.Mock(url=url)

        def perform_request(connection: mock.Mock, *_args: object) -> dict[str, int]:
            with calls_lock:
                calls_by_thread.setdefault(threading.get_ident(), []).append(connection.url)
            return {"total_ns": 1}

        urls = {
            "direct": "direct",
            "relay-minimal": "relay-minimal",
            "relay-file": "relay-file",
            "relay-otlp": "relay-otlp",
        }
        with (
            mock.patch.object(benchmarks, "connection_for", side_effect=connection_for),
            mock.patch.object(benchmarks, "make_request", return_value=b"request"),
            mock.patch.object(benchmarks, "perform_request", side_effect=perform_request),
        ):
            benchmarks.benchmark_scenario(
                urls,
                provider="openai",
                model="benchmark-model",
                request_fill="x",
                streaming=False,
                payload_bytes=4096,
                samples=4,
                warmup=0,
                concurrency=4,
            )

        self.assertEqual(
            {tuple(calls) for calls in calls_by_thread.values()},
            {
                ("direct", "relay-minimal", "relay-file", "relay-otlp"),
                ("relay-minimal", "relay-file", "relay-otlp", "direct"),
                ("relay-file", "relay-otlp", "direct", "relay-minimal"),
                ("relay-otlp", "direct", "relay-minimal", "relay-file"),
            },
        )

    def test_aborts_barrier_when_a_worker_fails_during_warmup(self) -> None:
        barrier = mock.Mock()
        connection = mock.Mock()

        with (
            mock.patch.object(benchmarks.threading, "Barrier", return_value=barrier),
            mock.patch.object(benchmarks, "connection_for", return_value=connection),
            mock.patch.object(benchmarks, "make_request", return_value=b"request"),
            mock.patch.object(benchmarks, "perform_request", side_effect=RuntimeError("warmup failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "warmup failed"):
                benchmarks.benchmark_scenario(
                    {"direct": "http://127.0.0.1:8000"},
                    provider="openai",
                    model="benchmark-model",
                    request_fill="x",
                    streaming=False,
                    payload_bytes=4096,
                    samples=2,
                    warmup=1,
                    concurrency=2,
                )

        barrier.abort.assert_called()


if __name__ == "__main__":
    unittest.main()
