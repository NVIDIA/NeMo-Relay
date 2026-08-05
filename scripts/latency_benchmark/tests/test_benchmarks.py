# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for latency benchmark measurement coordination."""

import unittest
from unittest import mock

from scripts.latency_benchmark.src import benchmarks


class GatewayBenchmarkTests(unittest.TestCase):
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
