# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep benchmark comparisons complete and their latency direction unambiguous."""

import copy
import unittest

from scripts.native_plugin_benchmark.run import report


class NativePluginBenchmarkTests(unittest.TestCase):
    def measurements(self):
        results = [
            {
                "path": path,
                "mode": mode,
                "payload": payload,
                "median_ns": 1000,
                "host_allocations": 4,
                "plugin_allocations": 2,
            }
            for path in ("request", "execute")
            for mode in ("ready", "pending", "wakers", "fanout")
            for payload in ("tiny", "objects", "large")
        ]
        return {
            "platform": "test",
            "compiler": "test",
            "runs": {label: [{"results": copy.deepcopy(results)}] for label in ("baseline", "checked")},
        }

    def test_reports_regression_in_the_correct_direction(self):
        data = self.measurements()
        for result in data["runs"]["checked"][0]["results"]:
            result["median_ns"] = 2000
            result["host_allocations"] = 8
        output = report(data)
        self.assertEqual(output.count("| 1.00 | 2.00 | 2.00 | 4 → 8 | 2 → 2 |"), 24)

    def test_rejects_missing_or_duplicate_scenarios(self):
        for change in ("missing", "duplicate", "empty"):
            with self.subTest(change=change):
                data = self.measurements()
                results = data["runs"]["checked"][0]["results"]
                if change == "missing":
                    results.pop()
                elif change == "duplicate":
                    results[-1] = results[0]
                else:
                    data["runs"]["checked"] = []
                with self.assertRaises(ValueError):
                    report(data)


if __name__ == "__main__":
    unittest.main()
