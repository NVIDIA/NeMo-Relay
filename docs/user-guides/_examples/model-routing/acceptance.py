# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small task acceptance checks, run from outside the agent's working directory."""

import importlib.util
import sys
from pathlib import Path


def rejects(function, *arguments):
    try:
        function(*arguments)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError for {arguments!r}")


def main():
    directory, task = Path(sys.argv[1]), sys.argv[2]
    spec = importlib.util.spec_from_file_location("example_service", directory / "service.py")
    if spec is None or spec.loader is None:
        raise SystemExit("Cannot load service.py")
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    if task == "ticket-id":
        normalize = service.normalize_ticket_id
        assert normalize(" inc-1042 ") == "INC-1042"
        assert normalize("INC-0000") == "INC-0000"
        for value in [
            None,
            1042,
            True,
            b"INC-1042",
            "",
            "INC-123",
            "INC-12345",
            "INC-１２３４",
            "INC-١٢٣٤",
            "INC-12３4",
            "ınc-1234",
            "İNC-1234",
            "XINC-1042",
            "INC-1042x",
            "INC-1042\nx",
        ]:
            rejects(normalize, value)
    elif task == "retry-policy":
        delays = service.retry_delays
        assert delays(0, 1, 10) == []
        assert delays(5, 2, 7) == [2, 4, 7, 7, 7]
        assert delays(3, 0.5, 2) == [0.5, 1, 2]
        assert delays(2, 5, 2) == [2, 2]
        for arguments in [
            (-1, 1, 10),
            (1.5, 1, 10),
            (True, 1, 10),
            (1, 0, 10),
            (1, 1, -1),
            (1, "1", 10),
            (1, None, 10),
        ]:
            rejects(delays, *arguments)
    elif task == "dependency-plan":
        import copy

        plan = service.deployment_plan
        graph = {"web": ["api"], "api": ["db", "db"], "db": [], "auth": [], "z": []}
        original = copy.deepcopy(graph)
        assert plan(graph) == ["auth", "db", "api", "web", "z"]
        assert plan(graph, ["db"]) == ["auth", "z"]
        assert graph == original
        assert plan({}) == []
        for invalid in (None, [], {"a": "a"}, {"a": ["missing"]}, {1: []}, {"": []}, {"a": [1]}):
            rejects(plan, invalid)
        for blocked in ("a", ["missing"], [1], None):
            rejects(plan, {"a": []}, blocked)
        rejects(plan, {"a": ["a"]})
        rejects(plan, {"a": ["b"], "b": ["a"]}, ["a"])
        chain = {f"s{i:05}": [] if i == 0 else [f"s{i - 1:05}"] for i in range(10000)}
        assert plan(chain) == list(chain)
        assert plan(chain, ["s00001"]) == ["s00000"]
    else:
        raise SystemExit("Unknown task")
    print(f"Accepted: {task}")


if __name__ == "__main__":
    main()
