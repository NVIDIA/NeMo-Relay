# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the task contract independently of tests written by the coding agent."""

import importlib.util
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "ticket_id.py"
spec = importlib.util.spec_from_file_location("ticket_id", path)
if spec is None or spec.loader is None:
    raise SystemExit("Cannot load ticket_id.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
normalize = module.normalize_ticket_id
for value, expected in [(" inc-1042 ", "INC-1042"), ("INC-0000", "INC-0000")]:
    if normalize(value) != expected:
        raise SystemExit(f"Incorrect normalization: {value!r}")
invalid = [
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
    "INC- 1042",
    "INC -1042",
    "INC-+104",
    "INC-1.42",
]
for value in invalid:
    try:
        normalize(value)
    except ValueError:
        continue
    raise SystemExit(f"Expected ValueError: {value!r}")
print(f"Accepted {2 + len(invalid)} independent task cases")
