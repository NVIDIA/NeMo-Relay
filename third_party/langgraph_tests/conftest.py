# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared import guards for LangGraph integration tests.

All tests in this directory require:
1. ``nemo_flow`` to be installed (the NeMo Flow Python bindings)
2. ``langgraph`` to be installed with the NeMo Flow integration patch applied

Tests are automatically skipped when either dependency is unavailable.
"""

from __future__ import annotations

import pytest


def _langgraph_patched() -> bool:
    """Return True if langgraph is installed with the NeMo Flow integration patch."""
    try:
        from langgraph import _nemo_flow  # noqa: F401

        return True
    except ImportError:
        return False


# Skip the entire directory when langgraph is not installed or not patched.
if not _langgraph_patched():
    collect_ignore_glob = ["test_*.py"]

pytestmark = pytest.mark.skipif(
    not _langgraph_patched(),
    reason="langgraph not installed or NeMo Flow integration patch not applied",
)
