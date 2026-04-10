# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for runtime public-surface behavior."""

import nemo_flow


class TestRuntime:
    def test_runtime_control_module_is_not_exported(self):
        assert not hasattr(nemo_flow, "runtime")

    def test_scope_stack_helpers_remain_available(self):
        stack = nemo_flow.create_scope_stack()

        assert stack is not None
