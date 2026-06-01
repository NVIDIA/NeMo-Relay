# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_relay import plugin


def test_layer_plugin_config_round_trips_merge():
    # Smoke test only: merge semantics are covered by the core crate. This
    # verifies the binding forwards both documents and returns merged JSON.
    assert plugin.layer({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
