# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrate NeMo Relay with Google ADK.

Add `NemoRelayPlugin` to `App.plugins` to record ADK runs. Pass a `Workflow` to `wrap_node` to record the runs of its
graph nodes as scopes.
"""

from __future__ import annotations

from nemo_relay.integrations.google_adk._nodes import wrap_node
from nemo_relay.integrations.google_adk.plugin import NemoRelayPlugin

__all__ = ["NemoRelayPlugin", "wrap_node"]
