# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deep Agents callback handler for NeMo Flow observability."""

from __future__ import annotations

from typing import Any

from nemo_flow.integrations.deepagents._events import emit_mark, mark_base_name
from nemo_flow.integrations.langgraph.callbacks import NemoFlowCallbackHandler as LangGraphNemoFlowCallbackHandler


class NemoFlowDeepAgentsCallbackHandler(LangGraphNemoFlowCallbackHandler):
    """Bridge Deep Agents LangGraph lifecycle events to NeMo Flow marks."""

    def _emit_graph_mark(self, name: str, data: dict[str, Any]) -> None:
        phase = {
            "Graph Interrupt": "interrupt",
            "Graph Resume": "resume",
        }.get(name, "mark")

        emit_mark(
            mark_base_name("human_in_the_loop"),
            "human_in_the_loop",
            phase,
            data,
            metadata={"langgraph_event": name},
        )


__all__ = ["NemoFlowDeepAgentsCallbackHandler"]
