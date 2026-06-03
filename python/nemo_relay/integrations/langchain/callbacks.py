# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain callback handler that maps run hierarchy to NeMo Relay scopes."""

from __future__ import annotations

import logging
import typing

from langchain_core.callbacks.base import BaseCallbackHandler

import nemo_relay
from nemo_relay.integrations.langchain._serialization import _prepare_lc_payloads

if typing.TYPE_CHECKING:
    from uuid import UUID

_logger = logging.getLogger(__name__)


class NemoRelayCallbackHandler(BaseCallbackHandler):
    """Bridge LangChain chain run IDs to NeMo Relay Agent scopes."""

    # We need to run inline to ensure scopes are pushed and popped in the correct order.
    run_inline = True

    def __init__(self) -> None:
        super().__init__()
        self._scope_handles: dict[UUID, typing.Any] = {}

    def on_chain_start(
        self,
        serialized: dict[str, typing.Any],
        inputs: dict[str, typing.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, typing.Any] | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Push a NeMo Relay Agent scope for a LangChain chain run."""
        try:
            name = kwargs.get("name")

            if serialized is not None:
                name = name or serialized.get("name")
                if name is None:
                    id_list = serialized.get("id")
                    if isinstance(id_list, list) and len(id_list) > 0:
                        name = id_list[-1]

            if name is None:
                name = "Unknown"

            parent = None
            if parent_run_id is not None:
                parent = self._scope_handles.get(parent_run_id)

            scope_metadata = metadata.copy() if metadata else {}
            scope_metadata["langchain_run_id"] = str(run_id)
            prepared_inputs = _prepare_lc_payloads(inputs)
            handle = nemo_relay.scope.push(
                name,
                nemo_relay.ScopeType.Agent,
                handle=parent,
                input=prepared_inputs,
                metadata=scope_metadata,
            )
            self._scope_handles[run_id] = handle
        except Exception:
            _logger.error("NeMo Relay: on_chain_start failed", exc_info=True)

    def on_chain_end(
        self,
        outputs: dict[str, typing.Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Pop the NeMo Relay scope associated with a LangChain chain run."""
        self._pop_scope(run_id, output=outputs)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        """Pop the NeMo Relay scope associated with a failed LangChain chain run."""
        self._pop_scope(run_id, output={"error": repr(error)})

    def _pop_scope(self,
                   run_id: UUID,
                   *,
                   output: dict[str, typing.Any] | None = None) -> None:
        handle = self._scope_handles.pop(run_id, None)
        if handle is None:
            return

        try:
            prepared_outputs = _prepare_lc_payloads(
                output) if output is not None else None
            nemo_relay.scope.pop(handle, output=prepared_outputs)
        except Exception:
            _logger.error("NeMo Relay: scope.pop failed", exc_info=True)
