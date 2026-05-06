# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain callback handler that maps run hierarchy to NeMo Flow scopes."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from langchain_nemo_flow._nemo_flow import get_nemo_flow

_logger = logging.getLogger(__name__)


class NemoFlowCallbackHandler(BaseCallbackHandler):
    """Bridge LangChain chain run IDs to NeMo Flow Agent scopes."""

    def __init__(self) -> None:
        super().__init__()
        self._scope_handles: dict[UUID, Any] = {}
        self._nnex = get_nemo_flow()

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Push a NeMo Flow Agent scope for a LangChain chain run."""
        if self._nnex is None:
            return None
        try:
            name = serialized.get("name") or serialized.get("id", ["Unknown"])[-1]
            parent = self._scope_handles.get(parent_run_id) if parent_run_id else None
            handle = self._nnex.scope.push(
                name,
                self._nnex.ScopeType.Agent,
                handle=parent,
                input=inputs,
                metadata={"langchain_run_id": str(run_id), **(metadata or {})},
            )
            self._scope_handles[run_id] = handle
        except Exception:
            _logger.debug("NeMo Flow: on_chain_start failed", exc_info=True)
        return None

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Pop the NeMo Flow scope associated with a LangChain chain run."""
        self._pop_scope(run_id, output=outputs)
        return None

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Pop the NeMo Flow scope associated with a failed LangChain chain run."""
        self._pop_scope(run_id, output={"error": repr(error)})
        return None

    def _pop_scope(self, run_id: UUID, *, output: Any | None = None) -> None:
        if self._nnex is None:
            return
        handle = self._scope_handles.pop(run_id, None)
        if handle is None:
            return
        try:
            self._nnex.scope.pop(handle, output=output)
        except Exception:
            _logger.debug("NeMo Flow: scope.pop failed", exc_info=True)
