# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend wrappers for Deep Agents filesystem and sandbox observability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nemo_flow.integrations.deepagents._events import backend_event_data, backend_kind, emit_mark, mark_base_name


class NemoFlowDeepAgentsBackend:
    """Proxy a Deep Agents backend and emit NeMo Flow marks for backend operations."""

    def __init__(self, backend: Any, *, name: str | None = None) -> None:
        self._backend = backend
        self._name = name or type(backend).__name__

    @property
    def __wrapped__(self) -> Any:
        """Return the wrapped backend."""
        return self._backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def ls(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("ls", args, kwargs)

    async def als(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("als", args, kwargs)

    def read_file(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("read_file", args, kwargs)

    async def aread_file(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("aread_file", args, kwargs)

    def write_file(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("write_file", args, kwargs)

    async def awrite_file(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("awrite_file", args, kwargs)

    def edit_file(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("edit_file", args, kwargs)

    async def aedit_file(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("aedit_file", args, kwargs)

    def glob(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("glob", args, kwargs)

    async def aglob(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("aglob", args, kwargs)

    def grep(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("grep", args, kwargs)

    async def agrep(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("agrep", args, kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("execute", args, kwargs)

    async def aexecute(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("aexecute", args, kwargs)

    def upload_files(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("upload_files", args, kwargs)

    async def aupload_files(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("aupload_files", args, kwargs)

    def download_file(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_sync("download_file", args, kwargs)

    async def adownload_file(self, *args: Any, **kwargs: Any) -> Any:
        return await self._call_async("adownload_file", args, kwargs)

    def _call_sync(self, method_name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        kind = backend_kind(method_name)
        base_name = mark_base_name(kind)
        emit_mark(base_name, kind, "start", backend_event_data(self._name, method_name, args, kwargs))
        try:
            result = getattr(self._backend, method_name)(*args, **kwargs)
        except Exception as error:
            emit_mark(
                base_name,
                kind,
                "error",
                backend_event_data(self._name, method_name, args, kwargs, error=error),
            )
            raise
        emit_mark(base_name, kind, "end", backend_event_data(self._name, method_name, args, kwargs, result=result))
        return result

    async def _call_async(self, method_name: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
        kind = backend_kind(method_name)
        base_name = mark_base_name(kind)
        emit_mark(base_name, kind, "start", backend_event_data(self._name, method_name, args, kwargs))
        try:
            result = await getattr(self._backend, method_name)(*args, **kwargs)
        except Exception as error:
            emit_mark(
                base_name,
                kind,
                "error",
                backend_event_data(self._name, method_name, args, kwargs, error=error),
            )
            raise
        emit_mark(base_name, kind, "end", backend_event_data(self._name, method_name, args, kwargs, result=result))
        return result


def observe_backend(backend: Any, *, name: str | None = None) -> NemoFlowDeepAgentsBackend:
    """Wrap a Deep Agents backend so direct backend calls emit NeMo Flow marks."""
    if isinstance(backend, NemoFlowDeepAgentsBackend):
        return backend
    return NemoFlowDeepAgentsBackend(backend, name=name)
