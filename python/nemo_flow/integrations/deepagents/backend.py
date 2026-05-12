# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend wrappers for Deep Agents filesystem and sandbox observability."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    SandboxBackendProtocol,
    WriteResult,
    execute_accepts_timeout,
)

from nemo_flow.integrations.deepagents._events import backend_event_data, backend_kind, emit_mark, mark_base_name


class NemoFlowDeepAgentsBackend(BackendProtocol):
    """Proxy a Deep Agents backend and emit NeMo Flow marks for backend operations."""

    def __init__(self, backend: BackendProtocol, *, name: str | None = None) -> None:
        self._backend = backend
        self._name = name or type(backend).__name__

    @property
    def __wrapped__(self) -> BackendProtocol:
        """Return the wrapped backend."""
        return self._backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def ls(self, path: str) -> LsResult:
        return self._call_sync("ls", (path,), {})

    async def als(self, path: str) -> LsResult:
        return await self._call_async("als", (path,), {})

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._call_sync("read", (file_path,), {"offset": offset, "limit": limit})

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return await self._call_async("aread", (file_path,), {"offset": offset, "limit": limit})

    def write(self, file_path: str, content: str) -> WriteResult:
        return self._call_sync("write", (file_path, content), {})

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return await self._call_async("awrite", (file_path, content), {})

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return self._call_sync(
            "edit",
            (file_path, old_string, new_string),
            {"replace_all": replace_all},
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return await self._call_async(
            "aedit",
            (file_path, old_string, new_string),
            {"replace_all": replace_all},
        )

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        return self._call_sync("glob", (pattern,), {"path": path})

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        return await self._call_async("aglob", (pattern,), {"path": path})

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self._call_sync("grep", (pattern,), {"path": path, "glob": glob})

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return await self._call_async("agrep", (pattern,), {"path": path, "glob": glob})

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self._call_sync("upload_files", (files,), {})

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return await self._call_async("aupload_files", (files,), {})

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self._call_sync("download_files", (paths,), {})

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return await self._call_async("adownload_files", (paths,), {})

    def read_file(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self._call_sync("read", (file_path,), {"offset": offset, "limit": limit}, event_name="read_file")

    async def aread_file(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return await self._call_async(
            "aread",
            (file_path,),
            {"offset": offset, "limit": limit},
            event_name="aread_file",
        )

    def write_file(self, file_path: str, content: str) -> WriteResult:
        return self._call_sync("write", (file_path, content), {}, event_name="write_file")

    async def awrite_file(self, file_path: str, content: str) -> WriteResult:
        return await self._call_async("awrite", (file_path, content), {}, event_name="awrite_file")

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return self._call_sync(
            "edit",
            (file_path, old_string, new_string),
            {"replace_all": replace_all},
            event_name="edit_file",
        )

    async def aedit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return await self._call_async(
            "aedit",
            (file_path, old_string, new_string),
            {"replace_all": replace_all},
            event_name="aedit_file",
        )

    def download_file(self, path: str) -> FileDownloadResponse:
        return self._call_sync("download_files", ([path],), {}, event_name="download_file")[0]

    async def adownload_file(self, path: str) -> FileDownloadResponse:
        return (await self._call_async("adownload_files", ([path],), {}, event_name="adownload_file"))[0]

    def _call_sync(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        event_name: str | None = None,
    ) -> Any:
        event_method_name = event_name or method_name
        kind = backend_kind(event_method_name)
        base_name = mark_base_name(kind)
        emit_mark(base_name, kind, "start", backend_event_data(self._name, event_method_name, args, kwargs))
        try:
            result = getattr(self._backend, method_name)(*args, **kwargs)
        except Exception as error:
            emit_mark(
                base_name,
                kind,
                "error",
                backend_event_data(self._name, event_method_name, args, kwargs, error=error),
            )
            raise
        emit_mark(
            base_name,
            kind,
            "end",
            backend_event_data(self._name, event_method_name, args, kwargs, result=result),
        )
        return result

    async def _call_async(
        self,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        event_name: str | None = None,
    ) -> Any:
        event_method_name = event_name or method_name
        kind = backend_kind(event_method_name)
        base_name = mark_base_name(kind)
        emit_mark(base_name, kind, "start", backend_event_data(self._name, event_method_name, args, kwargs))
        try:
            result = await getattr(self._backend, method_name)(*args, **kwargs)
        except Exception as error:
            emit_mark(
                base_name,
                kind,
                "error",
                backend_event_data(self._name, event_method_name, args, kwargs, error=error),
            )
            raise
        emit_mark(
            base_name,
            kind,
            "end",
            backend_event_data(self._name, event_method_name, args, kwargs, result=result),
        )
        return result


class NemoFlowDeepAgentsSandboxBackend(NemoFlowDeepAgentsBackend, SandboxBackendProtocol):
    """Proxy a Deep Agents sandbox backend while preserving sandbox protocol checks."""

    @property
    def id(self) -> str:
        return self._backend.id  # type: ignore[attr-defined]

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        kwargs = _execute_kwargs(self._backend, timeout)
        return self._call_sync("execute", (command,), kwargs)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        kwargs = _execute_kwargs(self._backend, timeout)
        return await self._call_async("aexecute", (command,), kwargs)


def _execute_kwargs(backend: BackendProtocol, timeout: int | None) -> dict[str, int]:
    if timeout is None:
        return {}
    if not execute_accepts_timeout(type(backend)):  # type: ignore[arg-type]
        raise ValueError("This sandbox backend does not support per-command timeout overrides.")
    return {"timeout": timeout}


def observe_backend(backend: Any, *, name: str | None = None) -> NemoFlowDeepAgentsBackend:
    """Wrap a Deep Agents backend so direct backend calls emit NeMo Flow marks."""
    if isinstance(backend, NemoFlowDeepAgentsBackend):
        return backend
    if isinstance(backend, SandboxBackendProtocol):
        return NemoFlowDeepAgentsSandboxBackend(backend, name=name)
    return NemoFlowDeepAgentsBackend(backend, name=name)
