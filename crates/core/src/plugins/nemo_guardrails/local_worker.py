# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import sys
import traceback
from collections.abc import AsyncIterator, MutableSet
from typing import Any, TypeAlias

DEFAULT_MODULE_NAME = "nemoguardrails"
SUPPORTED_NEMOGUARDRAILS_VERSION = "0.22.0"
STREAM_QUEUE_MAXSIZE = 32

JsonObject: TypeAlias = dict[str, Any]
StreamQueue: TypeAlias = asyncio.Queue[str | None]
PendingTasks: TypeAlias = MutableSet[asyncio.Task[None]]

_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def send(message: JsonObject) -> None:
    _PROTOCOL_STDOUT.write(json.dumps(message, separators=(",", ":")) + "\n")
    _PROTOCOL_STDOUT.flush()


def response(request_id: str, result: Any | None = None) -> None:
    payload = {"id": request_id, "ok": True}
    if result is not None:
        payload["result"] = result
    send(payload)


def error_response(request_id: str, error: BaseException) -> None:
    send({"id": request_id, "ok": False, "error": str(error)})


def stream_event(request_id: str, event: str, **fields: Any) -> None:
    payload = {"id": request_id, "ok": True, "event": event}
    payload.update(fields)
    send(payload)


def stream_error(request_id: str, error: BaseException) -> None:
    send({"id": request_id, "ok": False, "event": "error", "error": str(error)})


def status_value(status: Any) -> str:
    value = getattr(status, "value", status)
    return str(value).lower()


def optional_string_attr(obj: Any, attr: str) -> str | None:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    return str(value)


def string_attr_or_empty(obj: Any, attr: str) -> str:
    return optional_string_attr(obj, attr) or ""


def guardrails_stream_error_message(chunk: str) -> str | None:
    try:
        payload = json.loads(chunk)
    except Exception:
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    if error.get("type") != "guardrails_violation":
        return None
    return error.get("message") or "Blocked by output rails."


class AsyncTextStream:
    def __init__(self, queue: StreamQueue) -> None:
        self._queue = queue

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        value = await self._queue.get()
        if value is None:
            raise StopAsyncIteration
        return value


class GuardrailsWorker:
    def __init__(self, config: JsonObject) -> None:
        if sys.version_info < (3, 11):
            raise RuntimeError("NeMo Guardrails local backend requires python3 >= 3.11")

        local = config.get("local") or {}
        root_module = (local.get("python_module") or DEFAULT_MODULE_NAME).strip()
        guardrails = self._import_dependency(root_module, root_module)
        options = self._import_dependency(f"{root_module}.rails.llm.options", root_module)

        version = getattr(guardrails, "__version__", None)
        if version != SUPPORTED_NEMOGUARDRAILS_VERSION:
            raise RuntimeError(
                "NeMo Guardrails local backend requires "
                f"nemoguardrails=={SUPPORTED_NEMOGUARDRAILS_VERSION}, but found {version!r}. "
                f"Install it with: pip install nemoguardrails=={SUPPORTED_NEMOGUARDRAILS_VERSION}"
            )

        self._rail_type = options.RailType
        self._rail_status = options.RailStatus
        guardrails_config = self._build_guardrails_config(guardrails.RailsConfig, config)
        self._rails = guardrails.LLMRails(guardrails_config)

    def _import_dependency(self, module_name: str, root_module: str) -> Any:
        try:
            return importlib.import_module(module_name)
        except ImportError as err:
            missing = getattr(err, "name", None)
            if missing == root_module:
                raise RuntimeError(
                    "NeMo Guardrails is required for the built-in NeMo Guardrails local backend. "
                    f"Install it with: pip install nemoguardrails=={SUPPORTED_NEMOGUARDRAILS_VERSION}"
                ) from err
            raise RuntimeError(
                "NeMo Guardrails local backend could not import a required dependency: "
                f"{missing or err}. Install the full NeMo Guardrails runtime dependencies."
            ) from err

    def _build_guardrails_config(self, rails_config_cls: Any, config: JsonObject) -> Any:
        config_path = config.get("config_path")
        if config_path:
            return rails_config_cls.from_path(config_path)

        config_yaml = config.get("config_yaml")
        if config_yaml is None:
            raise ValueError("config_yaml is required when config_path is not provided")
        return rails_config_cls.from_content(
            colang_content=config.get("colang_content"),
            yaml_content=config_yaml,
        )

    def _rail_kind(self, rail_type: str | None) -> Any:
        if rail_type == "input":
            return self._rail_type.INPUT
        if rail_type == "output":
            return self._rail_type.OUTPUT
        raise ValueError(f"unsupported rail_type {rail_type!r}")

    async def check(self, messages: list[JsonObject], rail_type: str | None) -> JsonObject:
        result = await self._rails.check_async(
            messages,
            rail_types=[self._rail_kind(rail_type)],
        )
        return {
            "status": status_value(result.status),
            "content": string_attr_or_empty(result, "content"),
            "rail": optional_string_attr(result, "rail"),
        }

    def has_streaming_output_rails(self) -> bool:
        output = self._output_rails_config()
        flows = getattr(output, "flows", None) if output is not None else None
        return bool(flows)

    def ensure_streaming_output_supported(self) -> None:
        output = self._output_rails_config()
        if output is None:
            return

        streaming = getattr(output, "streaming", None)
        if streaming is None or not bool(getattr(streaming, "enabled", False)):
            raise RuntimeError(
                "local NeMo Guardrails streaming output rails require "
                "rails.output.streaming.enabled = true in the Guardrails config."
            )

        if not bool(getattr(streaming, "stream_first", True)):
            raise RuntimeError(
                "local NeMo Guardrails streaming output rails currently require "
                "rails.output.streaming.stream_first = true."
            )

    def _output_rails_config(self) -> Any:
        config = getattr(self._rails, "config", None)
        rails = getattr(config, "rails", None)
        return getattr(rails, "output", None)

    async def monitor_stream(
        self, request_id: str, messages: list[JsonObject], queue: StreamQueue, streams: dict[str, StreamQueue]
    ) -> None:
        try:
            async for chunk in self._rails.stream_async(
                messages=messages,
                generator=AsyncTextStream(queue),
                include_metadata=False,
            ):
                if not isinstance(chunk, str):
                    continue
                message = guardrails_stream_error_message(chunk)
                if message:
                    stream_event(request_id, "blocked", message=message)
                    return
            stream_event(request_id, "done")
        except Exception as err:
            stream_error(request_id, err)
        finally:
            streams.pop(request_id, None)


worker: GuardrailsWorker | None = None
streams: dict[str, StreamQueue] = {}


def track_task(pending_tasks: PendingTasks, task: asyncio.Task[None]) -> asyncio.Task[None]:
    pending_tasks.add(task)
    task.add_done_callback(pending_tasks.discard)
    return task


async def handle_message(message: JsonObject, pending_tasks: PendingTasks) -> None:
    global worker

    request_id = str(message.get("id", ""))
    command = message.get("command")
    try:
        if command == "init":
            worker = _initialize_worker(message)
            response(request_id, _worker_details())
        elif worker is None:
            raise RuntimeError("NeMo Guardrails local Python worker is not initialized")
        else:
            await _handle_worker_command(worker, command, request_id, message, pending_tasks)
    except Exception as err:
        if command and command.startswith("stream_"):
            stream_error(request_id, err)
        else:
            error_response(request_id, err)


def _initialize_worker(message: JsonObject) -> GuardrailsWorker:
    return GuardrailsWorker(message.get("config") or {})


def _worker_details() -> JsonObject:
    return {"python": sys.executable, "version": ".".join(str(part) for part in sys.version_info[:3])}


async def _handle_worker_command(
    worker: GuardrailsWorker,
    command: str | None,
    request_id: str,
    message: JsonObject,
    pending_tasks: PendingTasks,
) -> None:
    if command == "check":
        response(request_id, await worker.check(message.get("messages") or [], message.get("rail_type")))
    elif command == "has_streaming_output_rails":
        response(request_id, {"enabled": worker.has_streaming_output_rails()})
    elif command == "ensure_streaming_output_supported":
        worker.ensure_streaming_output_supported()
        response(request_id)
    elif command == "stream_start":
        _start_stream(worker, request_id, message, pending_tasks)
    elif command in {"stream_text", "stream_end"}:
        await _write_stream(command, request_id, message)
    else:
        raise RuntimeError(f"unknown worker command {command!r}")


def _start_stream(worker: GuardrailsWorker, request_id: str, message: JsonObject, pending_tasks: PendingTasks) -> None:
    queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
    streams[request_id] = queue
    task = worker.monitor_stream(request_id, message.get("messages") or [], queue, streams)
    track_task(pending_tasks, asyncio.create_task(task))


async def _write_stream(command: str | None, request_id: str, message: JsonObject) -> None:
    queue = streams.get(request_id)
    if queue is not None:
        await queue.put(message.get("text") or "" if command == "stream_text" else None)


async def main() -> None:
    pending_tasks = set()
    try:
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                return
            try:
                message = json.loads(line)
            except Exception:
                traceback.print_exc(file=sys.stderr)
                continue
            if str(message.get("command", "")).startswith("stream_"):
                await handle_message(message, pending_tasks)
            else:
                track_task(
                    pending_tasks,
                    asyncio.create_task(handle_message(message, pending_tasks)),
                )
    finally:
        for task in tuple(pending_tasks):
            task.cancel()
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)


asyncio.run(main())
