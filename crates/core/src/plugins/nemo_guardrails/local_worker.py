# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import json
import sys
import traceback

DEFAULT_MODULE_NAME = "nemoguardrails"
SUPPORTED_NEMOGUARDRAILS_VERSION = "0.22.0"
STREAM_QUEUE_MAXSIZE = 32

_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def send(message):
    _PROTOCOL_STDOUT.write(json.dumps(message, separators=(",", ":")) + "\n")
    _PROTOCOL_STDOUT.flush()


def response(request_id, result=None):
    payload = {"id": request_id, "ok": True}
    if result is not None:
        payload["result"] = result
    send(payload)


def error_response(request_id, error):
    send({"id": request_id, "ok": False, "error": str(error)})


def stream_event(request_id, event, **fields):
    payload = {"id": request_id, "ok": True, "event": event}
    payload.update(fields)
    send(payload)


def stream_error(request_id, error):
    send({"id": request_id, "ok": False, "event": "error", "error": str(error)})


def status_value(status):
    value = getattr(status, "value", status)
    return str(value).lower()


def optional_string_attr(obj, attr):
    value = getattr(obj, attr, None)
    if value is None:
        return None
    return str(value)


def string_attr_or_empty(obj, attr):
    return optional_string_attr(obj, attr) or ""


def guardrails_stream_error_message(chunk):
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
    def __init__(self, queue):
        self._queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self._queue.get()
        if value is None:
            raise StopAsyncIteration
        return value


class GuardrailsWorker:
    def __init__(self, config):
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

    def _import_dependency(self, module_name, root_module):
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

    def _build_guardrails_config(self, rails_config_cls, config):
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

    def _rail_kind(self, rail_type):
        if rail_type == "input":
            return self._rail_type.INPUT
        if rail_type == "output":
            return self._rail_type.OUTPUT
        raise ValueError(f"unsupported rail_type {rail_type!r}")

    async def check(self, messages, rail_type):
        result = await self._rails.check_async(
            messages,
            rail_types=[self._rail_kind(rail_type)],
        )
        return {
            "status": status_value(result.status),
            "content": string_attr_or_empty(result, "content"),
            "rail": optional_string_attr(result, "rail"),
        }

    def has_streaming_output_rails(self):
        output = self._output_rails_config()
        flows = getattr(output, "flows", None) if output is not None else None
        return bool(flows)

    def ensure_streaming_output_supported(self):
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

    def _output_rails_config(self):
        config = getattr(self._rails, "config", None)
        rails = getattr(config, "rails", None)
        return getattr(rails, "output", None)

    async def monitor_stream(self, request_id, messages, queue, streams):
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


worker = None
streams = {}


async def handle_message(message):
    global worker

    request_id = str(message.get("id", ""))
    command = message.get("command")
    try:
        if command == "init":
            worker = GuardrailsWorker(message.get("config") or {})
            response(
                request_id,
                {
                    "python": sys.executable,
                    "version": ".".join(str(part) for part in sys.version_info[:3]),
                },
            )
        elif worker is None:
            raise RuntimeError("NeMo Guardrails local Python worker is not initialized")
        elif command == "check":
            response(
                request_id,
                await worker.check(message.get("messages") or [], message.get("rail_type")),
            )
        elif command == "has_streaming_output_rails":
            response(request_id, {"enabled": worker.has_streaming_output_rails()})
        elif command == "ensure_streaming_output_supported":
            worker.ensure_streaming_output_supported()
            response(request_id)
        elif command == "stream_start":
            queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAXSIZE)
            streams[request_id] = queue
            asyncio.create_task(worker.monitor_stream(request_id, message.get("messages") or [], queue, streams))
        elif command == "stream_text":
            queue = streams.get(request_id)
            if queue is not None:
                await queue.put(message.get("text") or "")
        elif command == "stream_end":
            queue = streams.get(request_id)
            if queue is not None:
                await queue.put(None)
        else:
            raise RuntimeError(f"unknown worker command {command!r}")
    except Exception as err:
        if command and command.startswith("stream_"):
            stream_error(request_id, err)
        else:
            error_response(request_id, err)


async def main():
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
            await handle_message(message)
        else:
            asyncio.create_task(handle_message(message))


asyncio.run(main())
