# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic OpenAI and Anthropic request/response fixtures."""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any


def split_text(text: str, chunks: int) -> list[str]:
    chunk_size = max(1, math.ceil(len(text) / chunks))
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def response_document(model: str, text: str) -> dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex}"
    item_id = f"msg_{uuid.uuid4().hex}"
    created = int(time.time())
    item = {
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
                "logprobs": [],
            }
        ],
    }
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "completed_at": created,
        "status": "completed",
        "background": False,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "model": model,
        "output": [item],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "prompt_cache_key": None,
        "reasoning": {"effort": "medium", "summary": None},
        "safety_identifier": None,
        "service_tier": "default",
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 0,
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
        "user": None,
        "metadata": {},
    }


def openai_events(model: str, text: str, chunks: int) -> list[dict[str, Any]]:
    response = response_document(model, text)
    response_id = response["id"]
    item = response["output"][0]
    item_id = item["id"]
    in_progress = {**response, "completed_at": None, "status": "in_progress", "output": []}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": in_progress},
        {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {**item, "status": "in_progress", "content": []},
        },
        {
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []},
        },
    ]
    for delta in split_text(text, chunks):
        events.append(
            {
                "type": "response.output_text.delta",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
                "logprobs": [],
            }
        )
    events.extend(
        [
            {
                "type": "response.output_text.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
            {
                "type": "response.content_part.done",
                "response_id": response_id,
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "part": item["content"][0],
            },
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": 0,
                "item": item,
            },
            {"type": "response.completed", "response": response},
        ]
    )
    return events


def anthropic_events(model: str, text: str, chunks: int) -> list[dict[str, Any]]:
    message_id = f"msg_{uuid.uuid4().hex}"
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ]
    for delta in split_text(text, chunks):
        events.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta},
            }
        )
    events.extend(
        [
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )
    return events


def make_request(
    provider: str,
    streaming: bool,
    payload_bytes: int,
    *,
    model: str,
    request_fill: str,
) -> bytes:
    payload = request_fill * payload_bytes
    if provider == "openai":
        value = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": payload}],
                }
            ],
            "stream": streaming,
        }
    else:
        value = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": payload}],
            "stream": streaming,
        }
    return json.dumps(value, separators=(",", ":")).encode()


def request_path(provider: str) -> str:
    return "/v1/responses" if provider == "openai" else "/v1/messages"


def request_headers(provider: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        headers["Authorization"] = "Bearer relay-benchmark"
    else:
        headers["x-api-key"] = "relay-benchmark"
        headers["anthropic-version"] = "2023-06-01"
    return headers
