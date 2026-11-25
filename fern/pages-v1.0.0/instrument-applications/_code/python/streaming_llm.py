# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from dataclasses import dataclass

from nemo_relay import Json, LLMRequest
from nemo_relay.typed import DataclassCodec, llm_stream_execute


async def run_example() -> Json:
    """Read one stream chunk and close the producer."""

    @dataclass
    class Chunk:
        delta: str

    @dataclass
    class FinalResponse:
        text: str

    request = LLMRequest({}, {"messages": [{"role": "user", "content": "hello"}]})
    collected: list[Chunk] = []

    async def stream_impl(_request: LLMRequest) -> AsyncIterator[Chunk]:
        yield Chunk(delta="hi")

    stream = await llm_stream_execute(
        "demo-provider",
        request,
        stream_impl,
        collector=collected.append,
        finalizer=lambda: FinalResponse(text="".join(chunk.delta for chunk in collected)),
        chunk_json_codec=DataclassCodec(Chunk),
        response_json_codec=DataclassCodec(FinalResponse),
    )

    try:
        first_chunk = await anext(stream)
        # Stop early after deciding no more output is needed.
    finally:
        await stream.aclose()
    return first_chunk
