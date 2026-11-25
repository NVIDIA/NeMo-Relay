// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { LlmRequest } from 'nemo-relay-node';
import { typedLlmStreamExecute, type Codec } from 'nemo-relay-node/typed';

/** Read one stream chunk and close the producer. */
export async function runExample() {
  type Chunk = { delta: string };
  type FinalResponse = { text: string };

  const chunkCodec: Codec<Chunk> = {
    toJson: (value) => value,
    fromJson: (json) => json as Chunk,
  };
  const responseCodec: Codec<FinalResponse> = {
    toJson: (value) => value,
    fromJson: (json) => json as FinalResponse,
  };

  const request = new LlmRequest({}, { messages: [{ role: 'user', content: 'hello' }] });
  const collected: Chunk[] = [];

  const stream = await typedLlmStreamExecute(
    'demo-provider',
    request,
    async function* () {
      yield { delta: 'hi' };
    },
    (chunk) => {
      collected.push(chunk);
    },
    () => ({ text: collected.map((chunk) => chunk.delta).join('') }),
    chunkCodec,
    responseCodec,
  );

  try {
    const firstChunk = await stream.next();
    // Stop early after deciding no more output is needed.
    return firstChunk;
  } finally {
    await stream.close();
  }
}
