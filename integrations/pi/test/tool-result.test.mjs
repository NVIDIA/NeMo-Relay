// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Verifies how Pi's structured tool results are projected into Relay hooks. */
import assert from 'node:assert/strict';
import { after, before, beforeEach, describe, it } from 'node:test';

import { listen, load as loadExtension, named, stubGateway } from './harness.mjs';

const extension = (await import('../index.ts')).default;
const load = () => loadExtension(extension, { ctx: { mode: 'print', hasUI: false } });

describe('tool result projection', () => {
  let ctx;

  before(async () => {
    ctx = stubGateway();
    process.env.NEMO_RELAY_PI_GATEWAY_URL = await listen(ctx.server);
  });

  after(() => {
    ctx.server.close();
    delete process.env.NEMO_RELAY_PI_GATEWAY_URL;
  });

  beforeEach(() => {
    ctx.posts.length = 0;
  });

  async function project(result, isError = false) {
    const fire = load();
    await fire('tool_execution_end', {
      toolCallId: 'result-under-test',
      toolName: 'test-tool',
      result,
      isError,
    });
    await fire('session_shutdown', { reason: 'quit' });
    return named(ctx.posts, 'tool_execution_end')[0];
  }

  it('preserves text from a Pi Read result', async () => {
    const post = await project({
      content: [{ type: 'text', text: 'line one\nline two\n' }],
      details: { truncation: null },
    });

    assert.equal(post.result.content, 'line one\nline two\n');
    assert.deepEqual(post.result.result_keys, ['content', 'details']);
    assert.equal(post.status, 'ok');
  });

  it('preserves the diagnostic from a failed Pi Bash result', async () => {
    const post = await project(
      {
        content: [{ type: 'text', text: '(no output)\n\nCommand exited with code 42' }],
        details: {},
      },
      true,
    );

    assert.equal(post.result.content, '(no output)\n\nCommand exited with code 42');
    assert.equal(post.status, 'error');
  });

  it('preserves the error state for unsupported result types', async () => {
    const post = await project(() => undefined, true);

    assert.equal(post.result.content, 'Tool failed with an unsupported result type.');
    assert.equal(post.status, 'error');
  });

  it('joins text in order, omits images, and bounds the aggregate', async () => {
    const prefix = 'a'.repeat(1998);
    const post = await project({
      content: [
        { type: 'text', text: prefix },
        { type: 'image', data: 'binary-image-data', mimeType: 'image/png' },
        { type: 'text', text: 'BC' },
      ],
      details: {},
    });

    assert.equal(post.result.content, `${prefix}\nB... [truncated 1 chars]`);
    assert.doesNotMatch(post.result.content, /binary-image-data/);
  });

  it('does not split Unicode surrogate pairs at the truncation boundary', async () => {
    const prefix = 'a'.repeat(1999);
    const value = `${prefix}😀tail`;

    for (const result of [value, { content: [{ type: 'text', text: value }] }]) {
      ctx.posts.length = 0;
      const post = await project(result);
      assert.equal(post.result.content, `${prefix}... [truncated 6 chars]`);
    }
  });
});
