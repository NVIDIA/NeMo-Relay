// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The redirect decision matrix.
 *
 * `decideRedirect` is pure so this suite can cover every branch without a pi
 * runtime. The branch that matters most is the mismatch guard: the gateway
 * forwards to one statically configured upstream per API family, so a redirect
 * is only correct when that upstream is the endpoint the model would otherwise
 * have called. Getting this wrong does not cost spans, it breaks the session.
 *
 * Run: node --test integrations/pi/test/*.test.mjs
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

const { decideRedirect, isNotable, normalizeBaseUrl } = await import('../src/provider-redirect.ts');

const GATEWAY = 'http://127.0.0.1:4040';

/** The real NVIDIA catalog entry, which is what this was developed against. */
const nvidiaModel = {
  id: 'nvidia/nemotron-3-super-120b-a12b',
  api: 'openai-completions',
  provider: 'nvidia',
  baseUrl: 'https://integrate.api.nvidia.com/v1',
};

const anthropicModel = {
  id: 'claude-sonnet-4-5',
  api: 'anthropic-messages',
  provider: 'anthropic',
  baseUrl: 'https://api.anthropic.com',
};

const matchConfig = (extra = {}) => ({
  gatewayUrl: GATEWAY,
  mode: 'match',
  openaiUpstream: 'https://integrate.api.nvidia.com/v1',
  anthropicUpstream: 'https://api.anthropic.com',
  ...extra,
});

const none = new Set();

describe('redirect decision', () => {
  it('redirects when the gateway forwards to the model’s own endpoint', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig(), none);
    assert.equal(decision.kind, 'redirect');
    assert.equal(decision.provider, 'nvidia');
    assert.equal(decision.upstream, 'https://integrate.api.nvidia.com/v1');
  });

  // The failure this guard exists for: pi's catalog says NVIDIA, the gateway
  // forwards to OpenAI, and redirecting would send the request to a provider
  // that has never heard of the model or the key.
  it('refuses when the gateway forwards somewhere else', () => {
    const decision = decideRedirect(
      nvidiaModel,
      matchConfig({ openaiUpstream: 'https://api.openai.com/v1' }),
      none,
    );
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'upstream-mismatch');
    assert.match(decision.reason, /wrong provider/);
  });

  it('picks the upstream matching the model’s API family, not the other one', () => {
    // Anthropic model, anthropic upstream matches, openai upstream does not.
    const decision = decideRedirect(
      anthropicModel,
      matchConfig({ openaiUpstream: 'https://api.openai.com/v1' }),
      none,
    );
    assert.equal(decision.kind, 'redirect');
    assert.equal(decision.api, 'anthropic-messages');
  });

  it('refuses an API the gateway has no route for', () => {
    for (const api of ['google-generative-ai', 'bedrock-converse-stream', 'mistral-conversations']) {
      const decision = decideRedirect({ ...nvidiaModel, api }, matchConfig(), none);
      assert.equal(decision.kind, 'skip', api);
      assert.equal(decision.code, 'unserviceable-api', api);
    }
  });

  // Launched outside `nemo-relay run --agent pi`, so nothing told the extension
  // what the gateway fronts. Staying put costs spans; guessing costs the session.
  it('refuses when the gateway upstream is unknown', () => {
    const decision = decideRedirect(
      nvidiaModel,
      { gatewayUrl: GATEWAY, mode: 'match' },
      none,
    );
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'unknown-upstream');
    assert.match(decision.reason, /NEMO_RELAY_PI_REDIRECT=force/);
  });

  it('force skips the match check, match does not', () => {
    const forced = decideRedirect(
      nvidiaModel,
      { gatewayUrl: GATEWAY, mode: 'force', openaiUpstream: 'https://api.openai.com/v1' },
      none,
    );
    assert.equal(forced.kind, 'redirect');
    assert.match(forced.reason, /not checked/);
  });

  it('off disables redirection entirely', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig({ mode: 'off' }), none);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'disabled');
  });

  it('has nothing to decide before a model is resolved', () => {
    const decision = decideRedirect(undefined, matchConfig(), none);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'no-model');
  });

  // Without this, the second call compares the gateway URL against the upstream
  // and reports a mismatch for a provider it redirected itself.
  it('is idempotent once a provider has been redirected', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig(), new Set(['nvidia']));
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'already-redirected');
  });
});

describe('what reaches the trace', () => {
  // A mark per session for "no model yet" is noise; a mark explaining why LLM
  // spans are absent is the whole point.
  it('records outcomes that explain a trace, not bookkeeping', () => {
    assert.equal(isNotable(decideRedirect(nvidiaModel, matchConfig(), none)), true);
    assert.equal(
      isNotable(decideRedirect(nvidiaModel, matchConfig({ openaiUpstream: 'https://x.test' }), none)),
      true,
    );
    assert.equal(isNotable(decideRedirect(undefined, matchConfig(), none)), false);
    assert.equal(isNotable(decideRedirect(nvidiaModel, matchConfig(), new Set(['nvidia']))), false);
  });
});

describe('base URL comparison', () => {
  it('ignores trailing slashes and host case', () => {
    assert.equal(
      normalizeBaseUrl('https://API.Nvidia.com/v1/'),
      normalizeBaseUrl('https://api.nvidia.com/v1'),
    );
  });

  // `/v1` is a real path segment, not noise: providers host several API
  // versions, and equating them would redirect into the wrong one.
  it('does not treat a /v1 suffix as equivalent to its absence', () => {
    assert.notEqual(
      normalizeBaseUrl('https://api.anthropic.com'),
      normalizeBaseUrl('https://api.anthropic.com/v1'),
    );
  });

  it('keeps ports distinct', () => {
    assert.notEqual(
      normalizeBaseUrl('http://127.0.0.1:4040'),
      normalizeBaseUrl('http://127.0.0.1:4141'),
    );
  });
});
