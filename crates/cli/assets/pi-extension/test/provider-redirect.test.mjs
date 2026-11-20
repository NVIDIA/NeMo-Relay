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
 * Run: npm test --workspace=nemo-relay-pi
 */
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

const { decideRedirect, isNotable, normalizeBaseUrl, redactedUrl } = await import('../src/provider-redirect.ts');

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
  mode: 'match',
  canNameUpstream: false,
  openaiUpstream: 'https://integrate.api.nvidia.com/v1',
  anthropicUpstream: 'https://api.anthropic.com',
  ...extra,
});

const none = new Set();

describe('redirect decision', () => {
  it('refuses to name an upstream the gateway would reject', () => {
    const naming = matchConfig({ canNameUpstream: true });
    for (const [baseUrl, reason] of [
      ['https://10.20.30.40/v1', /public or loopback/],
      ['https://[fd00::1]/v1', /public or loopback/],
      ['https://[::ffff:10.0.0.1]/v1', /public or loopback/],
      ['https://[64:ff9b::a00:1]/v1', /public or loopback/],
      ['https://[64:ff9b::169.254.169.254]/v1', /public or loopback/],
      ['http://remote.example/v1', /https unless/],
      ['https://user:secret@llm.example/v1', /credentials/],
      ['https://llm.example/v1?team=a', /query/],
      ['https://llm.example/v1#frag', /fragment/],
    ]) {
      const model = { ...nvidiaModel, baseUrl };
      const decision = decideRedirect(model, naming, none, [model]);
      assert.equal(decision.kind, 'skip', baseUrl);
      assert.equal(decision.code, 'upstream-not-nameable', baseUrl);
      assert.match(decision.reason, reason);
    }
    for (const baseUrl of [
      'https://llm.example/v1',
      'https://[::ffff:8.8.8.8]/v1',
      'https://[64:ff9b::808:808]/v1',
      'https://[64:ff9b::127.0.0.1]/v1',
      'http://127.0.0.1:11434/v1',
      'http://localhost:8080',
      'http://[::1]:8080',
    ]) {
      const model = { ...nvidiaModel, baseUrl };
      assert.equal(decideRedirect(model, naming, none, [model]).kind, 'redirect', baseUrl);
    }
  });

  it('keeps credentials in a model endpoint out of every reason', () => {
    const secret = { ...nvidiaModel, baseUrl: 'https://user:secret@llm.example/v1' };
    const sibling = { ...nvidiaModel, id: 'nvidia/other', baseUrl: 'https://user:secret@other.example/v1' };
    const decisions = [
      decideRedirect(secret, matchConfig({ canNameUpstream: true }), none, [secret]),
      decideRedirect(secret, matchConfig({ openaiUpstream: 'https://api.openai.com/v1' }), none, [secret]),
      decideRedirect(nvidiaModel, matchConfig({ canNameUpstream: true, openaiUpstream: undefined }), none, [
        nvidiaModel,
        sibling,
      ]),
      decideRedirect(nvidiaModel, matchConfig(), none, [nvidiaModel, sibling]),
      decideRedirect(
        nvidiaModel,
        matchConfig({ openaiUpstream: 'https://user:secret@integrate.api.nvidia.com/v1' }),
        none,
        [],
      ),
    ];
    assert.deepEqual(
      decisions.map((decision) => (decision.kind === 'skip' ? decision.code : decision.kind)),
      [
        'upstream-not-nameable',
        'upstream-mismatch',
        'provider-mixed-endpoints',
        'provider-mixed-endpoints',
        'redirect',
      ],
    );
    for (const decision of decisions) {
      assert.doesNotMatch(decision.reason, /secret/, decision.reason);
      assert.match(decision.reason, /https:\/\//);
    }
    assert.equal(redactedUrl('https://user:secret@llm.example/v1'), 'https://llm.example/v1');
    assert.equal(redactedUrl('https://llm.example/v1?note=a@b'), 'https://llm.example/v1?note=a@b');
  });

  it('skips a natively registered provider, which an overlay would replace', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig({ canNameUpstream: true }), none, [nvidiaModel], true);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'native-provider');
    assert.equal(isNotable(decision), true);
  });

  it('skips a model that already points at the relay, whatever the mode', () => {
    for (const mode of ['match', 'force']) {
      const decision = decideRedirect(
        { ...nvidiaModel, baseUrl: 'http://127.0.0.1:4040/' },
        matchConfig({ mode, canNameUpstream: true, relayBaseUrl: 'http://127.0.0.1:4040' }),
        none,
        [],
      );
      assert.equal(decision.kind, 'skip', mode);
      assert.equal(decision.code, 'already-redirected', mode);
    }
  });

  it('redirects when the gateway forwards to the model’s own endpoint', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig(), none, []);
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
      [],
    );
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'upstream-mismatch');
    assert.match(decision.reason, /wrong provider/);
  });

  // pi's `modelRegistry.getAll()` returns the selected model too, so with a real pi
  // runtime a model is always one of its own siblings. While the whole-provider scan
  // ran first, this -- the ordinary mismatch, and the commonest outcome there is --
  // was reported as `provider-mixed-endpoints`, naming the selected model as the
  // sibling blocking itself. Same decision; the code has to be the actionable one.
  it('reports an ordinary mismatch as itself when the model is its own sibling', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig({ openaiUpstream: 'https://api.openai.com/v1' }), none, [
      nvidiaModel,
    ]);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'upstream-mismatch');
  });

  it('picks the upstream matching the model’s API family, not the other one', () => {
    // Anthropic model, anthropic upstream matches, openai upstream does not.
    const decision = decideRedirect(
      anthropicModel,
      matchConfig({ openaiUpstream: 'https://api.openai.com/v1' }),
      none,
      [],
    );
    assert.equal(decision.kind, 'redirect');
    assert.equal(decision.api, 'anthropic-messages');
  });

  it('refuses an API the gateway has no route for', () => {
    for (const api of ['google-generative-ai', 'bedrock-converse-stream', 'mistral-conversations']) {
      const decision = decideRedirect({ ...nvidiaModel, api }, matchConfig(), none, []);
      assert.equal(decision.kind, 'skip', api);
      assert.equal(decision.code, 'unserviceable-api', api);
    }
  });

  // `model.api` is a free-form string in pi, so the serviceable-API lookup must not
  // answer for names every JavaScript object inherits. On an object-literal map
  // `constructor` and `__proto__` read back truthy, and this model -- whose endpoint
  // happens to be the anthropic upstream -- was redirected into a route the gateway
  // does not have, with the inherited value rendered into the reason string.
  it('refuses an API named after an inherited object property', () => {
    for (const api of ['constructor', 'toString', 'valueOf', '__proto__']) {
      const decision = decideRedirect({ ...anthropicModel, api }, matchConfig(), none, []);
      assert.equal(decision.kind, 'skip', api);
      assert.equal(decision.code, 'unserviceable-api', api);
    }
  });

  // Launched outside `nemo-relay run --agent pi`, so nothing told the extension
  // what the gateway fronts. Staying put costs spans; guessing costs the session.
  it('refuses when the gateway upstream is unknown', () => {
    const decision = decideRedirect(nvidiaModel, { mode: 'match', canNameUpstream: false }, none, []);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'unknown-upstream');
    assert.match(decision.reason, /NEMO_RELAY_PI_REDIRECT=force/);
  });

  it('force skips the match check, match does not', () => {
    const forced = decideRedirect(
      nvidiaModel,
      { mode: 'force', canNameUpstream: false, openaiUpstream: 'https://api.openai.com/v1' },
      none,
      [],
    );
    assert.equal(forced.kind, 'redirect');
    assert.match(forced.reason, /not checked/);
  });

  it('off disables redirection entirely', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig({ mode: 'off' }), none, []);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'disabled');
  });

  it('has nothing to decide before a model is resolved', () => {
    const decision = decideRedirect(undefined, matchConfig(), none, []);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'no-model');
  });

  // Without this, the second call compares the gateway URL against the upstream
  // and reports a mismatch for a provider it redirected itself.
  it('is idempotent once a provider has been redirected', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig(), new Set(['nvidia']), []);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'already-redirected');
  });
});

describe('what reaches the trace', () => {
  // A mark per session for "no model yet" is noise; a mark explaining why LLM
  // spans are absent is the whole point.
  it('records outcomes that explain a trace, not bookkeeping', () => {
    assert.equal(isNotable(decideRedirect(nvidiaModel, matchConfig(), none, [])), true);
    assert.equal(
      isNotable(decideRedirect(nvidiaModel, matchConfig({ openaiUpstream: 'https://x.test' }), none, [])),
      true,
    );
    assert.equal(isNotable(decideRedirect(undefined, matchConfig(), none, [])), false);
    assert.equal(isNotable(decideRedirect(nvidiaModel, matchConfig(), new Set(['nvidia']), [])), false);
  });
});

describe('base URL comparison', () => {
  it('ignores trailing slashes and host case', () => {
    assert.equal(normalizeBaseUrl('https://API.Nvidia.com/v1/'), normalizeBaseUrl('https://api.nvidia.com/v1'));
  });

  // `/v1` is a real path segment, not noise: providers host several API
  // versions, and equating them would redirect into the wrong one.
  it('does not treat a /v1 suffix as equivalent to its absence', () => {
    assert.notEqual(normalizeBaseUrl('https://api.anthropic.com'), normalizeBaseUrl('https://api.anthropic.com/v1'));
  });

  it('keeps ports distinct', () => {
    assert.notEqual(normalizeBaseUrl('http://127.0.0.1:4040'), normalizeBaseUrl('http://127.0.0.1:4141'));
  });
});

// The failure this exists for, built from pi 0.84.0's real Fireworks catalog:
// 18 anthropic-messages models at `/inference` and 4 openai-completions models at
// `/inference/v1`, under one provider. `registerProvider(name, {baseUrl})` rewrites
// every model of a provider, so deciding from the selected model alone moves its
// siblings to an endpoint that has never heard of them.
describe('a provider whose models span API families', () => {
  const ANTHROPIC = {
    id: 'accounts/fireworks/models/deepseek-v4-flash',
    api: 'anthropic-messages',
    provider: 'fireworks',
    baseUrl: 'https://api.fireworks.ai/inference',
  };
  const OPENAI = {
    id: 'accounts/fireworks/models/glm-5p2',
    api: 'openai-completions',
    provider: 'fireworks',
    baseUrl: 'https://api.fireworks.ai/inference/v1',
  };
  const catalog = [ANTHROPIC, OPENAI];

  it('refuses to redirect when a sibling would be sent somewhere the gateway does not front', () => {
    const config = {
      mode: 'match',
      canNameUpstream: false,
      anthropicUpstream: 'https://api.fireworks.ai/inference',
      openaiUpstream: 'https://api.openai.com/v1',
    };
    // Judged alone, the selected model looks perfectly safe.
    assert.equal(decideRedirect(ANTHROPIC, config, new Set(), []).kind, 'redirect');

    // Judged with its siblings, it is not: the openai-completions models would be
    // pointed at api.openai.com carrying a Fireworks key and a Fireworks model id.
    const decision = decideRedirect(ANTHROPIC, config, new Set(), catalog);
    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'provider-mixed-endpoints');
    assert.match(decision.reason, /glm-5p2/, 'the reason must name the sibling that fails');
    assert.match(decision.reason, /openai-completions/);
  });

  it('redirects once both upstreams point at that provider', () => {
    const config = {
      mode: 'match',
      canNameUpstream: false,
      anthropicUpstream: 'https://api.fireworks.ai/inference',
      openaiUpstream: 'https://api.fireworks.ai/inference/v1',
    };
    for (const model of catalog) {
      const decision = decideRedirect(model, config, new Set(), catalog);
      assert.equal(decision.kind, 'redirect', `${model.api} should redirect: ${decision.reason}`);
    }
  });

  it('is notable, so the trace explains why there are no LLM spans', () => {
    const config = {
      mode: 'match',
      canNameUpstream: false,
      anthropicUpstream: 'https://api.fireworks.ai/inference',
      openaiUpstream: 'https://api.openai.com/v1',
    };
    assert.equal(isNotable(decideRedirect(ANTHROPIC, config, new Set(), catalog)), true);
  });
});

describe('naming an upstream the gateway does not front', () => {
  /** A launched session holds the proxy credential the gateway requires before honouring the header. */
  const namingConfig = (extra = {}) => matchConfig({ canNameUpstream: true, ...extra });

  it('redirects past a mismatch by naming the model endpoint', () => {
    const decision = decideRedirect(nvidiaModel, namingConfig({ openaiUpstream: 'https://api.openai.com/v1' }), none, [
      nvidiaModel,
    ]);

    assert.equal(decision.kind, 'redirect');
    assert.equal(decision.namedUpstream, nvidiaModel.baseUrl);
  });

  it('redirects when the gateway upstream is unknown', () => {
    const decision = decideRedirect(nvidiaModel, { mode: 'match', canNameUpstream: true }, none, [nvidiaModel]);

    assert.equal(decision.kind, 'redirect');
    assert.equal(decision.namedUpstream, nvidiaModel.baseUrl);
  });

  // Without the credential the gateway ignores the header, so redirecting would land
  // on its configured upstream -- the wrong provider, which is the exact break the
  // static check exists to prevent.
  it('still refuses without the credential, because the header would be ignored', () => {
    const decision = decideRedirect(nvidiaModel, matchConfig({ openaiUpstream: 'https://api.openai.com/v1' }), none, [
      nvidiaModel,
    ]);

    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'upstream-mismatch');
  });

  it('names nothing when the gateway already fronts the endpoint', () => {
    const decision = decideRedirect(nvidiaModel, namingConfig(), none, [nvidiaModel]);

    assert.equal(decision.kind, 'redirect');
    assert.equal(decision.namedUpstream, undefined);
  });

  // One endpoint is named for the whole provider, because `registerProvider` rewrites
  // every model of it. A sibling elsewhere would be pointed at a host that has never
  // heard of it -- a broken session rather than a missing span.
  it('refuses a provider whose models do not share one endpoint', () => {
    const sibling = {
      id: 'nvidia/other',
      api: 'openai-completions',
      provider: 'nvidia',
      baseUrl: 'https://other.nvidia.example/v1',
    };
    const decision = decideRedirect(nvidiaModel, namingConfig({ openaiUpstream: 'https://api.openai.com/v1' }), none, [
      nvidiaModel,
      sibling,
    ]);

    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'provider-mixed-endpoints');
  });

  // An unserviceable sibling has no gateway route at all, so naming an endpoint for
  // the provider would move it somewhere the gateway cannot forward it.
  it('refuses a provider with a sibling the gateway cannot route', () => {
    const sibling = {
      id: 'nvidia/gemini-ish',
      api: 'google-generative-ai',
      provider: 'nvidia',
      baseUrl: nvidiaModel.baseUrl,
    };
    const decision = decideRedirect(nvidiaModel, namingConfig({ openaiUpstream: 'https://api.openai.com/v1' }), none, [
      nvidiaModel,
      sibling,
    ]);

    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'provider-mixed-endpoints');
  });

  // The gateway serves no route for the API at all, so there is nothing to name.
  it('still refuses an unserviceable API', () => {
    const decision = decideRedirect({ ...nvidiaModel, api: 'google-generative-ai' }, namingConfig(), none, []);

    assert.equal(decision.kind, 'skip');
    assert.equal(decision.code, 'unserviceable-api');
  });
});
