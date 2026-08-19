// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Points pi's model traffic at the NeMo Relay gateway.
 *
 * pi has no base-URL flag and no generic environment override -- it resolves a
 * base URL per model from a generated catalog -- so redirection has to happen
 * inside the extension. The mechanism is one call:
 *
 * ```ts
 * pi.registerProvider(providerId, { baseUrl: gatewayUrl });
 * ```
 *
 * With `baseUrl` and no `models`, pi rewrites the URL of every existing model
 * for that provider and keeps their API, headers, costs and context windows
 * (`applyExtension`, `core/provider-composer.ts:215`). That is far cheaper than
 * pi's own `custom-provider-*` examples, which register a `streamSimple` and
 * re-implement a provider protocol; the extension stays a thin client.
 *
 * **Redirection is conditional, and the condition is the whole design.** The
 * gateway forwards to one statically configured upstream per API family and a
 * client cannot override it per request -- inbound internal dispatch headers
 * are stripped. So sending a model's traffic to the gateway is only correct
 * when the gateway's upstream *is* the endpoint that model would otherwise
 * call. Redirecting an NVIDIA model into a gateway configured for
 * `api.openai.com` does not degrade to "no spans"; it breaks the session.
 *
 * The launcher therefore passes the gateway's own upstreams
 * (`NEMO_RELAY_PI_OPENAI_UPSTREAM`, `NEMO_RELAY_PI_ANTHROPIC_UPSTREAM`) and
 * this module redirects only on a match. Unmatched models keep their own
 * endpoint and produce no LLM spans, which is the honest outcome.
 */

/** The API families the gateway serves, mapped to the upstream that backs each. */
const SERVICEABLE_APIS: Record<string, 'openai' | 'anthropic'> = {
  'openai-completions': 'openai',
  'openai-responses': 'openai',
  'anthropic-messages': 'anthropic',
};

export type RedirectConfig = {
  /** Gateway base URL; the root, never root + `/v1`. */
  gatewayUrl: string;
  /** What the gateway forwards OpenAI-compatible traffic to, when known. */
  openaiUpstream?: string;
  /** What the gateway forwards Anthropic traffic to, when known. */
  anthropicUpstream?: string;
  /**
   * `match` (default) redirects only when the model's endpoint is the
   * gateway's upstream. `force` skips that check for operators who know their
   * gateway is correct but did not launch through `nemo-relay run`. `off`
   * disables redirection entirely.
   */
  mode: 'match' | 'force' | 'off';
};

/** A model, narrowed to the fields redirection depends on. */
export type RedirectModel = {
  id: string;
  api: string;
  provider: string;
  baseUrl: string;
};

/**
 * Why a redirect did or did not happen.
 *
 * `code` exists so callers can tell a transient skip from one worth recording.
 * `no-model` and `already-redirected` are bookkeeping -- a `model_select` is
 * either coming or the work is done -- while the rest are the reasons a user
 * ends up staring at a trace with no LLM spans, and belong in that trace.
 */
export type RedirectSkipCode =
  | 'disabled'
  | 'no-model'
  | 'already-redirected'
  | 'unserviceable-api'
  | 'unknown-upstream'
  | 'upstream-mismatch';

export type RedirectDecision =
  | { kind: 'redirect'; provider: string; api: string; upstream: string; reason: string }
  | { kind: 'skip'; code: RedirectSkipCode; provider?: string; api?: string; reason: string };

/** Whether this outcome explains something a trace reader would otherwise have to guess. */
export function isNotable(decision: RedirectDecision): boolean {
  return decision.kind === 'redirect' || !['no-model', 'already-redirected'].includes(decision.code);
}

/**
 * Normalize a base URL for comparison.
 *
 * Compares scheme, host, port and path with trailing slashes removed. A
 * trailing `/v1` is *not* stripped: pi's Anthropic catalog entry is
 * `https://api.anthropic.com` while its OpenAI entry is
 * `https://api.openai.com/v1`, and the gateway's two defaults match those
 * exactly, so treating `/v1` as noise would equate genuinely different
 * endpoints on providers that host several API versions.
 */
export function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  try {
    const url = new URL(trimmed);
    const path = url.pathname.replace(/\/+$/, '');
    return `${url.protocol}//${url.host.toLowerCase()}${path}`;
  } catch {
    return trimmed.toLowerCase();
  }
}

/**
 * Decide whether this model's provider should be pointed at the gateway.
 *
 * Pure, so the decision matrix is testable without a pi runtime. `redirected`
 * carries providers already pointed at the gateway, which makes the call
 * idempotent: once a provider is redirected, `model.baseUrl` reads back as the
 * gateway and the upstream comparison would otherwise fail on the second call.
 */
export function decideRedirect(
  model: RedirectModel | undefined,
  config: RedirectConfig,
  redirected: ReadonlySet<string>,
): RedirectDecision {
  if (config.mode === 'off') {
    return {
      kind: 'skip',
      code: 'disabled',
      reason: 'redirection disabled by NEMO_RELAY_PI_REDIRECT=off',
    };
  }
  if (!model) {
    return { kind: 'skip', code: 'no-model', reason: 'no model selected yet' };
  }
  if (redirected.has(model.provider)) {
    return {
      kind: 'skip',
      code: 'already-redirected',
      provider: model.provider,
      api: model.api,
      reason: 'already redirected',
    };
  }

  const family = SERVICEABLE_APIS[model.api];
  if (!family) {
    // Seven of pi's 39 built-in providers speak an API the gateway has no
    // route for: Bedrock, Azure OpenAI Responses, Google, Google Vertex,
    // Mistral, OpenAI Codex, and Radius (`pi-messages`). Redirecting them would
    // 404 rather than degrade.
    //
    // Counted from `builtinProviders()`, not from the 38 files in
    // `providers/data/` -- Radius is a purely dynamic provider with no static
    // catalog entry, so a file count silently loses it.
    return {
      kind: 'skip',
      code: 'unserviceable-api',
      provider: model.provider,
      api: model.api,
      reason: `the gateway serves no route for the ${model.api} API`,
    };
  }

  const upstream = family === 'openai' ? config.openaiUpstream : config.anthropicUpstream;

  if (config.mode === 'force') {
    return {
      kind: 'redirect',
      provider: model.provider,
      api: model.api,
      upstream: model.baseUrl,
      reason: 'NEMO_RELAY_PI_REDIRECT=force; upstream match not checked',
    };
  }

  if (!upstream) {
    // Launched outside `nemo-relay run --agent pi`, so the gateway's upstream
    // is unknown. Staying put is the safe default: a wrong redirect breaks the
    // session, a skipped one only costs spans.
    return {
      kind: 'skip',
      code: 'unknown-upstream',
      provider: model.provider,
      api: model.api,
      reason:
        `the gateway's ${family} upstream is unknown, so a redirect cannot be verified as safe; ` +
        `launch through \`nemo-relay run --agent pi\`, or set NEMO_RELAY_PI_REDIRECT=force`,
    };
  }

  if (normalizeBaseUrl(upstream) !== normalizeBaseUrl(model.baseUrl)) {
    return {
      kind: 'skip',
      code: 'upstream-mismatch',
      provider: model.provider,
      api: model.api,
      reason:
        `model targets ${model.baseUrl} but the gateway forwards ${family} traffic to ${upstream}; ` +
        `redirecting would send the request to the wrong provider`,
    };
  }

  return {
    kind: 'redirect',
    provider: model.provider,
    api: model.api,
    upstream: model.baseUrl,
    reason: `gateway forwards ${family} traffic to the same endpoint (${upstream})`,
  };
}

/** Read redirection configuration from the environment the launcher sets. */
export function redirectConfigFromEnv(gatewayUrl: string): RedirectConfig {
  const raw = process.env.NEMO_RELAY_PI_REDIRECT;
  const mode = raw === 'off' ? 'off' : raw === 'force' ? 'force' : 'match';
  // Spread conditionally rather than assigning `undefined`: the package builds
  // under `exactOptionalPropertyTypes`, where an explicit `undefined` is not
  // the same as an absent key.
  const openaiUpstream = process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM;
  const anthropicUpstream = process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM;
  return {
    gatewayUrl,
    mode,
    ...(openaiUpstream ? { openaiUpstream } : {}),
    ...(anthropicUpstream ? { anthropicUpstream } : {}),
  };
}
