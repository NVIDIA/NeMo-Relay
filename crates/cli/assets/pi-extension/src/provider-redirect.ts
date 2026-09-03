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
 * (`applyExtension`, pi `v0.84.0`, `core/provider-composer.ts:215`). That is far
 * cheaper than pi's own `custom-provider-*` examples, which register a
 * `streamSimple` and re-implement a provider protocol; the extension stays a
 * thin client.
 *
 * **The gateway has to know where to forward, and there are two ways to tell
 * it.** It sends each API family to one statically configured upstream, so
 * redirecting a model whose endpoint is not that upstream would reach the wrong
 * provider: an NVIDIA model in a gateway configured for `api.openai.com` does
 * not degrade to "no spans", it breaks the session.
 *
 * - **The endpoint already matches.** The launcher passes the gateway's own
 *   upstreams (`NEMO_RELAY_PI_OPENAI_UPSTREAM`,
 *   `NEMO_RELAY_PI_ANTHROPIC_UPSTREAM`), and a model already targeting one of
 *   them is redirected as it is.
 * - **The endpoint is named.** Otherwise, when this run holds the launcher's
 *   proxy credential, the redirect carries `x-nemo-relay-upstream-base-url` and
 *   the gateway forwards there instead. This is what makes pi's arbitrary
 *   providers reachable.
 *
 * Without that credential the header would be ignored, so a standalone daemon
 * redirects only on a match and unmatched models keep their own endpoint and
 * produce no LLM spans.
 */

/**
 * The API families the gateway serves, mapped to the upstream that backs each.
 *
 * A `Map` rather than an object literal because pi does not constrain this key:
 * `api` is `KnownApi | (string & {})` and reaches us from models.json, remote
 * catalogs and other extensions. On an object literal an `api` of `constructor`
 * or `__proto__` resolves through the prototype chain to something truthy, so
 * the unserviceable-API guard -- whose whole job is to keep a model the gateway
 * cannot route out of the gateway -- would not fire, and the model was scored
 * against the Anthropic upstream instead.
 */
const SERVICEABLE_APIS = new Map<string, 'openai' | 'anthropic'>([
  ['openai-completions', 'openai'],
  ['openai-responses', 'openai'],
  ['anthropic-messages', 'anthropic'],
]);

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
  /**
   * This invocation's transparent-proxy credential, when the launcher set one.
   *
   * A gateway started by `nemo-relay run` authenticates its own client before a
   * request intercept can rewrite the route, and rejects a provider call that
   * does not present it (`crates/cli/src/provider_auth.rs`). Hook posts are not
   * covered by that check -- only provider passthrough is -- so without this the
   * redirect succeeds and then every model call comes back 401, which is the one
   * outcome the redirect exists to produce spans for.
   *
   * Absent for a standalone `nemo-relay --bind` daemon, which requires no
   * credential, so the key is omitted rather than sent empty.
   */
  proxyToken?: string;
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
  | 'upstream-mismatch'
  | 'provider-mixed-endpoints';

export type RedirectDecision =
  | {
      kind: 'redirect';
      provider: string;
      api: string;
      upstream: string;
      reason: string;
      /**
       * The endpoint the gateway must be told to forward to, when it does not
       * already front it.
       *
       * Absent on a static redirect, where the gateway's configured upstream is
       * the destination and saying so again would only add a header. Present
       * means every request for this provider carries it, so the gateway can
       * reach a provider it was never configured for.
       */
      namedUpstream?: string;
    }
  | { kind: 'skip'; code: RedirectSkipCode; provider?: string; api?: string; reason: string };

/**
 * Whether one model could be pointed at the gateway without breaking it.
 *
 * The same two conditions the selected model must satisfy: the gateway serves its
 * API family, and it already targets the endpoint the gateway forwards that family
 * to. Applied to every sibling because the registration is provider-wide.
 */
function isSafeToRedirect(model: RedirectModel, config: RedirectConfig): boolean {
  const family = SERVICEABLE_APIS.get(model.api);
  if (!family) return false;
  const upstream = family === 'openai' ? config.openaiUpstream : config.anthropicUpstream;
  if (!upstream) return false;
  return normalizeBaseUrl(upstream) === normalizeBaseUrl(model.baseUrl);
}

/**
 * Whether this invocation may tell the gateway where to forward.
 *
 * The gateway honors a named upstream only from a request carrying this run's
 * proxy credential, so without one the header would be ignored and the redirect
 * would land on the gateway's own configured upstream -- the wrong provider,
 * which is the failure the static check exists to prevent. Having the credential
 * is therefore the precondition for redirecting past a mismatch, not a detail of
 * how the request is authenticated.
 */
function canNameUpstream(config: RedirectConfig): boolean {
  return Boolean(config.proxyToken);
}

/**
 * Redirect a provider the gateway does not front, by naming its endpoint.
 *
 * The sibling scan survives from the static path but changes its question. There
 * it asked whether every model already targets the gateway's configured
 * upstream; here one endpoint is named for the whole provider, so it asks
 * whether every model shares the endpoint about to be named. A provider mixing
 * endpoints is still refused -- naming one of them would point the rest at a host
 * that has never heard of them, which is the same broken session the static path
 * refuses to create.
 */
function nameUpstream(
  model: RedirectModel,
  siblings: readonly RedirectModel[],
): RedirectDecision {
  const target = normalizeBaseUrl(model.baseUrl);
  const unsafe = siblings.find(
    (sibling) =>
      !SERVICEABLE_APIS.has(sibling.api) || normalizeBaseUrl(sibling.baseUrl) !== target,
  );
  if (unsafe) {
    return {
      kind: 'skip',
      code: 'provider-mixed-endpoints',
      provider: model.provider,
      api: model.api,
      reason:
        `redirecting ${model.provider} would also move its ${unsafe.api} models, and ` +
        `${unsafe.id} targets ${unsafe.baseUrl} rather than ${model.baseUrl}; ` +
        `a provider is named one endpoint, so its models must share one`,
    };
  }
  return {
    kind: 'redirect',
    provider: model.provider,
    api: model.api,
    upstream: model.baseUrl,
    namedUpstream: model.baseUrl,
    reason: `the gateway does not front ${model.baseUrl}, so each request names it`,
  };
}

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
  /**
   * Every model of `model.provider`, as the catalog had them **before** any
   * redirect. Optional only so an older caller still type-checks; omitting it
   * restores the per-model check, which is unsound for a mixed provider.
   */
  siblings: readonly RedirectModel[] = [],
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

  const family = SERVICEABLE_APIS.get(model.api);
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
    // Not knowing the gateway's upstream stops mattering once we can name ours:
    // the comparison existed to prove a redirect lands on the right provider, and
    // naming the endpoint proves it directly.
    if (canNameUpstream(config)) {
      return nameUpstream(model, siblings);
    }
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

  // Ahead of the sibling scan, deliberately. pi's `modelRegistry.getAll()` returns
  // the selected model too, so it is one of its own siblings -- and while the scan
  // ran first, an ordinary endpoint mismatch was reported as
  // `provider-mixed-endpoints`, naming the selected model as the sibling that
  // blocked it. Same decision either way; this is the code an operator can act on.
  if (normalizeBaseUrl(upstream) !== normalizeBaseUrl(model.baseUrl)) {
    // A mismatch is the ordinary case for pi's arbitrary providers: 32 of its 39
    // speak an API the gateway serves, but only the two it is configured for are
    // the endpoint a model already calls. Naming the endpoint turns that from a
    // refusal into a redirect.
    if (canNameUpstream(config)) {
      return nameUpstream(model, siblings);
    }
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

  // `registerProvider(name, {baseUrl})` rewrites the URL of EVERY model of that
  // provider, so a decision made from the selected model alone is a decision made
  // on behalf of its siblings. Several pi 0.84 providers mix API families at
  // different paths -- Fireworks serves anthropic-messages at `/inference` and
  // openai-completions at `/inference/v1`; opencode adds Google models the gateway
  // cannot route at all -- so redirecting on the strength of one model points the
  // others somewhere that has never heard of them. That is not "no spans", it is a
  // broken session, mid-run, after the user changed only the model.
  const unsafe = siblings.find((sibling) => !isSafeToRedirect(sibling, config));
  if (unsafe) {
    return {
      kind: 'skip',
      code: 'provider-mixed-endpoints',
      provider: model.provider,
      api: model.api,
      reason:
        `redirecting ${model.provider} would also move its ${unsafe.api} models, and ` +
        `${unsafe.id} targets ${unsafe.baseUrl}, which the gateway does not front; ` +
        `point both upstreams at ${model.provider}, or accept no LLM spans for it`,
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
  let mode: RedirectConfig['mode'] = 'match';
  if (raw === 'off' || raw === 'force') mode = raw;
  // Spread conditionally rather than assigning `undefined`: the package builds
  // under `exactOptionalPropertyTypes`, where an explicit `undefined` is not
  // the same as an absent key.
  const openaiUpstream = process.env.NEMO_RELAY_PI_OPENAI_UPSTREAM;
  const anthropicUpstream = process.env.NEMO_RELAY_PI_ANTHROPIC_UPSTREAM;
  // Not a `NEMO_RELAY_PI_*` name, deliberately: the launcher exports this one for
  // every agent it starts, and Codex reads the same variable through its
  // `env_http_headers` provider configuration. A pi-specific alias would be a
  // second name for one value, and the two could drift.
  const proxyToken = process.env.NEMO_RELAY_PROXY_CREDENTIAL;
  return {
    gatewayUrl,
    mode,
    ...(openaiUpstream ? { openaiUpstream } : {}),
    ...(anthropicUpstream ? { anthropicUpstream } : {}),
    ...(proxyToken ? { proxyToken } : {}),
  };
}
