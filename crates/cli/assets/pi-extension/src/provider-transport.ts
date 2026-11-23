// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** The physical model Pi has resolved, including any request-time auth endpoint. */
type Model = { provider: string; id: string; api: string; baseUrl: string };
type StreamOptions = { fetch?: typeof fetch };
type Stream = (model: Model, context: unknown, options?: StreamOptions) => unknown;
export type ChatProvider = { id: string; stream: Stream; streamSimple: Stream };
export type ProviderRegistry = {
  getAll?(): Model[];
  getAvailable?(): Model[];
  getProvider?(name: string): ChatProvider | undefined;
  getModelsOfType?(type: 'chat' | 'image' | 'classifier', provider?: string): readonly unknown[];
};
type Route = { gateway: string; headers: Record<string, string> };

const paths = new Map([
  ['openai-completions', '/chat/completions'],
  ['openai-responses', '/responses'],
  ['anthropic-messages', '/v1/messages'],
]);

/**
 * Keep Pi's physical URL through authentication, payload construction and model
 * selection. Only the final HTTP POST goes to Relay. Other provider operations
 * retain their own transports and never receive the Relay headers.
 */
export function registerChatTransports(
  registry: ProviderRegistry | undefined,
  selected: Model | undefined,
  registered: Set<string>,
  register: (provider: ChatProvider) => void,
  routeFor: (model: Model) => Route | undefined,
): boolean {
  // The typed catalog arrived with virtual models. Older Pi uses the legacy redirect.
  if (!registry?.getModelsOfType || !registry.getProvider) return false;
  if (!selected || (selected.api !== 'pi-virtual' && !paths.has(selected.api))) return false;
  const candidates = selected.api === 'pi-virtual' ? registry.getAvailable?.() ?? [] : [selected];
  for (const candidate of candidates) {
    if (!paths.has(candidate.api) || registered.has(candidate.provider)) continue;
    const provider = registry.getProvider(candidate.provider);
    if (!provider) continue;
    const wrap = (stream: Stream): Stream => (model, context, options) => {
      const suffix = paths.get(model.api);
      const route = suffix && routeFor(model);
      if (!route || !suffix) return stream.call(provider, model, context, options);
      const expected = new URL(model.baseUrl.replace(/\/+$/, '') + suffix);
      const forward: typeof fetch = async (input, init) => {
        const request = new Request(input, init);
        const url = new URL(request.url);
        const send = options?.fetch ?? fetch;
        if (request.method !== 'POST' || url.origin !== expected.origin || url.pathname !== expected.pathname) {
          return send(request);
        }
        const target = new URL(route.gateway.replace(/\/+$/, '') + suffix);
        target.search = url.search;
        const headers = new Headers(request.headers);
        headers.delete('host');
        for (const [key, value] of Object.entries(route.headers)) headers.set(key, value);
        const forwarded = new Request(target, request);
        return send(forwarded, { headers, redirect: 'error' });
      };
      return stream.call(provider, model, context, { ...options, fetch: forward });
    };
    register({ ...provider, stream: wrap(provider.stream), streamSimple: wrap(provider.streamSimple) });
    registered.add(candidate.provider);
  }
  return true;
}
