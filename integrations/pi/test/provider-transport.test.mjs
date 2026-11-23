// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import test from 'node:test';
import { registerChatTransports } from '../src/provider-transport.ts';

test('virtual dispatch routes each physical chat POST and isolates other provider operations', async () => {
  const catalog = ['first', 'second'].map((provider) => ({
    id: provider, provider, api: 'openai-completions', baseUrl: 'https://' + provider + '.invalid/v1',
  }));
  const originals = new Map(catalog.map((model) => [model.provider, {
    id: model.provider,
    streamSimple(actual, _, options) {
      return options.fetch(actual.baseUrl + '/chat/completions?stream=1', { method: 'POST' });
    },
    generateImages: () => 'direct image',
    classify: () => 'direct classification',
  }]));
  const wrapped = new Map();
  registerChatTransports({
    getAvailable: () => catalog, getProvider: (id) => originals.get(id), getModelsOfType: () => [],
  }, { api: 'pi-virtual' }, new Set(), (provider) => wrapped.set(provider.id, provider), (model) => ({
    gateway: 'https://gateway.invalid', headers: { 'x-nemo-relay-session-id': model.id },
  }));
  for (const model of catalog) {
    const provider = wrapped.get(model.provider);
    await provider.streamSimple(model, {}, { fetch(input, init) {
      const request = new Request(input, init);
      assert.equal(request.url, 'https://gateway.invalid/chat/completions?stream=1');
      assert.equal(request.headers.get('x-nemo-relay-session-id'), model.id);
      assert.equal(request.redirect, 'error');
      return Promise.resolve(new Response());
    } });
    assert.equal(provider.generateImages, originals.get(model.provider).generateImages);
    assert.equal(provider.classify, originals.get(model.provider).classify);
  }
});
