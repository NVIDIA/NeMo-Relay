// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { test } from 'node:test';

import * as plugin from '../pkg/plugin.js';
import * as pricing from '../pkg/model_pricing.js';

function pricedEntry(overrides = {}) {
  return pricing.catalogEntry({
    provider: 'test',
    model_id: 'priced-model',
    pricing_as_of: '2026-06-15',
    pricing_source: 'https://example.com/pricing',
    rates: pricing.tokenRates({ input_per_million: 1, output_per_million: 2 }),
    ...overrides,
  });
}

test('WebAssembly model pricing wrappers expose helper defaults', () => {
  assert.deepEqual(pricing.defaultConfig(), { sources: [] });
  assert.deepEqual(pricing.tokenRates(), {
    input_per_million: 0,
    output_per_million: 0,
  });
  assert.deepEqual(pricing.promptCache(), {
    read_accounting: 'included_in_prompt_tokens',
  });
  assert.deepEqual(pricing.fileSource('/tmp/pricing.json'), {
    type: 'file',
    path: '/tmp/pricing.json',
  });
});

test('WebAssembly model pricing wrappers build component specs and validate config', () => {
  assert.equal(plugin.listKinds().includes(pricing.PRICING_PLUGIN_KIND), true);

  const component = pricing.ComponentSpec(
    pricing.pricingConfig([pricing.inlineSource(pricing.inlineCatalog([pricedEntry()]))]),
  );

  assert.equal(component.kind, 'pricing');
  assert.equal(component.enabled, true);
  assert.equal(component.config.sources[0].type, 'inline');
  assert.deepEqual(pricing.validateConfig(component.config).diagnostics, []);

  const invalid = pricing.validateConfig(
    pricing.pricingConfig([
      pricing.inlineSource(
        pricing.inlineCatalog([pricedEntry({ rates: pricing.tokenRates({ input_per_million: -1 }) })]),
      ),
    ]),
  );
  assert.deepEqual(
    invalid.diagnostics.map((diagnostic) => diagnostic.code),
    ['pricing.invalid_config'],
  );
});
