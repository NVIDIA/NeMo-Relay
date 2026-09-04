// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const require = createRequire(import.meta.url);
const plugin = require('../plugin.js');
const pricing = require('../model_pricing.js');

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

describe('model pricing plugin helpers', () => {
  it('builds defaults and plugin component shape', () => {
    assert.deepEqual(pricing.defaultConfig(), { sources: [] });
    assert.deepEqual(pricing.tokenRates(), {
      input_per_million: 0,
      output_per_million: 0,
    });
    assert.deepEqual(pricing.promptCache(), {
      read_accounting: 'included_in_prompt_tokens',
    });

    const component = pricing.ComponentSpec(
      pricing.pricingConfig([pricing.inlineSource(pricing.inlineCatalog([pricedEntry()]))]),
    );
    assert.equal(component.kind, pricing.PRICING_PLUGIN_KIND);
    assert.equal(component.enabled, true);
    assert.equal(component.config.sources[0].type, 'inline');
    assert.deepEqual(pricedEntry({ prompt_cache: {} }).prompt_cache, {
      read_accounting: 'included_in_prompt_tokens',
    });
  });

  it('builds file sources and threshold rate schedules', () => {
    assert.deepEqual(pricing.fileSource('/tmp/pricing.json'), {
      type: 'file',
      path: '/tmp/pricing.json',
    });
    assert.deepEqual(
      pricing.promptTokenThresholdRateSchedule([
        pricing.tokenRateTier(pricing.tokenRates({ input_per_million: 1, output_per_million: 2 }), {
          min_prompt_tokens: 128,
        }),
      ]),
      {
        type: 'prompt_token_threshold',
        applies_to: 'full_request',
        tiers: [
          {
            min_prompt_tokens: 128,
            rates: {
              input_per_million: 1,
              output_per_million: 2,
            },
          },
        ],
      },
    );
  });

  it('lists builtin pricing kind and validates config', () => {
    assert.equal(plugin.listKinds().includes(pricing.PRICING_PLUGIN_KIND), true);
    const report = pricing.validateConfig(
      pricing.pricingConfig([pricing.inlineSource(pricing.inlineCatalog([pricedEntry()]))]),
    );
    assert.deepEqual(report.diagnostics, []);

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

  it('validates only the supplied pricing config when ambient config is malformed', () => {
    const configHome = mkdtempSync(path.join(tmpdir(), 'nemo-relay-node-pricing-'));
    const relayConfig = path.join(configHome, 'nemo-relay');
    mkdirSync(relayConfig);
    writeFileSync(path.join(relayConfig, 'plugins.toml'), '{');
    const previous = process.env.XDG_CONFIG_HOME;
    process.env.XDG_CONFIG_HOME = configHome;
    try {
      const report = pricing.validateConfig(
        pricing.pricingConfig([pricing.inlineSource(pricing.inlineCatalog([pricedEntry()]))]),
      );
      assert.deepEqual(report.diagnostics, []);
    } finally {
      if (previous === undefined) {
        delete process.env.XDG_CONFIG_HOME;
      } else {
        process.env.XDG_CONFIG_HOME = previous;
      }
      rmSync(configHome, { recursive: true, force: true });
    }
  });
});
