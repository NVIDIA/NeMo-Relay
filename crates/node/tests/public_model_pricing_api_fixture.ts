// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  ComponentSpec,
  catalogEntry,
  fileSource,
  inlineCatalog,
  inlineSource,
  pricingConfig,
  promptCache,
  tokenRates,
  parseConfig,
  validateConfig,
  type ModelPricing,
  type PricingCatalog,
  type PricingConfig,
} from '../model_pricing.js';
import type { JsonObject } from '../index.js';

const entry: ModelPricing = catalogEntry({
  provider: 'openai',
  model_id: 'gpt-4o-mini',
  aliases: ['openai/openai/gpt-4o-mini'],
  currency: 'USD',
  unit: 'per_token',
  rates: tokenRates({ input_per_million: 0.15, output_per_million: 0.6, cache_read_per_million: 0.075 }),
  prompt_cache: promptCache({ read_accounting: 'included_in_prompt_tokens' }),
  pricing_as_of: '2026-06-06',
  pricing_source: 'internal-pricing-snapshot',
});

const catalog: PricingCatalog = inlineCatalog([entry, { vendor_specific: { tier: 'enterprise' } }], { version: 1 });

const config: PricingConfig = pricingConfig([fileSource('./pricing.json'), inlineSource(catalog), { type: 'remote' }]);

ComponentSpec(config);
ComponentSpec({ sources: [] });
validateConfig(config).diagnostics.map((diagnostic) => diagnostic.level);
declare const document: JsonObject;
ComponentSpec(parseConfig(document));

catalogEntry({
  provider: 'openai',
  model_id: 'gpt-4o-mini',
  // @ts-expect-error Model pricing supports only the documented billing units.
  unit: 'per_hour',
  pricing_as_of: '2026-06-06',
  pricing_source: 'snapshot',
});

// @ts-expect-error Prompt cache accounting supports only the documented modes.
promptCache({ read_accounting: 'unknown_mode' });

// @ts-expect-error Inline sources require a catalog payload.
inlineSource();
