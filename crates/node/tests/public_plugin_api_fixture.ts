// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComponentSpec as adaptiveComponentSpec, defaultConfig as adaptiveDefaultConfig } from '../adaptive.js';
import {
  listRuntimeRegistrations,
  registerConditionalMiddlewareGuardrail,
  RuntimeRegistrationKind,
  type RuntimeRegistrationKindName,
} from '../index.js';
import * as plugin from '../plugin.js';
import type { ComponentKind, KnownComponentSpec, PluginConfig } from '../plugin.js';

type AssertEqual<T, U> = [T] extends [U] ? ([U] extends [T] ? true : false) : false;

const kinds: AssertEqual<
  ComponentKind,
  'adaptive' | 'observability' | 'pii_redaction' | 'pricing' | 'resource_metrics'
> = true;

const pricing = plugin.ComponentSpec('pricing', { sources: [{ type: 'file', path: './pricing.json' }] });
const pricingKind: AssertEqual<typeof pricing.kind, 'pricing'> = true;
const open = plugin.ComponentSpec('example.add-tag', { tag: 'documentation' });
const openKind: AssertEqual<typeof open.kind, string> = true;

const config: PluginConfig = {
  version: 1,
  components: [
    adaptiveComponentSpec(adaptiveDefaultConfig()),
    {
      kind: 'resource_metrics',
      enabled: true,
      config: { polling: { enabled: true, interval_millis: 5000 }, units: { cpu: { user_time: 'seconds' } } },
    },
    pricing,
    open,
  ],
};

const subscribers = listRuntimeRegistrations(['subscriber']);
const subscriberKind: AssertEqual<(typeof subscribers)[number]['kind'], RuntimeRegistrationKind.Subscriber> = true;
const everyKind: AssertEqual<ReturnType<typeof listRuntimeRegistrations>[number]['kind'], RuntimeRegistrationKind> =
  true;
registerConditionalMiddlewareGuardrail(
  'pause',
  ['subscriber', RuntimeRegistrationKind.ToolRequestIntercept],
  'name',
  () => null,
);
const name: RuntimeRegistrationKindName = 'llm_execution_intercept';

// @ts-expect-error A known kind rejects another component's config.
plugin.ComponentSpec('pricing', { atof: { enabled: true } });

// @ts-expect-error Resource metrics config rejects unknown fields.
const malformed: KnownComponentSpec = { kind: 'resource_metrics', config: { polling: { interval: 1 } } };

// @ts-expect-error Registration kinds must be known names.
listRuntimeRegistrations(['subscribers']);

export const checked = [kinds, pricingKind, openKind, config, subscriberKind, everyKind, name, malformed];
