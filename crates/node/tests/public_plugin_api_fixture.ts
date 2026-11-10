// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { ComponentSpec as adaptiveComponentSpec, defaultConfig as adaptiveDefaultConfig } from '../adaptive.js';
import { ComponentSpec as pricingComponentSpec } from '../model_pricing.js';
import { ComponentSpec as piiComponentSpec, defaultConfig as piiDefaultConfig } from '../pii_redaction.js';
import {
  listRuntimeRegistrations,
  registerConditionalMiddlewareGuardrail,
  RuntimeRegistrationKind,
  type JsonObject,
  type RuntimeRegistrationKindName,
} from '../index.js';
import * as plugin from '../plugin.js';
import type {
  ComponentKind,
  ComponentSpecFor,
  KnownComponentSpec,
  OpenComponentSpec,
  PluginConfig,
} from '../plugin.js';

type AssertEqual<T, U> = [T] extends [U] ? ([U] extends [T] ? true : false) : false;

const kinds: AssertEqual<
  ComponentKind,
  'adaptive' | 'observability' | 'pii_redaction' | 'pricing' | 'resource_metrics'
> = true;

const pricing = plugin.ComponentSpec('pricing', { sources: [{ type: 'file', path: './pricing.json' }] });
const pricingKind: AssertEqual<typeof pricing.kind, 'pricing'> = true;
const open = plugin.ComponentSpec('example.add-tag', { tag: 'documentation' });
const openKind: AssertEqual<typeof open.kind, string> = true;
const openTag: string | undefined = open.config?.tag;
const erasedOpen: ComponentSpecFor<string> = open;
const forwardedOpen: JsonObject | undefined = erasedOpen.config;
declare const dynamicKind: string;
const dynamic = plugin.ComponentSpec(dynamicKind, { tag: 'documentation' });
const dynamicSpec: AssertEqual<typeof dynamic, OpenComponentSpec<{ tag: string }>> = true;

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
    pricingComponentSpec({ sources: [] }),
    piiComponentSpec(piiDefaultConfig()),
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

const malformedKnown: PluginConfig = {
  // @ts-expect-error A known kind inside a plugin config rejects unknown fields.
  components: [{ kind: 'pricing', config: { sources: [], typo: 1 } }],
};

const inlineOpen: PluginConfig = {
  // @ts-expect-error An open kind must be created with ComponentSpec.
  components: [{ kind: 'example.add-tag', config: { tag: 'documentation' } }],
};

export const checked = [
  kinds,
  pricingKind,
  openKind,
  openTag,
  config,
  subscriberKind,
  everyKind,
  name,
  malformed,
  malformedKnown,
  inlineOpen,
];
