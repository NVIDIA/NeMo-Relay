// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { registerEventMetadataInjector, scopeRegisterEventMetadataInjector } from '../index.js';
import type { EventMetadata, PluginContext } from '../plugin.js';

const acceptedMetadata: EventMetadata = {
  'fixture.string': 'value',
  'fixture.number': 42,
  'fixture.boolean': true,
  'fixture.strings': ['alpha', 'beta'],
  'fixture.numbers': [1, 2.5],
  'fixture.booleans': [true, false],
  'fixture.empty': [],
};

registerEventMetadataInjector('fixture-global', 0, () => acceptedMetadata);
scopeRegisterEventMetadataInjector(
  '00000000-0000-0000-0000-000000000001',
  'fixture-scope',
  0,
  async () => acceptedMetadata,
);

declare const context: PluginContext;
context.registerEventMetadataInjector('fixture-plugin', 0, () => acceptedMetadata);

// @ts-expect-error Global injector callbacks cannot return object values.
registerEventMetadataInjector('fixture-object', 0, () => ({ 'fixture.object': { nested: true } }));

// @ts-expect-error Global injector callbacks cannot return nested arrays.
registerEventMetadataInjector('fixture-nested', 0, () => ({ 'fixture.nested': [[1]] }));

scopeRegisterEventMetadataInjector('00000000-0000-0000-0000-000000000001', 'fixture-null', 0, () => ({
  // @ts-expect-error Scope-local injector callbacks cannot return null values.
  'fixture.null': null,
}));

// @ts-expect-error Plugin injector arrays must contain one primitive type.
context.registerEventMetadataInjector('fixture-mixed', 0, () => ({ 'fixture.mixed': [1, 'two'] }));
