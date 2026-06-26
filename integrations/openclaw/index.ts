// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * OpenClaw plugin entry point.
 *
 * This file should stay small: it declares the public plugin metadata and hands
 * registration to the runtime-state module, where lifecycle and hook wiring live.
 */
import { definePluginEntry, type OpenClawPluginApi } from 'openclaw/plugin-sdk/plugin-entry';

import { nemoRelayConfigSchema } from './src/config.js';
import { registerNemoRelayPlugin } from './src/runtime-state.js';

type NemoRelayPluginEntry = ReturnType<typeof definePluginEntry>;

const nemoRelayPluginEntry: NemoRelayPluginEntry = definePluginEntry({
  id: 'nemo-relay',
  name: 'NeMo Relay Observability',
  description: 'ATIF, OpenInference, and OpenTelemetry telemetry through NeMo Relay',
  configSchema: nemoRelayConfigSchema,
  register(api: OpenClawPluginApi) {
    registerNemoRelayPlugin(api);
  },
});

export default nemoRelayPluginEntry;
