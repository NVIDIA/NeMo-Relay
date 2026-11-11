// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { ScopeStack } from 'nemo-relay-node';
import type { Codec } from 'nemo-relay-node/typed';
import type { ComponentSpec as PluginComponentSpec } from 'nemo-relay-node/plugin';
import type { ComponentSpec as AdaptiveComponentSpec } from 'nemo-relay-node/adaptive';
import type { AtofConfig } from 'nemo-relay-node/observability';
import type { ComponentSpec as PiiComponentSpec } from 'nemo-relay-node/pii_redaction';
import type { TokenPricingRates } from 'nemo-relay-node/model_pricing';
import type { DiagnosticsReader } from 'nemo-relay-node/diagnostics';

export type EntryPoints = [
  ScopeStack,
  Codec<string>,
  PluginComponentSpec,
  AdaptiveComponentSpec,
  AtofConfig,
  PiiComponentSpec,
  TokenPricingRates,
  DiagnosticsReader,
];
