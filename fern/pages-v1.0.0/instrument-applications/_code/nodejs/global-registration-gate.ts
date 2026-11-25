// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  deregisterConditionalMiddlewareGuardrail,
  listRuntimeRegistrations,
  registerConditionalMiddlewareGuardrail,
} from 'nemo-relay-node';

/** Pause a configured OpenTelemetry subscriber during application work. */
export async function runExample(runApplication: () => Promise<void>) {
  const target = listRuntimeRegistrations(['subscriber']).find(
    (registration) => registration.localName === 'opentelemetry' && registration.owner.pluginKind === 'observability',
  );
  if (!target) throw new Error('OpenTelemetry subscriber was not found');

  let paused = true;
  registerConditionalMiddlewareGuardrail('pause-opentelemetry', ['subscriber'], target.effectiveName, () =>
    paused ? 'export paused' : null,
  );
  try {
    await runApplication();
  } finally {
    deregisterConditionalMiddlewareGuardrail('pause-opentelemetry');
  }
}
