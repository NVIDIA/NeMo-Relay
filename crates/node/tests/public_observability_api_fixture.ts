// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  event,
  metric,
  LogSeverity,
  MetricKind,
  MetricTemporality,
  MetricValueType,
  OpenTelemetryLogSubscriber,
  OpenTelemetryMetricSubscriber,
} from '../index.js';

const logSubscriber = new OpenTelemetryLogSubscriber({
  endpoint: 'http://localhost:4318/v1/logs',
  minimumSeverity: LogSeverity.Warn,
});
const metricSubscriber = new OpenTelemetryMetricSubscriber({
  endpoint: 'http://localhost:4318/v1/metrics',
  temporality: MetricTemporality.Delta,
});

event('fixture.log', null, { ready: true }, null, null, null, LogSeverity.Info);
metric(
  'fixture.metric',
  [
    {
      name: 'relay.tokens',
      kind: MetricKind.Counter,
      valueType: MetricValueType.U64,
      value: 1,
    },
  ],
  null,
  null,
  null,
);

const logDiagnostics: Array<{ code: string; message: string; count: number }> = logSubscriber.runtimeDiagnostics();
const metricDiagnostics: Array<{ code: string; message: string; count: number }> =
  metricSubscriber.runtimeDiagnostics();

void logDiagnostics;
void metricDiagnostics;
