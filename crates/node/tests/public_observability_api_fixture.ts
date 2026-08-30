// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import {
  type DataSchema,
  event,
  metric,
  LogSeverity,
  MetricKind,
  MetricTemporality,
  MetricValueType,
  OpenTelemetryLogSubscriber,
  OpenTelemetryMetricSubscriber,
  OpenTelemetrySubscriber,
} from '../index.js';

const dataSchema: DataSchema = { name: 'example.fixture', version: '1' };

const logSubscriber = new OpenTelemetryLogSubscriber({
  endpoint: 'http://localhost:4318/v1/logs',
  headerEnv: { authorization: 'OTEL_LOG_AUTHORIZATION' },
  minimumSeverity: LogSeverity.Warn,
});
const metricSubscriber = new OpenTelemetryMetricSubscriber({
  endpoint: 'http://localhost:4318/v1/metrics',
  headerEnv: { authorization: 'OTEL_METRIC_AUTHORIZATION' },
  temporality: MetricTemporality.Delta,
});
const traceSubscriber = new OpenTelemetrySubscriber({
  type: 'full',
  endpoint: 'http://localhost:4318/v1/traces',
  headerEnv: { authorization: 'OTEL_TRACE_AUTHORIZATION' },
  completedSpanContextTtlMillis: 4_294_967_296n,
});

event('fixture.log', null, { ready: true }, null, null, dataSchema, LogSeverity.Info);
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
const traceDiagnostics: Array<{ code: string; message: string; count: number }> = traceSubscriber.runtimeDiagnostics();
logSubscriber.register('fixture-log-subscriber');
const logDeregistered: boolean = logSubscriber.deregister('fixture-log-subscriber');
logSubscriber.forceFlush();
logSubscriber.shutdown();
metricSubscriber.register('fixture-metric-subscriber');
const metricDeregistered: boolean = metricSubscriber.deregister('fixture-metric-subscriber');
metricSubscriber.forceFlush();
metricSubscriber.shutdown();
traceSubscriber.shutdown();

if (!logDiagnostics || !metricDiagnostics || !logDeregistered || !metricDeregistered) {
  throw new Error('fixture observability operations failed');
}
