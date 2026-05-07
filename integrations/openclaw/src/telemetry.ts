// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { NemoFlowHookBackendConfig, TelemetrySinkConfig } from "./config.js";
import type { NemoFlowRuntimeModule, NemoFlowSubscriber } from "./modules.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";

export type TelemetrySubscriberEntry = {
  output: "otel" | "openInference";
  name: string;
  subscriber: NemoFlowSubscriber;
};

export type RegisterTelemetrySubscribersOptions = {
  nf: NemoFlowRuntimeModule;
  config: NemoFlowHookBackendConfig;
  logger: PluginLogger;
  markOutputDegraded: (output: "otel" | "openInference") => void;
};

export function registerTelemetrySubscribers(
  options: RegisterTelemetrySubscribersOptions,
): TelemetrySubscriberEntry[] {
  const entries: TelemetrySubscriberEntry[] = [];
  const outputs: Array<{
    name: string;
    configKey: "otel" | "openInference";
    sinkConfig: TelemetrySinkConfig;
    Ctor: typeof options.nf.OpenTelemetrySubscriber | typeof options.nf.OpenInferenceSubscriber;
  }> = [];

  if (options.config.telemetry.otel.enabled) {
    outputs.push({
      name: "openclaw.nemo-flow.otel",
      configKey: "otel",
      sinkConfig: options.config.telemetry.otel,
      Ctor: options.nf.OpenTelemetrySubscriber,
    });
  }

  if (options.config.telemetry.openInference.enabled) {
    outputs.push({
      name: "openclaw.nemo-flow.openinference",
      configKey: "openInference",
      sinkConfig: options.config.telemetry.openInference,
      Ctor: options.nf.OpenInferenceSubscriber,
    });
  }

  for (const output of outputs) {
    let subscriber: NemoFlowSubscriber | undefined;
    try {
      subscriber = new output.Ctor(toSubscriberConfig(output.sinkConfig));
      subscriber.register(output.name);
      entries.push({ output: output.configKey, name: output.name, subscriber });
    } catch (error) {
      options.markOutputDegraded(output.configKey);
      options.logger.warn?.(
        `nemo-flow failed to register subscriber ${output.name}: ${toMessage(error)}`,
      );
      if (subscriber) {
        try {
          subscriber.shutdown();
        } catch (cleanupError) {
          options.logger.warn?.(
            `nemo-flow failed to cleanup subscriber ${output.name} after registration failure: ${toMessage(cleanupError)}`,
          );
        }
      }
    }
  }

  return entries;
}

export function shutdownTelemetrySubscribers(params: {
  subscribers: TelemetrySubscriberEntry[];
  logger: PluginLogger;
  markOutputDegraded: (output: "otel" | "openInference") => void;
}): void {
  for (const { output, name, subscriber } of params.subscribers) {
    try {
      const removed = subscriber.deregister(name);
      if (!removed) {
        params.markOutputDegraded(output);
        params.logger.warn?.(`nemo-flow subscriber ${name} was already deregistered before shutdown`);
      }
    } catch (error) {
      params.markOutputDegraded(output);
      params.logger.warn?.(`nemo-flow failed to deregister subscriber ${name}: ${toMessage(error)}`);
    }

    try {
      subscriber.forceFlush();
    } catch (error) {
      params.markOutputDegraded(output);
      params.logger.warn?.(`nemo-flow failed to flush subscriber ${name}: ${toMessage(error)}`);
    }

    try {
      subscriber.shutdown();
    } catch (error) {
      params.markOutputDegraded(output);
      params.logger.warn?.(`nemo-flow failed to shutdown subscriber ${name}: ${toMessage(error)}`);
    }
  }
}

function toSubscriberConfig(config: TelemetrySinkConfig): Record<string, unknown> {
  const raw = {
    transport: config.transport,
    endpoint: config.endpoint,
    headers: config.headers,
    resourceAttributes: config.resourceAttributes,
    serviceName: config.serviceName,
    serviceNamespace: config.serviceNamespace,
    serviceVersion: config.serviceVersion,
    instrumentationScope: config.instrumentationScope,
    timeoutMillis: config.timeoutMillis,
  };
  return Object.fromEntries(Object.entries(raw).filter(([, value]) => value !== undefined));
}

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
