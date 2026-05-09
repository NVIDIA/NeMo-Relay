// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Telemetry subscriber shutdown tests for deregister/flush/shutdown failure paths.
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { shutdownTelemetrySubscribers, type TelemetrySubscriberEntry } from "../telemetry.js";
import type { NemoFlowSubscriber } from "../modules.js";
import type { PluginLogger } from "openclaw/plugin-sdk/plugin-entry";

describe("telemetry subscriber shutdown", () => {
  it("continues force flush and shutdown when deregister throws", () => {
    const subscriber = new TestSubscriber({ deregisterThrows: true });
    const degraded: string[] = [];

    shutdownTelemetrySubscribers({
      subscribers: [entry("otel", "otel-test", subscriber)],
      logger: createLogger(),
      markOutputDegraded: (output) => degraded.push(output),
    });

    assert.deepEqual(subscriber.actions, ["deregister:otel-test", "forceFlush", "shutdown"]);
    assert.deepEqual(degraded, ["otel"]);
  });

  it("marks output degraded and continues shutdown when forceFlush throws", () => {
    const subscriber = new TestSubscriber({ forceFlushThrows: true });
    const degraded: string[] = [];

    shutdownTelemetrySubscribers({
      subscribers: [entry("openInference", "oi-test", subscriber)],
      logger: createLogger(),
      markOutputDegraded: (output) => degraded.push(output),
    });

    assert.deepEqual(subscriber.actions, ["deregister:oi-test", "forceFlush", "shutdown"]);
    assert.deepEqual(degraded, ["openInference"]);
  });

  it("marks output degraded and continues shutting down other subscribers when shutdown throws", () => {
    const first = new TestSubscriber({ shutdownThrows: true });
    const second = new TestSubscriber();
    const degraded: string[] = [];

    shutdownTelemetrySubscribers({
      subscribers: [entry("otel", "otel-test", first), entry("openInference", "oi-test", second)],
      logger: createLogger(),
      markOutputDegraded: (output) => degraded.push(output),
    });

    assert.deepEqual(first.actions, ["deregister:otel-test", "forceFlush", "shutdown"]);
    assert.deepEqual(second.actions, ["deregister:oi-test", "forceFlush", "shutdown"]);
    assert.deepEqual(degraded, ["otel"]);
  });
});

function entry(
  output: "otel" | "openInference",
  name: string,
  subscriber: NemoFlowSubscriber,
): TelemetrySubscriberEntry {
  return { output, name, subscriber };
}

function createLogger(): PluginLogger {
  return {
    info: () => {},
    warn: () => {},
    error: () => {},
  };
}

class TestSubscriber implements NemoFlowSubscriber {
  readonly actions: string[] = [];

  constructor(
    private readonly failures: {
      deregisterThrows?: boolean;
      forceFlushThrows?: boolean;
      shutdownThrows?: boolean;
    } = {},
  ) {}

  register(name: string): void {
    this.actions.push(`register:${name}`);
  }

  deregister(name: string): boolean {
    this.actions.push(`deregister:${name}`);
    if (this.failures.deregisterThrows) {
      throw new Error("deregister failed");
    }
    return true;
  }

  forceFlush(): void {
    this.actions.push("forceFlush");
    if (this.failures.forceFlushThrows) {
      throw new Error("force flush failed");
    }
  }

  shutdown(): void {
    this.actions.push("shutdown");
    if (this.failures.shutdownThrows) {
      throw new Error("shutdown failed");
    }
  }
}
