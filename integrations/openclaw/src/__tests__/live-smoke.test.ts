// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { it } from "node:test";

import { makeSafeSessionId } from "../atif-capture.js";
import { registerNemoFlowPlugin } from "../runtime-state.js";
import type { NemoFlowHealthSnapshot } from "../health.js";
import type { NemoFlowModules } from "../modules.js";
import type { OpenClawHookHandlerLike, OpenClawPluginApiLike, PluginLoggerLike } from "../types.js";

const liveSmokeEnabled = process.env.NEMO_FLOW_OPENCLAW_LIVE_SMOKE === "1";

it(
  "runs a live NeMo Flow binding smoke for session ATIF export and hook replay",
  { skip: !liveSmokeEnabled },
  async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-openclaw-live-"));
    const modules = await loadRealNemoFlowModules();
    const api = createApi({
      pluginConfig: {
        atif: {
          enabled: true,
          outputDir,
        },
        telemetry: {
          otel: { enabled: false },
          openInference: { enabled: false },
        },
      },
    });
    let serviceStarted = false;

    try {
      registerNemoFlowPlugin(api, async () => modules);

      const service = api.calls.services[0];
      assert.ok(service, "expected OpenClaw service registration");
      await service.start({
        stateDir: outputDir,
        logger: api.logger,
      });
      serviceStarted = true;

      const sessionStart = api.calls.hooks.find((hook) => hook.hookName === "session_start");
      const llmInput = api.calls.hooks.find((hook) => hook.hookName === "llm_input");
      const llmOutput = api.calls.hooks.find((hook) => hook.hookName === "llm_output");
      const afterToolCall = api.calls.hooks.find((hook) => hook.hookName === "after_tool_call");
      const sessionEnd = api.calls.hooks.find((hook) => hook.hookName === "session_end");
      assert.ok(sessionStart, "expected session_start hook registration");
      assert.ok(llmInput, "expected llm_input hook registration");
      assert.ok(llmOutput, "expected llm_output hook registration");
      assert.ok(afterToolCall, "expected after_tool_call hook registration");
      assert.ok(sessionEnd, "expected session_end hook registration");

      await sessionStart.handler({ sessionId: "../live-session:1" }, { sessionId: "../live-session:1" });
      await llmInput.handler(
        {
          runId: "live-run-1",
          sessionId: "../live-session:1",
          provider: "openai",
          model: "gpt-live",
          systemPrompt: "be concise",
          prompt: "hello",
          historyMessages: [],
          imagesCount: 0,
        },
        { runId: "live-run-1", sessionId: "../live-session:1", agentId: "agent-live" },
      );
      await llmOutput.handler(
        {
          runId: "live-run-1",
          sessionId: "../live-session:1",
          provider: "openai",
          model: "gpt-live",
          assistantTexts: ["hi"],
          usage: { input: 1, output: 1 },
        },
        { runId: "live-run-1", sessionId: "../live-session:1", agentId: "agent-live" },
      );
      await afterToolCall.handler(
        {
          toolName: "read_file",
          params: { path: "README.md" },
          runId: "live-run-1",
          toolCallId: "tool-live-1",
          result: { text: "ok" },
          durationMs: 2,
        },
        {
          runId: "live-run-1",
          sessionId: "../live-session:1",
          toolName: "read_file",
          toolCallId: "tool-live-1",
        },
      );
      await sessionEnd.handler(
        { sessionId: "../live-session:1", messageCount: 1, reason: "idle" },
        { sessionId: "../live-session:1" },
      );

      const targetPath = path.join(outputDir, `${makeSafeSessionId("../live-session:1")}.json`);
      const exported = JSON.parse(await fs.readFile(targetPath, "utf8")) as unknown;
      assert.equal(typeof exported, "object");

      const status = api.calls.gatewayMethods[0]?.handler();
      assert.ok(status);
      assert.equal(status.outputs.atif, "enabled");
      assert.equal(status.counters.llmSpansReplayed, 1);
      assert.equal(status.counters.toolSpansReplayed, 1);
      assert.equal(status.counters.atifFilesWritten, 1);

    } finally {
      if (serviceStarted) {
        await api.calls.services[0]?.stop?.({
          stateDir: outputDir,
          logger: api.logger,
        });
      }
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  },
);

type TestApi = OpenClawPluginApiLike & {
  calls: {
    services: Parameters<OpenClawPluginApiLike["registerService"]>[0][];
    lifecycle: Parameters<OpenClawPluginApiLike["registerRuntimeLifecycle"]>[0][];
    gatewayMethods: Array<{
      method: string;
      handler: () => NemoFlowHealthSnapshot;
    }>;
    hooks: Array<{ hookName: string; handler: OpenClawHookHandlerLike }>;
  };
};

function createApi(params: { pluginConfig: Record<string, unknown> }): TestApi {
  const calls: TestApi["calls"] = {
    services: [],
    lifecycle: [],
    gatewayMethods: [],
    hooks: [],
  };
  const logger: PluginLoggerLike = {
    info: () => {},
    warn: () => {},
  };

  return {
    id: "nemo-flow",
    version: "live-smoke",
    registrationMode: "full",
    pluginConfig: params.pluginConfig,
    logger,
    resolvePath: (input) => input,
    registerService: (service) => calls.services.push(service),
    registerRuntimeLifecycle: (lifecycle) => calls.lifecycle.push(lifecycle),
    on: (hookName, handler) => calls.hooks.push({ hookName, handler }),
    registerGatewayMethod: (method, handler) =>
      calls.gatewayMethods.push({
        method,
        handler: handler as TestApi["calls"]["gatewayMethods"][number]["handler"],
      }),
    calls,
  };
}

async function loadRealNemoFlowModules(): Promise<NemoFlowModules> {
  let nf: unknown;
  let pluginHost: unknown;
  try {
    [nf, pluginHost] = await Promise.all([
      import("nemo-flow-node"),
      import("nemo-flow-node/plugin"),
    ]);
  } catch (error) {
    if (isMissingLocalNemoFlowNode(error)) {
      throw new Error(
        "Live smoke requires the local nemo-flow-node native package to be built. From the NeMo-Flow repo root, run `npm --prefix crates/node run build`, then rerun `npm --prefix integrations/openclaw run test:live`.",
      );
    }
    throw error;
  }

  return {
    nf: nf as unknown as NemoFlowModules["nf"],
    pluginHost: pluginHost as unknown as NemoFlowModules["pluginHost"],
  };
}

function isMissingLocalNemoFlowNode(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    error.code === "ERR_MODULE_NOT_FOUND" &&
    error.message.includes("nemo-flow-node")
  );
}
