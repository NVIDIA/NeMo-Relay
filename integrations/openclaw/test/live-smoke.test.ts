// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Opt-in live smoke test for exercising the real OpenClaw plugin runtime.
 */
import assert from "node:assert/strict";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { it } from "node:test";

import { registerNemoFlowPlugin } from "../src/runtime-state.js";
import {
  defaultNemoFlowModuleLoader,
  type NemoFlowModuleLoader,
  type NemoFlowModules,
} from "../src/modules.js";
import type { OpenClawPluginApi, PluginLogger } from "openclaw/plugin-sdk/plugin-entry";
import { callGatewayStatus, type TestGatewayMethodHandler } from "./gateway-status.js";

const liveSmokeEnabled = process.env.NEMO_FLOW_OPENCLAW_LIVE_SMOKE === "1";

it(
  "runs a live NeMo Flow binding smoke for session ATIF export and hook replay",
  { skip: !liveSmokeEnabled },
  async () => {
    const outputDir = await fs.mkdtemp(path.join(os.tmpdir(), "nemo-flow-openclaw-live-"));
    const modules = await loadRealNemoFlowModules();
    const api = createApi({
      pluginConfig: {
        plugins: {
          version: 1,
          components: [
            {
              kind: "observability",
              enabled: true,
              config: {
                version: 1,
                atif: {
                  enabled: true,
                  agent_name: "openclaw",
                  output_directory: outputDir,
                  filename_template: "live-{session_id}.json",
                },
              },
            },
          ],
        },
      },
    });
    let serviceStarted = false;

    try {
      registerPlugin(api, async () => modules);

      const service = api.calls.services[0];
      assert.ok(service, "expected OpenClaw service registration");
      await service.start({
        stateDir: outputDir,
        config: {} as never,
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

      const files = await fs.readdir(outputDir);
      const exportedPath = files.find((file) => file.startsWith("live-") && file.endsWith(".json"));
      assert.ok(exportedPath, "expected generic observability ATIF export");
      const exported = JSON.parse(await fs.readFile(path.join(outputDir, exportedPath), "utf8")) as unknown;
      assert.equal(typeof exported, "object");

      const status = await callGatewayStatus(api.calls.gatewayMethods[0]?.handler);
      assert.equal(status.outputs.atif, "enabled");
      assert.equal(status.counters.llmSpansReplayed, 1);
      assert.equal(status.counters.toolSpansReplayed, 1);

    } finally {
      if (serviceStarted) {
        await api.calls.services[0]?.stop?.({
          stateDir: outputDir,
          config: {} as never,
          logger: api.logger,
        });
      }
      await fs.rm(outputDir, { recursive: true, force: true });
    }
  },
);

type HookHandler = (event: unknown, ctx: unknown) => void | Promise<void>;

type TestApi = {
  id: string;
  version?: string;
  registrationMode: OpenClawPluginApi["registrationMode"];
  pluginConfig?: Record<string, unknown>;
  logger: PluginLogger;
  resolvePath: OpenClawPluginApi["resolvePath"];
  registerService: (service: Parameters<OpenClawPluginApi["registerService"]>[0]) => void;
  registerRuntimeLifecycle: (lifecycle: Parameters<OpenClawPluginApi["registerRuntimeLifecycle"]>[0]) => void;
  on: (hookName: string, handler: HookHandler) => void;
  registerGatewayMethod: (
    method: string,
    handler: TestGatewayMethodHandler,
    opts?: { scope?: string },
  ) => void;
  calls: {
    services: Parameters<OpenClawPluginApi["registerService"]>[0][];
    lifecycle: Parameters<OpenClawPluginApi["registerRuntimeLifecycle"]>[0][];
    gatewayMethods: Array<{
      method: string;
      handler: TestGatewayMethodHandler;
    }>;
    hooks: Array<{ hookName: string; handler: HookHandler }>;
  };
};

function createApi(params: { pluginConfig: Record<string, unknown> }): TestApi {
  const calls: TestApi["calls"] = {
    services: [],
    lifecycle: [],
    gatewayMethods: [],
    hooks: [],
  };
  const logger: PluginLogger = {
    info: () => {},
    warn: () => {},
    error: () => {},
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
    on: (hookName: string, handler: HookHandler) => calls.hooks.push({ hookName, handler }),
    registerGatewayMethod: (method, handler) => calls.gatewayMethods.push({ method, handler }),
    calls,
  };
}

function registerPlugin(api: TestApi, moduleLoader: NemoFlowModuleLoader): void {
  registerNemoFlowPlugin(api as unknown as OpenClawPluginApi, moduleLoader);
}

async function loadRealNemoFlowModules(): Promise<NemoFlowModules> {
  try {
    return await defaultNemoFlowModuleLoader();
  } catch (error) {
    if (isMissingLocalNemoFlowNode(error)) {
      throw new Error(
        "Live smoke requires the nemo-flow-node native package for this platform. Install workspace dependencies, or build local bindings when testing an unpublished version, then rerun `npm run test:live --workspace=nemo-flow-openclaw`.",
      );
    }
    throw error;
  }
}

function isMissingLocalNemoFlowNode(error: unknown): boolean {
  return (
    error instanceof Error &&
    "code" in error &&
    error.code === "ERR_MODULE_NOT_FOUND" &&
    error.message.includes("nemo-flow-node")
  );
}
