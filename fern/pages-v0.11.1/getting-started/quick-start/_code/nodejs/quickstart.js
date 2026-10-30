// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const fs = require("node:fs");
const relay = require("nemo-relay-node");
const plugin = require("nemo-relay-node/plugin");

async function main() {
  const activation = await plugin.initialize(plugin.defaultConfig(), "plugins.toml");

  try {
    await relay.withScope("demo-agent", relay.ScopeType.Agent, async (handle) => {
      relay.event("initialized", handle, { binding: "node" }, null);

      const toolResult = await relay.toolCallExecute(
        "emit_marker",
        { marker: "RELAY_QUICKSTART_TOOL_OK" },
        (args) => ({ result: { marker: args.marker } }),
        handle,
        null,
        null,
        null,
      );

      const llmResult = await relay.llmCallExecute(
        "demo-provider",
        new relay.LlmRequest(
          {},
          { messages: [{ role: "user", content: "RELAY_QUICKSTART_RESPONSE_OK" }] },
        ),
        () => ({ content: "RELAY_QUICKSTART_RESPONSE_OK" }),
        handle,
        null,
        null,
        null,
        null,
      );

      console.log(toolResult.result.marker);
      console.log(llmResult.content);
    });
  } finally {
    try {
      await relay.flushSubscribers();
    } finally {
      await activation.close();
    }
  }

  const eventText = fs.readFileSync("logs/events.jsonl", "utf8").trim();
  if (!eventText) throw new Error("Relay event file is empty");
  const events = eventText.split(/\r?\n/).map((line) => JSON.parse(line));
  if (events.some((event) => !event || event.atof_version !== "0.1" ||
      !["scope", "mark"].includes(event.kind))) {
    throw new Error("Expected ATOF 0.1 records");
  }
  const starts = new Map(events.filter((event) =>
    event.kind === "scope" && event.scope_category === "start",
  ).map((event) => [event.uuid, event]));
  const closed = events.filter((event) => {
    const start = starts.get(event.uuid);
    return event.kind === "scope" && event.scope_category === "end" && start &&
      ["name", "category", "parent_uuid"].every((key) => event[key] === start[key]);
  });
  if (!["agent", "tool", "llm"].every((category) =>
      closed.some((event) => event.category === category))) {
    throw new Error("Expected paired agent, tool, and LLM lifecycle records");
  }
  console.log("RELAY_QUICKSTART_EVENTS_OK");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
