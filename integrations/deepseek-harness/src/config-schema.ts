// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The plugin's `Config` schema for DeepSeek Harness.
 *
 * DeepSeek Harness validates the Loader row with this schema, renders it in
 * Settings, and keeps `guardrails.rules` and `guardrails.failMode` volatile so
 * a saved change applies in place. An installed bundle cannot import DeepSeek
 * Harness's own packages, so the bundle depends on `@deepseek-ai/schemastery`
 * itself; Cordis recognizes the schema through its Standard Schema interface
 * (vendor `schemastery`), not by object identity. If the import still fails,
 * the plugin loads and `parseConfig` validates the row alone.
 */
import type Schema from '@deepseek-ai/schemastery';

import {
  DEFAULT_CAPTURE,
  DEFAULT_PHOENIX_PROJECT,
  DEFAULT_PHOENIX_PYTHON,
  DEFAULT_PHOENIX_STARTUP_TIMEOUT_MS,
  DEFAULT_PHOENIX_URL,
  DEFAULT_PHOENIX_VERSION,
  DEFAULT_RETENTION_DAYS,
  MAX_RETENTION_DAYS,
  PHOENIX_STARTUP_TIMEOUT_RANGE,
} from './config.js';

type SchemaStatic = typeof Schema;

/**
 * Build the schema when `@deepseek-ai/schemastery` resolves.
 *
 * @returns The schema, or `undefined` when schemastery is unavailable.
 */
export async function loadConfigSchema(): Promise<unknown> {
  let z: SchemaStatic;
  try {
    z = (await import('@deepseek-ai/schemastery')).default;
  } catch {
    return undefined;
  }
  return createConfigSchema(z);
}

function createConfigSchema(z: SchemaStatic): unknown {
  const rule = z.object({
    id: z.string().description('Stable rule id: letters, digits, ".", "_" and "-".'),
    tool: z
      .union([z.string(), z.array(z.string())])
      .description('Tool name, comma-separated names, a list of names, "*" or empty for any tool, or "/regex/flags".'),
    match: z
      .string()
      .description(
        'Regular expression tested against the arguments as JSON and each top-level string argument; empty matches any arguments.',
      ),
    flags: z.string().description('Regular expression flags from "imsu"; default "i".'),
    reason: z.string().description('Message the model receives when this rule blocks a call.'),
    enabled: z.boolean().description('Turn the rule off without deleting it.'),
  });

  return z.object({
    enabled: z.boolean().default(true).description('Route model and tool calls through NeMo Relay.'),
    dataDir: z
      .string()
      .description('Directory for traces, trajectories, rules, and status; default <DSH_HOME>/nemo-relay.'),
    plugins: z
      .object({
        version: z.number().default(1),
        components: z.array(
          z.object({
            kind: z.string().required(),
            enabled: z.boolean(),
            config: z.dict(z.any()),
          }),
        ),
      })
      .description('NeMo Relay plugin-host document, layered over the plugins.toml files like any Relay host.'),
    capture: z.object({
      includePrompts: z.boolean().default(DEFAULT_CAPTURE.includePrompts).description('Record model requests.'),
      includeResponses: z.boolean().default(DEFAULT_CAPTURE.includeResponses).description('Record model responses.'),
      stripToolArgs: z
        .boolean()
        .default(DEFAULT_CAPTURE.stripToolArgs)
        .description('Record only the argument names of tool calls.'),
      stripToolResults: z
        .boolean()
        .default(DEFAULT_CAPTURE.stripToolResults)
        .description('Record only whether tool calls succeeded.'),
    }),
    view: z.object({
      enabled: z.boolean().default(true).description('Write the per-session trace behind the NeMo Relay view.'),
    }),
    redaction: z.object({
      secrets: z
        .boolean()
        .default(true)
        .description("Scrub credentials from every Relay artifact with Relay's pii_redaction secret detectors."),
    }),
    guardrails: z.object({
      enabled: z.boolean().default(true).description('Check tool calls against the NeMo Relay rules.'),
      failMode: z
        .union(['open', 'closed'])
        .default('open')
        .volatile()
        .description('open allows a call when the rules cannot be checked; closed blocks it.'),
      rules: z
        .array(rule)
        .extra('default', undefined)
        .volatile()
        .description('Rules that block tool calls; when omitted, the shipped default rules apply.'),
    }),
    phoenix: z.object({
      enabled: z.boolean().default(true).description('Offer a local Arize Phoenix for these traces.'),
      url: z.string().default(DEFAULT_PHOENIX_URL).description('Phoenix to use or start.'),
      project: z.string().default(DEFAULT_PHOENIX_PROJECT).description('Phoenix project receiving the traces.'),
      autoConnect: z.boolean().default(true).description('Reconnect at startup when Phoenix was connected before.'),
      launch: z.object({
        enabled: z.boolean().default(true).description('Start a local Phoenix when none answers at the URL.'),
        version: z.string().default(DEFAULT_PHOENIX_VERSION).description('arize-phoenix release run by uv.'),
        python: z.string().default(DEFAULT_PHOENIX_PYTHON).description('Python version used by uv.'),
        command: z
          .array(z.string())
          .extra('default', undefined)
          .description('Command that serves Phoenix instead of uv, such as ["phoenix", "serve"].'),
        startupTimeoutMs: z
          .natural()
          .min(PHOENIX_STARTUP_TIMEOUT_RANGE.min)
          .max(PHOENIX_STARTUP_TIMEOUT_RANGE.max)
          .default(DEFAULT_PHOENIX_STARTUP_TIMEOUT_MS)
          .description('How long a launch may take, including the first download.'),
      }),
    }),
    retention: z.object({
      days: z
        .natural()
        .max(MAX_RETENTION_DAYS)
        .default(DEFAULT_RETENTION_DAYS)
        .description(
          'Remove view traces and default ATIF trajectories that no turn has touched for this many days; 0 keeps them.',
        ),
    }),
    agent: z.object({
      enabled: z
        .boolean()
        .default(true)
        .description(
          'Let the agent inspect and change NeMo Relay with the nemo_relay tool and skill; changes that loosen protection ask you first.',
        ),
    }),
  });
}
