// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createRequire } from 'node:module';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const invokedDirectly = path.resolve(process.argv[1] ?? '') === fileURLToPath(import.meta.url);

const require = createRequire(import.meta.url);
const relay = require('nemo-relay-node');
const plugin = require('nemo-relay-node/plugin');
const typed = require('nemo-relay-node/typed');
const JSON_CODEC = new typed.JsonPassthrough();

export { plugin, relay };

export const DEFAULT_CONFIG = {
  tag: 'documentation',
  observe: { enabled: true, redact_keys: ['secret'] },
  requests: {
    enabled: true,
    mode: 'enforce',
    blocked_tools: ['dangerous_tool'],
    blocked_models: ['restricted-model'],
    header_name: 'x-nemo-relay-plugin',
    header_value: 'documentation',
    priority: 20,
    break_chain: false,
  },
  execution: { enabled: true, priority: 30, emit_pending_marks: true },
  runtime: { emit_marks: true, emit_isolated_scope: true },
  registration_control: {
    enabled: false,
    kinds: ['subscriber'],
    registration_name: 'documentation-controlled-subscriber',
    reason: 'disabled by documentation plugin',
  },
};

const GROUP_FIELDS = {
  observe: new Set(['enabled', 'redact_keys']),
  requests: new Set([
    'enabled',
    'mode',
    'blocked_tools',
    'blocked_models',
    'header_name',
    'header_value',
    'priority',
    'break_chain',
  ]),
  execution: new Set(['enabled', 'priority', 'emit_pending_marks']),
  runtime: new Set(['emit_marks', 'emit_isolated_scope']),
  registration_control: new Set(['enabled', 'kinds', 'registration_name', 'reason']),
};

function diagnostic(level, code, field, message) {
  return {
    level,
    code: `documentation-plugin.${code}`,
    component: 'documentation-plugin',
    ...(field === null ? {} : { field }),
    message,
  };
}

function normalizedConfig(config) {
  const settings = structuredClone(DEFAULT_CONFIG);
  if ('tag' in config) settings.tag = config.tag;
  for (const group of Object.keys(GROUP_FIELDS)) {
    const value = config[group];
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      Object.assign(settings[group], value);
    }
  }
  return settings;
}

function redactJson(value, redactKeys) {
  if (Array.isArray(value)) return value.map((item) => redactJson(item, redactKeys));
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        redactKeys.includes(key) ? '[REDACTED]' : redactJson(item, redactKeys),
      ]),
    );
  }
  return value;
}

function emitRuntimeEvents(runtime, tag) {
  if (runtime.emit_marks) {
    relay.event('documentation-plugin.request', null, { tag, secret: 'application-value' });
  }
  if (runtime.emit_isolated_scope) {
    const isolatedStack = relay.createScopeStack();
    relay.withScopeStack(isolatedStack, () => {
      const scope = relay.pushScope('documentation-plugin.isolated', relay.ScopeType.Custom);
      relay.popScope(scope);
    });
  }
}

function validateDocumentationConfig(config) {
  const diagnostics = [];
  if (config === null || typeof config !== 'object' || Array.isArray(config)) {
    return [diagnostic('error', 'invalid_config', null, 'plugin config must be a JSON object')];
  }
  const topLevel = new Set(['tag', ...Object.keys(GROUP_FIELDS)]);
  for (const key of Object.keys(config)) {
    if (!topLevel.has(key)) {
      diagnostics.push(diagnostic('warning', 'unknown_field', key, `unknown field '${key}' is not supported`));
    }
  }
  for (const [group, allowed] of Object.entries(GROUP_FIELDS)) {
    const value = config[group];
    if (value !== undefined && (value === null || typeof value !== 'object' || Array.isArray(value))) {
      diagnostics.push(diagnostic('error', 'invalid_config', group, `${group} must be an object`));
      continue;
    }
    if (value !== undefined) {
      for (const key of Object.keys(value)) {
        if (!allowed.has(key)) {
          const field = `${group}.${key}`;
          diagnostics.push(diagnostic('warning', 'unknown_field', field, `unknown field '${field}' is not supported`));
        }
      }
    }
  }

  const settings = normalizedConfig(config);
  const fields = {
    tag: settings.tag,
    'observe.enabled': settings.observe.enabled,
    'observe.redact_keys': settings.observe.redact_keys,
    'requests.enabled': settings.requests.enabled,
    'requests.mode': settings.requests.mode,
    'requests.blocked_tools': settings.requests.blocked_tools,
    'requests.blocked_models': settings.requests.blocked_models,
    'requests.header_name': settings.requests.header_name,
    'requests.header_value': settings.requests.header_value,
    'requests.priority': settings.requests.priority,
    'requests.break_chain': settings.requests.break_chain,
    'execution.enabled': settings.execution.enabled,
    'execution.priority': settings.execution.priority,
    'execution.emit_pending_marks': settings.execution.emit_pending_marks,
    'runtime.emit_marks': settings.runtime.emit_marks,
    'runtime.emit_isolated_scope': settings.runtime.emit_isolated_scope,
    'registration_control.enabled': settings.registration_control.enabled,
    'registration_control.kinds': settings.registration_control.kinds,
    'registration_control.registration_name': settings.registration_control.registration_name,
    'registration_control.reason': settings.registration_control.reason,
  };
  const stringFields = new Set([
    'tag',
    'requests.mode',
    'requests.header_name',
    'requests.header_value',
    'registration_control.registration_name',
    'registration_control.reason',
  ]);
  const arrayFields = new Set([
    'observe.redact_keys',
    'requests.blocked_tools',
    'requests.blocked_models',
    'registration_control.kinds',
  ]);
  const integerFields = new Set(['requests.priority', 'execution.priority']);
  for (const [field, value] of Object.entries(fields)) {
    const valid = stringFields.has(field)
      ? typeof value === 'string'
      : arrayFields.has(field)
        ? Array.isArray(value) && value.every((item) => typeof item === 'string')
        : integerFields.has(field)
          ? Number.isInteger(value)
          : typeof value === 'boolean';
    if (!valid) {
      diagnostics.push(diagnostic('error', 'invalid_config', field, `${field} has the wrong type`));
    }
  }
  if (typeof settings.tag === 'string' && settings.tag.length === 0) {
    diagnostics.push(diagnostic('error', 'invalid_tag', 'tag', 'tag must be a non-empty string'));
  }
  const supportedKinds = new Set(Object.values(relay.RuntimeRegistrationKind));
  if (
    Array.isArray(settings.registration_control.kinds) &&
    (settings.registration_control.kinds.length === 0 ||
      !settings.registration_control.kinds.every((kind) => supportedKinds.has(kind)))
  ) {
    diagnostics.push(
      diagnostic(
        'error',
        'invalid_config',
        'registration_control.kinds',
        'registration_control.kinds must be a non-empty array of supported registration kinds',
      ),
    );
  }
  for (const field of ['requests.header_name', 'requests.header_value']) {
    const value = fields[field];
    if (typeof value === 'string' && value.length === 0) {
      diagnostics.push(diagnostic('error', 'invalid_header', field, `${field} must be a non-empty string`));
    }
  }
  for (const field of ['registration_control.registration_name', 'registration_control.reason']) {
    const value = fields[field];
    if (typeof value === 'string' && value.length === 0) {
      diagnostics.push(diagnostic('error', 'invalid_config', field, `${field} must be a non-empty string`));
    }
  }
  if (typeof settings.requests.mode === 'string' && !new Set(['observe', 'enforce']).has(settings.requests.mode)) {
    diagnostics.push(
      diagnostic('error', 'unsupported_mode', 'requests.mode', 'requests.mode must be either observe or enforce'),
    );
  }
  return diagnostics;
}

export const documentationPlugin = {
  events: [],
  validate(config) {
    return validateDocumentationConfig(config);
  },
  register(config, context) {
    const settings = normalizedConfig(config);
    const { observe, requests, execution, runtime, registration_control: registrationControl } = settings;
    if (registrationControl.enabled) {
      context.registerConditionalMiddlewareGuardrail(
        'documentation-registration-control',
        registrationControl.kinds,
        registrationControl.registration_name,
        () => registrationControl.reason,
      );
    }
    if (observe.enabled) {
      context.registerSubscriber('events', (event) => documentationPlugin.events.push(event.name));
      const sanitizeEvent = (_event, fields) => ({
        data: redactJson(fields.data, observe.redact_keys),
        categoryProfile: redactJson(fields.categoryProfile, observe.redact_keys),
        metadata: { ...(redactJson(fields.metadata, observe.redact_keys) ?? {}), plugin_tag: settings.tag },
      });
      context.registerMarkSanitizeGuardrail('mark-sanitizer', 10, sanitizeEvent);
      context.registerScopeSanitizeStartGuardrail('scope-start-sanitizer', 10, sanitizeEvent);
      context.registerScopeSanitizeEndGuardrail('scope-end-sanitizer', 10, sanitizeEvent);
      context.registerToolSanitizeRequestGuardrail('tool-request-sanitizer', 10, (_name, value) =>
        redactJson(value, observe.redact_keys),
      );
      context.registerToolSanitizeResponseGuardrail('tool-response-sanitizer', 10, (_name, value) =>
        redactJson(value, observe.redact_keys),
      );
      context.registerLlmSanitizeRequestGuardrail('llm-request-sanitizer', 10, (request) => ({
        ...request,
        content: redactJson(request.content, observe.redact_keys),
      }));
      context.registerLlmSanitizeResponseGuardrail('llm-response-sanitizer', 10, (response) =>
        redactJson(response, observe.redact_keys),
      );
    }
    if (requests.enabled) {
      context.registerToolConditionalExecutionGuardrail('tool-policy', 10, (name) =>
        requests.mode === 'enforce' && requests.blocked_tools.includes(name) ? `tool '${name}' is blocked` : null,
      );
      context.registerToolRequestIntercept('tool-request', requests.priority, requests.break_chain, (_name, args) => {
        return { ...args, plugin_tag: settings.tag };
      });
      context.registerLlmConditionalExecutionGuardrail('llm-policy', 10, (request) => {
        const model = request?.content?.model;
        return requests.mode === 'enforce' && requests.blocked_models.includes(model)
          ? `model '${model}' is blocked`
          : null;
      });
      context.registerLlmRequestIntercept(
        'llm-request',
        requests.priority,
        requests.break_chain,
        ({ request, annotated }) => ({
          request: {
            ...request,
            headers: {
              ...request.headers,
              [requests.header_name]: requests.header_value,
            },
          },
          annotated,
        }),
      );
    }
    if (runtime.emit_marks || runtime.emit_isolated_scope) {
      context.registerToolExecutionIntercept('runtime-events', 0, async (args, next) => {
        emitRuntimeEvents(runtime, settings.tag);
        return await next(args);
      });
    }
    if (execution.enabled) {
      context.registerToolExecutionIntercept('tool-execution', execution.priority, async (args, next) => {
        const downstream = await next(args);
        return {
          ...downstream,
          pendingMarks: execution.emit_pending_marks ? [{ name: 'documentation-plugin.tool-complete' }] : [],
        };
      });
      context.registerLlmExecutionIntercept('llm-execution', execution.priority, async (request, next) =>
        next(request),
      );
      context.registerLlmStreamExecutionIntercept('llm-stream', execution.priority, async function* (request, next) {
        for await (const chunk of await next(request)) {
          yield { ...chunk, plugin_stream: true };
        }
      });
    }
  },
};

export function isolateExampleEnvironment() {
  const previousDirectory = process.cwd();
  const previousConfigHome = process.env.XDG_CONFIG_HOME;
  const isolationDirectory = mkdtempSync(path.join(tmpdir(), 'nemo-relay-language-plugin-'));
  process.chdir(isolationDirectory);
  process.env.XDG_CONFIG_HOME = isolationDirectory;
  return () => {
    process.chdir(previousDirectory);
    if (previousConfigHome === undefined) delete process.env.XDG_CONFIG_HOME;
    else process.env.XDG_CONFIG_HOME = previousConfigHome;
    rmSync(isolationDirectory, { recursive: true, force: true });
  };
}

export function config(mode, enabled = true) {
  const settings = structuredClone(DEFAULT_CONFIG);
  settings.requests.mode = mode;
  return { version: 1, components: [plugin.ComponentSpec('documentation-plugin', settings, { enabled })] };
}

export async function main() {
  const restoreEnvironment = isolateExampleEnvironment();
  documentationPlugin.events.length = 0;
  plugin.register('documentation-plugin', documentationPlugin);
  console.log('registered:', plugin.listKinds());
  const invalid = plugin.validate(config('invalid')).diagnostics;
  const disabledInvalid = plugin.validate(config('invalid', false)).diagnostics;
  if (disabledInvalid[0]?.code !== 'documentation-plugin.unsupported_mode') {
    throw new Error('disabled invalid configuration must still be validated');
  }
  console.log('invalid:', invalid);
  let summary;
  try {
    const report = await plugin.initialize(config('enforce'));
    console.log('active:', report);
    const toolResult = await relay.toolCallExecute('safe_tool', { value: 1 }, (args) => ({
      result: args,
      annotation: { source: 'application' },
    }));
    console.log('tool:', toolResult);
    const request = { headers: {}, content: { model: 'allowed-model' } };
    const llmResult = await relay.llmCallExecute('allowed-model', request, (rewritten) => ({
      headers: rewritten.headers,
    }));
    console.log('llm:', llmResult);
    const stream = await typed.typedLlmStreamExecute(
      'allowed-model',
      request,
      async function* streamChunks() {
        yield { chunk: 1 };
        yield { chunk: 2 };
      },
      () => {},
      () => ({ done: true }),
      JSON_CODEC,
      JSON_CODEC,
    );
    const streamResults = [];
    for (;;) {
      const chunk = await stream.next();
      if (chunk === null) break;
      streamResults.push(chunk);
      console.log('stream:', chunk);
    }
    relay.event('documentation-event', null, { emitted: true });
    await relay.flushSubscribers();
    console.log('event: documentation-event emitted through plugin sanitizer');
    summary = {
      invalid,
      report,
      tool: toolResult,
      llm: llmResult,
      stream: streamResults,
      events: [...documentationPlugin.events],
    };
  } finally {
    plugin.clear();
    plugin.deregister('documentation-plugin');
    restoreEnvironment();
  }
  console.log('teardown: complete');
  return summary;
}

if (invokedDirectly) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
