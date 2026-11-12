// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Projection between DeepSeek Harness tool results and NeMo Relay
 * `ToolExecutionResult` values.
 *
 * Relay receives the normalized `nemo.relay.tool_result@1` wrapper of the host
 * result. The host result object itself never crosses the native boundary:
 * the `tools/execute` listener keeps it and returns it to DeepSeek Harness, so
 * cancellation results, deferred contexts, and turn-concluding flags are
 * preserved exactly. When Relay middleware changes the wrapper, the change is
 * applied through the host's `tools/post-execute` contract instead.
 */
import { asRecord, jsonEqual, toJson } from './llm-projection.js';
import type { ContentBlock, HarnessToolResult, JsonValue } from './dsh-types.js';

export const TOOL_RESULT_SCHEMA = 'nemo.relay.tool_result@1';

/** Opaque annotation key that pairs a wrapper returned by Relay with the host result it came from. */
export const TOOL_RESULT_ANNOTATION_KEY = 'deepseek_harness';

export type ToolResultWrapper = {
  schema: typeof TOOL_RESULT_SCHEMA;
  content: string | null;
  is_error: boolean;
  result: JsonValue;
};

/** Text of the model-visible content blocks of a host tool result. */
export function contentText(content: readonly ContentBlock[]): string {
  return content
    .map((block) => {
      switch (block.type) {
        case 'text':
          return block.text;
        case 'image':
          return '[image]';
        case 'file':
          return `[file: ${block.attachment.name}]`;
        case 'tool-addition':
        case 'tool-removal':
          return `[${block.type}: ${block.toolName}]`;
        default:
          return `[${block.type}]`;
      }
    })
    .join('\n');
}

/** Wrap a host tool result in the normalized Relay tool-result format. */
export function wrapToolResult(result: HarnessToolResult): ToolResultWrapper {
  const text = contentText(result.content);
  return {
    schema: TOOL_RESULT_SCHEMA,
    content: text.length > 0 ? text : null,
    is_error: result.isError,
    result: result.isError
      ? {
          error: {
            message: result.error.message,
            ...(result.error.info === undefined ? {} : { info: toJson(result.error.info) }),
          },
        }
      : toJson(result.value),
  };
}

/** Annotation that marks a wrapper as the projection of one host result. */
export function resultAnnotation(token: string): JsonValue {
  return { [TOOL_RESULT_ANNOTATION_KEY]: { token } };
}

/** The pairing token of a result returned through Relay, if it carries one. */
export function resultToken(annotation: unknown): string | undefined {
  const marker = asRecord(asRecord(annotation)?.[TOOL_RESULT_ANNOTATION_KEY]);
  return typeof marker?.token === 'string' ? marker.token : undefined;
}

/** How Relay middleware changed a host tool result, expressed in host terms. */
export type ToolResultEdit =
  | { kind: 'none' }
  | { kind: 'content'; text: string }
  | { kind: 'value'; value: JsonValue }
  | { kind: 'replace'; result: HarnessToolResult };

/**
 * Compare the wrapper Relay returned with the wrapper the host result produced.
 *
 * A wrapper Relay produced without running the host body (for example a
 * response-cache hit) has no pairing token and replaces the host result.
 */
export function diffToolResult(
  original: ToolResultWrapper | undefined,
  returned: unknown,
  token: string,
  annotation: unknown,
): ToolResultEdit {
  const wrapper = parseWrapper(returned);
  if (original === undefined || resultToken(annotation) !== token) {
    return { kind: 'replace', result: hostResultFromRelay(wrapper, returned) };
  }
  if (wrapper === undefined) {
    // Middleware replaced the wrapper with a plain JSON value: treat it as the new canonical value.
    return jsonEqual(returned, original) ? { kind: 'none' } : { kind: 'value', value: toJson(returned) };
  }
  if (jsonEqual(wrapper, original)) {
    return { kind: 'none' };
  }
  if (!original.is_error && !wrapper.is_error && !jsonEqual(wrapper.result, original.result)) {
    return { kind: 'value', value: wrapper.result };
  }
  return { kind: 'content', text: wrapper.content ?? '' };
}

/** Convert a result Relay produced without the host body into a host tool result. */
export function hostResultFromRelay(wrapper: ToolResultWrapper | undefined, raw: unknown): HarnessToolResult {
  if (wrapper?.is_error === true) {
    const message = wrapper.content ?? 'NeMo Relay returned a failed tool result';
    return {
      isError: true,
      error: { message, info: { name: 'NemoRelayToolResult', code: 'NEMO_RELAY_RESULT' } },
      content: [{ type: 'text', text: message }],
    };
  }
  const value = wrapper === undefined ? toJson(raw) : wrapper.result;
  const text = wrapper?.content ?? (typeof value === 'string' ? value : JSON.stringify(value));
  return { isError: false, value, content: [{ type: 'text', text }] };
}

function parseWrapper(value: unknown): ToolResultWrapper | undefined {
  const record = asRecord(value);
  if (record?.schema !== TOOL_RESULT_SCHEMA || typeof record.is_error !== 'boolean' || !('result' in record)) {
    return undefined;
  }
  const content = displayText(record.content);
  if (content === undefined) {
    return undefined;
  }
  return { schema: TOOL_RESULT_SCHEMA, content, is_error: record.is_error, result: record.result ?? null };
}

/** Normalize wrapper `content` (text, an array of content parts, or null) to display text. */
function displayText(value: JsonValue | undefined): string | null | undefined {
  if (value === null || typeof value === 'string') {
    return value;
  }
  if (Array.isArray(value)) {
    return value
      .map((part) => {
        const record = asRecord(part);
        return typeof record?.text === 'string'
          ? record.text
          : `[${typeof record?.type === 'string' ? record.type : 'part'}]`;
      })
      .join('\n');
  }
  return undefined;
}
