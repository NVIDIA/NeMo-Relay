// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isRecord } from './gateway.ts';

const MAX_CONTENT_CHARS = 2000;

export function summarizeToolResult(result: unknown, isError: boolean): Record<string, unknown> {
  if (result === null || result === undefined) {
    return { content: isError ? 'Tool failed with no result.' : 'Tool completed with no result.' };
  }
  if (typeof result === 'string') return { content: truncate(result) };
  if (isRecord(result)) {
    const text = toolResultText(result.content ?? result.output ?? result.text);
    return {
      content: text ?? `Tool ${isError ? 'failed' : 'completed'}.`,
      result_keys: Object.keys(result).slice(0, 20),
    };
  }
  return { content: primitiveSummary(result, isError) };
}

function primitiveSummary(result: unknown, isError: boolean): string {
  switch (typeof result) {
    case 'boolean':
    case 'number':
    case 'bigint':
    case 'symbol':
      return truncate(String(result));
    default:
      return `Tool ${isError ? 'failed' : 'completed'} with an unsupported result type.`;
  }
}

function toolResultText(content: unknown): string | null {
  if (typeof content === 'string') return truncate(content);
  if (!Array.isArray(content)) return null;

  let text = '';
  let omittedChars = 0;
  let foundText = false;
  const append = (value: string): void => {
    if (omittedChars > 0) {
      omittedChars += value.length;
      return;
    }
    const kept = sliceAtCodePointBoundary(value, Math.max(0, MAX_CONTENT_CHARS - text.length));
    text += kept;
    omittedChars += value.length - kept.length;
  };
  for (const part of content) {
    if (!isRecord(part) || part.type !== 'text' || typeof part.text !== 'string') continue;
    if (foundText) append('\n');
    append(part.text);
    foundText = true;
  }
  if (!foundText) return null;
  return omittedChars === 0 ? text : `${text}... [truncated ${omittedChars} chars]`;
}

function sliceAtCodePointBoundary(value: string, limit: number): string {
  let end = Math.min(value.length, limit);
  if (
    end > 0 &&
    end < value.length &&
    value.charCodeAt(end - 1) >= 0xd800 &&
    value.charCodeAt(end - 1) <= 0xdbff &&
    value.charCodeAt(end) >= 0xdc00 &&
    value.charCodeAt(end) <= 0xdfff
  ) {
    end -= 1;
  }
  return value.slice(0, end);
}

function truncate(value: string): string {
  if (value.length <= MAX_CONTENT_CHARS) return value;
  const kept = sliceAtCodePointBoundary(value, MAX_CONTENT_CHARS);
  return `${kept}... [truncated ${value.length - kept.length} chars]`;
}
