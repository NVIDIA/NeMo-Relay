// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-session trace files for the DeepSeek Harness conversation view.
 *
 * The configured observability plugin owns NeMo Relay's exported outputs. The
 * conversation view additionally needs one session's events at a time, so a
 * subscriber routes each ATOF event to its DeepSeek Harness session: directly
 * through `metadata.session_id`, or through the event's propagation root,
 * which is first seen on an event that carries the session id.
 *
 * The files hold ATOF events with one reduction: an LLM start event keeps only
 * the part of its request the view shows (see {@link compactForView}). Every
 * model call otherwise repeats the turn's message history and the complete
 * tool list, twice, which grows a long conversation's file by megabytes per
 * call. Complete requests remain available from Relay's own ATOF and ATIF
 * exporters.
 */
import { appendFile, mkdir } from 'node:fs/promises';
import path from 'node:path';

import { asRecord } from './llm-projection.js';

const FLUSH_DELAY_MS = 100;
const MAX_ROOTS = 10_000;

/** Metadata key on a compacted LLM start event: `{ request_messages, request_tools }` of the full request. */
export const VIEW_REQUEST_SUMMARY_KEY = 'deepseek_harness.view';
/** Longest tool-result text kept in a compacted request; the view shows a one-line preview of each. */
export const VIEW_TOOL_RESULT_CHARS = 4000;

/** File-name stem for a session id. */
export function sessionFileStem(sessionId: string): string {
  return sessionId.replace(/[^A-Za-z0-9._-]/g, '_');
}

/**
 * Reduce an LLM start event to what the conversation view shows.
 *
 * The request keeps the messages after the last assistant message (the tool
 * results and user input this call reacts to), preceded by the latest user
 * message when none of them is one. The tool list, the earlier history, and
 * the codec's duplicate of the request (`category_profile.annotated_request`)
 * are dropped; their sizes are recorded under {@link VIEW_REQUEST_SUMMARY_KEY}
 * in the event metadata. Any other event is returned unchanged, and the input
 * is never modified.
 */
export function compactForView(event: unknown): unknown {
  const record = asRecord(event);
  if (record === undefined || record.category !== 'llm' || record.scope_category !== 'start') {
    return event;
  }
  const data = asRecord(record.data);
  const content = asRecord(data?.content);
  const messages = content?.messages;
  if (data === undefined || content === undefined || !Array.isArray(messages)) {
    return event;
  }
  let lastAssistant = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (asRecord(messages[index])?.role === 'assistant') {
      lastAssistant = index;
      break;
    }
  }
  let kept: unknown[] = messages.slice(lastAssistant + 1).filter((message) => asRecord(message)?.role !== 'system');
  if (!kept.some((message) => asRecord(message)?.role === 'user')) {
    for (let index = lastAssistant; index >= 0; index -= 1) {
      const message: unknown = messages[index];
      if (asRecord(message)?.role === 'user') {
        kept = [message, ...kept];
        break;
      }
    }
  }
  const { messages: _messages, tools, ...request } = content;
  const profile = asRecord(record.category_profile);
  const { annotated_request: _annotated, ...restProfile } = profile ?? {};
  return {
    ...record,
    data: { ...data, content: { ...request, messages: kept.map(capToolResult) } },
    ...(profile === undefined ? {} : { category_profile: restProfile }),
    metadata: {
      ...asRecord(record.metadata),
      [VIEW_REQUEST_SUMMARY_KEY]: {
        request_messages: messages.length,
        request_tools: Array.isArray(tools) ? tools.length : 0,
      },
    },
  };
}

/** Shorten a tool message's text; the tool's own end event carries the complete result. */
function capToolResult(message: unknown): unknown {
  const record = asRecord(message);
  if (record === undefined || record.role !== 'tool') {
    return message;
  }
  const content = record.content;
  if (typeof content === 'string') {
    return content.length > VIEW_TOOL_RESULT_CHARS ? { ...record, content: shorten(content) } : message;
  }
  if (!Array.isArray(content)) {
    return message;
  }
  return {
    ...record,
    content: content.map((part) => {
      const text = asRecord(part)?.text;
      return typeof text === 'string' && text.length > VIEW_TOOL_RESULT_CHARS
        ? { ...asRecord(part), text: shorten(text) }
        : part;
    }),
  };
}

function shorten(text: string): string {
  return `${text.slice(0, VIEW_TOOL_RESULT_CHARS)}…`;
}

export class SessionTraceWriter {
  private readonly roots = new Map<string, string>();
  private readonly pending = new Map<string, string[]>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private chain: Promise<void> = Promise.resolve();
  private directoryReady = false;

  constructor(
    readonly directory: string,
    private readonly warn: (message: string) => void = () => undefined,
  ) {}

  /** Absolute path of one session's trace file. */
  pathFor(sessionId: string): string {
    return path.join(this.directory, `${sessionFileStem(sessionId)}.atof.jsonl`);
  }

  /** Route one ATOF event to its session's file. */
  accept(event: unknown): void {
    const record = asRecord(event);
    if (record === undefined) {
      return;
    }
    // Per-chunk receipt marks are high-volume and carry no trace structure.
    if (record.kind === 'mark' && record.name === 'llm.chunk') {
      return;
    }
    const sessionId = asRecord(record.metadata)?.session_id;
    const root = record.propagation_root_uuid;
    if (typeof sessionId === 'string' && typeof root === 'string') {
      if (this.roots.size >= MAX_ROOTS) {
        this.roots.clear();
      }
      this.roots.set(root, sessionId);
    }
    const target =
      typeof sessionId === 'string' ? sessionId : typeof root === 'string' ? this.roots.get(root) : undefined;
    if (target === undefined) {
      return;
    }
    const lines = this.pending.get(target) ?? [];
    lines.push(JSON.stringify(compactForView(event)));
    this.pending.set(target, lines);
    this.timer ??= setTimeout(() => {
      this.timer = undefined;
      void this.flush();
    }, FLUSH_DELAY_MS);
  }

  /** Append every queued line; resolves when the writes settle. */
  flush(): Promise<void> {
    if (this.timer !== undefined) {
      clearTimeout(this.timer);
      this.timer = undefined;
    }
    const batches = [...this.pending.entries()];
    this.pending.clear();
    if (batches.length > 0) {
      this.chain = this.chain.then(async () => {
        try {
          if (!this.directoryReady) {
            await mkdir(this.directory, { recursive: true, mode: 0o700 });
            this.directoryReady = true;
          }
          for (const [sessionId, lines] of batches) {
            await appendFile(this.pathFor(sessionId), `${lines.join('\n')}\n`, { mode: 0o600 });
          }
        } catch (error) {
          this.warn(`NeMo Relay view trace write failed: ${error instanceof Error ? error.message : String(error)}`);
        }
      });
      // A failed write must not leave every later flush rejected, or reject a caller that ignores it.
      this.chain = this.chain.catch(() => undefined);
    }
    return this.chain;
  }
}
