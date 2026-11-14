// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { _ContentSnapshot } from '../binding.js';
import type { _ContentEntry, JsonCompatible, JsonValue } from '../binding.js';

/** Read-time location in a normalized snapshot; not an edit handle or provider-wire offset. */
export interface ContentLocation {
  readonly root: 'request_message' | 'message' | 'response' | 'content';
  readonly messageIndex: number | null;
  readonly field: 'content' | 'message' | 'tool_calls' | 'tool_call_item' | 'output' | 'value' | 'native_content';
  readonly index: number | null;
}

/** Tool payload in its original normalized representation; value materializes an independent JavaScript value. */
export type PayloadView =
  | { readonly kind: 'null'; readonly value: null }
  | { readonly kind: 'string'; readonly value: string }
  | { readonly kind: 'json' | 'content'; readonly value: JsonValue };

/** Tool invocation backed by selected Rust snapshot storage. */
export interface ToolCallView {
  readonly id: string | null;
  readonly callId: string | null;
  readonly name: string;
  readonly arguments: PayloadView;
}

/** Complete tool result, excluded from default text extraction. */
export interface ToolResultView {
  readonly id: string | null;
  readonly callId: string | null;
  readonly name: string | null;
  readonly isError: boolean | null;
  readonly payload: PayloadView;
}

/** Explicit normalized text or refusal. */
export interface TextEntry {
  readonly kind: 'text' | 'refusal';
  readonly location: ContentLocation;
  readonly text: string;
}

/** A typed tool-call entry. */
export interface ToolCallEntry {
  readonly kind: 'tool_call';
  readonly location: ContentLocation;
  readonly toolCall: ToolCallView;
}

/** A typed tool-result entry. */
export interface ToolResultEntry {
  readonly kind: 'tool_result';
  readonly location: ContentLocation;
  readonly toolResult: ToolResultView;
}

/** Media retained as an independent JavaScript value when value is accessed. */
export interface MediaEntry {
  readonly kind: 'image_url' | 'image' | 'audio' | 'file';
  readonly location: ContentLocation;
  readonly value: JsonValue;
}

/** Opaque provider-native content; default text permits only supported shallow fallbacks. */
export interface ProviderNativeEntry {
  readonly kind: 'provider_native';
  readonly location: ContentLocation;
  readonly provider: string;
  readonly nativeKind: string;
  readonly value: JsonValue;
}

/** Discriminated content union; checking kind narrows the available fields. */
export type ContentEntry = TextEntry | ToolCallEntry | ToolResultEntry | MediaEntry | ProviderNativeEntry;

type PayloadKind = PayloadView['kind'];
interface EntryMetadata {
  kind: ContentEntry['kind'];
  location: ContentLocation;
  text: string;
  provider: string;
  nativeKind: string;
  toolCall: { id: string | null; callId: string | null; name: string; payloadKind: PayloadKind };
  toolResult: {
    id: string | null;
    callId: string | null;
    name: string | null;
    isError: boolean | null;
    payloadKind: PayloadKind;
  };
}

function payload(part: _ContentEntry, kind: PayloadKind): PayloadView {
  // The native metadata and value come from the same immutable normalized entry.
  return Object.freeze({
    kind,
    get value() {
      return part.payloadValue();
    },
  }) as PayloadView;
}

function entry(part: _ContentEntry, metadata: EntryMetadata): ContentEntry {
  const location = Object.freeze(metadata.location);
  switch (metadata.kind) {
    case 'text':
    case 'refusal':
      return Object.freeze({ kind: metadata.kind, location, text: metadata.text });
    case 'tool_call': {
      const call = metadata.toolCall;
      const toolCall = Object.freeze({
        id: call.id,
        callId: call.callId,
        name: call.name,
        arguments: payload(part, call.payloadKind),
      });
      return Object.freeze({ kind: 'tool_call', location, toolCall });
    }
    case 'tool_result': {
      const result = metadata.toolResult;
      const toolResult = Object.freeze({
        id: result.id,
        callId: result.callId,
        name: result.name,
        isError: result.isError,
        payload: payload(part, result.payloadKind),
      });
      return Object.freeze({ kind: 'tool_result', location, toolResult });
    }
    case 'provider_native':
      return Object.freeze({
        kind: 'provider_native',
        location,
        provider: metadata.provider,
        nativeKind: metadata.nativeKind,
        get value() {
          return part.value();
        },
      });
    default:
      return Object.freeze({
        kind: metadata.kind,
        location,
        get value() {
          return part.value();
        },
      });
  }
}

/** Read-only Rust-owned snapshot. Iterators, entries, and payload views retain selected storage. */
export class MessageView {
  readonly #snapshot: _ContentSnapshot;

  private constructor(snapshot: _ContentSnapshot) {
    if (!(snapshot instanceof _ContentSnapshot)) {
      throw new TypeError('Use MessageView.fromMessage/fromContent or a codecs selection helper');
    }
    this.#snapshot = snapshot;
    Object.freeze(this);
  }

  /** Snapshot a normalized message, preserving its role and tool-result semantics. */
  static fromMessage<T extends JsonCompatible<T>>(value: T): MessageView {
    return new MessageView(new _ContentSnapshot('message', value as JsonValue));
  }

  /** Snapshot standalone normalized content: a string or content-part array. */
  static fromContent<T extends JsonCompatible<T>>(value: T): MessageView {
    return new MessageView(new _ContentSnapshot('content', value as JsonValue));
  }

  /** Iterate typed parts in normalized order, followed by separate tool-call fields. */
  *parts(): IterableIterator<ContentEntry> {
    const cursor = this.#snapshot.cursor();
    for (;;) {
      const part = cursor.nextEntry();
      if (part == null) return;
      yield entry(part, part.metadata() as unknown as EntryMetadata);
    }
  }

  /** Iterate text/refusal strings, including supported native fallbacks and excluding tool payloads. */
  *textParts(): IterableIterator<string> {
    const cursor = this.#snapshot.cursor();
    for (;;) {
      const text = cursor.nextText();
      if (text === null) return;
      yield text;
    }
  }

  /** Return null for absent text; otherwise preserve whitespace and newline-join empty/text/refusal parts. */
  text(): string | null {
    return this.#snapshot.text();
  }

  /** @internal Rust selects the source; this facade only validates JavaScript data properties. */
  static _selectRequest(request: unknown, selection: string, index?: number): MessageView | null {
    const annotation = object(request);
    const messages = history(request);
    const instructions = field(annotation, 'instructions');
    const selected = _ContentSnapshot.selectRequest(
      selection,
      {
        messages: Array.from({ length: messages.length }, (_, index) =>
          selectionMetadata(historyItem(messages, index)),
        ),
        instructions: instructions == null ? null : [],
      },
      index,
    ) as { messageIndex: number } | { instructions: true } | null;
    if (selected == null) return null;
    if ('instructions' in selected) return MessageView.fromContent(instructions as JsonValue);
    return new MessageView(
      new _ContentSnapshot('message', historyItem(messages, selected.messageIndex) as JsonValue, selected.messageIndex),
    );
  }

  /** @internal Snapshot current output without retaining unrelated response metadata. */
  static _output(value: JsonValue): MessageView {
    return new MessageView(new _ContentSnapshot('response', value));
  }
}

function object(value: unknown): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('Expected a normalized annotation object');
  }
  return value as Record<string, unknown>;
}

function field(value: Record<string, unknown>, key: string): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(value, key);
  if (descriptor === undefined) return undefined;
  if (!('value' in descriptor)) throw new TypeError('Normalized annotations must use data properties');
  return descriptor.value;
}

function history(request: unknown): unknown[] {
  const messages = field(object(request), 'messages');
  if (messages === undefined) return [];
  if (!Array.isArray(messages)) throw new TypeError('Normalized request messages must be an array');
  return messages;
}

function historyItem(messages: unknown[], index: number): unknown {
  const descriptor = Object.getOwnPropertyDescriptor(messages, String(index));
  if (!descriptor || !('value' in descriptor)) throw new TypeError('Message history must contain data entries');
  return descriptor.value;
}

// Structural metadata adapter only: provider/role recognition belongs to Rust.
// Content and payload fields never cross the boundary until an item is selected.
function selectionMetadata(value: unknown): JsonValue {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return null;
  const item = object(value);
  const envelope = field(item, 'value');
  const native =
    envelope !== null && typeof envelope === 'object' && !Array.isArray(envelope)
      ? {
          role: field(object(envelope), 'role') ?? null,
          type: Object.hasOwn(envelope, 'type') ? (field(object(envelope), 'type') ?? null) : undefined,
        }
      : null;
  return {
    role: field(item, 'role') ?? null,
    content: [],
    provider: field(item, 'provider') ?? null,
    kind: field(item, 'kind') ?? null,
    value:
      native === null
        ? null
        : {
            role: native.role,
            ...(native.type === undefined ? {} : { type: native.type }),
          },
  } as JsonValue;
}

/** Snapshot any normalized request item by integer index; negative indices count from the end. */
export function message<T extends JsonCompatible<T>>(request: T, index: number): MessageView {
  if (!Number.isSafeInteger(index)) throw new RangeError('Message index must be a safe integer');
  return MessageView._selectRequest(request, 'message', index)!;
}

/** Snapshot exactly the final supplied item, or null for empty history; never search backward for text. */
export function lastMessage<T extends JsonCompatible<T>>(request: T): MessageView | null {
  return MessageView._selectRequest(request, 'last_message');
}

/** Snapshot only top-level instructions, without falling back to history. */
export function instructionsView<T extends JsonCompatible<T>>(request: T): MessageView | null {
  return MessageView._selectRequest(request, 'instructions');
}

/** Snapshot current normalized message and separate tool calls, excluding response metadata. */
export function output<T extends JsonCompatible<T>>(response: T): MessageView {
  const annotation = object(response);
  return MessageView._output({
    message: (field(annotation, 'message') ?? null) as JsonValue,
    tool_calls: (field(annotation, 'tool_calls') ?? null) as JsonValue,
  });
}

/** Extract instructions, otherwise the first portable/native system/developer item, including textless selections. */
export function systemPrompt<T extends JsonCompatible<T>>(request: T): string | null {
  return MessageView._selectRequest(request, 'system_prompt')?.text() ?? null;
}

/** Extract the final portable/native user item without skipping a selected item that has no text. */
export function lastUserMessage<T extends JsonCompatible<T>>(request: T): string | null {
  return MessageView._selectRequest(request, 'last_user_message')?.text() ?? null;
}

/** Extract current normalized response text/refusals, excluding tool payloads, reasoning, and media. */
export function responseText<T extends JsonCompatible<T>>(response: T): string | null {
  return output(response).text();
}
