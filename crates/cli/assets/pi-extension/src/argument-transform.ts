// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Applies gateway-rewritten tool arguments to pi's `tool_call` event.
 *
 * A Relay request intercept can rewrite a tool's arguments. The gateway never
 * runs the tool, so the rewrite comes back in the hook response and this module
 * applies it to `event.input`.
 *
 * **Two pi facts make that dangerous, and shape everything here.**
 *
 * 1. *pi validates before the hook and never re-validates.*
 *    `validateToolArguments` returns a `structuredClone`, that clone is what
 *    `tool_call` receives, and pi's own types say "later `tool_call` handlers
 *    see earlier mutations. No re-validation is performed after mutation." So
 *    arguments that violate the tool's schema will execute.
 * 2. *The schema is reachable, and deliberately not used.* `pi.getAllTools()`
 *    returns every configured tool -- built-ins included -- with its TypeBox
 *    `parameters` schema (pi `v0.84.0`, `core/extensions/types.ts:1334`, impl
 *    `core/agent-session.ts:908`). So validating locally, or forwarding the
 *    schema to the gateway, are both possible. They are not done because the
 *    tool set is per-session mutable (`setActiveTools`, `registerTool`), so a
 *    forwarded schema can go stale mid-session, and carrying pi's tool
 *    vocabulary into the gateway buys precision this transform does not need.
 *
 * So the transform is constrained rather than validated: it may **rewrite the
 * values of existing keys, preserving each value's JSON type**, recursively.
 * Adding a key, removing a key, changing a type, or changing an array's length
 * is refused. An object that satisfied the schema before therefore still has
 * the required keys, of the required types, afterwards.
 *
 * **This is a structural guarantee, not schema validation.** Value-level
 * constraints -- `pattern`, `enum`, `minimum`, `format` -- are not checked. That
 * is a choice, not a limit: see point 2. A transform that rewrites a string to
 * one the schema would reject still executes.
 *
 * A refused transform **blocks the call**. Running the original arguments would
 * silently discard a policy decision, which is the failure the transform
 * existed to prevent; and this is a different axis from
 * `NEMO_RELAY_PI_FAIL`, which governs an unreachable gateway rather than a
 * gateway that answered with something unusable.
 */

/** What the gateway sent back on an allowed `tool_call`. */
export type TransformEnvelope = {
  tool_call_id?: unknown;
  input?: unknown;
};

export type TransformOutcome =
  | { kind: 'none' }
  | { kind: 'apply'; input: Record<string, unknown> }
  | { kind: 'refuse'; reason: string };

/** JSON type name used for the type-preservation check. `null` is its own type. */
function jsonType(value: unknown): string {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

/**
 * Check that `next` preserves the shape of `current`.
 *
 * Returns null when the shape holds, or a human-readable reason when it does
 * not. The reason reaches the model verbatim through pi, so it names the path.
 */
export function shapeViolation(current: unknown, next: unknown, path = 'input'): string | null {
  const currentType = jsonType(current);
  const nextType = jsonType(next);
  if (currentType !== nextType) {
    return `${path} changed type from ${currentType} to ${nextType}`;
  }
  if (currentType === 'object') return objectShapeViolation(current, next, path);
  if (currentType === 'array') return arrayShapeViolation(current, next, path);
  return null;
}

function objectShapeViolation(current: unknown, next: unknown, path: string): string | null {
  const currentRecord = current as Record<string, unknown>;
  const nextRecord = next as Record<string, unknown>;
  const currentKeys = Object.keys(currentRecord).sort((left, right) => left.localeCompare(right));
  const nextKeys = Object.keys(nextRecord).sort((left, right) => left.localeCompare(right));
  const added = nextKeys.filter((key) => !currentKeys.includes(key));
  const removed = currentKeys.filter((key) => !nextKeys.includes(key));
  if (added.length > 0) return `${path} added ${added.join(', ')}`;
  if (removed.length > 0) return `${path} removed ${removed.join(', ')}`;
  return firstViolation(currentKeys, (key) =>
    shapeViolation(currentRecord[key], nextRecord[key], `${path}.${key}`),
  );
}

function arrayShapeViolation(current: unknown, next: unknown, path: string): string | null {
  const currentItems = current as unknown[];
  const nextItems = next as unknown[];
  if (currentItems.length !== nextItems.length) {
    return `${path} changed length from ${currentItems.length} to ${nextItems.length}`;
  }
  return firstViolation(currentItems, (item, index) =>
    shapeViolation(item, nextItems[index], `${path}[${index}]`),
  );
}

function firstViolation<T>(items: T[], check: (item: T, index: number) => string | null): string | null {
  for (const [index, item] of items.entries()) {
    const violation = check(item, index);
    if (violation) return violation;
  }
  return null;
}

/**
 * Decide what to do with a hook response body for a given tool call.
 *
 * Pure, so the decision matrix is testable without pi or a gateway.
 */
export function decideTransform(
  body: { tool_call?: TransformEnvelope } | null,
  toolCallId: string,
  current: Record<string, unknown>,
): TransformOutcome {
  const envelope = body?.tool_call;
  if (envelope?.input === undefined) return { kind: 'none' };

  // The echoed id is what proves the transform belongs to the call we just posted, so it has to be
  // present and exact. Accepting a missing or non-string id would let a truncated or malformed body
  // rewrite the wrong call's arguments -- the failure the echo exists to prevent.
  if (typeof envelope.tool_call_id !== 'string' || envelope.tool_call_id !== toolCallId) {
    const named =
      typeof envelope.tool_call_id === 'string'
        ? envelope.tool_call_id
        : JSON.stringify(envelope.tool_call_id) ?? 'nothing';
    return {
      kind: 'refuse',
      reason: `the transform names tool call ${named}, not ${toolCallId}`,
    };
  }

  if (jsonType(envelope.input) !== 'object') {
    return { kind: 'refuse', reason: `the transform is ${jsonType(envelope.input)}, not an object` };
  }

  const violation = shapeViolation(current, envelope.input);
  if (violation) return { kind: 'refuse', reason: violation };

  return { kind: 'apply', input: envelope.input as Record<string, unknown> };
}

/**
 * Apply a transform to pi's event object **in place**.
 *
 * In place is required, not stylistic: pi passes the same object on to the tool
 * and to later handlers, so replacing the reference would be discarded. Because
 * the shape check has already run, this only ever overwrites existing keys.
 */
export function applyTransform(
  target: Record<string, unknown>,
  input: Record<string, unknown>,
): void {
  for (const [key, value] of Object.entries(input)) {
    target[key] = value;
  }
}

/** The block reason used when a transform arrives but cannot be applied safely. */
export function refusalReason(toolName: string, detail: string): string {
  return (
    `A NeMo Relay policy rewrote the arguments for this ${toolName} call, but the rewrite could ` +
    `not be applied safely: ${detail}. The call was blocked rather than run with the original ` +
    `arguments, because running them would ignore the policy. This is a configuration problem in ` +
    `the policy, not a judgment about your request.`
  );
}
