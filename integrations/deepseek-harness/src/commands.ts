// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The `/relay` command: the user's (and the NeMo Relay view's) single entry
 * point for status, rule edits, and Phoenix.
 *
 * - `/relay`, `/relay status`: human-readable status.
 * - `/relay json`: the status snapshot as one line of JSON.
 * - `/relay rules`: the rules as text.
 * - `/relay rules save|reset|test <json>`: rule edits and dry runs.
 * - `/relay rules failmode open|closed`: whether calls the rules cannot check run or are blocked.
 * - `/relay phoenix`: Phoenix as text; `/relay phoenix start|stop|connect|disconnect`.
 *
 * DeepSeek Harness shows every reply in the session's transcript, so replies
 * are sentences, and `kind` is `error` when an action failed. Only `/relay
 * json` answers with JSON. Each action also leaves its structured result in the
 * status snapshot's `actions` list, keyed by the request id (`rid`) the caller
 * may add to its JSON input; the status file is rewritten before the command
 * replies, so the NeMo Relay view reads its result from there. Rule edits
 * exist only here and in the agent tool, which asks the user before loosening.
 */
import type { GuardrailPolicy, RuleError, Verdict } from './policy.js';
import {
  formatPhoenix,
  formatRules,
  formatStatus,
  type PhoenixSnapshot,
  type RelayActionRecord,
  type StatusSnapshot,
} from './status.js';

export type RelayCommandReply = { kind: 'success' | 'error'; text: string };

export type PhoenixActionResult = { ok: boolean; error?: string; phoenix: PhoenixSnapshot };

/** Phoenix operations the command exposes. Implementations never reject. */
export interface PhoenixControl {
  snapshot(): PhoenixSnapshot;
  /** Use a Phoenix that answers at the configured URL, or launch one, then export to it. */
  start(): Promise<PhoenixActionResult>;
  /** Stop a Phoenix this plugin started; only disconnect from one someone else runs. */
  stop(): Promise<PhoenixActionResult>;
  /** Export to a Phoenix that already answers, without launching one. */
  connect(): Promise<PhoenixActionResult>;
  /** Stop exporting, leaving any Phoenix server running. */
  disconnect(): Promise<PhoenixActionResult>;
}

export type RelayCommandContext = {
  snapshot(): StatusSnapshot;
  /** Rewrite the status file. */
  publish(): void;
  /** Keep an action's structured result for the status snapshot (published before the reply). */
  recordAction?: (entry: RelayActionRecord) => void;
  policy: GuardrailPolicy;
  phoenix: PhoenixControl | undefined;
  /** The invoking session, when the command runs in one. */
  sessionId: string | undefined;
  /** This session's ATOF trace file, when the view trace is on. */
  sessionTrace: string | undefined;
};

export const RELAY_COMMAND_HINT = '[status|json|rules|phoenix]';

/** A request id: what the NeMo Relay view adds to an action so it can find the result in the status file. */
const REQUEST_ID = /^[A-Za-z0-9._:-]{1,64}$/;

const USAGE = [
  'Usage:',
  '/relay [status]            NeMo Relay status',
  '/relay json                status as JSON',
  '/relay rules               the guardrail rules',
  '/relay rules save {"base":"<revision>","rules":[...]}',
  '/relay rules reset {"base":"<revision>"}',
  '/relay rules test {"tool":"bash","arguments":{...}}',
  '/relay rules failmode open|closed',
  '/relay phoenix [start|stop|connect|disconnect]',
  'Actions accept a request id: add "rid":"<id>" to the JSON, or a {"rid":"<id>"} object after failmode and phoenix actions.',
].join('\n');

/** An action result before it is timestamped and recorded. */
type ActionOutcome = Omit<RelayActionRecord, 'at' | 'rid'> & { text: string };

/**
 * Run one `/relay` invocation.
 *
 * @param rawInput - Everything after `/relay`, exactly as DeepSeek Harness passes it.
 * @param context - Status, rules, and Phoenix of this plugin instance.
 * @returns The command result; never rejects.
 */
export async function runRelayCommand(rawInput: unknown, context: RelayCommandContext): Promise<RelayCommandReply> {
  try {
    return await dispatch(String(rawInput ?? '').trim(), context);
  } catch (error) {
    return {
      kind: 'error',
      text: `NeMo Relay could not run the command: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
}

async function dispatch(input: string, context: RelayCommandContext): Promise<RelayCommandReply> {
  const [command, rest] = splitWord(input);
  switch (command.toLowerCase()) {
    case '':
    case 'status': {
      context.publish();
      return { kind: 'success', text: formatStatus(context.snapshot(), sessionLines(context)) };
    }
    case 'json': {
      context.publish();
      return { kind: 'success', text: JSON.stringify(context.snapshot()) };
    }
    case 'rules':
      return rules(rest, context);
    case 'phoenix':
      return phoenix(rest, context);
    case 'help':
      return { kind: 'success', text: USAGE };
    default:
      return { kind: 'error', text: `Unknown /relay subcommand "${command}".\n${USAGE}` };
  }
}

async function rules(input: string, context: RelayCommandContext): Promise<RelayCommandReply> {
  const { policy } = context;
  const [action, payloadText] = splitWord(input);
  const name = action.toLowerCase();
  if (name === '') {
    context.publish();
    return { kind: 'success', text: formatRules(policy.snapshot()) };
  }
  if (!['save', 'reset', 'test', 'failmode'].includes(name)) {
    return finish(context, undefined, {
      action: `rules.${name}`,
      ok: false,
      code: 'invalid',
      error: `Unknown rules action "${action}".`,
      text: `Unknown rules action "${action}". Use /relay rules, or rules save, reset, test, or failmode.`,
    });
  }
  const label = `rules.${name}`;
  // The fail mode also takes a plain word, which a {"rid":"<id>"} object may follow: `/relay rules failmode closed`.
  const [word, trailing] =
    name === 'failmode' && !payloadText.startsWith('{') ? splitWord(payloadText) : [undefined, payloadText];
  const payload = word !== undefined && trailing.length === 0 ? {} : parseObject(trailing);
  if (typeof payload === 'string') {
    return finish(context, undefined, invalid(label, payload));
  }
  const rid = requestId(payload.rid);
  if (rid === false) {
    return finish(
      context,
      undefined,
      invalid(label, 'The request id (rid) must be 1-64 letters, digits, ".", "_", ":" or "-".'),
    );
  }
  switch (name) {
    case 'save': {
      const result = await policy.save(payload.base, payload.rules);
      if (!result.ok) {
        return finish(context, rid, rulesFailure(label, result));
      }
      const count = result.guardrails.rules.length;
      return finish(context, rid, {
        action: label,
        ok: true,
        revision: result.revision,
        text: `Saved ${count} NeMo Relay ${count === 1 ? 'rule' : 'rules'} (revision ${result.revision}).`,
      });
    }
    case 'reset': {
      const result = await policy.reset(payload.base);
      if (!result.ok) {
        return finish(context, rid, rulesFailure(label, result));
      }
      return finish(context, rid, {
        action: label,
        ok: true,
        revision: result.revision,
        text: `Restored the default NeMo Relay rules (revision ${result.revision}).`,
      });
    }
    case 'test': {
      if (typeof payload.tool !== 'string' || payload.tool.length === 0) {
        return finish(context, rid, invalid(label, 'The request needs a "tool" name.'));
      }
      let verdict: Verdict;
      try {
        verdict = policy.dryRun(payload.tool, payload.arguments ?? {});
      } catch (error) {
        const reason = `NeMo Relay could not check this call: ${error instanceof Error ? error.message : String(error)}`;
        return finish(context, rid, { action: label, ok: false, code: 'error', error: reason, text: reason });
      }
      if (verdict.verdict === 'block') {
        return finish(context, rid, {
          action: label,
          ok: true,
          verdict: 'block',
          ruleId: verdict.ruleId,
          reason: verdict.reason,
          text: `NeMo Relay would block this call: rule "${verdict.ruleId}".`,
        });
      }
      const timedOut = verdict.timedOut ?? [];
      return finish(context, rid, {
        action: label,
        ok: true,
        verdict: 'allow',
        ...(timedOut.length === 0 ? {} : { timedOut }),
        text:
          timedOut.length === 0
            ? 'NeMo Relay would allow this call.'
            : `NeMo Relay would allow this call, but could not check ${timedOut.map((id) => `"${id}"`).join(', ')} (out of time, or arguments too large).`,
      });
    }
    default: {
      const mode = (word ?? (typeof payload.mode === 'string' ? payload.mode : '')).toLowerCase();
      if (mode.length === 0) {
        return finish(context, rid, invalid(label, 'Use /relay rules failmode open or /relay rules failmode closed.'));
      }
      const result = await policy.setFailMode(mode);
      return finish(
        context,
        rid,
        result.ok
          ? { action: label, ok: true, failMode: result.failMode, text: `Fail mode is now ${result.failMode}.` }
          : { action: label, ok: false, code: result.code, error: result.error, text: result.error },
      );
    }
  }
}

async function phoenix(input: string, context: RelayCommandContext): Promise<RelayCommandReply> {
  const [action, payloadText] = splitWord(input);
  const control = context.phoenix;
  const name = action.toLowerCase();
  if (name === '' || name === 'status') {
    context.publish();
    const snapshot = context.snapshot().phoenix;
    const link = sessionLink(snapshot, context.sessionId);
    return {
      kind: 'success',
      text: [formatPhoenix(snapshot), ...(link === undefined ? [] : [`This session: ${link}`])].join('\n'),
    };
  }
  const label = `phoenix.${name}`;
  if (!['start', 'stop', 'connect', 'disconnect'].includes(name)) {
    return finish(
      context,
      undefined,
      invalid(label, `Unknown phoenix action "${action}". Use start, stop, connect, or disconnect.`),
    );
  }
  const payload = payloadText.length === 0 ? {} : parseObject(payloadText);
  if (typeof payload === 'string') {
    return finish(context, undefined, invalid(label, payload));
  }
  const rid = requestId(payload.rid);
  if (rid === false) {
    return finish(
      context,
      undefined,
      invalid(label, 'The request id (rid) must be 1-64 letters, digits, ".", "_", ":" or "-".'),
    );
  }
  if (control === undefined) {
    const error = 'Phoenix is turned off in the plugin configuration (phoenix.enabled: false).';
    return finish(context, rid, {
      action: label,
      ok: false,
      code: 'unavailable',
      error,
      phoenix: context.snapshot().phoenix,
      text: error,
    });
  }
  const before = control.snapshot();
  const result = await control[name as 'start' | 'stop' | 'connect' | 'disconnect']();
  return finish(context, rid, {
    action: label,
    ok: result.ok,
    ...(result.ok ? {} : { code: 'error', error: result.error ?? formatPhoenix(result.phoenix) }),
    phoenix: result.phoenix,
    text: result.ok
      ? phoenixSentence(name, before, result.phoenix)
      : `Phoenix: ${result.error ?? formatPhoenix(result.phoenix)}`,
  });
}

/** What a successful Phoenix action did, in one sentence. */
function phoenixSentence(action: string, before: PhoenixSnapshot, after: PhoenixSnapshot): string {
  if (action === 'stop') {
    return before.mode === 'owned' || before.state === 'starting'
      ? `Stopped Phoenix at ${before.url}.`
      : `Stopped sending traces to Phoenix at ${before.url}; it keeps running.`;
  }
  if (action === 'disconnect') {
    return `Stopped sending traces to Phoenix at ${before.url}.`;
  }
  switch (after.state) {
    case 'running':
      return after.connected
        ? `Phoenix is running at ${after.url} and receiving traces.`
        : `Phoenix is running at ${after.url}, but is not receiving traces.`;
    case 'starting':
      return `Phoenix is starting at ${after.url}; traces follow once it answers.`;
    case 'waiting':
      return `Waiting for Phoenix at ${after.url}; traces follow once it answers.`;
    default:
      return formatPhoenix(after);
  }
}

/** Record an action's structured result, rewrite the status file, then reply in words. */
function finish(context: RelayCommandContext, rid: string | undefined, outcome: ActionOutcome): RelayCommandReply {
  const { text, ...record } = outcome;
  try {
    context.recordAction?.({ ...(rid === undefined ? {} : { rid }), ...record, at: new Date().toISOString() });
  } catch {
    // The reply still says what happened.
  }
  context.publish();
  return { kind: outcome.ok ? 'success' : 'error', text };
}

function invalid(action: string, error: string): ActionOutcome {
  return { action, ok: false, code: 'invalid', error, text: error };
}

function rulesFailure(
  action: string,
  result: { ok: false; code: 'stale' | 'invalid' | 'unavailable'; error: string; errors?: RuleError[] },
): ActionOutcome {
  return {
    action,
    ok: false,
    code: result.code,
    error: result.error,
    ...(result.errors === undefined ? {} : { errors: result.errors }),
    text: result.code === 'stale' ? 'Rules changed elsewhere; reload and try again.' : result.error,
  };
}

/** The request id, `undefined` when absent, or `false` when present but malformed. */
function requestId(value: unknown): string | undefined | false {
  if (value === undefined) {
    return undefined;
  }
  return typeof value === 'string' && REQUEST_ID.test(value) ? value : false;
}

function sessionLines(context: RelayCommandContext): string[] {
  const lines: string[] = [];
  if (context.sessionTrace !== undefined) {
    lines.push(`Session trace (ATOF): ${context.sessionTrace}`);
  }
  const link = sessionLink(context.snapshot().phoenix, context.sessionId);
  if (link !== undefined) {
    lines.push(`This session in Phoenix: ${link}`);
  }
  return lines;
}

/** This session's Phoenix deep link while Phoenix receives traces. */
export function sessionLink(phoenix: PhoenixSnapshot, sessionId: string | undefined): string | undefined {
  if (sessionId === undefined || phoenix.sessionUrl === undefined || !phoenix.connected) {
    return undefined;
  }
  return phoenix.sessionUrl.split('{sessionId}').join(encodeURIComponent(sessionId));
}

/** Split off the first whitespace-delimited word; the rest keeps its inner spacing. */
function splitWord(input: string): [string, string] {
  const match = /^(\S*)\s*([\s\S]*)$/.exec(input.trim());
  return [match?.[1] ?? '', match?.[2]?.trim() ?? ''];
}

/** Parse a JSON object payload, or describe why it is not one. */
function parseObject(text: string): Record<string, unknown> | string {
  if (text.length === 0) {
    return 'The request needs a JSON object.';
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    return `The request is not valid JSON: ${error instanceof Error ? error.message : String(error)}`;
  }
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : 'The request must be a JSON object.';
}
