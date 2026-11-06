// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NeMo Relay conversation view for DeepSeek Harness (Client half).
 *
 * The Host half runs DeepSeek Harness model streams and tool calls through NeMo
 * Relay's managed execution wrappers, and a Relay subscriber appends each
 * session's ATOF 0.1 events to one JSONL file. This view reads that file
 * incrementally through the workspaceFiles Remote, follows it live through the
 * `file` resource, and renders it as turns of LLM calls, tool calls, guardrail
 * decisions, and session marks, plus the raw ATOF event list.
 *
 * Above the trace, the Host process's status file (contract v4, named by the
 * session projection) drives the Relay state, version, exporters, redaction, and
 * configuration layers, the guardrail rules editor, and the Arize Phoenix menu.
 * Every change made in this view goes through the user-invoked `/relay` command.
 *
 * Plain JavaScript without a build step; DeepSeek Harness loads it through
 * `window.__ModuleLoader__`.
 */
window.__ModuleLoader__.load({
  id: 'nemo-relay-deepseek-harness',
  factory(require) {
    const React = require('react');
    const h = React.createElement;
    const {
      useState,
      useEffect,
      useMemo,
      useRef,
      useCallback,
      useContext,
      useId,
      useSyncExternalStore,
      memo,
      Fragment,
    } = React;

    const NS = 'nemo-relay-deepseek-harness';
    const VIEW_ID = 'nemo-relay';
    const PROJECTION_KEY = 'nemoRelayTrace';
    /** One read window; the Host refuses byte windows above its `maxBytes` (2 MiB by default). */
    const CHUNK_BYTES = 1024 * 1024;
    /** A Host with a smaller `maxBytes` refuses the window: it is halved, down to this, until one is accepted. */
    const MIN_CHUNK_BYTES = 64 * 1024;
    /** Longest single event the view assembles from several windows before it reports the trace unreadable. */
    const MAX_EVENT_BYTES = 64 * 1024 * 1024;
    const RAW_CAP = 4000;
    /** Metadata key the Host writes on a compacted LLM start event: the full request's message and tool counts. */
    const REQUEST_SUMMARY_KEY = 'deepseek_harness.view';
    /** Longest tool-result text kept from an LLM request; details show a one-line preview of each. */
    const TOOL_RESULT_CHARS = 4000;
    const RAW_PAGE = 300;
    /** Name prefix of the session marks the Host half emits. */
    const MARK_PREFIX = 'deepseek-harness.';
    /** Session marks (prefix stripped) that have a localized label. */
    const KNOWN_MARKS = new Set([
      'user_message',
      'approval_asked',
      'approval_decided',
      'subagent',
      'request_edit_dropped',
      'compaction',
    ]);
    const TOOL_RESULT_SCHEMA = 'nemo.relay.tool_result@1';
    /** Exporter kinds of the status file: [kind, display name, key in the older `outputs` map]. */
    const EXPORTERS = [
      ['atof', 'ATOF', 'atof'],
      ['atif', 'ATIF', 'atif'],
      ['otel', 'OTel', 'otel'],
      ['openinference', 'OpenInference', 'openInference'],
    ];
    const RELAY_STATES = new Set(['ready', 'degraded', 'not_initialized', 'disabled', 'stopping', 'stopped']);
    /** Runtime states in which the Host half routes nothing through Relay. */
    const INACTIVE_STATES = new Set(['not_initialized', 'disabled', 'stopping', 'stopped']);
    const PHOENIX_STATES = new Set(['disabled', 'stopped', 'starting', 'waiting', 'running', 'error']);
    /** Finish reasons that need no chip on a model-call row. */
    const NORMAL_FINISH = new Set(['stop', 'tool_calls', 'end_turn', 'tool_use']);
    /** Rows shown per turn before "show earlier rows". */
    const ROW_CAP = 300;
    /** Suffix of a rule's rejection reason: `<reason> (NeMo Relay rule "<id>")`. */
    const RULE_SUFFIX = /\s*\(NeMo Relay rule "([^"]*)"\)/g;
    /** The shipped safety rule; the rules panel says so when it is off. */
    const SAFETY_RULE_ID = 'no-recursive-delete-of-root-or-home';
    /** The Host's message for a rule without its own reason (sent to the agent verbatim). */
    const DEFAULT_REASON = 'This call is blocked by NeMo Relay policy.';
    const RULE_LIMITS = { rules: 100, id: 64, tool: 300, pattern: 1000, reason: 500, preview: 2000, pills: 8 };
    const TOOL_NAME_RE = /^[A-Za-z0-9_.:-]+$/;
    const RULE_FLAGS = new Set(['i', 'm', 's', 'u']);
    /** Repetition counts above this on a group that itself repeats count as unbounded. */
    const MAX_GROUP_REPEAT = 5;
    // The browser preview runs a draft's patterns during render, where nothing can interrupt one,
    // so it only runs what is cheap. The Host's `/relay rules test` checks saved rules under a time limit.
    /**
     * The preview skips a pattern with more unbounded quantifiers (`*`, `+`, `{n,}`) than this. Two
     * already cost the cube of the text length, because an unanchored test retries every start position.
     */
    const PREVIEW_MAX_UNBOUNDED = 2;
    /**
     * The preview skips a pattern whose bounded choices (`?`, ranges like `{2,5}`) can double the work
     * more often than this, by its number of unbounded quantifiers (0, 1, 2).
     */
    const PREVIEW_MAX_OPTIONAL = [12, 8, 2];
    /** The preview does not run on a text longer than this. */
    const PREVIEW_MAX_TEXT = 600;
    /** The preview starts no further rule once it has run this long. */
    const PREVIEW_BUDGET_MS = 50;
    /** How long the view waits for a `/relay` action before it reports that NeMo Relay did not answer. */
    const TIMING = { relayDeadlineMs: 30000 };
    /** Unsaved rule drafts kept for this many status files (Host processes) while their views are unmounted. */
    const DRAFT_STORE_CAP = 8;

    /* ------------------------------------------------------------------ */
    /* Formatting                                                          */
    /* ------------------------------------------------------------------ */

    function toMicros(ts) {
      if (typeof ts === 'number' && Number.isFinite(ts)) return ts;
      if (typeof ts !== 'string') return 0;
      const m = /^(.*?T\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:?\d\d)?$/.exec(ts);
      if (m === null) {
        const value = Date.parse(ts);
        return Number.isFinite(value) ? value * 1000 : 0;
      }
      const base = Date.parse(m[1] + (m[3] || 'Z'));
      const frac = `${m[2] || ''}000000`.slice(0, 6);
      return Number.isFinite(base) ? base * 1000 + Number(frac) : 0;
    }

    function fmtDuration(us) {
      if (typeof us !== 'number' || !Number.isFinite(us) || us < 0) return '—';
      const ms = us / 1000;
      if (ms < 1) return '<1 ms';
      // Each unit ends where its rounded value would reach the next unit: never "1000 ms", "60.0 s", or "1m 60s".
      if (ms < 999.5) return `${Math.round(ms)} ms`;
      const s = ms / 1000;
      if (s < 9.995) return `${s.toFixed(2)} s`;
      if (s < 59.95) return `${s.toFixed(1)} s`;
      const seconds = Math.round(s);
      const m = Math.floor(seconds / 60);
      if (m < 60) return `${m}m ${seconds % 60}s`;
      return `${Math.floor(m / 60)}h ${m % 60}m`;
    }

    /**
     * A count phrase: `<key>.one` for exactly one, `<key>.other` otherwise. Every locale defines
     * both forms (Chinese uses the same words for both), so the message never says "1 tools".
     */
    function plural(t, key, n, params = {}) {
      return t(`${key}.${Number(n) === 1 ? 'one' : 'other'}`, { n, ...params });
    }

    function fmtCount(n) {
      if (typeof n !== 'number' || !Number.isFinite(n)) return '0';
      if (n < 1000) return String(n);
      // Each form ends where its rounded value would need another digit: never "10.0k" or "1000k".
      if (n < 9950) return `${(n / 1e3).toFixed(1)}k`;
      if (n < 999500) return `${(n / 1e3).toFixed(0)}k`;
      if (n < 9995000) return `${(n / 1e6).toFixed(2)}M`;
      return `${(n / 1e6).toFixed(1)}M`;
    }

    function fmtClock(us) {
      if (!us) return '';
      const d = new Date(us / 1000);
      const pad = (v, n = 2) => String(v).padStart(n, '0');
      return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
    }

    function oneLine(text, max = 160) {
      if (typeof text !== 'string') return '';
      const flat = text.replace(/\s+/g, ' ').trim();
      return flat.length > max ? `${flat.slice(0, max)}…` : flat;
    }

    function pretty(value) {
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return String(value);
      }
    }

    function shortId(uuid) {
      return typeof uuid === 'string' ? uuid.slice(-8) : '';
    }

    function fileAddress(sessionId, path) {
      const normalized = String(path).replace(/\\/g, '/');
      return `dsh-resource://file/session/${encodeURIComponent(sessionId)}/${normalized
        .split('/')
        .map((segment) => encodeURIComponent(segment))
        .join('/')}`;
    }

    function asRecord(value) {
      return value !== null && typeof value === 'object' && !Array.isArray(value) ? value : undefined;
    }

    function asText(value) {
      return typeof value === 'string' && value.length > 0 ? value : undefined;
    }

    function toCount(value) {
      const n = Number(value ?? 0);
      return Number.isFinite(n) ? n : 0;
    }

    function joinParts(parts, separator = ' · ') {
      return parts.filter((part) => typeof part === 'string' && part.length > 0).join(separator);
    }

    /** Text of an OpenAI message `content`: a string, or an array of content parts. */
    function contentText(content) {
      if (typeof content === 'string') return content;
      if (!Array.isArray(content)) return '';
      return content
        .map((part) => {
          if (typeof part === 'string') return part;
          const record = asRecord(part);
          if (record && typeof record.text === 'string') return record.text;
          return record && typeof record.type === 'string' ? `[${record.type}]` : '';
        })
        .filter((text) => text.length > 0)
        .join('\n');
    }

    /* ------------------------------------------------------------------ */
    /* Trace model (incremental, mutated in place, revisioned)             */
    /* ------------------------------------------------------------------ */
    //
    // Turns are scopes with `metadata.nemo_relay_scope_role === 'turn'`. Every other scope or
    // mark belongs to the turn its parent chain reaches; anything else is "loose" (for example a
    // session-title LLM call, whose parent is the stack root). Guardrail evaluation scopes are not
    // rows: a passing evaluation is folded into the call it guards, and a rejection becomes one
    // "block" item that merges the rejecting evaluation with the rejection mark Relay emits
    // instead of starting the guarded call.

    function emptyStats() {
      return { llm: 0, tools: 0, errors: 0, blocked: 0, prompt: 0, completion: 0, cached: 0, reasoning: 0 };
    }

    function createModel() {
      return {
        spans: new Map(),
        turns: [],
        loose: [],
        pending: new Map(),
        raw: [],
        count: 0,
        version: null,
        stats: emptyStats(),
        turnRev: new Map(),
        rev: 0,
        firstUs: 0,
        lastUs: 0,
      };
    }

    /** Build a complete model from parsed ATOF events. */
    function buildModel(events) {
      const model = createModel();
      for (const ev of events) ingest(model, ev);
      return model;
    }

    function bump(model, turn) {
      model.rev += 1;
      if (turn) model.turnRev.set(turn.uuid, model.rev);
    }

    function metaOf(ev) {
      return asRecord(ev && ev.metadata) ?? {};
    }

    function statusCode(ev) {
      return metaOf(ev)['otel.status_code'];
    }

    function timeOf(item) {
      return item.kind === 'span' ? item.startUs : item.ts;
    }

    function byTime(a, b) {
      return timeOf(a) - timeOf(b) || a.index - b.index;
    }

    function createSpan(model, ev, ts, index) {
      const meta = metaOf(ev);
      const profile = asRecord(ev.category_profile) ?? {};
      const span = {
        kind: 'span',
        uuid: ev.uuid,
        category: typeof ev.category === 'string' ? ev.category : 'unknown',
        name: typeof ev.name === 'string' ? ev.name : '',
        isTurn: meta.nemo_relay_scope_role === 'turn',
        callId: asText(profile.tool_call_id) ?? asText(meta.tool_call_id),
        parentCallId: asText(meta.parent_tool_call_id),
        startUs: ts,
        endUs: null,
        start: null,
        end: null,
        children: [],
        marks: [],
        blocks: [],
        guards: [],
        index,
        error: false,
        parentSpan: null,
        turn: null,
      };
      model.spans.set(span.uuid, span);
      if (span.isTurn) {
        span.turn = span;
        model.turns.push(span);
        return span;
      }
      const parent = model.spans.get(ev.parent_uuid);
      if (parent) {
        span.parentSpan = parent;
        span.turn = parent.turn;
      }
      if (span.category === 'guardrail') return span;
      if (parent) parent.children.push(span);
      else model.loose.push(span);
      if (span.category === 'llm' || span.category === 'tool') {
        if (span.category === 'llm') model.stats.llm += 1;
        else model.stats.tools += 1;
        span.guards.push(
          ...takeGuards(model, span.turn, { target: span.category, callId: span.callId, name: span.name }),
        );
      }
      return span;
    }

    /** What one guardrail evaluation guards: a tool call (by call id or tool name) or a model call. */
    function guardInfo(span) {
      const data = asRecord(span.start && span.start.data) ?? {};
      const meta = { ...metaOf(span.start), ...metaOf(span.end) };
      const kind = typeof data.kind === 'string' ? data.kind : '';
      const callId = asText(meta.tool_call_id);
      const target = kind.startsWith('tool') ? 'tool' : kind.startsWith('llm') ? 'llm' : callId ? 'tool' : 'llm';
      return { kind, target, callId, name: asText(data.target_name) };
    }

    function sameCall(info, call) {
      if (info.target !== call.target) return false;
      if (call.target === 'llm') return true;
      if (info.callId && call.callId) return info.callId === call.callId;
      return !info.name || !call.name || info.name === call.name;
    }

    function pendingKey(turn) {
      return turn ? turn.uuid : '';
    }

    /** Remove and return the passing evaluations waiting for this call in its turn. */
    function takeGuards(model, turn, call) {
      const key = pendingKey(turn);
      const list = model.pending.get(key);
      if (!list || list.length === 0) return [];
      const taken = [];
      const rest = [];
      for (const entry of list) {
        if (sameCall(entry.info, call)) taken.push(entry.span);
        else rest.push(entry);
      }
      model.pending.set(key, rest);
      return taken;
    }

    function createBlock(model, turn, uuid, ts, index) {
      const block = {
        kind: 'block',
        uuid,
        target: 'tool',
        callId: undefined,
        name: undefined,
        guardName: undefined,
        reason: undefined,
        ts,
        guard: null,
        mark: null,
        guards: [],
        index,
        error: true,
        turn,
      };
      if (turn) turn.blocks.push(block);
      else model.loose.push(block);
      model.stats.blocked += 1;
      return block;
    }

    function findBlock(model, turn, predicate) {
      const list = turn ? turn.blocks : model.loose;
      for (let i = list.length - 1; i >= 0; i -= 1) {
        const item = list[i];
        if (item.kind === 'block' && predicate(item)) return item;
      }
      return undefined;
    }

    /** A guardrail evaluation ended: queue a pass for its call, or record a rejection. */
    function settleGuard(model, span) {
      const data = asRecord(span.end.data) ?? {};
      const info = guardInfo(span);
      if (data.rejected !== true && data.allowed !== false) {
        const key = pendingKey(span.turn);
        const list = model.pending.get(key) ?? [];
        list.push({ span, info });
        model.pending.set(key, list);
        return;
      }
      let block = findBlock(model, span.turn, (candidate) => candidate.guard === null && sameCall(info, candidate));
      if (block === undefined) {
        block = createBlock(model, span.turn, span.uuid, span.endUs, span.index);
        block.target = info.target;
        block.callId = info.callId;
        block.name = info.name;
      }
      block.guard = span;
      block.guardName = span.name;
      // A guardrail that failed (fail-closed) reports `error` instead of a rejection reason.
      block.reason = block.reason ?? asText(data.rejection_reason) ?? asText(data.error);
      block.guards.push(...takeGuards(model, span.turn, block));
    }

    /** Relay's rejection mark (named after the tool or provider) for a call that never started. */
    function settleRejection(model, ev, ts, index, parent) {
      const data = asRecord(ev.data) ?? {};
      const callId = asText(metaOf(ev).tool_call_id);
      const name = asText(ev.name);
      const turn = parent ? parent.turn : null;
      let block = findBlock(model, turn, (candidate) => {
        if (candidate.mark !== null) return false;
        if (callId && candidate.callId) return candidate.callId === callId;
        if (candidate.target === 'llm') return !callId;
        return !candidate.name || !name || candidate.name === name;
      });
      if (block === undefined) {
        block = createBlock(model, turn, ev.uuid, ts, index);
        block.target = callId ? 'tool' : 'llm';
        block.callId = callId;
        block.name = name;
        block.guards.push(...takeGuards(model, turn, block));
      }
      block.mark = { kind: 'mark', uuid: ev.uuid, name: ev.name, ts, ev, index };
      block.name = name ?? block.name;
      block.callId = block.callId ?? callId;
      block.reason = block.reason ?? asText(data.rejection_reason);
      return block;
    }

    function normalizeUsage(value) {
      const usage = asRecord(value);
      if (!usage || (usage.prompt_tokens === undefined && usage.completion_tokens === undefined)) return undefined;
      return {
        prompt: toCount(usage.prompt_tokens),
        completion: toCount(usage.completion_tokens),
        cached: toCount(asRecord(usage.prompt_tokens_details)?.cached_tokens ?? usage.cache_read_tokens),
        reasoning: toCount(asRecord(usage.completion_tokens_details)?.reasoning_tokens ?? usage.reasoning_tokens),
      };
    }

    /** Token usage of an ended LLM scope; falls back to the codec's annotated response. */
    function usageOfSpan(span) {
      const end = span.end;
      if (!end) return undefined;
      const annotated = asRecord(asRecord(end.category_profile)?.annotated_response);
      return normalizeUsage(asRecord(end.data)?.usage) ?? normalizeUsage(annotated && annotated.usage);
    }

    function addUsage(target, usage) {
      if (!usage) return;
      target.prompt += usage.prompt;
      target.completion += usage.completion;
      target.cached += usage.cached;
      target.reasoning += usage.reasoning;
    }

    function capToolResult(message) {
      const record = asRecord(message);
      if (!record || record.role !== 'tool') return message;
      const cut = (text) => `${text.slice(0, TOOL_RESULT_CHARS)}…`;
      const content = record.content;
      if (typeof content === 'string')
        return content.length > TOOL_RESULT_CHARS ? { ...record, content: cut(content) } : message;
      if (!Array.isArray(content)) return message;
      return {
        ...record,
        content: content.map((part) => {
          const text = asRecord(part)?.text;
          return typeof text === 'string' && text.length > TOOL_RESULT_CHARS ? { ...part, text: cut(text) } : part;
        }),
      };
    }

    /**
     * Keep of an LLM start event what the view shows: the latest user message and the messages
     * after the last assistant message, plus the counts of what is dropped. The Host half writes
     * events in this form; this handles a trace written before it did, whose every model call
     * repeats the turn's history and the full tool list.
     */
    function compactLlmStart(ev) {
      if (!ev || ev.category !== 'llm' || ev.scope_category !== 'start') return ev;
      if (asRecord(asRecord(ev.metadata)?.[REQUEST_SUMMARY_KEY])) return ev;
      const data = asRecord(ev.data);
      const content = asRecord(data?.content);
      const messages = content?.messages;
      if (!data || !content || !Array.isArray(messages)) return ev;
      let lastAssistant = -1;
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        if (asRecord(messages[i])?.role === 'assistant') {
          lastAssistant = i;
          break;
        }
      }
      let kept = messages.slice(lastAssistant + 1).filter((message) => asRecord(message)?.role !== 'system');
      if (!kept.some((message) => asRecord(message)?.role === 'user')) {
        for (let i = lastAssistant; i >= 0; i -= 1) {
          if (asRecord(messages[i])?.role === 'user') {
            kept = [messages[i], ...kept];
            break;
          }
        }
      }
      const { messages: _messages, tools, ...request } = content;
      const profile = asRecord(ev.category_profile);
      const { annotated_request: _annotated, ...restProfile } = profile ?? {};
      return {
        ...ev,
        data: { ...data, content: { ...request, messages: kept.map(capToolResult) } },
        ...(profile ? { category_profile: restProfile } : {}),
        metadata: {
          ...asRecord(ev.metadata),
          [REQUEST_SUMMARY_KEY]: {
            request_messages: messages.length,
            request_tools: Array.isArray(tools) ? tools.length : 0,
          },
        },
      };
    }

    function ingest(model, raw) {
      if (!raw || typeof raw !== 'object' || typeof raw.kind !== 'string') return;
      const ev = compactLlmStart(raw);
      // Per-chunk receipt marks carry no trace structure; the Host half does not write them.
      if (ev.kind === 'mark' && ev.name === 'llm.chunk') return;
      const index = model.count;
      model.count += 1;
      model.raw.push(ev);
      if (model.raw.length > RAW_CAP) model.raw.splice(0, model.raw.length - RAW_CAP);
      if (model.version === null && typeof ev.atof_version === 'string') model.version = ev.atof_version;
      const ts = toMicros(ev.timestamp);
      if (ts && (!model.firstUs || ts < model.firstUs)) model.firstUs = ts;
      if (ts > model.lastUs) model.lastUs = ts;

      if (ev.kind === 'mark') {
        const parent = model.spans.get(ev.parent_uuid);
        if (asRecord(ev.data)?.rejected === true) {
          const block = settleRejection(model, ev, ts, index, parent);
          bump(model, block.turn);
          return;
        }
        const mark = {
          kind: 'mark',
          uuid: ev.uuid,
          name: typeof ev.name === 'string' ? ev.name : '',
          ts,
          ev,
          index,
          error: statusCode(ev) === 'ERROR',
        };
        if (parent) parent.marks.push(mark);
        else model.loose.push(mark);
        bump(model, parent ? parent.turn : null);
        return;
      }

      if (ev.kind !== 'scope') return;
      let span = model.spans.get(ev.uuid);
      if (ev.scope_category === 'start') {
        if (span === undefined) span = createSpan(model, ev, ts, index);
        span.start = ev;
        span.startUs = ts;
        bump(model, span.turn);
        return;
      }
      if (ev.scope_category !== 'end') return;
      if (span === undefined) span = createSpan(model, ev, ts, index);
      const first = span.end === null;
      span.end = ev;
      span.endUs = ts;
      span.error = statusCode(ev) === 'ERROR' || (span.category === 'tool' && asRecord(ev.data)?.is_error === true);
      if (first) {
        if (span.category === 'guardrail') settleGuard(model, span);
        else if (span.category === 'llm') addUsage(model.stats, usageOfSpan(span));
        if (span.error && !span.isTurn && span.category !== 'guardrail') model.stats.errors += 1;
      }
      bump(model, span.turn);
    }

    function turnNumber(turn) {
      const value = metaOf(turn.start).turn_index ?? metaOf(turn.end).turn_index;
      return typeof value === 'number' || typeof value === 'string' ? value : undefined;
    }

    function turnSummary(turn) {
      const summary = {
        llm: 0,
        tools: 0,
        errors: 0,
        blocked: turn.blocks.length,
        open: false,
        prompt: 0,
        completion: 0,
        cached: 0,
        reasoning: 0,
      };
      const visit = (spans) => {
        for (const span of spans) {
          if (span.category === 'llm') {
            summary.llm += 1;
            addUsage(summary, usageOfSpan(span));
          } else if (span.category === 'tool') summary.tools += 1;
          if (span.error) summary.errors += 1;
          if (span.endUs === null) summary.open = true;
          visit(span.children);
        }
      };
      visit(turn.children);
      return summary;
    }

    /**
     * Display rows of one container (a turn, or a run of events outside turns): calls, marks, and
     * blocked calls in time order. Nested calls follow their parent call one level deeper: run_code
     * sub-calls through `metadata.parent_tool_call_id`, and any scope opened inside another scope.
     */
    function flattenItems(spans, marks, blocks) {
      const callIds = new Set();
      for (const span of spans) if (span.callId) callIds.add(span.callId);
      const nestedOf = new Map();
      const top = [];
      for (const span of spans) {
        const parentCall = span.parentCallId;
        if (parentCall && parentCall !== span.callId && callIds.has(parentCall)) {
          const list = nestedOf.get(parentCall) ?? [];
          list.push(span);
          nestedOf.set(parentCall, list);
        } else top.push(span);
      }
      const rows = [];
      const seen = new Set();
      const visit = (item, depth) => {
        if (seen.has(item)) return;
        seen.add(item);
        rows.push({ item, depth });
        if (item.kind !== 'span') return;
        const nested = [
          ...((item.callId && nestedOf.get(item.callId)) || []),
          ...item.children,
          ...item.marks,
          ...item.blocks,
        ];
        for (const child of nested.sort(byTime)) visit(child, depth + 1);
      };
      for (const item of [...top, ...marks, ...blocks].sort(byTime)) visit(item, 0);
      return rows;
    }

    /** Turns and runs of loose items in time order. */
    function timelineOf(model) {
      const entries = [
        ...model.turns.map((turn) => ({ kind: 'turn', at: turn.startUs, index: turn.index, turn })),
        ...model.loose.map((item) => ({ kind: 'loose', at: timeOf(item), index: item.index, item })),
      ].sort((a, b) => a.at - b.at || a.index - b.index);
      const out = [];
      let group = null;
      for (const entry of entries) {
        if (entry.kind === 'turn') {
          group = null;
          out.push(entry);
        } else {
          if (group === null) {
            group = { kind: 'loose', key: `loose-${entry.item.uuid}`, items: [] };
            out.push(group);
          }
          group.items.push(entry.item);
        }
      }
      return out;
    }

    /* ------------------------------------------------------------------ */
    /* Call details                                                        */
    /* ------------------------------------------------------------------ */

    function toolCallsOf(value) {
      if (!Array.isArray(value)) return [];
      return value.map((call) => {
        const record = asRecord(call) ?? {};
        const fn = asRecord(record.function);
        const args = fn ? fn.arguments : record.arguments;
        return {
          id: asText(record.id),
          name: asText(fn ? fn.name : record.name) ?? '?',
          args: typeof args === 'string' ? args : args === undefined ? '' : JSON.stringify(args),
        };
      });
    }

    /** One LLM scope's request and response (OpenAI Chat Completions), with codec fallbacks. */
    function llmInfo(span) {
      const start = span.start;
      const end = span.end;
      const startProfile = asRecord(start && start.category_profile) ?? {};
      const endProfile = asRecord(end && end.category_profile) ?? {};
      const annotated = asRecord(endProfile.annotated_response) ?? {};
      const request =
        asRecord(asRecord(start && start.data)?.content) ?? asRecord(startProfile.annotated_request) ?? {};
      const response = asRecord(end && end.data) ?? {};
      const choice = asRecord(Array.isArray(response.choices) ? response.choices[0] : undefined) ?? {};
      const message = asRecord(choice.message) ?? {};
      const meta = { ...metaOf(start), ...metaOf(end) };
      const summary = asRecord(metaOf(start)[REQUEST_SUMMARY_KEY]) ?? {};
      const ttft = endProfile.time_to_first_chunk;
      return {
        model:
          asText(startProfile.model_name) ??
          asText(endProfile.model_name) ??
          asText(response.model) ??
          asText(annotated.model) ??
          asText(request.model),
        purpose: asText(meta.request_purpose),
        step: meta.step_index,
        messages: Array.isArray(request.messages) ? request.messages : [],
        // A compacted request keeps only the messages the view shows; the summary has the full counts.
        messageCount: toCount(
          summary.request_messages ?? (Array.isArray(request.messages) ? request.messages.length : 0),
        ),
        tools: toCount(summary.request_tools ?? (Array.isArray(request.tools) ? request.tools.length : 0)),
        text: asText(message.content) ?? asText(annotated.message),
        reasoning: asText(message.reasoning_content),
        toolCalls: toolCallsOf(Array.isArray(message.tool_calls) ? message.tool_calls : annotated.tool_calls),
        finish: asText(choice.finish_reason) ?? asText(annotated.finish_reason),
        usage: usageOfSpan(span),
        ttftUs: typeof ttft === 'number' && Number.isFinite(ttft) ? ttft * 1e6 : undefined,
        description: asText(meta['otel.status_description']),
      };
    }

    function lastUserText(messages) {
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        const message = asRecord(messages[i]);
        if (message && message.role === 'user') return contentText(message.content);
      }
      return undefined;
    }

    /** Tool results the request sends after the last assistant message (what this call reacts to). */
    function newToolResults(messages) {
      const out = [];
      for (let i = messages.length - 1; i >= 0; i -= 1) {
        const message = asRecord(messages[i]);
        if (!message || message.role === 'assistant') break;
        if (message.role === 'tool') out.unshift(message);
      }
      return out;
    }

    function markKey(name) {
      const text = typeof name === 'string' ? name : '';
      return text.startsWith(MARK_PREFIX) ? text.slice(MARK_PREFIX.length) : text;
    }

    function markLabel(mark, t) {
      const key = markKey(mark.name);
      return KNOWN_MARKS.has(key) ? t(`mark.${key}`) : key;
    }

    function markSubtitle(mark, t) {
      const key = markKey(mark.name);
      const data = asRecord(mark.ev.data);
      if (key === 'user_message')
        return data && typeof data.text === 'string' ? oneLine(data.text, 140) : t('mark.notCaptured');
      if (!data) return '';
      if (key === 'approval_asked') return oneLine(joinParts([data.toolName, data.reason]), 140);
      if (key === 'approval_decided') return joinParts([data.outcome]);
      // The child session id renders as its own chip.
      if (key === 'subagent') return joinParts([data.label, data.mode]);
      if (key === 'request_edit_dropped') return Array.isArray(data.fields) ? data.fields.join(', ') : '';
      if (typeof data.text === 'string') return oneLine(data.text, 140);
      if (typeof data.preview === 'string') return oneLine(data.preview, 140);
      if (typeof data.title === 'string') return oneLine(data.title, 140);
      if (typeof data.model === 'string') return `${data.provider ? `${data.provider}/` : ''}${data.model}`;
      if (typeof data.outcome === 'string') return data.outcome;
      if (typeof data.error === 'string') return oneLine(data.error, 140);
      return oneLine(JSON.stringify(data), 140);
    }

    function spanSubtitle(span) {
      if (span.category === 'llm') {
        // Token counts and unusual finish reasons render as chips beside this text.
        const info = llmInfo(span);
        const parts = [];
        if (info.purpose) parts.push(info.purpose);
        if (info.toolCalls.length > 0) parts.push(info.toolCalls.map((call) => call.name).join(', '));
        if (span.error && info.description) parts.push(info.description);
        return oneLine(parts.join(' · '), 160);
      }
      if (span.category === 'tool') {
        const data = asRecord(span.start && span.start.data);
        if (data && data.stripped === true) {
          const keys = Array.isArray(data.argKeys) ? data.argKeys.filter((key) => typeof key === 'string') : [];
          return keys.map((key) => `${key}: …`).join(', ');
        }
        if (data) {
          const first =
            data.command ??
            data.file_path ??
            data.path ??
            data.pattern ??
            data.query ??
            data.url ??
            data.description ??
            data.prompt ??
            data.name;
          if (typeof first === 'string') return oneLine(first, 120);
          return oneLine(JSON.stringify(data), 120);
        }
      }
      return '';
    }

    function itemName(item, t) {
      if (item.kind === 'block')
        return item.name ?? (item.target === 'llm' ? t('block.targetLlm') : t('block.targetTool'));
      if (item.kind === 'mark') return markLabel(item, t);
      if (item.category === 'llm') return llmInfo(item).model ?? item.name;
      return item.name;
    }

    function itemSubtitle(item, t) {
      if (item.kind === 'block') return item.target === 'llm' ? t('block.llm') : t('block.tool');
      if (item.kind === 'mark') return markSubtitle(item, t);
      return spanSubtitle(item);
    }

    /* ------------------------------------------------------------------ */
    /* Trace source: incremental reader over the workspaceFiles Remote      */
    /* ------------------------------------------------------------------ */

    function errorCode(result) {
      const error = result && result.error;
      return (error && (error.code || error.name)) || 'error';
    }

    function errorMessage(result) {
      const error = result && result.error;
      return (error && (error.message || error.code)) || 'unknown error';
    }

    class TraceSource {
      constructor(workspaceFiles, sessionId, path) {
        this.files = workspaceFiles;
        this.sessionId = sessionId;
        this.path = path;
        this.listeners = new Set();
        this.controller = new AbortController();
        this.decoder = new TextDecoder();
        this.reading = false;
        this.dirty = false;
        // The read window; halved when the Host refuses it as too large.
        this.chunk = CHUNK_BYTES;
        this.reset();
        this.snapshot = { status: 'loading', model: this.model, rev: 0, error: null, retry: true };
      }

      reset() {
        this.model = createModel();
        this.consumed = 0;
        // The first bytes of the file as last read, to notice a replaced file.
        this.head = null;
      }

      static startsWith(bytes, prefix) {
        if (bytes.length < prefix.length) return false;
        for (let i = 0; i < prefix.length; i += 1) if (bytes[i] !== prefix[i]) return false;
        return true;
      }

      /** Join the windows of one event that is longer than a read window. */
      static join(parts, last) {
        if (parts.length === 0) return last;
        const out = new Uint8Array(parts.reduce((total, part) => total + part.length, last.length));
        let at = 0;
        for (const part of [...parts, last]) {
          out.set(part, at);
          at += part.length;
        }
        return out;
      }

      subscribe = (listener) => {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
      };

      getSnapshot = () => this.snapshot;

      /**
       * Publish a state. An `error` keeps the model: events read before the failure stay on screen.
       * `retry` is false for a failure that reading again cannot fix, so the view does not repeat it.
       */
      publish(status, error = null, retry = true) {
        const next = { status, model: this.model, rev: this.model.rev, error, retry };
        const prev = this.snapshot;
        if (
          prev.status === next.status &&
          prev.model === next.model &&
          prev.rev === next.rev &&
          prev.error === next.error &&
          prev.retry === next.retry
        )
          return;
        this.snapshot = next;
        for (const listener of this.listeners) listener();
      }

      dispose() {
        this.controller.abort();
        this.listeners.clear();
      }

      async read(offset, length) {
        return this.files.readBytes(this.sessionId, this.path, { range: { offset, length } }, this.controller.signal);
      }

      /** Bring the model up to date with the file; coalesces concurrent requests. */
      async sync() {
        if (this.controller.signal.aborted) return;
        if (this.reading) {
          this.dirty = true;
          return;
        }
        this.reading = true;
        try {
          do {
            this.dirty = false;
            await this.syncOnce();
          } while (this.dirty && !this.controller.signal.aborted);
        } catch (error) {
          if (!this.controller.signal.aborted) this.publish('error', String((error && error.message) || error));
        } finally {
          this.reading = false;
        }
      }

      async syncOnce() {
        if (!this.files || typeof this.files.readBytes !== 'function') {
          this.publish('error', 'workspaceFiles.readBytes is unavailable in this client', false);
          return;
        }
        const head = await this.read(0, 4096);
        if (this.controller.signal.aborted) return;
        if (!head || !head.ok) {
          const code = errorCode(head);
          if (code === 'workspace-file/not-found') {
            if (this.model.count > 0) this.reset();
            this.publish('missing');
          } else this.publish('error', errorMessage(head));
          return;
        }
        const headBytes = head.value.data;
        // The file is append-only, so its first bytes never change: different ones mean it was
        // replaced, and the view starts over. (A first event can be longer than this window.)
        if (this.head !== null && !TraceSource.startsWith(headBytes, this.head)) this.reset();
        if (this.head === null || headBytes.length > this.head.length) this.head = headBytes.slice();
        const total = typeof head.value.bytes === 'number' ? head.value.bytes : undefined;
        if (total !== undefined && total < this.consumed) this.reset();

        let changed = false;
        // Windows of an event longer than one read window, kept until the window with its line break.
        let carry = [];
        let carried = 0;
        for (let guard = 0; guard < 65536; guard += 1) {
          const size = this.chunk;
          const chunk = await this.read(this.consumed + carried, size);
          if (this.controller.signal.aborted) return;
          if (!chunk || !chunk.ok) {
            const code = errorCode(chunk);
            if (code === 'workspace-file/not-found') {
              this.reset();
              this.publish('missing');
              return;
            }
            // The Host's byte-window cap is configurable: ask for half as much, from the same offset.
            if (code === 'workspace-file/too-large' && size > MIN_CHUNK_BYTES) {
              this.chunk = Math.max(MIN_CHUNK_BYTES, Math.floor(size / 2));
              continue;
            }
            // A window the Host refuses even at the smallest size stays refused until its configuration changes.
            this.publish('error', errorMessage(chunk), code !== 'workspace-file/too-large');
            return;
          }
          const data = chunk.value.data;
          const lastBreak = data.lastIndexOf(10);
          if (lastBreak < 0) {
            // The file ends inside this event (it is still being written): read it again on the next sync.
            if (data.length < size || chunk.value.eof) break;
            carry.push(data);
            carried += data.length;
            if (carried > MAX_EVENT_BYTES) {
              // Reading again would fetch the same windows and fail the same way.
              this.publish(
                'error',
                `one trace event is larger than ${Math.round(MAX_EVENT_BYTES / 1048576)} MiB`,
                false,
              );
              return;
            }
            continue;
          }
          const text = this.decoder.decode(TraceSource.join(carry, data.subarray(0, lastBreak + 1)));
          this.consumed += carried + lastBreak + 1;
          carry = [];
          carried = 0;
          for (const line of text.split('\n')) {
            if (line.length === 0) continue;
            let event;
            try {
              event = JSON.parse(line);
            } catch {
              continue;
            }
            ingest(this.model, event);
            changed = true;
          }
          if (chunk.value.eof && lastBreak + 1 >= data.length) break;
          if (data.length < size) break;
        }
        if (changed || this.snapshot.status !== 'ready') this.publish(this.model.count > 0 ? 'ready' : 'empty');
      }
    }

    /* ------------------------------------------------------------------ */
    /* Status file (contract v4; older health snapshots degrade gracefully) */
    /* ------------------------------------------------------------------ */

    function thrown(error) {
      return String((error && error.message) || error);
    }

    function parseJsonObject(text) {
      if (typeof text !== 'string') return undefined;
      const trimmed = text.trim();
      if (!trimmed.startsWith('{')) return undefined;
      try {
        return asRecord(JSON.parse(trimmed));
      } catch {
        return undefined;
      }
    }

    function stringList(value) {
      return Array.isArray(value) ? value.filter((entry) => typeof entry === 'string' && entry.length > 0) : [];
    }

    /** Exporters grouped by kind, known kinds first; v3 files only report enabled outputs. */
    function readExporters(value, outputs) {
      const groups = new Map();
      const add = (kind, target) => {
        let group = groups.get(kind);
        if (group === undefined) {
          const row = EXPORTERS.find((entry) => entry[0] === kind);
          group = {
            kind,
            label: row ? row[1] : kind,
            targets: [],
            health: row && outputs[row[2]] === 'degraded' ? 'degraded' : 'enabled',
          };
          groups.set(kind, group);
        }
        if (target && !group.targets.includes(target)) group.targets.push(target);
      };
      if (Array.isArray(value)) {
        for (const raw of value) {
          const record = asRecord(raw);
          const kind = record ? asText(record.kind) : undefined;
          if (kind) add(kind.toLowerCase(), asText(record.target));
        }
      } else {
        for (const [kind, , key] of EXPORTERS) {
          if (outputs[key] === 'enabled' || outputs[key] === 'degraded') add(kind);
        }
      }
      const order = (kind) => {
        const index = EXPORTERS.findIndex((entry) => entry[0] === kind);
        return index < 0 ? EXPORTERS.length : index;
      };
      return [...groups.values()].sort((a, b) => order(a.kind) - order(b.kind));
    }

    /** Rule problems the Host reports: `{ index, id?, field, message }`; others are dropped. */
    function readRuleErrors(value) {
      if (!Array.isArray(value)) return [];
      return value
        .map(asRecord)
        .filter((error) => error && Number.isInteger(error.index) && asText(error.message))
        .map((error) => ({
          index: error.index,
          id: asText(error.id),
          field: asText(error.field) ?? 'rules',
          message: error.message,
        }));
    }

    /** The `guardrails` section: rules normalized for display and editing. */
    function readGuardrails(value) {
      const record = asRecord(value);
      if (!record) return null;
      const revision =
        asText(record.revision) ?? (typeof record.revision === 'number' ? String(record.revision) : undefined);
      // A Map: a rule may be named like an Object.prototype member (`constructor`, `__proto__`).
      const timeouts = new Map();
      for (const [id, count] of Object.entries(asRecord(record.timeouts) ?? {})) {
        if (typeof count === 'number' && count > 0) timeouts.set(id, count);
      }
      return {
        enabled: record.enabled !== false,
        failMode: record.failMode === 'closed' ? 'closed' : 'open',
        source: asText(record.source) ?? 'config',
        editable: record.editable === true && revision !== undefined,
        revision,
        rules: Array.isArray(record.rules) ? record.rules.slice(0, RULE_LIMITS.rules).map(normalizeRule) : [],
        rejections: typeof record.rejections === 'number' ? record.rejections : undefined,
        location: asText(record.location),
        errors: readRuleErrors(record.errors),
        loadError: asText(record.loadError),
        timeouts,
      };
    }

    function readPhoenix(value) {
      const record = asRecord(value);
      if (!record) return null;
      const url = asText(record.url);
      return {
        enabled: record.enabled !== false,
        state: PHOENIX_STATES.has(record.state) ? record.state : 'stopped',
        // Older status files have no `connected`; a running Phoenix then counts as connected.
        connected: typeof record.connected === 'boolean' ? record.connected : undefined,
        mode: record.mode === 'owned' || record.mode === 'external' ? record.mode : undefined,
        url: url && /^https?:\/\//i.test(url) ? url.replace(/\/+$/, '') : undefined,
        version: asText(record.version),
        project: asText(record.project),
        error: asText(record.error),
        sessionUrl: asText(record.sessionUrl),
        phase: asText(record.phase),
        launchHint: asText(record.launchHint),
        log: asText(record.log),
      };
    }

    function readRedaction(value) {
      const record = asRecord(value);
      if (!record) return null;
      return {
        secrets: record.secrets === true,
        detectors: stringList(record.detectors),
        source: asText(record.source),
        note: asText(record.note),
      };
    }

    /** Normalize the Host status file into what the view renders; `null` when it is not an object. */
    function readStatus(value) {
      const record = asRecord(value);
      if (!record) return null;
      const runtime = asRecord(record.status) ?? {};
      const relay = asRecord(record.relay) ?? {};
      const outputs = asRecord(record.outputs) ?? {};
      return {
        v: typeof record.v === 'number' ? record.v : 3,
        state: asText(record.state) ?? asText(runtime.state),
        reason: asText(record.lastError) ?? asText(runtime.reason),
        updatedAt: asText(record.updatedAt),
        relayVersion: asText(relay.version),
        nodeVersion: asText(relay.node),
        pluginHost: typeof record.initializedPluginHost === 'boolean' ? record.initializedPluginHost : undefined,
        configPaths: stringList(record.configPaths),
        exporters: readExporters(record.exporters, outputs),
        counters: asRecord(record.counters),
        activeSessions: typeof record.activeSessions === 'number' ? record.activeSessions : undefined,
        redaction: readRedaction(record.redaction),
        guardrails: readGuardrails(record.guardrails),
        phoenix: readPhoenix(record.phoenix),
      };
    }

    /**
     * Phoenix deep link for one session: `sessionUrl` with `{sessionId}` filled in, else Phoenix's
     * redirect route. None while Relay is not exporting to that Phoenix: it would not have the session.
     */
    function phoenixSessionUrl(phoenix, sessionId) {
      if (!phoenix || typeof sessionId !== 'string' || phoenix.connected === false) return undefined;
      const id = encodeURIComponent(sessionId);
      const url = phoenix.sessionUrl
        ? phoenix.sessionUrl.split('{sessionId}').join(id)
        : phoenix.url
          ? `${phoenix.url}/redirects/sessions/${id}`
          : undefined;
      return url && /^https?:\/\//i.test(url) ? url : undefined;
    }

    function isLoopbackUrl(url) {
      return (
        typeof url === 'string' && /^https?:\/\/(?:127\.\d+\.\d+\.\d+|localhost|\[::1\])(?::\d+)?(?:\/|$)/i.test(url)
      );
    }

    /** Open an http(s) address in a new browser tab or window (the desktop app routes it to the system browser). */
    function openExternal(url) {
      if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) return false;
      try {
        if (typeof window !== 'undefined' && typeof window.open === 'function') {
          window.open(url, '_blank', 'noopener,noreferrer');
          return true;
        }
      } catch {
        /* blocked by the page */
      }
      return false;
    }

    /** A configuration layer installed by an operator rather than the user. */
    function isSystemLayer(path) {
      return /^\/etc\/|^\/Library\/|^[A-Za-z]:\\ProgramData\\/i.test(path);
    }

    /* ------------------------------------------------------------------ */
    /* Guardrail rules: validation and preview mirror the Host's rules     */
    /* ------------------------------------------------------------------ */

    function ruleIdOf(value, index) {
      const id = String(value ?? '')
        .trim()
        .replace(/\s+/g, '-')
        .replace(/[^A-Za-z0-9._-]/g, '-')
        .replace(/-{2,}/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, RULE_LIMITS.id);
      return id.length > 0 ? id : `rule-${index + 1}`;
    }

    function cleanFlags(flags) {
      const raw = typeof flags === 'string' ? flags : 'i';
      return [...new Set([...raw].filter((flag) => RULE_FLAGS.has(flag)))].sort().join('');
    }

    /** One rule in its stored shape `{ id, tool, match, flags, reason, enabled }`; a tool list becomes a comma list. */
    function normalizeRule(raw, index) {
      const rule = asRecord(raw) ?? {};
      const tool = Array.isArray(rule.tool)
        ? rule.tool
            .map((name) => String(name).trim())
            .filter(Boolean)
            .join(', ')
        : typeof rule.tool === 'string'
          ? rule.tool.trim()
          : '';
      return {
        id: ruleIdOf(rule.id, index),
        tool: tool === '*' ? '' : tool,
        match: typeof rule.match === 'string' ? rule.match : '',
        flags: cleanFlags(rule.flags),
        reason: typeof rule.reason === 'string' && rule.reason.trim().length > 0 ? rule.reason.trim() : DEFAULT_REASON,
        enabled: rule.enabled !== false,
      };
    }

    function rulesKey(rules) {
      return JSON.stringify(rules.map(normalizeRule));
    }

    function braceQuantifier(source, at) {
      const match = /^\{(\d+)(,(\d*))?\}/.exec(source.slice(at));
      if (!match) return null;
      const unbounded = match[2] !== undefined && match[3] === '';
      const max = unbounded ? Infinity : Number(match[2] === undefined ? match[1] : match[3]);
      return { unbounded, max, min: Number(match[1]), length: match[0].length };
    }

    /**
     * Scan a pattern's groups. `nested`: a repeating group repeated again without bound or more than
     * a few times, like (a+)+, which the Host refuses. The rest describes patterns the Host runs
     * under its time limit but this browser preview skips:
     * - `alternation`: a group of alternatives repeated without bound or more than a few times;
     * - `repeated`: a group with `*`, `+`, or `{n,}` in it repeated at all, like (.*a){5}, or a group
     *   with a bounded choice in it repeated without bound, like (a?b?)*;
     * - `unbounded`: how many quantifiers are `*`, `+`, or `{n,}`;
     * - `optional`: how often the bounded choices can double the work: 1 for each `?`, 2 for a range
     *   of four counts like `{2,5}`, and so on, times the repetitions of a group they are in.
     *   Alternatives count by their largest, since a match follows one of them.
     */
    function scanPattern(source) {
      // The pattern is the outermost frame. `optional` counts for the alternative being read,
      // `widest` is the largest count among the alternatives before it; `open`: `*`, `+`, or `{n,}` inside.
      const frame = () => ({ repeats: false, alt: false, open: false, optional: 0, widest: 0 });
      const top = frame();
      const groups = [];
      const current = () => groups[groups.length - 1] ?? top;
      let inClass = false;
      let closed = null;
      // What the previous character was, to read a "?": 'open' for "(", 'quantifier', or ''.
      let last = '';
      const found = { nested: false, alternation: false, repeated: false, unbounded: 0, optional: 0 };
      for (let i = 0; i < source.length; i += 1) {
        const c = source[i];
        if (c === '\\') {
          i += 1;
          closed = null;
          last = '';
          continue;
        }
        if (inClass) {
          if (c === ']') inClass = false;
          continue;
        }
        if (c === '[') {
          inClass = true;
          closed = null;
          last = '';
          continue;
        }
        if (c === '(') {
          groups.push(frame());
          closed = null;
          last = 'open';
          continue;
        }
        if (c === ')') {
          closed = groups.pop() ?? frame();
          closed.optional = Math.max(closed.optional, closed.widest);
          const parent = groups[groups.length - 1];
          if (parent) {
            if (closed.repeats) parent.repeats = true;
            if (closed.alt) parent.alt = true;
            if (closed.open) parent.open = true;
          }
          current().optional += closed.optional;
          last = '';
          continue;
        }
        if (c === '|') {
          const group = current();
          if (groups.length > 0) group.alt = true;
          group.widest = Math.max(group.widest, group.optional);
          group.optional = 0;
          closed = null;
          last = '';
          continue;
        }
        if (c === '?') {
          // After "(" it is group syntax and after a quantifier it makes that one lazy; otherwise
          // the item before it is optional.
          if (last === '') current().optional += 1;
          last = last === '' ? 'quantifier' : '';
          closed = null;
          continue;
        }
        const quantifier =
          c === '*' || c === '+'
            ? { unbounded: true, max: Infinity, min: 0, length: 1 }
            : c === '{'
              ? braceQuantifier(source, i)
              : null;
        if (quantifier) {
          if (quantifier.unbounded) {
            found.unbounded += 1;
            current().open = true;
          } else current().optional += Math.log2(Math.max(1, quantifier.max - quantifier.min + 1));
          if (closed) {
            const many = quantifier.unbounded || quantifier.max > MAX_GROUP_REPEAT;
            if (closed.repeats && many) found.nested = true;
            if (closed.alt && many) found.alternation = true;
            if (quantifier.max > 1 && (closed.open || (quantifier.unbounded && closed.optional > 0)))
              found.repeated = true;
            // The group's choices were counted once when it closed; each further repetition makes them again.
            else if (quantifier.max > 1 && !quantifier.unbounded)
              current().optional += closed.optional * (quantifier.max - 1);
          }
          if (groups.length > 0) groups[groups.length - 1].repeats = true;
          i += quantifier.length - 1;
          closed = null;
          last = 'quantifier';
          continue;
        }
        closed = null;
        last = '';
      }
      found.optional = Math.max(top.optional, top.widest);
      return found;
    }

    /** Patterns this browser preview skips because they can run slowly; the Host still enforces them. */
    function previewRisk(source) {
      const scan = scanPattern(source);
      return (
        /\\[1-9]|\\k</.test(source) ||
        scan.alternation ||
        scan.repeated ||
        scan.unbounded > PREVIEW_MAX_UNBOUNDED ||
        scan.optional > PREVIEW_MAX_OPTIONAL[scan.unbounded]
      );
    }

    function compilePattern(source, flags) {
      if (source.length > RULE_LIMITS.pattern) return { error: ['rules.err.long', { n: RULE_LIMITS.pattern }] };
      if (scanPattern(source).nested) return { error: ['rules.err.nested'] };
      try {
        return { regex: new RegExp(source, flags) };
      } catch (error) {
        return { error: ['rules.err.regex', { detail: thrown(error).replace(/^Invalid regular expression: /, '') }] };
      }
    }

    function compileRule(rule) {
      const errors = {};
      let tool = { any: true };
      let risky = false;
      if (rule.tool.length > RULE_LIMITS.tool) errors.tool = ['rules.err.long', { n: RULE_LIMITS.tool }];
      else if (rule.tool.startsWith('/') && rule.tool.lastIndexOf('/') > 0) {
        const slash = rule.tool.lastIndexOf('/');
        const source = rule.tool.slice(1, slash);
        const compiled = compilePattern(source, cleanFlags(rule.tool.slice(slash + 1)));
        if (compiled.error) errors.tool = compiled.error;
        else {
          tool = { regex: compiled.regex };
          risky = previewRisk(source);
        }
      } else if (rule.tool.length > 0) {
        const names = rule.tool
          .split(',')
          .map((name) => name.trim())
          .filter(Boolean);
        const bad = names.find((name) => !TOOL_NAME_RE.test(name));
        if (bad !== undefined) errors.tool = ['rules.err.toolName', { name: bad }];
        else if (names.length > 0) tool = { names: new Set(names.map((name) => name.toLowerCase())) };
      }
      let match = null;
      if (rule.match.length > 0) {
        const compiled = compilePattern(rule.match, rule.flags);
        if (compiled.error) errors.match = compiled.error;
        else {
          match = compiled.regex;
          risky = risky || previewRisk(rule.match);
        }
      }
      if (rule.tool.length === 0 && rule.match.length === 0) errors.match = ['rules.err.everything'];
      if (rule.reason.length > RULE_LIMITS.reason) errors.reason = ['rules.err.long', { n: RULE_LIMITS.reason }];
      return { rule, tool, match, risky, errors, ok: Object.keys(errors).length === 0 };
    }

    /** Validate a draft as the Host will; `rules` is exactly what a save sends. */
    function validateRules(draftRules) {
      const rules = draftRules.slice(0, RULE_LIMITS.rules).map(normalizeRule);
      const compiled = rules.map(compileRule);
      const seen = new Set();
      rules.forEach((rule, index) => {
        const key = rule.id.toLowerCase();
        if (seen.has(key)) {
          compiled[index].errors.id = ['rules.err.duplicate', { id: rule.id }];
          compiled[index].ok = false;
        } else seen.add(key);
      });
      const tooMany = draftRules.length > RULE_LIMITS.rules;
      return { rules, compiled, tooMany, errorCount: compiled.filter((entry) => !entry.ok).length + (tooMany ? 1 : 0) };
    }

    /** The texts a pattern is tested against: the arguments as JSON, then each top-level string. */
    function argumentTexts(args) {
      const texts = [];
      if (typeof args === 'string') texts.push(args);
      else {
        try {
          texts.push(JSON.stringify(args ?? {}) ?? '');
        } catch {
          texts.push(String(args));
        }
        const record = asRecord(args);
        if (record) {
          for (const value of Object.values(record)) {
            for (const item of Array.isArray(value) ? value : [value]) if (typeof item === 'string') texts.push(item);
          }
        }
      }
      return texts.map((text) => text.slice(0, RULE_LIMITS.preview));
    }

    function clockMs() {
      return typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : Date.now();
    }

    /**
     * The first enabled, valid rule that blocks this call, as far as the browser can tell cheaply.
     * It runs synchronously, so it runs no pattern on a long text (`tooLong`), and it skips and lists
     * (`skipped`) the rules whose patterns can run slowly and those left when its time is used up.
     */
    function previewCall(compiled, toolName, args) {
      const skipped = [];
      const name = String(toolName ?? '');
      const texts = argumentTexts(args);
      if (name.length > PREVIEW_MAX_TEXT || texts.some((text) => text.length > PREVIEW_MAX_TEXT))
        return { blocked: null, skipped, tooLong: true };
      const started = clockMs();
      const late = () => clockMs() - started > PREVIEW_BUDGET_MS;
      for (const entry of compiled) {
        if (!entry.rule.enabled || !entry.ok) continue;
        if (entry.risky || late()) {
          skipped.push(entry.rule.id);
          continue;
        }
        let toolMatches = true;
        if (entry.tool.names) toolMatches = entry.tool.names.has(name.toLowerCase());
        else if (entry.tool.regex) {
          entry.tool.regex.lastIndex = 0;
          toolMatches = entry.tool.regex.test(name);
        }
        if (!toolMatches) continue;
        if (entry.match === null) return { blocked: entry.rule, skipped, tooLong: false };
        let checked = 0;
        for (const text of texts) {
          if (checked > 0 && late()) break;
          entry.match.lastIndex = 0;
          if (entry.match.test(text)) return { blocked: entry.rule, skipped, tooLong: false };
          checked += 1;
        }
        if (checked < texts.length) skipped.push(entry.rule.id);
      }
      return { blocked: null, skipped, tooLong: false };
    }

    /** "Try" input: a JSON object is used as the arguments; any other text is a shell command. */
    function tryArguments(text) {
      const value = parseJsonObject(text);
      return value !== undefined ? value : { command: text };
    }

    function uniqueRuleId(rules) {
      const taken = new Set(rules.map((rule, index) => ruleIdOf(rule.id, index).toLowerCase()));
      for (let n = 1; ; n += 1) {
        const id = n === 1 ? 'new-rule' : `new-rule-${n}`;
        if (!taken.has(id)) return id;
      }
    }

    /** The rule id and the reason text of a rejection reason `<reason> (NeMo Relay rule "<id>")`. */
    function ruleRefOf(reason) {
      if (typeof reason !== 'string' || reason.length === 0) return { id: undefined, text: undefined };
      let id;
      for (const match of reason.matchAll(RULE_SUFFIX)) id = match[1];
      const text = reason.replace(RULE_SUFFIX, '').trim();
      return { id: id || undefined, text: text.length > 0 ? text : undefined };
    }

    /* ------------------------------------------------------------------ */
    /* `/relay` commands                                                    */
    /* ------------------------------------------------------------------ */

    /** How often, and how far apart, the view looks for an action's result in the status file. */
    const RESULT_TRIES = 12;
    const RESULT_RETRY_MS = 250;

    /** A request id the Host echoes into the status file (`[A-Za-z0-9._:-]{1,64}`). */
    function newRequestId() {
      const random =
        typeof globalThis.crypto?.randomUUID === 'function'
          ? globalThis.crypto.randomUUID()
          : `${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
      return `nrv-${Date.now().toString(36)}-${random}`.slice(0, 64);
    }

    /** The parsed status file, or null when it cannot be read. */
    async function readStatusFile(relay) {
      if (!relay.files || typeof relay.files.readBytes !== 'function' || !relay.statusPath) return null;
      try {
        const result = await relay.files.readBytes(relay.sessionId, relay.statusPath, {});
        if (!result || !result.ok || !result.value) return null;
        return asRecord(JSON.parse(new TextDecoder().decode(result.value.data)));
      } catch {
        return null;
      }
    }

    /**
     * The structured result the Host recorded for request `rid` in the status file, with the rules
     * and Phoenix state the status reports. The Host rewrites the file before it replies, so the
     * first read normally finds it; a few retries cover a slow file system.
     */
    async function readActionResult(relay, rid) {
      for (let attempt = 0; attempt < RESULT_TRIES; attempt += 1) {
        if (attempt > 0) await new Promise((resolve) => setTimeout(resolve, RESULT_RETRY_MS));
        const status = await readStatusFile(relay);
        if (status === null) continue;
        const record = (Array.isArray(status.actions) ? status.actions : [])
          .map(asRecord)
          .find((entry) => entry && entry.rid === rid);
        if (!record) continue;
        const result = {
          ...record,
          ...(record.guardrails === undefined && status.guardrails !== undefined
            ? { guardrails: status.guardrails }
            : {}),
          ...(record.phoenix === undefined && status.phoenix !== undefined ? { phoenix: status.phoenix } : {}),
        };
        return record.ok === true
          ? { ...result, ok: true }
          : { ...result, ok: false, code: asText(record.code) ?? 'error', error: asText(record.error) };
      }
      return null;
    }

    /**
     * Run one `/relay` action (`command` such as `rules save`, with its JSON `payload`) for this
     * session. The command replies in words, which DeepSeek Harness shows in the transcript; the
     * structured result comes from the status file, found by the request id (`rid`) this view adds.
     * An older Host replies with single-line JSON instead. `relay` carries the commands service,
     * the workspace files service, the status file path, and the session id.
     *
     * A command that does not settle (a dropped gateway connection, say) resolves as `timeout`
     * after the deadline, so the caller's controls do not stay disabled; the Host may still have
     * carried the action out.
     */
    async function runRelay(relay, command, payload) {
      if (!relay || !relay.commands || typeof relay.commands.execute !== 'function') {
        return { ok: false, code: 'unavailable' };
      }
      const limit = TIMING.relayDeadlineMs;
      let timer;
      const deadline = new Promise((resolve) => {
        timer = setTimeout(() => resolve({ ok: false, code: 'timeout', seconds: Math.round(limit / 1000) }), limit);
      });
      try {
        return await Promise.race([sendRelay(relay, command, payload), deadline]);
      } finally {
        clearTimeout(timer);
      }
    }

    /** `runRelay` without the deadline: send the command, then find its result. */
    async function sendRelay(relay, command, payload) {
      const rid = newRequestId();
      let result;
      try {
        result = await relay.commands.execute(
          relay.sessionId,
          `/relay ${command} ${JSON.stringify({ ...payload, rid })}`,
          [],
        );
      } catch (error) {
        return { ok: false, code: 'transport', error: thrown(error) };
      }
      if (!result || result.ok !== true) return { ok: false, code: 'transport', error: errorMessage(result) };
      const outcome = asRecord(asRecord(result.value)?.result);
      if (!outcome) return { ok: false, code: 'unavailable' };
      const text = typeof outcome.text === 'string' ? outcome.text : '';
      const legacy = parseJsonObject(text);
      if (legacy && typeof legacy.ok === 'boolean') {
        return legacy.ok
          ? { ...legacy, ok: true }
          : { ...legacy, ok: false, code: asText(legacy.code) ?? 'error', error: asText(legacy.error) };
      }
      const recorded = await readActionResult(relay, rid);
      if (recorded) return recorded;
      return {
        ok: false,
        code: outcome.kind === 'success' ? 'no-result' : 'error',
        error: asText(oneLine(text, 240)),
      };
    }

    /** Localized text for a failed `/relay` reply. */
    function replyFailure(reply, t) {
      switch (reply.code) {
        case 'unavailable':
          // The Host explains why it cannot do this here; without a reply there is nothing to quote.
          return reply.error ?? t('reply.unavailable');
        case 'not-json':
          return reply.error ? t('reply.notJsonText', { text: reply.error }) : t('reply.notJson');
        case 'no-result':
          return reply.error ? t('reply.noResultText', { text: reply.error }) : t('reply.noResult');
        case 'transport':
          return t('reply.transport', { message: reply.error ?? t('reply.unknown') });
        case 'timeout':
          return t('reply.timeout', { seconds: reply.seconds ?? Math.round(TIMING.relayDeadlineMs / 1000) });
        case 'stale':
          return t('rules.staleBody');
        case 'invalid':
          return reply.error ? t('reply.invalidText', { message: reply.error }) : t('reply.invalid');
        default:
          return reply.error ?? t('reply.unknown');
      }
    }

    /* ------------------------------------------------------------------ */
    /* Styles: DeepSeek Harness theme tokens; fallbacks follow the scheme  */
    /* ------------------------------------------------------------------ */
    //
    // Every color, radius, and font comes from a `--dsw-*` / `--dsh-*` token. A fallback only applies
    // if the host renames a token, and uses CSS system colors so it still follows light and dark.
    // The NVIDIA artwork lives in the logo glyph only.

    const CSS = `
.nrv-root{--nrv-err:var(--dsw-alias-state-error-primary,#d64545);--nrv-warn:var(--dsw-alias-state-warn-primary,#c98500);--nrv-warn-text:var(--dsw-alias-state-warn-label,var(--nrv-warn));--nrv-ok:var(--dsw-alias-state-success-primary,#2f9a5d);--nrv-accent:var(--dsw-alias-state-business-primary,LinkText);--nrv-brand:var(--dsw-alias-brand-primary,var(--nrv-accent));--nrv-idle:var(--dsw-alias-state-idle-primary,var(--nrv-fg4));--nrv-fg:var(--dsw-alias-label-primary,CanvasText);--nrv-fg2:var(--dsw-alias-label-secondary,color-mix(in srgb,CanvasText 76%,Canvas));--nrv-fg3:var(--dsw-alias-label-tertiary,color-mix(in srgb,CanvasText 56%,Canvas));--nrv-fg4:var(--dsw-alias-label-caption,color-mix(in srgb,CanvasText 44%,Canvas));--nrv-dim:var(--dsw-alias-label-dimmed,var(--nrv-fg4));--nrv-on-fill:var(--dsw-alias-label-primary-foreground,Canvas);--nrv-on-solid:var(--dsw-alias-label-primary-inverted,Canvas);--nrv-bg1:var(--dsw-alias-bg-layer-1,Canvas);--nrv-bg2:var(--dsw-alias-bg-layer-2,color-mix(in srgb,CanvasText 5%,Canvas));--nrv-bg3:var(--dsw-alias-bg-layer-3,color-mix(in srgb,CanvasText 9%,Canvas));--nrv-bubble:var(--dsw-specific-bubble,var(--nrv-bg2));--nrv-line:var(--dsw-alias-border-l2,color-mix(in srgb,CanvasText 14%,transparent));--nrv-line2:var(--dsw-alias-border-l3,color-mix(in srgb,CanvasText 9%,transparent));--nrv-line3:var(--dsw-alias-border-l4,color-mix(in srgb,CanvasText 22%,transparent));--nrv-hover:var(--dsw-alias-interactive-bg-hover,color-mix(in srgb,CanvasText 6%,transparent));--nrv-press:var(--dsw-alias-interactive-bg-active,color-mix(in srgb,CanvasText 10%,transparent));--nrv-hover-danger:var(--dsw-alias-interactive-bg-hover-danger,color-mix(in srgb,var(--nrv-err) 10%,transparent));--nrv-fill:var(--dsw-alias-button-primary-fill,var(--nrv-accent));--nrv-fill-hover:var(--dsw-alias-button-primary-hover,var(--nrv-fill));--nrv-ring:var(--dsw-focus-ring-color,var(--nrv-accent));--nrv-ring-w:var(--dsw-focus-ring-width,2px);--nrv-r-xs:var(--dsw-radius-xs,4px);--nrv-r-sm:var(--dsw-radius-sm,6px);--nrv-r-md:var(--dsw-radius-md,8px);--nrv-r-lg:var(--dsw-radius-lg,10px);--nrv-hair:var(--dsh-hairline-width,.5px);--nrv-mono:var(--dsh-font-family-code,ui-monospace,SFMono-Regular,Menlo,monospace);--nrv-shadow:var(--dsw-shadow-lv2,0 12px 32px rgba(0,0,0,.22));--nrv-llm:var(--nrv-brand);--nrv-tool:var(--nrv-ok);box-sizing:border-box;width:100%;padding:12px 24px 28px;color:var(--nrv-fg);font-size:var(--dsw-font-xs-13-font-size,13px);line-height:var(--dsw-font-xs-13-line-height,20px);container-type:inline-size}
.nrv-root *,.nrv-root *::before,.nrv-root *::after{box-sizing:border-box}
.nrv-column{max-width:var(--dsh-chat-content-width,860px);margin:0 auto}
.nrv-mono{font-family:var(--nrv-mono);font-size:12px}
.nrv-spacer{flex:1}
.nrv-top{border:var(--nrv-hair) solid var(--nrv-line);border-radius:var(--nrv-r-lg);background:var(--nrv-bg1);margin-bottom:12px;position:relative}
.nrv-head{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 12px 6px}
.nrv-title{display:inline-flex;align-items:center;gap:8px;font-size:15px;line-height:22px;font-weight:600;color:var(--nrv-fg);white-space:nowrap}
.nrv-logo{width:18px;height:18px;flex:none}
.nrv-meta-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:0 12px 10px;min-width:0}
.nrv-meta-label{color:var(--nrv-fg3);font-size:11.5px;margin-right:2px}
.nrv-chip{display:inline-flex;align-items:center;gap:5px;height:20px;max-width:100%;padding:0 7px;border-radius:var(--nrv-r-sm);background:var(--nrv-bg2);color:var(--nrv-fg2);font-size:11px;line-height:20px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.nrv-chip svg{flex:none}
.nrv-chip.quiet{background:transparent;color:var(--nrv-fg3);box-shadow:inset 0 0 0 var(--nrv-hair) var(--nrv-line)}
.nrv-chip.live{color:var(--nrv-ok)}
.nrv-chip.err{color:var(--nrv-err);background:color-mix(in srgb,var(--nrv-err) 10%,transparent)}
.nrv-chip.warn{color:var(--nrv-warn-text);background:color-mix(in srgb,var(--nrv-warn) 12%,transparent)}
.nrv-chip.mono{font-family:var(--nrv-mono);font-size:10.5px}
.nrv-edot{width:6px;height:6px;border-radius:50%;flex:none;background:var(--nrv-ok)}
.nrv-chip.err .nrv-edot{background:var(--nrv-err)}
.nrv-state-pill{display:inline-flex;align-items:center;gap:6px;height:22px;padding:0 9px 0 8px;border-radius:999px;background:var(--nrv-bg2);color:var(--nrv-fg2);font-size:12px;font-weight:500;white-space:nowrap;cursor:default}
.nrv-sdot{width:7px;height:7px;border-radius:50%;flex:none;background:var(--nrv-idle)}
.nrv-state-pill.ok .nrv-sdot{background:var(--nrv-ok)}
.nrv-state-pill.warn{color:var(--nrv-warn-text);background:color-mix(in srgb,var(--nrv-warn) 12%,transparent)}
.nrv-state-pill.warn .nrv-sdot{background:var(--nrv-warn)}
.nrv-state-pill.err{color:var(--nrv-err);background:color-mix(in srgb,var(--nrv-err) 10%,transparent)}
.nrv-state-pill.err .nrv-sdot{background:var(--nrv-err)}
.nrv-btn,.nrv-seg button,.nrv-link,.nrv-disclose,.nrv-flag,.nrv-rpill,.nrv-switch{font:inherit;-webkit-appearance:none;appearance:none}
.nrv-btn{display:inline-flex;align-items:center;justify-content:center;gap:5px;height:24px;padding:0 9px;border:none;border-radius:var(--nrv-r-sm);background:transparent;color:var(--nrv-fg2);font-size:12px;line-height:24px;white-space:nowrap;cursor:pointer;transition:background-color .12s,color .12s}
.nrv-btn svg{flex:none}
.nrv-btn:hover:not(:disabled){background:var(--nrv-hover);color:var(--nrv-fg)}
.nrv-btn:active:not(:disabled){background:var(--nrv-press)}
.nrv-btn:disabled{opacity:.45;cursor:default}
.nrv-btn.outline{box-shadow:inset 0 0 0 var(--nrv-hair) var(--nrv-line3)}
.nrv-btn.icon{padding:0 5px}
.nrv-btn.primary{background:var(--nrv-fill);color:var(--nrv-on-fill);font-weight:500;padding:0 12px}
.nrv-btn.primary:hover:not(:disabled){background:var(--nrv-fill-hover);color:var(--nrv-on-fill)}
.nrv-btn.primary:disabled{opacity:.4}
.nrv-btn.danger{color:var(--nrv-err)}
.nrv-btn.danger:hover:not(:disabled){background:var(--nrv-hover-danger);color:var(--nrv-err)}
.nrv-link{display:inline-flex;align-items:center;gap:4px;height:20px;padding:0 4px;border:none;border-radius:var(--nrv-r-xs);background:transparent;color:var(--nrv-accent);font-size:12px;cursor:pointer;white-space:nowrap}
.nrv-link:hover{background:var(--nrv-hover);text-decoration:underline}
.nrv-root :is(button,input,[role=button],[tabindex]):focus-visible{outline:var(--nrv-ring-w) solid var(--nrv-ring);outline-offset:1px}
.nrv-disclose{display:inline-flex;align-items:center;gap:4px;height:20px;padding:0 6px 0 7px;border:none;border-radius:var(--nrv-r-sm);background:transparent;color:var(--nrv-fg3);font-size:11px;cursor:pointer;box-shadow:inset 0 0 0 var(--nrv-hair) var(--nrv-line)}
.nrv-disclose:hover{background:var(--nrv-hover);color:var(--nrv-fg)}
.nrv-disclose[aria-expanded=true]{background:var(--nrv-bg2);color:var(--nrv-fg)}
.nrv-disclose .nrv-chev{width:10px}
.nrv-layers{margin:-4px 12px 10px;padding:8px 10px;border-radius:var(--nrv-r-md);background:var(--nrv-bg2);font-size:12px;animation:nrv-in .14s ease-out}
.nrv-layers p{margin:0 0 6px;color:var(--nrv-fg3);font-size:11.5px;line-height:16px}
.nrv-layers ol{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:3px}
.nrv-layers li{display:flex;align-items:center;gap:8px;min-width:0}
.nrv-layers code{font-family:var(--nrv-mono);font-size:11.5px;color:var(--nrv-fg);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.nrv-layer-kind{flex:none;color:var(--nrv-fg3);font-size:11px}
.nrv-layer-kind.system{color:var(--nrv-warn-text)}
.nrv-seg{display:inline-flex;padding:2px;border-radius:var(--nrv-r-md);background:var(--nrv-bg2)}
.nrv-seg button{height:22px;padding:0 10px;border:none;border-radius:var(--nrv-r-sm);background:transparent;color:var(--nrv-fg2);font-size:12px;cursor:pointer}
.nrv-seg button[aria-pressed=true]{background:var(--nrv-bg1);color:var(--nrv-fg);box-shadow:0 0 0 var(--nrv-hair) var(--nrv-line)}
.nrv-seg button:hover{color:var(--nrv-fg)}
.nrv-banner{display:flex;align-items:flex-start;gap:8px;margin:0 12px 10px;padding:7px 10px;border-radius:var(--nrv-r-md);background:color-mix(in srgb,var(--nrv-warn) 12%,transparent);color:var(--nrv-warn-text);font-size:12px;line-height:18px}
.nrv-banner>svg{flex:none;margin-top:2px}
.nrv-banner.err{background:color-mix(in srgb,var(--nrv-err) 10%,transparent);color:var(--nrv-err)}
.nrv-banner.info{background:var(--nrv-bg2);color:var(--nrv-fg2)}
.nrv-banner.ok{background:color-mix(in srgb,var(--nrv-ok) 11%,transparent);color:var(--nrv-ok)}
.nrv-banner-text{flex:1;min-width:0;overflow-wrap:anywhere}
.nrv-banner .nrv-btn{margin:-3px -4px -3px 0;color:inherit}
.nrv-banner.nrv-trace-error{margin:0 0 8px}
.nrv-rules{border-top:var(--nrv-hair) solid var(--nrv-line);padding:8px 12px 10px}
.nrv-rules-bar{display:flex;align-items:center;gap:8px;min-height:28px;flex-wrap:wrap}
.nrv-rules-toggle{display:inline-flex;align-items:center;gap:6px;height:26px;margin-left:-6px;padding:0 6px;border:none;border-radius:var(--nrv-r-sm);background:transparent;color:var(--nrv-fg);font:inherit;font-weight:600;cursor:pointer;white-space:nowrap}
.nrv-rules-toggle:hover:not(:disabled){background:var(--nrv-hover)}
.nrv-rules-toggle:disabled{cursor:default}
.nrv-rules-toggle>svg:first-of-type{color:var(--nrv-fg3)}
.nrv-tag{display:inline-flex;align-items:center;gap:4px;height:20px;padding:0 8px;border-radius:999px;background:var(--nrv-bg2);color:var(--nrv-fg2);font-size:11.5px;white-space:nowrap}
.nrv-tag b{font-weight:600;color:var(--nrv-fg)}
.nrv-tag.warn{background:color-mix(in srgb,var(--nrv-warn) 12%,transparent);color:var(--nrv-warn-text)}
.nrv-tag.err{background:color-mix(in srgb,var(--nrv-err) 10%,transparent);color:var(--nrv-err)}
.nrv-rpills{display:flex;gap:6px;flex-wrap:wrap;min-width:0;flex:1}
.nrv-rpill{display:inline-flex;align-items:center;gap:6px;height:22px;max-width:240px;padding:0 9px;border:var(--nrv-hair) solid var(--nrv-line);border-radius:999px;background:var(--nrv-bg2);color:var(--nrv-fg);font-size:12px;cursor:pointer;transition:background-color .12s}
.nrv-rpill span:last-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nrv-rpill:hover{background:var(--nrv-hover)}
.nrv-rpill.off{color:var(--nrv-fg3);background:transparent;border-style:dashed}
.nrv-rpill.bad{border-color:var(--nrv-err);color:var(--nrv-err);background:transparent}
.nrv-rdot{width:7px;height:7px;border-radius:50%;flex:none;background:var(--nrv-ok)}
.nrv-rpill.off .nrv-rdot{background:transparent;box-shadow:inset 0 0 0 1.5px var(--nrv-fg3)}
.nrv-rpill.bad .nrv-rdot{background:var(--nrv-err)}
.nrv-note{display:inline-flex;align-items:center;gap:5px;color:var(--nrv-fg3);font-size:12px}
.nrv-note.warn{color:var(--nrv-warn-text)}
.nrv-note.err{color:var(--nrv-err)}
.nrv-rules-body{margin-top:8px;animation:nrv-in .16s ease-out}
.nrv-rules-sub{margin:0 0 10px;color:var(--nrv-fg3);font-size:12px;line-height:18px}
.nrv-rules-body .nrv-banner{margin:0 0 8px}
.nrv-rule{border:var(--nrv-hair) solid var(--nrv-line);border-radius:var(--nrv-r-md);background:var(--nrv-bg1);margin-bottom:6px;overflow:hidden;transition:border-color .12s,box-shadow .12s}
.nrv-rule.open{box-shadow:0 0 0 1px color-mix(in srgb,var(--nrv-accent) 35%,transparent)}
.nrv-rule.bad{border-color:color-mix(in srgb,var(--nrv-err) 60%,transparent)}
.nrv-rule-head{display:flex;align-items:center;gap:10px;min-height:34px;padding:4px 6px 4px 10px}
.nrv-rule-head:hover{background:var(--nrv-hover)}
.nrv-rule-open{flex:1;display:grid;grid-template-columns:minmax(90px,1.1fr) minmax(60px,.7fr) minmax(0,1.6fr) auto;align-items:center;gap:10px;min-width:0;min-height:26px;padding:0 4px;border:none;border-radius:var(--nrv-r-sm);background:transparent;color:var(--nrv-fg);font:inherit;text-align:left;cursor:pointer}
.nrv-rule-id{font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.nrv-rule.off .nrv-rule-id{color:var(--nrv-fg3);font-weight:500}
.nrv-rule-tool,.nrv-rule-match{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--nrv-fg2)}
.nrv-rule-match{color:var(--nrv-fg3)}
.nrv-rule-end{display:flex;align-items:center;gap:6px}
.nrv-rule-body{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px 12px;padding:8px 12px 12px 50px;border-top:var(--nrv-hair) solid var(--nrv-line2)}
.nrv-field{display:flex;flex-direction:column;gap:4px;min-width:0}
.nrv-field.wide{grid-column:1/-1}
.nrv-label{color:var(--nrv-fg3);font-size:11px;letter-spacing:.02em}
.nrv-input{width:100%;height:28px;padding:0 9px;border:var(--nrv-hair) solid var(--nrv-line3);border-radius:var(--nrv-r-sm);background:var(--nrv-bg2);color:var(--nrv-fg);font:inherit;font-size:12.5px;outline:none;transition:border-color .12s,background-color .12s}
.nrv-input::placeholder{color:var(--nrv-fg4)}
.nrv-input:focus{border-color:var(--nrv-accent);background:var(--nrv-bg1)}
.nrv-root .nrv-input:focus-visible{outline:none}
.nrv-input.mono{font-family:var(--nrv-mono);font-size:12px}
.nrv-input.bad{border-color:var(--nrv-err)}
.nrv-input:disabled{opacity:.6}
.nrv-inline{display:flex;align-items:center;gap:6px;min-width:0}
.nrv-flag{flex:none;height:28px;min-width:34px;padding:0 6px;border:var(--nrv-hair) solid var(--nrv-line3);border-radius:var(--nrv-r-sm);background:transparent;color:var(--nrv-fg3);font-size:12px;font-weight:600;cursor:pointer}
.nrv-flag[aria-pressed=true]{color:var(--nrv-on-fill);background:var(--nrv-fill);border-color:var(--nrv-fill)}
.nrv-flag:disabled{opacity:.5;cursor:default}
.nrv-hint{color:var(--nrv-fg3);font-size:11px;line-height:15px}
.nrv-hint.err{color:var(--nrv-err)}
.nrv-hint.warn{color:var(--nrv-warn-text)}
.nrv-switch{position:relative;flex:none;width:28px;height:16px;padding:0;border:none;border-radius:999px;background:var(--nrv-line3);cursor:pointer;transition:background-color .12s}
.nrv-switch-thumb{position:absolute;top:2px;left:2px;width:12px;height:12px;border-radius:50%;background:var(--dsw-alias-switch-thumb,var(--nrv-bg1));box-shadow:0 1px 2px rgba(0,0,0,.25);transition:transform .12s}
.nrv-switch[aria-checked=true]{background:var(--nrv-accent)}
.nrv-switch[aria-checked=true] .nrv-switch-thumb{transform:translateX(12px)}
.nrv-switch:disabled{opacity:.5;cursor:default}
.nrv-add{width:100%;height:30px;border:var(--nrv-hair) dashed var(--nrv-line3)}
.nrv-add:hover:not(:disabled){border-style:solid}
.nrv-try{margin-top:12px;padding:10px;border-radius:var(--nrv-r-md);background:var(--nrv-bg2)}
.nrv-try-row{display:grid;grid-template-columns:minmax(80px,130px) 1fr auto;gap:6px;margin-top:6px}
.nrv-try .nrv-input{background:var(--nrv-bg1)}
.nrv-verdict{display:flex;align-items:center;gap:6px;margin-top:8px;min-height:20px;font-size:12.5px;flex-wrap:wrap}
.nrv-verdict svg{flex:none}
.nrv-verdict.allow{color:var(--nrv-ok);font-weight:600}
.nrv-verdict.block{color:var(--nrv-err);font-weight:600}
.nrv-verdict.muted{color:var(--nrv-fg3)}
.nrv-verdict code{font-family:var(--nrv-mono);font-size:12px}
.nrv-why{color:var(--nrv-fg2);font-weight:400}
.nrv-foot{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:12px;padding-top:10px;border-top:var(--nrv-hair) solid var(--nrv-line2)}
.nrv-failmode{display:inline-flex;align-items:center;gap:6px;margin-left:8px;color:var(--nrv-fg3);font-size:12px}
.nrv-failmode-label{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.nrv-saved{display:inline-flex;align-items:center;gap:4px;color:var(--nrv-ok);font-size:12px;animation:nrv-in .16s ease-out}
.nrv-flash{animation:nrv-flash 1.6s ease-out}
.nrv-phx{position:relative;display:inline-flex}
.nrv-split{display:inline-flex;align-items:stretch;border-radius:var(--nrv-r-sm);box-shadow:inset 0 0 0 var(--nrv-hair) var(--nrv-line3);transition:background-color .15s}
.nrv-split.on{background:var(--nrv-bg2)}
.nrv-split>.nrv-btn{border-radius:0}
.nrv-split>.nrv-btn:first-child{border-radius:var(--nrv-r-sm) 0 0 var(--nrv-r-sm)}
.nrv-split>.nrv-btn:last-child{border-radius:0 var(--nrv-r-sm) var(--nrv-r-sm) 0;gap:4px}
.nrv-split-div{width:var(--nrv-hair);margin:4px 0;background:var(--nrv-line3)}
.nrv-pdot{width:7px;height:7px;border-radius:50%;flex:none;background:transparent;box-shadow:inset 0 0 0 1.5px var(--nrv-fg3)}
.nrv-pdot.on{background:var(--nrv-ok);box-shadow:none}
.nrv-pdot.busy{background:var(--nrv-warn);box-shadow:none;animation:nrv-pulse 1s ease-in-out infinite}
.nrv-pdot.err{background:var(--nrv-err);box-shadow:none}
.nrv-pdot.warn{background:var(--nrv-warn);box-shadow:none}
.nrv-spin{width:11px;height:11px;flex:none;border-radius:50%;border:1.5px solid color-mix(in srgb,var(--nrv-accent) 25%,transparent);border-top-color:var(--nrv-accent);animation:nrv-rot .8s linear infinite}
.nrv-menu{position:absolute;right:0;top:calc(100% + 6px);z-index:40;display:flex;flex-direction:column;min-width:272px;max-width:340px;padding:6px;border:var(--nrv-hair) solid var(--nrv-line);border-radius:var(--nrv-r-lg);background:var(--nrv-bg1);box-shadow:var(--nrv-shadow);animation:nrv-in .14s ease-out}
.nrv-menu .nrv-btn{width:100%;justify-content:flex-start;height:28px;padding:0 8px;color:var(--nrv-fg)}
.nrv-menu .nrv-btn svg{color:var(--nrv-fg3)}
.nrv-menu .nrv-btn.danger,.nrv-menu .nrv-btn.danger svg{color:var(--nrv-err)}
.nrv-menu-head{padding:6px 8px 8px;margin-bottom:4px;border-bottom:var(--nrv-hair) solid var(--nrv-line2)}
.nrv-menu-title{display:flex;align-items:center;gap:6px;font-weight:600;color:var(--nrv-fg)}
.nrv-menu-title.err{color:var(--nrv-err)}
.nrv-menu-sub{margin-top:2px;color:var(--nrv-fg3);font-size:11.5px;line-height:16px;overflow-wrap:anywhere}
.nrv-menu-sub.err{color:var(--nrv-err)}
.nrv-menu-sep{height:var(--nrv-hair);margin:4px 2px;background:var(--nrv-line2)}
.nrv-menu-foot{margin-top:4px;padding:6px 8px 2px;border-top:var(--nrv-hair) solid var(--nrv-line2);color:var(--nrv-fg3);font-size:11px;line-height:15px}
.nrv-toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 8px}
.nrv-stats{display:flex;flex-wrap:wrap;gap:4px 14px;color:var(--nrv-fg3);font-size:12px}
.nrv-stats b{color:var(--nrv-fg2);font-weight:500}
.nrv-path{display:flex;align-items:center;gap:6px;min-width:0;margin-bottom:6px;color:var(--nrv-fg4);font-size:11px}
.nrv-path code{font-family:var(--nrv-mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.nrv-path .nrv-btn{height:20px;font-size:11px;padding:0 6px}
.nrv-state{padding:28px 8px;text-align:center;color:var(--nrv-fg3)}
.nrv-state strong{display:block;margin-bottom:4px;color:var(--nrv-fg2);font-weight:500}
.nrv-state.err strong{color:var(--nrv-err)}
.nrv-crash{display:flex;align-items:center;gap:8px;margin:8px 0;padding:8px 10px;border-radius:var(--nrv-r-md);background:color-mix(in srgb,var(--nrv-err) 8%,transparent);color:var(--nrv-err);font-size:12px}
.nrv-loose{margin:8px 0;color:var(--nrv-fg3);font-size:12px}
.nrv-loose-head{display:flex;align-items:center;gap:6px;padding:2px 4px;border-radius:var(--nrv-r-sm);cursor:pointer;user-select:none}
.nrv-loose-head:hover{background:var(--nrv-hover)}
.nrv-turn{margin:10px 0;border:var(--nrv-hair) solid var(--nrv-line);border-radius:var(--nrv-r-lg);background:var(--nrv-bg1);overflow:hidden}
.nrv-turn-head{display:flex;align-items:center;gap:8px;min-width:0;padding:9px 12px;cursor:pointer;user-select:none}
.nrv-turn-head:hover{background:var(--nrv-hover)}
.nrv-chev{width:12px;flex:none;color:var(--nrv-fg4);transition:transform .15s ease}
.nrv-chev.open{transform:rotate(90deg)}
.nrv-turn-title{font-weight:600;white-space:nowrap}
.nrv-turn-meta{display:flex;align-items:center;gap:10px;min-width:0;overflow:hidden;color:var(--nrv-fg3);font-size:12px;white-space:nowrap}
.nrv-status{width:8px;height:8px;border-radius:50%;flex:none;background:var(--nrv-ok)}
.nrv-status.err{background:var(--nrv-err)}
.nrv-status.open{background:var(--nrv-warn);animation:nrv-pulse 1.2s ease-in-out infinite}
.nrv-status.idle{background:var(--nrv-idle)}
.nrv-prompt{margin:0 12px 8px 32px;padding:7px 10px;max-height:7.5em;overflow:hidden;border-radius:var(--nrv-r-md);background:var(--nrv-bubble);color:var(--nrv-fg);white-space:pre-wrap;word-break:break-word}
.nrv-prompt.collapsed{max-height:none;padding:0 10px 0 0;background:transparent;color:var(--nrv-fg2);white-space:nowrap;text-overflow:ellipsis}
.nrv-prompt.none{padding:0 10px;background:transparent;color:var(--nrv-fg3);font-style:italic}
.nrv-rows{padding:2px 0 6px;border-top:var(--nrv-hair) solid var(--nrv-line2)}
.nrv-axis,.nrv-row{display:grid;grid-template-columns:18px minmax(140px,38%) 1fr 76px;gap:8px}
.nrv-axis{padding:4px 12px 2px;color:var(--nrv-fg4);font-size:10.5px}
.nrv-axis span:last-child{text-align:right}
.nrv-row{align-items:center;min-height:24px;padding:2px 12px;cursor:pointer}
.nrv-row:hover{background:var(--nrv-hover)}
.nrv-row.sel{background:var(--nrv-press)}
.nrv-row.blocked{background:color-mix(in srgb,var(--nrv-err) 7%,transparent)}
.nrv-row.blocked:hover{background:color-mix(in srgb,var(--nrv-err) 12%,transparent)}
.nrv-row.blocked.sel{background:color-mix(in srgb,var(--nrv-err) 16%,transparent)}
.nrv-row.blocked .nrv-name .n{color:var(--nrv-err);font-weight:600}
.nrv-block-reason{display:flex;align-items:center;gap:6px;flex-wrap:wrap;padding:0 12px 6px 38px;background:color-mix(in srgb,var(--nrv-err) 7%,transparent);color:var(--nrv-err);font-size:12px;line-height:18px}
.nrv-block-reason svg{flex:none}
.nrv-block-reason code{font-family:var(--nrv-mono);font-size:11.5px;font-weight:600}
.nrv-block-reason .nrv-why{color:var(--nrv-fg2);overflow-wrap:anywhere;min-width:0}
.nrv-icon{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:var(--nrv-r-xs);color:var(--nrv-on-solid);font-size:10px;font-weight:700}
.nrv-icon.llm{background:var(--nrv-llm)}
.nrv-icon.tool{background:var(--nrv-tool)}
.nrv-icon.mark{background:transparent;color:var(--nrv-warn);font-size:11px}
.nrv-icon.user{background:transparent;color:var(--nrv-fg2)}
.nrv-icon.child{background:transparent;color:var(--nrv-accent);font-size:12px}
.nrv-icon.other{background:var(--nrv-fg3)}
.nrv-icon.err{background:var(--nrv-err)}
.nrv-name{display:flex;align-items:center;gap:6px;min-width:0;overflow:hidden;white-space:nowrap}
.nrv-name .n{flex:none;max-width:60%;overflow:hidden;text-overflow:ellipsis;color:var(--nrv-fg)}
.nrv-name .s{min-width:0;overflow:hidden;text-overflow:ellipsis;color:var(--nrv-fg3);font-size:12px}
.nrv-guard{flex:none;display:inline-flex;align-items:center;height:16px;padding:0 5px;border-radius:var(--nrv-r-xs);background:color-mix(in srgb,var(--nrv-ok) 14%,transparent);color:var(--nrv-ok);font-size:10.5px;font-weight:600;line-height:16px}
.nrv-tok{flex:none;display:inline-flex;align-items:center;gap:3px;height:16px;padding:0 5px;border-radius:var(--nrv-r-xs);background:var(--nrv-bg2);color:var(--nrv-fg2);font-family:var(--nrv-mono);font-size:10.5px;line-height:16px}
.nrv-tok i{font-style:normal;color:var(--nrv-fg4)}
.nrv-fin{flex:none;display:inline-flex;align-items:center;height:16px;padding:0 5px;border-radius:var(--nrv-r-xs);background:color-mix(in srgb,var(--nrv-warn) 13%,transparent);color:var(--nrv-warn-text);font-size:10.5px;line-height:16px}
.nrv-fin.err{background:color-mix(in srgb,var(--nrv-err) 12%,transparent);color:var(--nrv-err)}
.nrv-child{flex:none;display:inline-flex;align-items:center;height:16px;padding:0 5px;border-radius:var(--nrv-r-xs);background:color-mix(in srgb,var(--nrv-accent) 12%,transparent);color:var(--nrv-accent);font-family:var(--nrv-mono);font-size:10.5px;line-height:16px}
.nrv-track{position:relative;height:10px;border-radius:3px;background-color:color-mix(in srgb,var(--nrv-bg2) 70%,transparent);background-image:linear-gradient(to right,var(--nrv-line2) 1px,transparent 1px);background-size:25% 100%}
.nrv-bar{position:absolute;top:1px;height:8px;min-width:2px;border-radius:3px;background:var(--nrv-fg3)}
.nrv-bar.llm{background:color-mix(in srgb,var(--nrv-llm) 78%,transparent)}
.nrv-bar.llm.ttft{background:linear-gradient(to right,color-mix(in srgb,var(--nrv-llm) 32%,transparent) 0 var(--nrv-ttft),color-mix(in srgb,var(--nrv-llm) 82%,transparent) var(--nrv-ttft) 100%)}
.nrv-bar.tool{background:color-mix(in srgb,var(--nrv-tool) 72%,transparent)}
.nrv-bar.err{background:var(--nrv-err)}
.nrv-bar.open{background:repeating-linear-gradient(-45deg,var(--nrv-warn) 0 4px,transparent 4px 8px);animation:nrv-slide 1s linear infinite}
.nrv-tick{position:absolute;top:0;width:2px;height:10px;margin-left:-1px;border-radius:1px;background:var(--nrv-warn)}
.nrv-tick.err{width:3px;background:var(--nrv-err)}
.nrv-tick.child{background:var(--nrv-accent)}
.nrv-dur{text-align:right;color:var(--nrv-fg3);font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}
.nrv-dur .nrv-chip{height:18px;padding:0 6px}
.nrv-detail{margin:2px 12px 8px 38px;padding:8px 10px;border-left:2px solid var(--nrv-line);color:var(--nrv-fg2);font-size:12px}
.nrv-blockedby{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;color:var(--nrv-err);font-weight:600}
.nrv-blockedby code{font-family:var(--nrv-mono);font-size:12px}
.nrv-kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 12px;margin:0 0 6px}
.nrv-kv dt{color:var(--nrv-fg3)}
.nrv-kv dd{margin:0;color:var(--nrv-fg);word-break:break-word}
.nrv-kv dd .nrv-btn{height:18px;margin-left:6px;padding:0 5px;font-size:11px;vertical-align:baseline}
.nrv-sec{margin:8px 0 4px;color:var(--nrv-fg3);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.nrv-dnote{margin:2px 0 4px;color:var(--nrv-fg3);font-style:italic}
.nrv-pre{margin:0;max-height:320px;overflow:auto;padding:8px;border-radius:var(--nrv-r-sm);background:var(--nrv-bg2);color:var(--nrv-fg2);font-family:var(--nrv-mono);font-size:11.5px;line-height:17px;white-space:pre-wrap;word-break:break-word}
.nrv-pre.err{color:var(--nrv-err);background:color-mix(in srgb,var(--nrv-err) 7%,transparent)}
.nrv-msg{margin:4px 0;max-height:200px;overflow:auto;padding:6px 8px;border-radius:var(--nrv-r-sm);background:var(--nrv-bg2);white-space:pre-wrap;word-break:break-word}
.nrv-msg em{margin-right:6px;font-style:normal;color:var(--nrv-fg3)}
.nrv-answer{margin:2px 12px 10px 32px;max-height:9em;overflow:hidden;padding:7px 10px;border:var(--nrv-hair) dashed var(--nrv-line);border-radius:var(--nrv-r-md);color:var(--nrv-fg2);white-space:pre-wrap;word-break:break-word}
.nrv-error{margin:2px 12px 10px 32px;max-height:9em;overflow:auto;padding:7px 10px;border:var(--nrv-hair) solid var(--nrv-err);border-radius:var(--nrv-r-md);color:var(--nrv-err);white-space:pre-wrap;word-break:break-word}
.nrv-more{display:flex;margin:2px 12px 6px 32px}
.nrv-awaiting{margin:2px 12px 8px 38px;color:var(--nrv-fg3);font-size:12px}
.nrv-raw{font-family:var(--nrv-mono);font-size:11.5px}
.nrv-raw-row{display:grid;grid-template-columns:96px 92px 64px 1fr 72px;gap:8px;padding:2px 6px;border-radius:var(--nrv-r-sm);cursor:pointer;white-space:nowrap}
.nrv-raw-row:hover{background:var(--nrv-hover)}
.nrv-raw-row span{overflow:hidden;text-overflow:ellipsis}
.nrv-raw-row .c-llm{color:var(--nrv-llm)}
.nrv-raw-row .c-tool{color:var(--nrv-tool)}
.nrv-raw-row .c-guard{color:var(--nrv-accent)}
.nrv-raw-row .c-mark{color:var(--nrv-warn-text)}
.nrv-raw-row .c-err{color:var(--nrv-err)}
.nrv-raw-row .dim{color:var(--nrv-fg4)}
.nrv-blocked{color:var(--nrv-err)}
@keyframes nrv-in{from{opacity:0;transform:translateY(-3px)}to{opacity:1;transform:none}}
@keyframes nrv-flash{0%{box-shadow:0 0 0 3px color-mix(in srgb,var(--nrv-accent) 70%,transparent)}100%{box-shadow:0 0 0 3px transparent}}
@keyframes nrv-pulse{0%,100%{opacity:1}50%{opacity:.35}}
@keyframes nrv-slide{to{background-position:16px 0}}
@keyframes nrv-rot{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.nrv-root *,.nrv-root *::before,.nrv-root *::after{animation:none!important;transition:none!important}}
@container (max-width:560px){.nrv-root{padding:10px 12px 24px}.nrv-rule-open{grid-template-columns:minmax(0,1fr) auto}.nrv-rule-tool,.nrv-rule-match{display:none}.nrv-rule-body{padding-left:12px}.nrv-rpills{display:none}.nrv-axis,.nrv-row{grid-template-columns:18px minmax(0,1fr) 64px}.nrv-axis span:nth-child(3),.nrv-row>.nrv-track{display:none}.nrv-try-row{grid-template-columns:1fr auto}.nrv-try-row .nrv-input:first-child{grid-column:1/-1}}
`;

    /* ------------------------------------------------------------------ */
    /* Shared UI pieces                                                     */
    /* ------------------------------------------------------------------ */

    const ICON_PATHS = {
      shield: 'M8 1.6 2.6 3.6v4.1c0 3.1 2.2 5.4 5.4 6.7 3.2-1.3 5.4-3.6 5.4-6.7V3.6L8 1.6Z',
      check: 'M3.5 8.4 6.6 11.4 12.5 4.6',
      close: 'M4.2 4.2l7.6 7.6M11.8 4.2l-7.6 7.6',
      plus: 'M8 3.2v9.6M3.2 8h9.6',
      pencil: 'M10.4 2.6l3 3L6 13H3v-3l7.4-7.4Z',
      trash: 'M3 4.6h10M6.4 4.6V3h3.2v1.6M4.6 4.6l.6 8.8h5.6l.6-8.8',
      warn: 'M8 2.2 14.4 13.4H1.6L8 2.2ZM8 6.4v3.4M8 11.6v.1',
      reset: 'M3.2 8a4.8 4.8 0 1 0 1.4-3.4M3.2 2.6v2.8H6',
      external: 'M6.6 3.4H3.4v9.2h9.2V9.4M9.4 2.6h4v4M13.4 2.6 7.6 8.4',
      chevronDown: 'M4.4 6.2 8 9.8l3.6-3.6',
      stop: 'M4.6 4.6h6.8v6.8H4.6z',
      play: 'M5.2 3.6v8.8L12.2 8 5.2 3.6Z',
      copy: 'M5.6 5.6h7.2v7.2H5.6zM3.2 10.4V3.2h7.2',
      lock: 'M4.2 7.2h7.6v6H4.2zM5.8 7.2V5.4a2.2 2.2 0 0 1 4.4 0v1.8',
      layers: 'M8 2.6 13.8 5.6 8 8.6 2.2 5.6 8 2.6ZM2.2 8.4 8 11.4l5.8-3M2.2 11 8 14l5.8-3',
      flame:
        'M8 14.4c-2.5 0-4.4-1.9-4.4-4.3 0-2.7 2.2-3.8 3.1-7 1.6 1.3 2.5 2.7 2.7 4.1.5-.5.9-1.2 1-2.1 1.4 1.2 2.1 2.9 2.1 4.8 0 2.6-1.9 4.5-4.5 4.5Z',
      unlink:
        'M6.4 9.6 4.8 11.2a2 2 0 0 1-2.8-2.8L3.6 6.8M9.6 6.4l1.6-1.6a2 2 0 0 1 2.8 2.8l-1.6 1.6M2.6 2.6l10.8 10.8',
    };

    function Icon({ name, size = 14 }) {
      return h(
        'svg',
        {
          viewBox: '0 0 16 16',
          width: size,
          height: size,
          'aria-hidden': true,
          fill: 'none',
          stroke: 'currentColor',
          strokeWidth: 1.5,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
        },
        h('path', { d: ICON_PATHS[name] ?? ICON_PATHS.warn }),
      );
    }

    function Spinner() {
      return h('span', { className: 'nrv-spin', 'aria-hidden': true });
    }

    /** A host-style toggle: a `button` with `role="switch"` and `aria-checked`; `action` names it for tests. */
    function Switch({ checked, onChange, label, disabled, action }) {
      return h(
        'button',
        {
          type: 'button',
          role: 'switch',
          className: 'nrv-switch',
          'aria-checked': checked === true,
          'aria-label': label,
          title: label,
          disabled: disabled === true,
          'data-nrv-action': action,
          onClick: (event) => {
            event.stopPropagation();
            if (disabled !== true) onChange(!checked);
          },
        },
        h('span', { className: 'nrv-switch-thumb', 'aria-hidden': true }),
      );
    }

    /** Props for an element that toggles on click, Enter, or Space; keys from inner controls are ignored. */
    function pressable(expanded, onPress) {
      return {
        role: 'button',
        tabIndex: 0,
        'aria-expanded': expanded,
        onClick: onPress,
        onKeyDown: (event) => {
          if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) return;
          event.preventDefault();
          onPress();
        },
      };
    }

    function useMounted() {
      const mounted = useRef(true);
      useEffect(() => {
        mounted.current = true;
        return () => {
          mounted.current = false;
        };
      }, []);
      return mounted;
    }

    /** A notice; an `ok` notice clears itself after `ms`. */
    function useNotice(ms) {
      const [notice, setNotice] = useState(null);
      useEffect(() => {
        if (!notice || notice.kind !== 'ok') return undefined;
        const timer = setTimeout(() => setNotice((current) => (current === notice ? null : current)), ms);
        return () => clearTimeout(timer);
      }, [notice, ms]);
      return [notice, setNotice];
    }

    /** Copy text to the clipboard; `copied` is true for a moment afterwards. */
    function useCopy() {
      const [copied, setCopied] = useState(false);
      const mounted = useMounted();
      /** Copy `text`; `copied` becomes `tag` (or true) for a moment, so one hook can serve several buttons. */
      const copy = (text, tag) => {
        try {
          const clipboard = typeof navigator !== 'undefined' ? navigator.clipboard : undefined;
          if (!clipboard || typeof clipboard.writeText !== 'function') return;
          void clipboard.writeText(String(text)).then(
            () => {
              if (!mounted.current) return;
              setCopied(tag ?? true);
              setTimeout(() => {
                if (mounted.current) setCopied(false);
              }, 1400);
            },
            () => undefined,
          );
        } catch {
          /* clipboard unavailable */
        }
      };
      return [copied, copy];
    }

    function CopyButton({ text, t }) {
      const [copied, copy] = useCopy();
      return h(
        'button',
        {
          type: 'button',
          className: 'nrv-btn',
          title: t('copy.title'),
          onClick: (event) => {
            event.stopPropagation();
            copy(text);
          },
        },
        copied ? t('copy.done') : t('copy.label'),
      );
    }

    /**
     * Error boundary around one area of the view: a rendering failure shows a short message in
     * place of that area instead of blanking the whole slot entry. Offline checks evaluate this
     * module with a minimal React stand-in, so the class is only defined when React provides one.
     */
    const Guard =
      typeof React.Component === 'function'
        ? class Guard extends React.Component {
            constructor(props) {
              super(props);
              this.state = { error: null };
            }

            static getDerivedStateFromError(error) {
              return { error };
            }

            componentDidCatch(error) {
              try {
                console.warn('[nemo-relay-deepseek-harness] a NeMo Relay view section failed to render', error);
              } catch {
                /* console unavailable */
              }
            }

            render() {
              const error = this.state.error;
              if (error === null) return this.props.children;
              const t = this.props.t;
              const message = h(
                'div',
                { className: 'nrv-crash', role: 'alert' },
                h(Icon, { name: 'warn' }),
                h('span', { className: 'nrv-banner-text' }, t('crash.body', { message: oneLine(thrown(error), 200) })),
                h(
                  'button',
                  { type: 'button', className: 'nrv-btn', onClick: () => this.setState({ error: null }) },
                  t('crash.retry'),
                ),
              );
              return this.props.frame
                ? h(
                    'div',
                    { className: 'nrv-root' },
                    h('style', null, CSS),
                    h('div', { className: 'nrv-column' }, message),
                  )
                : message;
            }
          }
        : function Guard(props) {
            return props.children;
          };

    /** Opens a guardrail rule in the rules panel; provided by the trace panel. */
    const RulesContext = typeof React.createContext === 'function' ? React.createContext(null) : null;

    /* ------------------------------------------------------------------ */
    /* Trace components                                                     */
    /* ------------------------------------------------------------------ */

    /** NeMo Relay mark (artwork; NVIDIA colors appear only here and in icon.svg). */
    function RelayGlyph() {
      return h(
        'svg',
        { className: 'nrv-logo', viewBox: '0 0 64 64', 'aria-hidden': true },
        h('rect', { x: 2, y: 2, width: 60, height: 60, rx: 14, fill: '#14210a' }),
        h('rect', { x: 12, y: 15, width: 30, height: 7, rx: 3.5, fill: '#76b900' }),
        h('rect', { x: 20, y: 28.5, width: 22, height: 7, rx: 3.5, fill: '#9fd13a' }),
        h('rect', { x: 28, y: 42, width: 18, height: 7, rx: 3.5, fill: '#c9ec7c' }),
        h('path', {
          d: 'M46 18.5h4.5a3 3 0 0 1 3 3V40a3 3 0 0 1-3 3H50',
          stroke: '#e8f6c8',
          strokeWidth: 3.2,
          strokeLinecap: 'round',
          fill: 'none',
        }),
        h('path', {
          d: 'M51.5 39.5 48 43l3.5 3.5',
          stroke: '#e8f6c8',
          strokeWidth: 3.2,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
          fill: 'none',
        }),
        h('circle', { cx: 12, cy: 45.5, r: 3, fill: '#e8f6c8' }),
      );
    }

    function Chevron({ open }) {
      return h(
        'svg',
        { className: `nrv-chev${open ? ' open' : ''}`, viewBox: '0 0 12 12', 'aria-hidden': true },
        h('path', {
          d: 'M4 2.5 7.5 6 4 9.5',
          stroke: 'currentColor',
          strokeWidth: 1.5,
          fill: 'none',
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
        }),
      );
    }

    function childSessionOf(mark) {
      return markKey(mark.name) === 'subagent' ? asText(asRecord(mark.ev.data)?.child_session_id) : undefined;
    }

    function iconFor(item, t) {
      if (item.kind === 'block') return h('span', { className: 'nrv-icon err', title: t('block.icon') }, '⛨');
      if (item.kind === 'mark') {
        const key = markKey(item.name);
        if (key === 'user_message') return h('span', { className: 'nrv-icon user' }, '›');
        if (key === 'subagent') return h('span', { className: 'nrv-icon child' }, '↳');
        return h('span', { className: `nrv-icon ${item.error ? 'err' : 'mark'}` }, item.error ? '!' : '◆');
      }
      if (item.error) return h('span', { className: 'nrv-icon err' }, '!');
      if (item.category === 'llm') return h('span', { className: 'nrv-icon llm' }, 'L');
      if (item.category === 'tool') return h('span', { className: 'nrv-icon tool' }, 'T');
      return h('span', { className: 'nrv-icon other' }, (item.category || '?').slice(0, 1).toUpperCase());
    }

    /** "⛨ n" indicator of the guardrail evaluations a call passed; names in the tooltip. */
    function GuardBadge({ guards, t }) {
      if (!guards || guards.length === 0) return null;
      const names = guards.map((guard) => guard.name);
      return h(
        'span',
        { className: 'nrv-guard', title: `${plural(t, 'guard.title', guards.length)}\n${names.join('\n')}` },
        `⛨ ${guards.length}`,
      );
    }

    function tokenTitle(usage, t) {
      return t('detail.tokenLine', {
        prompt: fmtCount(usage.prompt),
        cached: fmtCount(usage.cached),
        completion: fmtCount(usage.completion),
        reasoning: fmtCount(usage.reasoning),
      });
    }

    /** Token chip of one model call: input → output, details in the tooltip. */
    function TokenChip({ usage, t }) {
      if (!usage) return null;
      return h(
        'span',
        { className: 'nrv-tok', title: tokenTitle(usage, t) },
        fmtCount(usage.prompt),
        h('i', null, '→'),
        fmtCount(usage.completion),
      );
    }

    /** Finish reasons worth a chip on the row: anything but a normal stop or a tool-call hand-off. */
    function FinishChip({ finish, t }) {
      if (!finish || NORMAL_FINISH.has(finish)) return null;
      const error = finish === 'error' || finish === 'content_filter';
      return h(
        'span',
        { className: `nrv-fin${error ? ' err' : ''}`, title: t('detail.finishTitle', { reason: finish }) },
        finish,
      );
    }

    function KeyValues({ rows }) {
      const items = rows.filter((row) => row[1] !== undefined && row[1] !== null && row[1] !== '');
      if (items.length === 0) return null;
      return h(
        'dl',
        { className: 'nrv-kv' },
        items.map(([key, value]) =>
          h(Fragment, { key }, h('dt', null, key), h('dd', null, React.isValidElement(value) ? value : String(value))),
        ),
      );
    }

    function section(title, ...nodes) {
      return h(Fragment, null, h('div', { className: 'nrv-sec' }, title), ...nodes);
    }

    function guardNames(item) {
      return item.guards.length > 0 ? item.guards.map((guard) => guard.name).join(', ') : undefined;
    }

    function itemEvents(item) {
      if (item.kind === 'mark') return [item.ev];
      const guards = item.guards.flatMap((guard) => [guard.start, guard.end]);
      if (item.kind === 'block')
        return [
          ...guards,
          item.guard && item.guard.start,
          item.guard && item.guard.end,
          item.mark && item.mark.ev,
        ].filter(Boolean);
      return [...guards, item.start, item.end].filter(Boolean);
    }

    function RawEvents({ item, t }) {
      return section(
        t('detail.atof'),
        h(
          'pre',
          { className: 'nrv-pre' },
          itemEvents(item)
            .map((ev) => pretty(ev))
            .join('\n\n'),
        ),
      );
    }

    /** "Edit rule": opens the rules panel at this rule. */
    function EditRuleButton({ id, t }) {
      const rules = useContext(RulesContext);
      if (!rules || !id) return null;
      return h(
        'button',
        {
          type: 'button',
          className: 'nrv-link',
          'data-nrv-action': 'edit-rule',
          'data-rule-id': id,
          onClick: (event) => {
            event.stopPropagation();
            rules.openRule(id);
          },
        },
        h(Icon, { name: 'pencil', size: 12 }),
        t('block.editRule'),
      );
    }

    /** The reason line under a blocked call: which rule blocked it and why. */
    function BlockReason({ block, t }) {
      const ref = ruleRefOf(block.reason);
      return h(
        'div',
        { className: 'nrv-block-reason', 'data-nrv': 'block-reason' },
        h(Icon, { name: 'shield', size: 12 }),
        ref.id
          ? h(Fragment, null, h('span', null, t('block.byRule')), h('code', null, ref.id))
          : h('span', null, block.guardName ?? t('block.unknown')),
        h('span', { className: 'nrv-why' }, ref.text ?? (ref.id ? '' : t('block.noReason'))),
        h(EditRuleButton, { id: ref.id, t }),
      );
    }

    function BlockDetail({ block, t }) {
      const meta = { ...metaOf(block.guard && block.guard.start), ...metaOf(block.mark && block.mark.ev) };
      const target = block.target === 'llm' ? t('block.targetLlm') : t('block.targetTool');
      const ref = ruleRefOf(block.reason);
      return h(
        'div',
        { className: 'nrv-detail' },
        ref.id
          ? h(
              'div',
              { className: 'nrv-blockedby' },
              h(Icon, { name: 'shield' }),
              h('span', null, t('block.byRule')),
              h('code', null, ref.id),
              h(EditRuleButton, { id: ref.id, t }),
            )
          : null,
        h(KeyValues, {
          rows: [
            [t('block.reason'), ref.text ?? block.reason],
            [t('block.guardrail'), block.guardName],
            [t('block.target'), block.name ? `${target} · ${block.name}` : target],
            [t('detail.callId'), block.callId],
            [t('detail.step'), meta.step_index],
            [
              t('block.evaluated'),
              block.guard && block.guard.endUs !== null
                ? fmtDuration(block.guard.endUs - block.guard.startUs)
                : undefined,
            ],
            [t('guard.label'), guardNames(block)],
            [t('detail.time'), fmtClock(block.ts)],
            [t('detail.uuid'), block.uuid],
          ],
        }),
        h(RawEvents, { item: block, t }),
      );
    }

    function ToolResult({ item, t }) {
      const result = item.end ? item.end.data : undefined;
      const record = asRecord(result);
      if (record && record.schema === TOOL_RESULT_SCHEMA) {
        const content = typeof record.content === 'string' ? record.content : pretty(record.content ?? null);
        const value = record.result;
        const failed = record.is_error === true;
        const failure = failed ? asText(asRecord(asRecord(value)?.error)?.message) : undefined;
        return h(
          Fragment,
          null,
          section(
            failed ? t('detail.resultError') : t('detail.result'),
            h(
              'pre',
              { className: `nrv-pre${failed ? ' err' : ''}` },
              failure && content === 'null' ? failure : content,
            ),
          ),
          value !== null && value !== undefined && value !== record.content && !failure
            ? section(t('detail.resultValue'), h('pre', { className: 'nrv-pre' }, pretty(value)))
            : null,
        );
      }
      if (record && record.stripped === true) {
        return section(
          t('detail.result'),
          typeof record.content === 'string' ? h('pre', { className: 'nrv-pre' }, record.content) : null,
          h('div', { className: 'nrv-dnote' }, t('detail.resultStripped')),
        );
      }
      if (result === undefined || result === null) return null;
      const text =
        record && typeof record.content === 'string'
          ? record.content
          : typeof result === 'string'
            ? result
            : pretty(result);
      return section(t('detail.result'), h('pre', { className: 'nrv-pre' }, text));
    }

    function MarkDetail({ item, t }) {
      const key = markKey(item.name);
      const data = asRecord(item.ev.data) ?? {};
      const child = childSessionOf(item);
      const rows =
        key === 'subagent'
          ? [
              [
                t('subagent.child'),
                child
                  ? h(Fragment, null, h('code', { className: 'nrv-mono' }, child), h(CopyButton, { text: child, t }))
                  : undefined,
              ],
              [t('subagent.label'), asText(data.label)],
              [t('subagent.mode'), asText(data.mode)],
            ]
          : [];
      return h(
        'div',
        { className: 'nrv-detail' },
        h(KeyValues, {
          rows: [
            ...rows,
            [t('detail.mark'), item.name],
            [t('detail.time'), fmtClock(item.ts)],
            [t('detail.uuid'), item.uuid],
          ],
        }),
        section(t('detail.data'), h('pre', { className: 'nrv-pre' }, pretty(item.ev.data ?? null))),
      );
    }

    function SpanDetail({ item, t, nowUs }) {
      if (item.kind === 'block') return h(BlockDetail, { block: item, t });
      if (item.kind === 'mark') return h(MarkDetail, { item, t });
      const duration = item.endUs !== null ? item.endUs - item.startUs : nowUs - item.startUs;
      const status = item.endUs === null ? t('status.open') : item.error ? t('status.error') : t('status.ok');
      const description = asText(metaOf(item.end)['otel.status_description']);
      const statusText = description ? `${status} — ${description}` : status;
      const step = metaOf(item.start).step_index ?? metaOf(item.end).step_index;
      if (item.category === 'llm') {
        const info = llmInfo(item);
        const lastUser = lastUserText(info.messages);
        const results = newToolResults(info.messages);
        const usage = info.usage;
        let request;
        if (info.messageCount > 0)
          request = `${plural(t, 'detail.messages', info.messageCount)} · ${plural(t, 'detail.tools', info.tools)}`;
        else if (item.start && item.start.data === null) request = t('detail.requestNotCaptured');
        return h(
          'div',
          { className: 'nrv-detail' },
          h(KeyValues, {
            rows: [
              [t('detail.model'), info.model],
              [t('detail.provider'), item.name],
              [t('detail.step'), step],
              [t('detail.purpose'), info.purpose],
              [t('detail.status'), statusText],
              [t('detail.duration'), fmtDuration(duration)],
              [t('detail.ttft'), info.ttftUs !== undefined ? fmtDuration(info.ttftUs) : undefined],
              [t('detail.tokens'), usage ? tokenTitle(usage, t) : undefined],
              [t('detail.finish'), info.finish],
              [t('detail.request'), request],
              [t('guard.label'), guardNames(item)],
              [t('detail.uuid'), item.uuid],
            ],
          }),
          item.error && description ? h('pre', { className: 'nrv-pre err' }, description) : null,
          lastUser ? section(t('detail.lastUser'), h('div', { className: 'nrv-msg' }, lastUser)) : null,
          results.length > 0
            ? section(
                t('detail.toolResults', { n: results.length }),
                results.map((message, index) =>
                  h(
                    'div',
                    { key: index, className: 'nrv-msg' },
                    h('em', null, `tool${message.tool_call_id ? ` ${shortId(String(message.tool_call_id))}` : ''}`),
                    oneLine(contentText(message.content), 600),
                  ),
                ),
              )
            : null,
          info.reasoning ? section(t('detail.reasoning'), h('div', { className: 'nrv-msg' }, info.reasoning)) : null,
          info.text ? section(t('detail.response'), h('div', { className: 'nrv-msg' }, info.text)) : null,
          info.toolCalls.length > 0
            ? section(
                t('detail.toolCalls'),
                info.toolCalls.map((call, index) =>
                  h('div', { key: index, className: 'nrv-msg' }, h('em', null, call.name), oneLine(call.args, 400)),
                ),
              )
            : null,
          item.end && item.end.data === null
            ? h('div', { className: 'nrv-dnote' }, t('detail.responseNotCaptured'))
            : null,
          h(RawEvents, { item, t }),
        );
      }
      if (item.category === 'tool') {
        const args = item.start ? item.start.data : undefined;
        const argsRecord = asRecord(args);
        const stripped = argsRecord !== undefined && argsRecord.stripped === true;
        const keys = stripped && Array.isArray(argsRecord.argKeys) ? argsRecord.argKeys.join(', ') : '';
        const command = !stripped && argsRecord ? asText(argsRecord.command) : undefined;
        return h(
          'div',
          { className: 'nrv-detail' },
          h(KeyValues, {
            rows: [
              [t('detail.tool'), item.name],
              [t('detail.callId'), item.callId],
              [t('detail.parentCall'), item.parentCallId],
              [t('detail.step'), step],
              [t('detail.status'), statusText],
              [t('detail.duration'), fmtDuration(duration)],
              [t('guard.label'), guardNames(item)],
              [t('detail.uuid'), item.uuid],
            ],
          }),
          command ? section(t('detail.command'), h('pre', { className: 'nrv-pre' }, command)) : null,
          section(
            t('detail.arguments'),
            stripped
              ? h('div', { className: 'nrv-dnote' }, t('detail.argsStripped', { keys: keys || '—' }))
              : h('pre', { className: 'nrv-pre' }, pretty(args ?? null)),
          ),
          h(ToolResult, { item, t }),
          h(RawEvents, { item, t }),
        );
      }
      return h(
        'div',
        { className: 'nrv-detail' },
        h(KeyValues, {
          rows: [
            [t('detail.status'), statusText],
            [t('detail.duration'), fmtDuration(duration)],
            [t('detail.uuid'), item.uuid],
          ],
        }),
        h(RawEvents, { item, t }),
      );
    }

    function Row({ item, origin, span, nowUs, selected, onSelect, depth = 0, compact = false, t }) {
      const isSpan = item.kind === 'span';
      const isBlock = item.kind === 'block';
      const startUs = isSpan ? item.startUs : item.ts;
      const endUs = isSpan ? (item.endUs ?? nowUs) : item.ts;
      const left = span > 0 ? Math.min(100, Math.max(0, ((startUs - origin) / span) * 100)) : 0;
      const width = span > 0 ? Math.max(0, Math.min(100 - left, ((endUs - startUs) / span) * 100)) : 0;
      const running = isSpan && item.endUs === null;
      const info = isSpan && item.category === 'llm' ? llmInfo(item) : undefined;
      const child = item.kind === 'mark' ? childSessionOf(item) : undefined;
      const name = itemName(item, t);
      const subtitle = itemSubtitle(item, t);
      const label = info ? `${t('row.llm')} · ${name}` : name;
      const toggle = () => onSelect(selected ? null : item.uuid);
      let track;
      if (compact) track = h('span');
      else if (isSpan) {
        const elapsed = endUs - startUs;
        const ttft =
          info && info.ttftUs !== undefined && elapsed > 0 && !running
            ? Math.min(100, Math.max(0, (info.ttftUs / elapsed) * 100))
            : undefined;
        const barClass = `nrv-bar ${item.category}${item.error ? ' err' : ''}${running ? ' open' : ''}${ttft !== undefined ? ' ttft' : ''}`;
        const style = { left: `${left}%`, width: `${width}%` };
        if (ttft !== undefined) style['--nrv-ttft'] = `${ttft.toFixed(1)}%`;
        track = h(
          'span',
          { className: 'nrv-track', 'aria-hidden': true },
          h('span', {
            className: barClass,
            style,
            title: info && info.ttftUs !== undefined ? t('row.ttft', { ttft: fmtDuration(info.ttftUs) }) : undefined,
          }),
        );
      } else {
        track = h(
          'span',
          { className: 'nrv-track', 'aria-hidden': true },
          h('span', { className: `nrv-tick${isBlock ? ' err' : child ? ' child' : ''}`, style: { left: `${left}%` } }),
        );
      }
      let duration;
      if (isBlock) duration = h('span', { className: 'nrv-chip err' }, t('row.blocked'));
      else if (!isSpan) duration = fmtClock(item.ts).slice(0, 8);
      else if (running) duration = t('status.running');
      else duration = compact && !item.endUs ? '' : fmtDuration(item.endUs - item.startUs);
      return h(
        Fragment,
        null,
        h(
          'div',
          {
            className: `nrv-row${selected ? ' sel' : ''}${depth > 0 ? ' nested' : ''}${isBlock ? ' blocked' : ''}`,
            title: `${label}${subtitle ? ` — ${subtitle}` : ''}${isBlock && item.reason ? `\n${item.reason}` : ''}`,
            'data-nrv-kind': isBlock ? 'block' : isSpan ? item.category : 'mark',
            ...pressable(selected, toggle),
          },
          iconFor(item, t),
          h(
            'span',
            { className: 'nrv-name', style: depth > 0 ? { paddingLeft: `${14 * depth}px` } : undefined },
            h('span', { className: 'n' }, name),
            item.kind === 'mark' ? null : h(GuardBadge, { guards: item.guards, t }),
            info ? h(TokenChip, { usage: info.usage, t }) : null,
            info ? h(FinishChip, { finish: info.finish, t }) : null,
            child
              ? h('span', { className: 'nrv-child', title: t('subagent.childTitle', { id: child }) }, shortId(child))
              : null,
            subtitle ? h('span', { className: 's' }, subtitle) : null,
          ),
          track,
          h('span', { className: 'nrv-dur' }, duration),
        ),
        isBlock ? h(BlockReason, { block: item, t }) : null,
        selected ? h(SpanDetail, { item, t, nowUs }) : null,
      );
    }

    const TurnCard = memo(function TurnCard({ turn, rev, open, onToggle, onReading, nowUs, running, t }) {
      const [selected, setSelected] = useState(null);
      // The panel keeps a turn open while one of its rows is selected.
      const select = useCallback(
        (uuid) => {
          setSelected(uuid);
          onReading(turn, uuid !== null);
        },
        [turn, onReading],
      );
      const [showAll, setShowAll] = useState(false);
      const summary = turnSummary(turn);
      const number = turnNumber(turn);
      const endUs = turn.endUs ?? nowUs;
      const origin = turn.startUs;
      const span = Math.max(1, endUs - origin);
      const endData = asRecord(turn.end && turn.end.data) ?? {};
      const endMeta = metaOf(turn.end);
      const reason = asText(endData.reason) ?? asText(endMeta.turn_end_reason);
      const errorText = turn.error
        ? (asText(endData.error) ?? asText(endMeta['otel.status_description']) ?? reason)
        : undefined;
      let statusClass = '';
      if (turn.endUs === null) statusClass = running ? 'open' : 'idle';
      else if (turn.error) statusClass = 'err';
      else if (endMeta['otel.status_code'] === 'UNSET') statusClass = 'idle';
      const startData = asRecord(turn.start && turn.start.data);
      const prompt = startData && typeof startData.prompt === 'string' ? startData.prompt : undefined;
      const source = asText(metaOf(turn.start).turn_source) ?? asText(endMeta.turn_source);
      const finalText = asText(endData.final_text);
      const rows = open ? flattenItems(turn.children, turn.marks, turn.blocks) : [];
      const hidden = showAll ? 0 : Math.max(0, rows.length - ROW_CAP);
      const awaiting = open && turn.endUs === null && running && !summary.open;
      let promptNode = null;
      if (prompt !== undefined)
        promptNode = h(
          'div',
          { className: `nrv-prompt${open ? '' : ' collapsed'}` },
          open ? prompt : oneLine(prompt, 220),
        );
      else if (open && turn.start) promptNode = h('div', { className: 'nrv-prompt none' }, t('turn.noPrompt'));
      const usage =
        summary.prompt + summary.completion > 0
          ? {
              prompt: summary.prompt,
              completion: summary.completion,
              cached: summary.cached,
              reasoning: summary.reasoning,
            }
          : undefined;
      return h(
        'section',
        { className: 'nrv-turn', 'data-rev': rev },
        h(
          'div',
          { className: 'nrv-turn-head', ...pressable(open, () => onToggle(turn, open)) },
          h(Chevron, { open }),
          h('span', {
            className: `nrv-status ${statusClass}`,
            title: errorText ?? reason ?? (turn.endUs === null ? t('status.open') : ''),
          }),
          h(
            'span',
            { className: 'nrv-turn-title' },
            number !== undefined ? t('turn.title', { n: number }) : t('turn.untitled'),
          ),
          source && source !== 'user' ? h('span', { className: 'nrv-chip' }, source) : null,
          h(
            'span',
            { className: 'nrv-turn-meta' },
            h(
              'span',
              null,
              turn.endUs === null
                ? running
                  ? t('status.running')
                  : t('status.open')
                : fmtDuration(turn.endUs - turn.startUs),
            ),
            h('span', null, plural(t, 'turn.llm', summary.llm)),
            h('span', null, plural(t, 'turn.tools', summary.tools)),
            summary.errors > 0 ? h('span', null, plural(t, 'turn.errors', summary.errors)) : null,
            summary.blocked > 0
              ? h('span', { className: 'nrv-blocked' }, t('turn.blocked', { n: summary.blocked }))
              : null,
            usage ? h(TokenChip, { usage, t }) : null,
            reason && reason !== 'completed'
              ? h('span', { className: turn.error ? 'nrv-blocked' : undefined }, reason)
              : null,
          ),
        ),
        promptNode,
        open
          ? h(
              'div',
              { className: 'nrv-rows' },
              h(
                'div',
                { className: 'nrv-axis', 'aria-hidden': true },
                h('span'),
                h('span', null, t('axis.scope')),
                h('span', null, `${fmtClock(origin).slice(0, 8)} → ${fmtDuration(span)}`),
                h('span', null, t('axis.duration')),
              ),
              hidden > 0
                ? h(
                    'button',
                    { type: 'button', className: 'nrv-btn nrv-more', onClick: () => setShowAll(true) },
                    plural(t, 'turn.moreRows', hidden),
                  )
                : null,
              rows.slice(hidden).map(({ item, depth }) =>
                h(Row, {
                  key: item.uuid,
                  item,
                  origin,
                  span,
                  nowUs: endUs,
                  depth,
                  selected: selected === item.uuid,
                  onSelect: select,
                  t,
                }),
              ),
              awaiting ? h('div', { className: 'nrv-awaiting' }, t('turn.awaiting')) : null,
            )
          : null,
        open && finalText ? h('div', { className: 'nrv-answer', title: t('turn.answer') }, finalText) : null,
        open && errorText ? h('div', { className: 'nrv-error', title: t('turn.error') }, errorText) : null,
      );
    });

    /** Compact run of items outside any turn (session-title calls, agentless tools, session marks). */
    function LooseItems({ items, nowUs, t }) {
      const [open, setOpen] = useState(false);
      const [selected, setSelected] = useState(null);
      if (items.length === 0) return null;
      const names = [...new Set(items.map((item) => itemName(item, t)))];
      const blocked = items.filter((item) => item.kind === 'block').length;
      const rows = open
        ? flattenItems(
            items.filter((item) => item.kind === 'span'),
            items.filter((item) => item.kind === 'mark'),
            items.filter((item) => item.kind === 'block'),
          )
        : [];
      return h(
        'div',
        { className: 'nrv-loose' },
        h(
          'div',
          { className: 'nrv-loose-head', ...pressable(open, () => setOpen(!open)) },
          h(Chevron, { open }),
          h('span', null, plural(t, 'loose.title', items.length)),
          blocked > 0 ? h('span', { className: 'nrv-blocked' }, t('turn.blocked', { n: blocked })) : null,
          h('span', null, `· ${oneLine(names.join(', '), 120)}`),
        ),
        rows.map(({ item, depth }) =>
          h(Row, {
            key: item.uuid,
            item,
            origin: 0,
            span: 0,
            nowUs,
            depth,
            compact: true,
            selected: selected === item.uuid,
            onSelect: setSelected,
            t,
          }),
        ),
      );
    }

    function RawList({ model, t }) {
      const [limit, setLimit] = useState(RAW_PAGE);
      const [selected, setSelected] = useState(null);
      const events = model.raw;
      const start = Math.max(0, events.length - limit);
      const shown = events.slice(start);
      return h(
        'div',
        { className: 'nrv-raw' },
        start > 0
          ? h(
              'button',
              { type: 'button', className: 'nrv-btn nrv-more', onClick: () => setLimit(limit + RAW_PAGE) },
              t('raw.more', { n: start }),
            )
          : null,
        model.count > events.length
          ? h(
              'div',
              { className: 'nrv-dnote' },
              plural(t, 'raw.capped', model.count - events.length, { count: fmtCount(model.count - events.length) }),
            )
          : null,
        shown.map((ev, index) => {
          const key = `${ev.uuid}:${ev.scope_category ?? 'm'}:${start + index}`;
          const isErr = metaOf(ev)['otel.status_code'] === 'ERROR' || asRecord(ev.data)?.rejected === true;
          let cls = '';
          if (isErr) cls = 'c-err';
          else if (ev.kind === 'mark') cls = 'c-mark';
          else if (ev.category === 'llm') cls = 'c-llm';
          else if (ev.category === 'tool') cls = 'c-tool';
          else if (ev.category === 'guardrail') cls = 'c-guard';
          const toggle = () => setSelected(selected === key ? null : key);
          return h(
            Fragment,
            { key },
            h(
              'div',
              { className: 'nrv-raw-row', ...pressable(selected === key, toggle) },
              h('span', { className: 'dim' }, fmtClock(toMicros(ev.timestamp))),
              h('span', { className: cls }, ev.kind === 'scope' ? `scope.${ev.scope_category}` : 'mark'),
              h('span', { className: cls }, typeof ev.category === 'string' ? ev.category : '—'),
              h('span', null, typeof ev.name === 'string' ? ev.name : ''),
              h('span', { className: 'dim' }, shortId(ev.uuid)),
            ),
            selected === key ? h('pre', { className: 'nrv-pre' }, pretty(ev)) : null,
          );
        }),
      );
    }

    function StateMessage({ title, body, error }) {
      return h(
        'div',
        { className: `nrv-state${error ? ' err' : ''}` },
        h('strong', null, title),
        body ? h('span', null, body) : null,
      );
    }

    /* ------------------------------------------------------------------ */
    /* Header: Relay state, version, exporters, redaction, config layers   */
    /* ------------------------------------------------------------------ */

    function relayStateLabel(state, t) {
      if (!state) return t('relay.unknown');
      return RELAY_STATES.has(state) ? t(`relay.${state}`) : t('relay.other', { state });
    }

    function StatePill({ status, statusFile, statusPath, t }) {
      if (!status) {
        if (statusFile.status === 'loading') return null;
        const missing = statusFile.status === 'missing';
        return h(
          'span',
          {
            className: 'nrv-state-pill',
            'data-nrv': 'relay-state',
            'data-state': missing ? 'missing' : 'unreadable',
            title: missing
              ? t('status.missingTitle', { path: statusPath ?? '' })
              : t('status.unreadableTitle', { message: statusFile.error ?? '' }),
          },
          h('span', { className: 'nrv-sdot' }),
          t(missing ? 'status.missing' : 'status.unreadable'),
        );
      }
      const state = status.state;
      let tone = 'idle';
      if (state === 'ready') tone = 'ok';
      else if (state === 'degraded' || state === 'not_initialized') tone = 'warn';
      else if (state === 'disabled' && status.reason) tone = 'err';
      const counters = status.counters;
      const tip = [
        status.reason,
        status.pluginHost === true
          ? t('relay.hostReady')
          : status.pluginHost === false
            ? t('relay.hostMissing')
            : undefined,
        counters
          ? [
              t('relay.counters', { opened: toCount(counters.turnsOpened), closed: toCount(counters.turnsClosed) }),
              plural(t, 'relay.llmCalls', toCount(counters.llmCalls)),
              plural(t, 'relay.toolCalls', toCount(counters.toolCalls)),
              plural(t, 'relay.rejections', toCount(counters.guardrailRejections)),
              plural(t, 'relay.marks', toCount(counters.marksEmitted)),
              plural(t, 'relay.middlewareErrors', toCount(counters.middlewareErrors)),
            ].join(' · ')
          : undefined,
        status.activeSessions !== undefined ? plural(t, 'relay.sessions', status.activeSessions) : undefined,
        status.updatedAt ? t('relay.updated', { time: status.updatedAt }) : undefined,
      ].filter(Boolean);
      return h(
        'span',
        { className: `nrv-state-pill ${tone}`, title: tip.join('\n'), 'data-nrv': 'relay-state', 'data-state': state },
        h('span', { className: 'nrv-sdot' }),
        relayStateLabel(state, t),
      );
    }

    function ExporterChips({ exporters, t }) {
      if (exporters.length === 0)
        return h('span', { className: 'nrv-chip quiet', title: t('exporters.noneTitle') }, t('exporters.none'));
      return exporters.map((exporter) =>
        h(
          'span',
          {
            key: exporter.kind,
            className: `nrv-chip${exporter.health === 'degraded' ? ' err' : ''}`,
            'data-nrv-exporter': exporter.kind,
            title: [
              t(exporter.health === 'degraded' ? 'exporters.degraded' : 'exporters.enabled', { name: exporter.label }),
              ...exporter.targets,
            ].join('\n'),
          },
          h('span', { className: 'nrv-edot', 'aria-hidden': true }),
          exporter.targets.length > 1 ? `${exporter.label} ×${exporter.targets.length}` : exporter.label,
        ),
      );
    }

    /** Secret redaction: a badge while credentials are scrubbed; a quiet chip when the Host says otherwise. */
    function RedactionBadge({ redaction, t }) {
      if (!redaction) return null;
      const detectors = redaction.detectors;
      let by = null;
      if (redaction.source === 'default') by = t('redaction.byDefault');
      else if (redaction.source === 'plugins') by = t('redaction.byPlugins');
      const title = (...lines) => lines.filter(Boolean).join('\n');
      const list = detectors.length > 0 ? t('redaction.detectors', { list: detectors.join(', ') }) : null;
      if (redaction.secrets)
        return h(
          'span',
          { className: 'nrv-chip', 'data-nrv': 'redaction', title: title(t('redaction.title'), by, list) },
          h(Icon, { name: 'lock', size: 11 }),
          t('redaction.on'),
        );
      if (redaction.source === 'none')
        return h(
          'span',
          {
            className: 'nrv-chip quiet',
            'data-nrv': 'redaction',
            title: title(t('redaction.offTitle'), redaction.note),
          },
          t('redaction.off'),
        );
      if (redaction.source === 'plugins')
        return h(
          'span',
          { className: 'nrv-chip quiet', 'data-nrv': 'redaction', title: title(t('redaction.partialTitle'), by, list) },
          t('redaction.partial'),
        );
      return null;
    }

    function LayersToggle({ paths, open, onToggle, listId, t }) {
      let label = t('layers.one');
      if (paths.length === 0) label = t('layers.none');
      else if (paths.length > 1) label = t('layers.n', { n: paths.length });
      const system = paths.some(isSystemLayer);
      return h(
        'button',
        {
          type: 'button',
          className: 'nrv-disclose',
          'aria-expanded': open,
          'aria-controls': listId,
          'data-nrv-action': 'layers',
          title: t('layers.title'),
          onClick: onToggle,
        },
        h(Icon, { name: 'layers', size: 11 }),
        label,
        system ? h('span', { className: 'nrv-layer-kind system' }, t('layers.systemShort')) : null,
        h(Chevron, { open }),
      );
    }

    function LayersList({ paths, listId, t }) {
      return h(
        'div',
        { className: 'nrv-layers', id: listId, 'data-nrv': 'layers' },
        h('p', null, paths.length > 0 ? t('layers.explain') : t('layers.noneExplain')),
        paths.length > 0
          ? h(
              'ol',
              null,
              paths.map((path) =>
                h(
                  'li',
                  { key: path },
                  h('code', { title: path }, path),
                  h(
                    'span',
                    { className: `nrv-layer-kind${isSystemLayer(path) ? ' system' : ''}` },
                    t(isSystemLayer(path) ? 'layers.system' : 'layers.user'),
                  ),
                ),
              ),
            )
          : null,
      );
    }

    /* ------------------------------------------------------------------ */
    /* Guardrail rules panel                                                */
    /* ------------------------------------------------------------------ */

    function makeDraft(server) {
      const rules = server ? server.rules.map((rule) => ({ ...rule })) : [];
      return { rules, base: server ? server.revision : undefined, baseKey: rulesKey(rules) };
    }

    /**
     * Unsaved rule drafts, `{ draft, expanded, open }`, by status-file path (the rules belong to
     * that Host process) or, without one, by session id. DeepSeek Harness unmounts this view when
     * the user opens another tab, so the rules panel keeps a draft with unsaved changes here and
     * takes it back on its next mount. A draft leaves the store when it no longer differs from the
     * saved rules: after a save, a discard, a reset, or a reload.
     */
    const DRAFTS = new Map();

    function keepDraft(key, entry) {
      DRAFTS.delete(key);
      DRAFTS.set(key, entry);
      while (DRAFTS.size > DRAFT_STORE_CAP) DRAFTS.delete(DRAFTS.keys().next().value);
    }

    function RuleItem({
      rule,
      index,
      idPrefix,
      entry,
      hostErrors,
      warning,
      open,
      flash,
      disabled,
      nodeRef,
      onToggle,
      onChange,
      onRemove,
      t,
    }) {
      // This view's own checks first; then what NeMo Relay reported when it refused the last save.
      const errorOf = (field) => {
        const error = entry && entry.errors[field];
        if (error) return t(error[0], error[1]);
        return hostErrors && hostErrors[field] ? hostErrors[field] : null;
      };
      const bad = (entry ? !entry.ok : false) || Boolean(hostErrors);
      const errors = { id: errorOf('id'), tool: errorOf('tool'), match: errorOf('match'), reason: errorOf('reason') };
      const flags = typeof rule.flags === 'string' ? rule.flags : 'i';
      const matchCase = !flags.includes('i');
      const toolLabel = rule.tool.trim() || t('rules.anyTool');
      const noMatch = rule.match.trim() === '';
      const name = rule.id || t('rules.unnamed');
      const fieldId = (field) => `${idPrefix}-rule-${index}-${field}`;
      const input = (field, props, mono) =>
        h('input', {
          id: fieldId(field),
          className: `nrv-input${mono ? ' mono' : ''}${errors[field] ? ' bad' : ''}`,
          value: rule[field],
          disabled,
          spellCheck: false,
          autoComplete: 'off',
          'aria-invalid': errors[field] ? true : undefined,
          'aria-describedby': `${fieldId(field)}-hint`,
          'data-nrv-field': field,
          'data-nrv-input': `${index}:${field}`,
          onChange: (event) => onChange(index, { [field]: event.target.value }),
          ...props,
        });
      const field = (key, label, control, hint, wide, tone) =>
        h(
          'div',
          { className: `nrv-field${wide ? ' wide' : ''}` },
          h('label', { className: 'nrv-label', htmlFor: fieldId(key) }, label),
          control,
          h(
            'span',
            {
              id: `${fieldId(key)}-hint`,
              className: `nrv-hint${errors[key] ? ' err' : tone ? ` ${tone}` : ''}`,
              role: errors[key] ? 'alert' : undefined,
              'data-nrv-error': errors[key] ? key : undefined,
            },
            errors[key] ?? hint,
          ),
        );
      return h(
        'div',
        {
          ref: nodeRef,
          className: `nrv-rule${open ? ' open' : ''}${rule.enabled ? '' : ' off'}${bad ? ' bad' : ''}${flash ? ' nrv-flash' : ''}`,
          'data-rule-id': rule.id,
          'data-highlight': flash ? 'true' : undefined,
        },
        h(
          'div',
          { className: 'nrv-rule-head' },
          h(Switch, {
            checked: rule.enabled,
            label: t('rules.toggle', { id: name }),
            disabled,
            onChange: (enabled) => onChange(index, { enabled }),
          }),
          h(
            'button',
            {
              type: 'button',
              className: 'nrv-rule-open',
              'aria-expanded': open,
              'aria-label': t(open ? 'rules.collapseRule' : 'rules.expandRule', { id: name }),
              onClick: () => onToggle(index),
            },
            h('span', { className: 'nrv-rule-id', title: rule.id }, name),
            h('span', { className: 'nrv-rule-tool nrv-mono', title: toolLabel }, toolLabel),
            h(
              'span',
              { className: 'nrv-rule-match nrv-mono', title: rule.match || t('rules.anyArgs') },
              rule.match ? `/${rule.match}/${flags}` : t('rules.anyArgs'),
            ),
            h(
              'span',
              { className: 'nrv-rule-end' },
              bad
                ? h('span', { className: 'nrv-note err', title: t('rules.hasErrors') }, h(Icon, { name: 'warn' }))
                : null,
              !bad && warning
                ? h(
                    'span',
                    { className: 'nrv-note warn', title: warning, 'data-nrv': 'rule-warning' },
                    h(Icon, { name: 'warn' }),
                  )
                : null,
              h(Chevron, { open }),
            ),
          ),
        ),
        open
          ? h(
              'div',
              { className: 'nrv-rule-body' },
              field(
                'id',
                t('rules.f.name'),
                input('id', {
                  maxLength: RULE_LIMITS.id + 8,
                  placeholder: 'no-force-push',
                  autoFocus: rule.isNew === true,
                  onBlur: () => onChange(index, { id: ruleIdOf(rule.id, index) }),
                }),
                t('rules.hint.name'),
                false,
              ),
              field(
                'tool',
                t('rules.f.tool'),
                input('tool', { maxLength: RULE_LIMITS.tool, placeholder: t('rules.anyTool') }, true),
                t('rules.hint.tool'),
                false,
              ),
              field(
                'match',
                t('rules.f.match'),
                h(
                  'div',
                  { className: 'nrv-inline' },
                  input('match', { maxLength: RULE_LIMITS.pattern, placeholder: t('rules.anyArgs') }, true),
                  h(
                    'button',
                    {
                      type: 'button',
                      className: 'nrv-flag',
                      'aria-pressed': matchCase,
                      'aria-label': t('rules.matchCase'),
                      title: t('rules.matchCase'),
                      disabled,
                      onClick: () =>
                        onChange(index, { flags: matchCase ? cleanFlags(`${flags}i`) : flags.replace(/i/g, '') }),
                    },
                    'Aa',
                  ),
                ),
                noMatch ? t('rules.blocksAll', { tool: toolLabel }) : t('rules.hint.match'),
                true,
                noMatch ? 'warn' : '',
              ),
              field(
                'reason',
                t('rules.f.reason'),
                input('reason', {
                  value: rule.reason === DEFAULT_REASON ? '' : rule.reason,
                  maxLength: RULE_LIMITS.reason,
                  placeholder: t('rules.reasonPlaceholder'),
                }),
                t('rules.hint.reason', { text: DEFAULT_REASON }),
                true,
              ),
              disabled && !warning
                ? null
                : h(
                    'div',
                    { className: 'nrv-field wide' },
                    h(
                      'div',
                      { className: 'nrv-inline' },
                      warning ? h('span', { className: 'nrv-hint warn' }, warning) : null,
                      h('span', { className: 'nrv-spacer' }),
                      disabled
                        ? null
                        : h(
                            'button',
                            {
                              type: 'button',
                              className: 'nrv-btn danger',
                              'data-nrv-action': 'rule-delete',
                              onClick: () => onRemove(index),
                            },
                            h(Icon, { name: 'trash' }),
                            t('rules.delete'),
                          ),
                    ),
                  ),
            )
          : null,
      );
    }

    /** "Try a call": asks NeMo Relay (`/relay rules test`) whether the saved rules would block a call. */
    function TryBox({ relay, compiled, dirty, onShowRule, t }) {
      const commands = relay ? relay.commands : undefined;
      const [tool, setTool] = useState('bash');
      const [text, setText] = useState('');
      const [busy, setBusy] = useState(false);
      const [result, setResult] = useState(null);
      const mounted = useMounted();
      const key = `${tool.trim()}\n${text}`;
      const canRun = Boolean(commands) && text.trim().length > 0 && tool.trim().length > 0 && !busy;
      const run = async () => {
        if (!canRun) return;
        setBusy(true);
        const request = { tool: tool.trim(), arguments: tryArguments(text.slice(0, RULE_LIMITS.preview)) };
        const reply = await runRelay(relay, 'rules test', request);
        if (!mounted.current) return;
        setBusy(false);
        setResult({ key, reply });
      };
      // What the unsaved rules would do, as far as this page can tell without risking a stall.
      const local = useMemo(
        () =>
          dirty && text.trim().length > 0 && tool.trim().length > 0
            ? previewCall(compiled, tool.trim(), tryArguments(text.slice(0, RULE_LIMITS.preview)))
            : null,
        [dirty, compiled, tool, text],
      );
      let draftNote = null;
      if (local) {
        const unsure = local.tooLong || local.skipped.length > 0;
        let outcome = 'allow';
        const parts = [];
        if (local.tooLong) {
          outcome = 'too-long';
          parts.push(t('rules.try.tooLong', { n: PREVIEW_MAX_TEXT }));
        } else if (local.blocked) {
          outcome = 'block';
          parts.push(t('rules.try.draftBlocked', { id: local.blocked.id }));
        } else if (unsure) {
          outcome = 'partial';
          parts.push(t('rules.try.draftUnsure'));
        } else parts.push(t('rules.try.draftAllowed'));
        if (local.skipped.length > 0) parts.push(t('rules.try.skipped', { ids: local.skipped.join(', ') }));
        // Whether those rules block the call is NeMo Relay's to say, under its time limit.
        if (unsure && !local.blocked) parts.push(t('rules.try.saveFirst'));
        draftNote = h(
          'div',
          { className: 'nrv-hint', 'data-nrv': 'try-draft', 'data-preview': outcome },
          parts.join(' '),
        );
      }
      const onKeyDown = (event) => {
        if (event.key !== 'Enter') return;
        event.preventDefault();
        void run();
      };
      const reply = result && result.key === key ? result.reply : null;
      let verdict;
      if (busy)
        verdict = h('div', { className: 'nrv-verdict muted', role: 'status' }, h(Spinner), t('rules.try.checking'));
      else if (reply && reply.ok && reply.verdict === 'block') {
        const ref = ruleRefOf(asText(reply.reason));
        const id = asText(reply.ruleId) ?? ref.id;
        verdict = h(
          'div',
          { className: 'nrv-verdict block', role: 'status', 'data-nrv': 'try-verdict', 'data-verdict': 'block' },
          h(Icon, { name: 'shield' }),
          id
            ? h(Fragment, null, h('span', null, t('rules.try.blockedBy')), h('code', null, id))
            : h('span', null, t('rules.try.blocked')),
          ref.text ? h('span', { className: 'nrv-why' }, ref.text) : null,
          id
            ? h(
                'button',
                {
                  type: 'button',
                  className: 'nrv-link',
                  'data-nrv-action': 'try-show-rule',
                  onClick: () => onShowRule(id),
                },
                t('rules.try.show'),
              )
            : null,
        );
      } else if (reply && reply.ok && reply.verdict === 'allow') {
        // Rules whose check ran out of time did not block; NeMo Relay names them.
        const timedOut = stringList(reply.timedOut);
        verdict = h(
          'div',
          { className: 'nrv-verdict allow', role: 'status', 'data-nrv': 'try-verdict', 'data-verdict': 'allow' },
          h(Icon, { name: 'check' }),
          h('span', null, t('rules.try.allowed')),
          timedOut.length > 0
            ? h('span', { className: 'nrv-why' }, t('rules.try.timedOut', { ids: timedOut.join(', ') }))
            : null,
        );
      } else if (reply && reply.ok)
        verdict = h('div', { className: 'nrv-verdict muted', role: 'status' }, t('rules.try.unknown'));
      else if (reply)
        verdict = h(
          'div',
          { className: 'nrv-verdict block', role: 'alert', 'data-nrv': 'try-verdict', 'data-verdict': 'error' },
          h(Icon, { name: 'warn' }),
          h('span', { className: 'nrv-why' }, replyFailure(reply, t)),
        );
      else verdict = h('div', { className: 'nrv-verdict muted' }, t(commands ? 'rules.try.hint' : 'rules.try.offline'));
      return h(
        'div',
        { className: 'nrv-try', 'data-nrv': 'try' },
        h('span', { className: 'nrv-label' }, t('rules.try')),
        h(
          'div',
          { className: 'nrv-try-row' },
          h('input', {
            className: 'nrv-input mono',
            value: tool,
            maxLength: 64,
            placeholder: 'bash',
            spellCheck: false,
            autoComplete: 'off',
            'aria-label': t('rules.try.tool'),
            'data-nrv-field': 'try-tool',
            onChange: (event) => setTool(event.target.value),
            onKeyDown,
          }),
          h('input', {
            className: 'nrv-input mono',
            value: text,
            maxLength: RULE_LIMITS.preview,
            placeholder: t('rules.try.placeholder'),
            spellCheck: false,
            autoComplete: 'off',
            'aria-label': t('rules.try.args'),
            'data-nrv-field': 'try-args',
            onChange: (event) => setText(event.target.value),
            onKeyDown,
          }),
          h(
            'button',
            {
              type: 'button',
              className: 'nrv-btn outline',
              disabled: !canRun,
              'data-nrv-action': 'rules-try',
              onClick: () => void run(),
            },
            busy ? h(Spinner) : h(Icon, { name: 'play', size: 12 }),
            t('rules.try.run'),
          ),
        ),
        verdict,
        draftNote,
      );
    }

    /**
     * Guardrail rules at the top of the tab: a summary bar (rule pills, on/total, fail mode) that
     * expands into the editor. Saves go through `/relay rules save` with the revision the draft was
     * based on, so a change made elsewhere is reported as stale instead of being overwritten.
     */
    function RulesPanel({ guardrails, statusState, relayState, relay, draftKey, focus, onRefresh, t }) {
      const commands = relay ? relay.commands : undefined;
      // Unsaved changes from before this view was last unmounted, if any.
      const [kept] = useState(() => (draftKey && DRAFTS.get(draftKey)) || null);
      const [open, setOpen] = useState(kept ? kept.open : false);
      const [draft, setDraft] = useState(kept ? kept.draft : null);
      const [expanded, setExpanded] = useState(kept ? kept.expanded : null);
      const [flash, setFlash] = useState(null);
      const [busy, setBusy] = useState(null);
      const [notice, setNotice] = useNotice(4000);
      const [stale, setStale] = useState(false);
      const [confirmReset, setConfirmReset] = useState(false);
      const [replied, setReplied] = useState(null);
      const [hostErrors, setHostErrors] = useState(null);
      const mounted = useMounted();
      const nodes = useRef(new Map());
      const panelRef = useRef(null);
      const pendingFocus = useRef(null);
      // The store key under which this panel last kept its draft.
      const keptUnder = useRef(null);
      // The Host revision the effect below last compared the draft with.
      const seenRevision = useRef(undefined);
      const uid = useId();

      // A command reply carries the newest rules until the next status-file read replaces it.
      useEffect(() => setReplied(null), [guardrails]);
      const server = replied ?? guardrails;
      const validation = useMemo(() => (draft ? validateRules(draft.rules) : null), [draft]);
      const dirty = Boolean(draft && validation && rulesKey(validation.rules) !== draft.baseKey);
      const canEdit = Boolean(server && server.editable && commands);
      const inactive = relayState !== undefined && INACTIVE_STATES.has(relayState);

      // The status path, and with it the key, can arrive after the first render.
      useEffect(() => {
        const entry = draftKey && !dirty ? DRAFTS.get(draftKey) : undefined;
        if (!entry || entry.draft === draft) return;
        setDraft(entry.draft);
        setExpanded(entry.expanded);
        setOpen(entry.open);
      }, [draftKey]);
      // Keep unsaved changes across an unmount (see DRAFTS); drop them once nothing is unsaved.
      useEffect(() => {
        if (!draftKey) return;
        if (dirty) {
          keepDraft(draftKey, { draft, expanded, open });
          keptUnder.current = draftKey;
        } else if (keptUnder.current === draftKey) {
          DRAFTS.delete(draftKey);
          keptUnder.current = null;
        }
      }, [draftKey, dirty, draft, expanded, open]);

      // Follow the Host while there are no unsaved changes; otherwise say the rules changed elsewhere.
      // Not while an action of this panel is in flight: the Host rewrites the status file before it
      // replies, and the reply settles the draft.
      const acting = busy !== null;
      useEffect(() => {
        if (!server) return;
        const moved = seenRevision.current !== server.revision;
        seenRevision.current = server.revision;
        if (!draft || acting) return;
        if (server.revision === draft.base) {
          // The rules came back to what the draft is based on. (A save the Host refused as stale
          // without the revision having moved here keeps its banner.)
          if (moved) setStale(false);
          return;
        }
        // The same rules as the draft, such as this panel's own save whose reply never arrived, are not a conflict.
        if (dirty && rulesKey(server.rules) !== rulesKey(validation.rules)) setStale(true);
        else {
          setDraft(makeDraft(server));
          setStale(false);
        }
      }, [server && server.revision, acting]);

      const openPanel = () => {
        if (!server) return null;
        const current = draft ?? makeDraft(server);
        if (!draft) setDraft(current);
        setOpen(true);
        return current;
      };

      const focusRule = (id) => {
        const current = openPanel();
        if (current === null) {
          pendingFocus.current = id;
          return;
        }
        const wanted = String(id).toLowerCase();
        const index = current.rules.findIndex((rule) => String(rule.id).toLowerCase() === wanted);
        if (index < 0) {
          setNotice({ kind: 'warn', text: t('rules.missingRule', { id }) });
          setFlash({ index: -1, at: Date.now() });
          return;
        }
        setExpanded(index);
        setFlash({ index, id: current.rules[index].id, at: Date.now() });
      };

      // "Edit rule" from a blocked call in the trace.
      useEffect(() => {
        if (focus) focusRule(focus.id);
      }, [focus]);
      useEffect(() => {
        if (server && pendingFocus.current !== null) {
          const id = pendingFocus.current;
          pendingFocus.current = null;
          focusRule(id);
        }
      }, [server]);
      // Bring the focused rule (or the panel) into view, then let the highlight fade.
      useEffect(() => {
        if (!flash) return undefined;
        const node = (flash.index >= 0 && nodes.current.get(flash.index)) || panelRef.current;
        if (node && typeof node.scrollIntoView === 'function') {
          try {
            node.scrollIntoView({ behavior: 'smooth', block: flash.index >= 0 ? 'center' : 'start' });
          } catch {
            /* detached node */
          }
        }
        const timer = setTimeout(() => setFlash(null), 1800);
        return () => clearTimeout(timer);
      }, [flash]);

      const update = (mutate) =>
        setDraft((current) => (current ? { ...current, rules: mutate(current.rules) } : current));
      // A rule focuses its name field only when it is first added.
      const changeRule = (index, patch) =>
        update((rules) => rules.map((rule, i) => (i === index ? { ...rule, ...patch, isNew: false } : rule)));
      const removeRule = (index) => {
        update((rules) => rules.filter((_, i) => i !== index));
        setExpanded(null);
      };
      const addRule = () => {
        if (!draft) return;
        const id = uniqueRuleId(draft.rules);
        update((rules) => [
          ...rules,
          { id, tool: 'bash', match: '', flags: 'i', reason: '', enabled: true, isNew: true },
        ]);
        setExpanded(draft.rules.length);
      };
      const toggleRule = (index) => setExpanded((current) => (current === index ? null : index));

      /** Adopt a reply's rules as the saved state. */
      const adopt = (reply) => {
        const latest = readGuardrails(reply.guardrails);
        if (latest) setReplied(latest);
        return latest;
      };

      const save = async () => {
        if (!draft || !validation || !canEdit || !dirty || busy || stale || validation.errorCount > 0) return;
        setBusy('save');
        setNotice(null);
        const rules = validation.rules;
        const reply = await runRelay(relay, 'rules save', { base: draft.base ?? '', rules });
        if (!mounted.current) return;
        setBusy(null);
        const latest = adopt(reply);
        if (reply.ok) {
          const saved = latest ?? {
            ...server,
            rules,
            revision: asText(reply.revision) ?? (server ? server.revision : undefined),
          };
          setDraft(makeDraft(saved));
          setExpanded(null);
          setStale(false);
          setNotice({ kind: 'ok', text: t('rules.saved') });
        } else if (reply.code === 'stale') setStale(true);
        else {
          // Show NeMo Relay's own checks on the fields they concern until the draft changes.
          const byIndex = {};
          for (const error of readRuleErrors(reply.errors)) {
            byIndex[error.index] = { ...byIndex[error.index], [error.field]: error.message };
          }
          if (Object.keys(byIndex).length > 0) setHostErrors({ key: rulesKey(rules), byIndex });
          setNotice({ kind: 'err', text: replyFailure(reply, t) });
        }
        onRefresh();
      };

      const reset = async () => {
        if (!server || !canEdit || busy) return;
        setBusy('reset');
        setConfirmReset(false);
        setNotice(null);
        const reply = await runRelay(relay, 'rules reset', { base: server.revision ?? '' });
        if (!mounted.current) return;
        setBusy(null);
        const latest = adopt(reply);
        if (reply.ok) {
          // The edits are gone either way; without the new rules, the editor closes until they are read.
          setDraft(latest ? makeDraft(latest) : null);
          if (!latest) setOpen(false);
          setExpanded(null);
          setStale(false);
          setNotice({ kind: 'ok', text: t('rules.restored') });
        } else if (reply.code === 'stale') setStale(true);
        else setNotice({ kind: 'err', text: replyFailure(reply, t) });
        onRefresh();
      };

      /** `/relay rules failmode`: whether a call the rules cannot check runs (open) or is blocked (closed). */
      const changeFailMode = async (mode) => {
        if (!server || !canEdit || busy || server.failMode === mode) return;
        setBusy('failmode');
        setNotice(null);
        const reply = await runRelay(relay, 'rules failmode', { mode });
        if (!mounted.current) return;
        setBusy(null);
        const latest = adopt(reply);
        if (reply.ok) {
          // A rule draft keeps its edits; only the fail mode changes.
          if (latest && !dirty) setDraft(makeDraft(latest));
          setNotice({ kind: 'ok', text: t(mode === 'closed' ? 'rules.failClosedSaved' : 'rules.failOpenSaved') });
        } else setNotice({ kind: 'err', text: replyFailure(reply, t) });
        onRefresh();
      };

      /** Drop the draft for the Host's current rules. */
      const reload = () => {
        if (server) setDraft(makeDraft(server));
        setStale(false);
        setExpanded(null);
        setNotice(null);
        onRefresh();
      };
      /** Keep the edits and base them on the Host's current rules, so the next save replaces them. */
      const keepMine = () => {
        if (!server || !draft) return;
        setDraft({ ...draft, base: server.revision, baseKey: rulesKey(server.rules) });
        setStale(false);
      };

      const onKeyDown = (event) => {
        if ((event.metaKey || event.ctrlKey) && !event.altKey && (event.key === 's' || event.key === 'S')) {
          event.preventDefault();
          event.stopPropagation();
          void save();
        }
      };

      const rules = server ? server.rules : [];
      const onCount = rules.filter((rule) => rule.enabled).length;
      const safety = rules.find((rule) => rule.id === SAFETY_RULE_ID);
      const safetyOff = Boolean(server) && (!safety || !safety.enabled);
      const draftSafety = validation ? validation.rules.find((rule) => rule.id === SAFETY_RULE_ID) : undefined;
      const draftSafetyOff = Boolean(dirty && !safetyOff && (!draftSafety || !draftSafety.enabled));
      const bodyId = `${uid}-rules`;
      // Problems NeMo Relay reports for its current rules (it skips those rules) and for a refused save.
      const serverErrors = server ? server.errors : [];
      const problemsOf = (index) => serverErrors.filter((error) => error.index === index).map((error) => error.message);
      const refused =
        hostErrors && validation && rulesKey(validation.rules) === hostErrors.key ? hostErrors.byIndex : null;
      const slow = (id) => {
        const count = server ? server.timeouts.get(id) : undefined;
        return count ? t('rules.slow', { n: count }) : null;
      };

      let summary = null;
      if (!server) {
        let key = 'rules.loading';
        if (statusState === 'ready') key = 'rules.notReported';
        else if (statusState === 'missing' || statusState === 'error') key = 'rules.unknown';
        summary = h('span', { className: 'nrv-note', 'data-nrv': 'rules-summary' }, t(key));
      }

      const pills = [];
      if (server && !open) {
        rules.slice(0, RULE_LIMITS.pills).forEach((rule, index) => {
          const problems = problemsOf(index);
          const tip = [
            rule.id,
            `${t('rules.f.tool')}: ${rule.tool || t('rules.anyTool')}`,
            `${t('rules.f.match')}: ${rule.match || t('rules.anyArgs')}`,
            rule.reason,
            rule.enabled ? null : t('rules.isOff'),
            ...problems,
            slow(rule.id),
          ];
          pills.push(
            h(
              'button',
              {
                key: `${index}:${rule.id}`,
                type: 'button',
                className: `nrv-rpill${rule.enabled ? '' : ' off'}${problems.length > 0 ? ' bad' : ''}`,
                title: tip.filter(Boolean).join('\n'),
                'data-rule-id': rule.id,
                onClick: () => focusRule(rule.id),
              },
              h('span', { className: 'nrv-rdot', 'aria-hidden': true }),
              h('span', null, rule.id),
            ),
          );
        });
        if (rules.length > RULE_LIMITS.pills)
          pills.push(
            h(
              'button',
              { key: 'more', type: 'button', className: 'nrv-rpill off', onClick: () => openPanel() },
              h('span', null, t('rules.more', { n: rules.length - RULE_LIMITS.pills })),
            ),
          );
        if (rules.length === 0) pills.push(h('span', { key: 'none', className: 'nrv-note' }, t('rules.none')));
      }

      const bar = h(
        'div',
        { className: 'nrv-rules-bar' },
        h(
          'button',
          {
            type: 'button',
            className: 'nrv-rules-toggle',
            'aria-expanded': open,
            'aria-controls': bodyId,
            disabled: !server,
            'data-nrv-action': 'rules-toggle',
            onClick: () => (open ? setOpen(false) : openPanel()),
          },
          h(Icon, { name: 'shield', size: 15 }),
          t('rules.title'),
          server ? h(Chevron, { open }) : null,
        ),
        summary,
        server
          ? h(
              'span',
              {
                className: `nrv-tag${inactive || !server.enabled ? ' warn' : ''}`,
                title: t('rules.onTotalTitle', { on: onCount, total: rules.length }),
                'data-nrv': 'rules-count',
              },
              // One text node: the tag's flex gap must not split "1/1".
              h('span', null, h('b', null, String(onCount)), t('rules.ofTotal', { total: rules.length })),
            )
          : null,
        server
          ? h(
              'span',
              {
                className: 'nrv-tag',
                title: t(server.failMode === 'closed' ? 'rules.failClosedTitle' : 'rules.failOpenTitle'),
                'data-nrv': 'fail-mode',
              },
              t(server.failMode === 'closed' ? 'rules.failClosed' : 'rules.failOpen'),
            )
          : null,
        server && !server.enabled ? h('span', { className: 'nrv-tag warn' }, t('rules.disabledTag')) : null,
        server && safetyOff
          ? h(
              'span',
              {
                className: 'nrv-tag warn',
                title: t('rules.safetyOffTitle', { id: SAFETY_RULE_ID }),
                'data-nrv': 'safety-off',
              },
              h(Icon, { name: 'warn', size: 11 }),
              t('rules.safetyOff'),
            )
          : null,
        dirty ? h('span', { className: 'nrv-tag warn', 'data-nrv': 'rules-dirty' }, t('rules.unsaved')) : null,
        pills.length > 0 ? h('div', { className: 'nrv-rpills' }, pills) : h('span', { className: 'nrv-spacer' }),
        server && server.rejections !== undefined && server.rejections > 0
          ? h(
              'span',
              { className: 'nrv-note', title: t('rules.rejectionsTitle') },
              t('rules.rejections', { n: server.rejections }),
            )
          : null,
      );

      let body = null;
      if (open && draft && validation && server) {
        const banner = (key, tone, text, ...actions) =>
          h(
            'div',
            {
              key,
              className: `nrv-banner ${tone}`,
              role: tone === 'err' ? 'alert' : 'status',
              'data-nrv-banner': key,
            },
            h(Icon, { name: tone === 'info' ? 'lock' : 'warn' }),
            h('span', { className: 'nrv-banner-text' }, text),
            ...actions,
          );
        const banners = [
          inactive
            ? banner('inactive', 'err', t('rules.notEnforced', { state: relayStateLabel(relayState, t) }))
            : null,
          !server.enabled ? banner('disabled', 'warn', t('rules.disabledBody')) : null,
          !canEdit ? banner('readonly', 'info', t(server.editable ? 'rules.noCommands' : 'rules.readOnly')) : null,
          stale
            ? banner(
                'stale',
                'warn',
                t('rules.stale'),
                h(
                  'button',
                  {
                    key: 'reload',
                    type: 'button',
                    className: 'nrv-btn',
                    'data-nrv-action': 'rules-reload',
                    onClick: reload,
                  },
                  h(Icon, { name: 'reset', size: 12 }),
                  t('rules.reload'),
                ),
                dirty
                  ? h(
                      'button',
                      {
                        key: 'keep',
                        type: 'button',
                        className: 'nrv-btn',
                        'data-nrv-action': 'rules-keep',
                        onClick: keepMine,
                      },
                      t('rules.keepMine'),
                    )
                  : null,
              )
            : null,
          draftSafetyOff ? banner('safety', 'warn', t('rules.safetyDraftOff', { id: SAFETY_RULE_ID })) : null,
          validation.tooMany ? banner('too-many', 'err', t('rules.err.tooMany', { n: RULE_LIMITS.rules })) : null,
          server.loadError ? banner('load-error', 'err', server.loadError) : null,
          serverErrors.length > 0 && !dirty
            ? banner(
                'skipped',
                'warn',
                plural(t, 'rules.skipped', new Set(serverErrors.map((error) => error.index)).size, {
                  list: serverErrors
                    .slice(0, 3)
                    .map((error) => `${error.id ?? `#${error.index + 1}`}: ${error.message}`)
                    .join(' · '),
                }),
              )
            : null,
          notice && notice.kind !== 'ok' ? banner('notice', notice.kind === 'err' ? 'err' : 'warn', notice.text) : null,
        ];
        const source = server.source === 'file' || server.source === 'default' ? server.source : 'config';
        const footRight = confirmReset
          ? [
              h('span', { key: 'q', className: 'nrv-note warn' }, t('rules.resetConfirm')),
              h(
                'button',
                { key: 'no', type: 'button', className: 'nrv-btn', onClick: () => setConfirmReset(false) },
                t('rules.cancel'),
              ),
              h(
                'button',
                {
                  key: 'yes',
                  type: 'button',
                  className: 'nrv-btn danger',
                  disabled: Boolean(busy),
                  'data-nrv-action': 'rules-reset-confirm',
                  onClick: () => void reset(),
                },
                h(Icon, { name: 'reset' }),
                t('rules.resetDo'),
              ),
            ]
          : [
              notice && notice.kind === 'ok'
                ? h(
                    'span',
                    { key: 'ok', className: 'nrv-saved', role: 'status' },
                    h(Icon, { name: 'check' }),
                    notice.text,
                  )
                : null,
              canEdit
                ? h(
                    'button',
                    {
                      key: 'reset',
                      type: 'button',
                      className: 'nrv-btn',
                      disabled: Boolean(busy),
                      'data-nrv-action': 'rules-reset',
                      title: t('rules.resetTitle'),
                      onClick: () => setConfirmReset(true),
                    },
                    busy === 'reset' ? h(Spinner) : h(Icon, { name: 'reset' }),
                    busy === 'reset' ? t('rules.resetting') : t('rules.reset'),
                  )
                : null,
              dirty
                ? h(
                    'button',
                    {
                      key: 'discard',
                      type: 'button',
                      className: 'nrv-btn',
                      disabled: Boolean(busy),
                      'data-nrv-action': 'rules-discard',
                      onClick: reload,
                    },
                    t('rules.discard'),
                  )
                : null,
              canEdit
                ? h(
                    'button',
                    {
                      key: 'save',
                      type: 'button',
                      className: 'nrv-btn primary',
                      disabled: !dirty || Boolean(busy) || stale || validation.errorCount > 0,
                      'data-nrv-action': 'rules-save',
                      title:
                        validation.errorCount > 0
                          ? plural(t, 'rules.fixFirst', validation.errorCount)
                          : stale
                            ? t('rules.reloadFirst')
                            : t('rules.shortcut'),
                      onClick: () => void save(),
                    },
                    busy === 'save' ? h(Spinner) : null,
                    busy === 'save' ? t('rules.saving') : t('rules.save'),
                  )
                : null,
            ];
        body = h(
          'div',
          {
            className: 'nrv-rules-body',
            id: bodyId,
            onKeyDown,
            'data-nrv': 'rules-editor',
            'aria-busy': Boolean(busy),
          },
          h('p', { className: 'nrv-rules-sub' }, t('rules.sub')),
          banners,
          draft.rules.map((rule, index) =>
            h(RuleItem, {
              idPrefix: uid,
              key: index,
              rule,
              index,
              entry: validation.compiled[index],
              hostErrors: refused ? refused[index] : undefined,
              warning: slow(rule.id),
              open: expanded === index,
              flash: flash !== null && flash.index === index,
              disabled: !canEdit || Boolean(busy),
              nodeRef: (node) => {
                if (node) nodes.current.set(index, node);
                else nodes.current.delete(index);
              },
              onToggle: toggleRule,
              onChange: changeRule,
              onRemove: removeRule,
              t,
            }),
          ),
          draft.rules.length === 0 ? h('div', { className: 'nrv-note' }, t('rules.noneEditor')) : null,
          canEdit
            ? h(
                'button',
                {
                  type: 'button',
                  className: 'nrv-btn nrv-add',
                  disabled: Boolean(busy) || draft.rules.length >= RULE_LIMITS.rules,
                  'data-nrv-action': 'rule-add',
                  onClick: addRule,
                },
                h(Icon, { name: 'plus' }),
                t('rules.add'),
              )
            : null,
          h(TryBox, {
            relay,
            compiled: validation.compiled,
            dirty,
            onShowRule: (id) => focusRule(id),
            t,
          }),
          h(
            'div',
            { className: 'nrv-foot' },
            h(
              'span',
              {
                className: 'nrv-note',
                'data-nrv': 'rules-source',
                title:
                  [
                    server.location ? t('rules.location', { location: server.location }) : null,
                    server.revision ? t('rules.revision', { rev: server.revision }) : null,
                  ]
                    .filter(Boolean)
                    .join('\n') || undefined,
              },
              t(`rules.source.${source}`),
            ),
            h(
              'span',
              {
                className: 'nrv-failmode',
                'data-nrv': 'failmode-control',
                title: t(server.failMode === 'closed' ? 'rules.failClosedTitle' : 'rules.failOpenTitle'),
              },
              h(Switch, {
                checked: server.failMode === 'closed',
                onChange: (closed) => void changeFailMode(closed ? 'closed' : 'open'),
                label: t('rules.failClosedSwitch'),
                disabled: !canEdit || Boolean(busy),
                action: 'failmode-toggle',
              }),
              h(
                'span',
                { className: 'nrv-failmode-label' },
                busy === 'failmode' ? h(Spinner) : null,
                t(server.failMode === 'closed' ? 'rules.failClosedLabel' : 'rules.failOpenLabel'),
              ),
            ),
            h('span', { className: 'nrv-spacer' }),
            footRight,
          ),
        );
      }

      return h(
        'div',
        { className: 'nrv-rules', ref: panelRef, 'data-nemo-relay-rules': '' },
        bar,
        !open && notice && notice.kind !== 'ok'
          ? h('div', { className: `nrv-note ${notice.kind === 'err' ? 'err' : 'warn'}`, role: 'alert' }, notice.text)
          : null,
        body,
      );
    }

    /* ------------------------------------------------------------------ */
    /* Phoenix menu                                                         */
    /* ------------------------------------------------------------------ */

    function PhoenixMenu({ phoenix, sessionId, relay, onRefresh, onNotice, t }) {
      const commands = relay ? relay.commands : undefined;
      const [busy, setBusy] = useState(null);
      const [menu, setMenu] = useState(false);
      const [replied, setReplied] = useState(null);
      const [copied, copy] = useCopy();
      const mounted = useMounted();
      const boxRef = useRef(null);
      const toggleRef = useRef(null);
      // A command reply carries the newest Phoenix state until the next status-file read replaces it.
      useEffect(() => setReplied(null), [phoenix]);
      // While the menu is open, a click outside it or Escape closes it.
      useEffect(() => {
        if (!menu || typeof document === 'undefined' || typeof document.addEventListener !== 'function')
          return undefined;
        const onDown = (event) => {
          if (boxRef.current && typeof boxRef.current.contains === 'function' && !boxRef.current.contains(event.target))
            setMenu(false);
        };
        const onKey = (event) => {
          if (event.key !== 'Escape') return;
          setMenu(false);
          if (toggleRef.current && typeof toggleRef.current.focus === 'function') toggleRef.current.focus();
        };
        document.addEventListener('mousedown', onDown);
        document.addEventListener('keydown', onKey);
        return () => {
          document.removeEventListener('mousedown', onDown);
          document.removeEventListener('keydown', onKey);
        };
      }, [menu]);

      const px = replied ?? phoenix;
      if (!px || px.enabled === false || px.state === 'disabled') return null;

      const state = px.state;
      const running = state === 'running';
      // A running Phoenix that NeMo Relay does not export to (older status files say nothing: assume it does).
      const linked = running && px.connected !== false;
      const unlinked = running && px.connected === false;
      const starting = state === 'starting' || busy === 'start' || busy === 'connect';
      const stopping = busy === 'stop' || busy === 'disconnect';
      const working = starting || stopping || state === 'waiting';
      const sessionUrl = linked ? phoenixSessionUrl(px, sessionId) : undefined;
      const address = px.url ? px.url.replace(/^https?:\/\//i, '') : undefined;
      const external = px.mode === 'external';
      const stopAction = external || state === 'waiting' ? 'disconnect' : 'stop';
      const downloading = state === 'starting' && px.phase === 'downloading';

      const run = async (action) => {
        setMenu(false);
        if (busy) return;
        setBusy(action);
        const reply = await runRelay(relay, `phoenix ${action}`, {});
        if (!mounted.current) return;
        setBusy(null);
        const next = readPhoenix(reply.phoenix);
        if (next) setReplied(next);
        if (!reply.ok) {
          const message = asText(reply.error) ?? (next && next.error) ?? replyFailure(reply, t);
          onNotice({ kind: 'err', text: t('phoenix.failed', { message }) });
        }
        onRefresh();
      };
      const open = (url) => {
        setMenu(false);
        if (!openExternal(url)) onNotice({ kind: 'err', text: t('phoenix.openFailed', { url: url ?? '' }) });
      };

      let label = t('phoenix.start');
      if (stopping) label = t('phoenix.stopping');
      else if (busy === 'connect') label = t('phoenix.connecting');
      else if (downloading) label = t('phoenix.downloading');
      else if (starting) label = t('phoenix.starting');
      else if (state === 'waiting') label = t('phoenix.waiting');
      else if (linked) label = t('phoenix.open');
      else if (unlinked) label = t('phoenix.connect');
      else if (state === 'error') label = t('phoenix.retry');
      let dot = 'off';
      if (linked) dot = 'on';
      else if (state === 'error') dot = 'err';
      else if (working) dot = 'busy';
      else if (unlinked) dot = 'warn';
      let mainTitle = t('phoenix.startHint', { address: address ?? '127.0.0.1:6006' });
      if (linked) mainTitle = t('phoenix.openHint', { address: address ?? '' });
      else if (unlinked) mainTitle = t('phoenix.notLinkedBody', { address: address ?? '' });
      else if (state === 'waiting') mainTitle = t('phoenix.waitingBody', { address: address ?? '' });
      else if (state === 'error' && px.error) mainTitle = px.error;

      const item = (key, icon, text, onClick, options = {}) =>
        h(
          'button',
          {
            key,
            type: 'button',
            role: 'menuitem',
            className: `nrv-btn${options.danger ? ' danger' : ''}`,
            disabled: options.disabled === true,
            'data-nrv-action': `phoenix-menu-${key}`,
            onClick,
          },
          h(Icon, { name: icon }),
          text,
        );
      const separator = h('div', { key: 'sep', className: 'nrv-menu-sep', role: 'separator' });
      const stopItem = external
        ? item('disconnect', 'unlink', t('phoenix.disconnect'), () => void run('disconnect'), { disabled: working })
        : item('stop', 'stop', t('phoenix.stop'), () => void run('stop'), { disabled: working, danger: true });
      const copyItems = [
        item('copy', 'copy', copied === 'url' ? t('phoenix.copied') : t('phoenix.copyUrl'), () => copy(px.url, 'url'), {
          disabled: !px.url,
        }),
        px.log
          ? item('log', 'copy', copied === 'log' ? t('phoenix.copied') : t('phoenix.copyLog'), () =>
              copy(px.log, 'log'),
            )
          : null,
      ];
      const items = [];
      if (linked) {
        items.push(
          item('session', 'external', t('phoenix.openSession'), () => open(sessionUrl), { disabled: !sessionUrl }),
          item('project', 'external', t('phoenix.openPhoenix'), () => open(px.url), { disabled: !px.url }),
          ...copyItems,
          separator,
          stopItem,
        );
      } else if (unlinked) {
        items.push(
          item('connect', 'play', t('phoenix.connect'), () => void run('connect'), { disabled: working || !commands }),
          item('project', 'external', t('phoenix.openPhoenix'), () => open(px.url), { disabled: !px.url }),
          ...copyItems,
          ...(external ? [] : [separator, stopItem]),
        );
      } else if (state === 'waiting' || state === 'starting') {
        items.push(
          stopAction === 'disconnect'
            ? item('disconnect', 'unlink', t('phoenix.stopWaiting'), () => void run('disconnect'), {
                disabled: stopping,
              })
            : item('stop', 'stop', t('phoenix.cancel'), () => void run('stop'), { disabled: stopping, danger: true }),
          px.log ? copyItems[1] : null,
        );
      } else {
        items.push(
          item('start', 'play', state === 'error' ? t('phoenix.retry') : t('phoenix.start'), () => void run('start'), {
            disabled: working || !commands,
          }),
          state === 'error' && px.log ? copyItems[1] : null,
        );
      }

      const sub = (key, text, extra = {}) =>
        text
          ? h('div', { key, className: `nrv-menu-sub${extra.err ? ' err' : ''}`, 'data-nrv': extra.nrv }, text)
          : null;
      let head;
      if (running)
        head = [
          h(
            'div',
            { key: 't', className: 'nrv-menu-title' },
            h('span', { className: `nrv-pdot ${linked ? 'on' : 'warn'}`, 'aria-hidden': true }),
            t('phoenix.runningAt', { version: px.version ? ` ${px.version}` : '', address: address ?? '' }),
          ),
          unlinked ? sub('l', t('phoenix.notLinked')) : null,
          sub('m', t(external ? 'phoenix.external' : 'phoenix.owned')),
          px.project ? sub('p', t('phoenix.project', { project: px.project })) : null,
        ];
      else if (starting)
        head = [
          h(
            'div',
            { key: 't', className: 'nrv-menu-title' },
            h(Spinner),
            downloading ? t('phoenix.downloading') : label,
          ),
          sub('s', t(downloading ? 'phoenix.downloadingBody' : 'phoenix.startingBody')),
        ];
      else if (state === 'waiting')
        head = [
          h('div', { key: 't', className: 'nrv-menu-title' }, h(Spinner), t('phoenix.waitingTitle')),
          sub('s', t('phoenix.waitingBody', { address: address ?? '' })),
        ];
      else if (state === 'error')
        head = [
          h('div', { key: 't', className: 'nrv-menu-title err' }, t('phoenix.failedTitle')),
          sub('s', px.error, { err: true, nrv: 'phoenix-error' }),
          sub('h', px.launchHint),
        ];
      else
        head = [
          h('div', { key: 't', className: 'nrv-menu-title' }, t('phoenix.notConnected')),
          sub('s', t('phoenix.firstRun', { address: address ?? '127.0.0.1:6006' })),
          sub('h', px.launchHint, { nrv: 'phoenix-hint' }),
        ];
      const local = !px.url || isLoopbackUrl(px.url);

      return h(
        'span',
        { className: 'nrv-phx', ref: boxRef, 'data-nemo-relay-phoenix': '', 'data-state': state },
        h(
          'span',
          { className: `nrv-split${linked ? ' on' : ''}` },
          h(
            'button',
            {
              type: 'button',
              className: 'nrv-btn',
              // Opening the session is a link; starting or connecting needs the `/relay` command.
              disabled: linked ? !sessionUrl : working || !commands,
              title: mainTitle,
              'data-nrv-action': linked ? 'phoenix-open' : unlinked ? 'phoenix-reconnect' : 'phoenix-start',
              onClick: () => {
                if (linked) open(sessionUrl);
                else void run(unlinked ? 'connect' : 'start');
              },
            },
            working ? h(Spinner) : h(Icon, { name: 'flame' }),
            label,
            linked && !working ? h(Icon, { name: 'external', size: 12 }) : null,
          ),
          h('span', { className: 'nrv-split-div', 'aria-hidden': true }),
          h(
            'button',
            {
              type: 'button',
              ref: toggleRef,
              className: 'nrv-btn icon',
              'aria-label': t('phoenix.menu'),
              title: t('phoenix.menu'),
              'aria-haspopup': 'menu',
              'aria-expanded': menu,
              'data-nrv-action': 'phoenix-menu',
              onClick: () => setMenu(!menu),
            },
            h('span', { className: `nrv-pdot ${dot}`, 'aria-hidden': true }),
            h(Icon, { name: 'chevronDown', size: 12 }),
          ),
        ),
        menu
          ? h(
              'div',
              { className: 'nrv-menu', role: 'menu', 'aria-label': t('phoenix.menu') },
              h('div', { className: 'nrv-menu-head' }, head),
              items,
              h(
                'div',
                { className: 'nrv-menu-foot' },
                local ? t('phoenix.local') : t('phoenix.remote', { address: address ?? '' }),
              ),
            )
          : null,
      );
    }

    /* ------------------------------------------------------------------ */
    /* Trace panel                                                          */
    /* ------------------------------------------------------------------ */

    /**
     * Read one small JSON side file (the Host status file) through the workspaceFiles Remote and
     * re-read it whenever the `file` resource reports a new version or `nonce` changes. The
     * resource hook is always called (hook order).
     */
    function useJsonFile(workspaceFiles, sessionId, path, useResource, nonce) {
      const resource =
        typeof useResource === 'function'
          ? useResource(path ? fileAddress(sessionId, path) : 'about:blank')
          : { status: 'none' };
      const token =
        resource && resource.status === 'live' && resource.value
          ? `${resource.value.version}:${resource.value.bytes ?? ''}`
          : (resource?.status ?? 'none');
      const [state, setState] = useState({ status: 'loading', value: null, error: null });
      const [poll, setPoll] = useState(0);
      useEffect(() => {
        if (!path || !workspaceFiles || typeof workspaceFiles.readBytes !== 'function') {
          setState({ status: 'missing', value: null, error: null });
          return undefined;
        }
        const controller = new AbortController();
        let pending;
        try {
          pending = Promise.resolve(workspaceFiles.readBytes(sessionId, path, {}, controller.signal));
        } catch (error) {
          pending = Promise.reject(error);
        }
        pending.then(
          (result) => {
            if (controller.signal.aborted) return;
            if (!result || !result.ok) {
              setState({
                status: errorCode(result) === 'workspace-file/not-found' ? 'missing' : 'error',
                value: null,
                error: errorMessage(result),
              });
              return;
            }
            try {
              setState({
                status: 'ready',
                value: JSON.parse(new TextDecoder().decode(result.value.data)),
                error: null,
              });
            } catch (error) {
              setState({ status: 'error', value: null, error: thrown(error) });
            }
          },
          (error) => {
            if (!controller.signal.aborted) setState({ status: 'error', value: null, error: thrown(error) });
          },
        );
        return () => controller.abort();
      }, [workspaceFiles, sessionId, path, token, poll, nonce]);
      // A failed read is tried again until it works: with a live `file` resource nothing else
      // would, since a failed read changes no file version.
      const failed = state.status === 'error';
      useEffect(() => {
        if (!failed) return undefined;
        const timer = setInterval(() => setPoll((value) => value + 1), 1500);
        return () => clearInterval(timer);
      }, [failed]);
      // Without a resource provider, poll slowly so the side file still refreshes.
      useEffect(() => {
        if (!path || (resource && resource.status !== 'none')) return undefined;
        const timer = setInterval(() => setPoll((value) => value + 1), 5000);
        return () => clearInterval(timer);
      }, [path, resource && resource.status]);
      return state;
    }

    function useNow(active) {
      const [now, setNow] = useState(() => Date.now() * 1000);
      useEffect(() => {
        if (!active) return undefined;
        const timer = setInterval(() => setNow(Date.now() * 1000), 1000);
        return () => clearInterval(timer);
      }, [active]);
      return active ? now : 0;
    }

    function TracePanel({ sessionId, view, workspaceFiles, commands, useResource, useSession, t }) {
      const source = useMemo(
        () => new TraceSource(workspaceFiles, sessionId, view.file),
        [workspaceFiles, sessionId, view.file],
      );
      useEffect(() => () => source.dispose(), [source]);
      const snapshot = useSyncExternalStore(source.subscribe, source.getSnapshot);
      const resource =
        typeof useResource === 'function' ? useResource(fileAddress(sessionId, view.file)) : { status: 'none' };
      const version =
        resource && resource.status === 'live' && resource.value
          ? `${resource.value.version}:${resource.value.bytes ?? ''}`
          : (resource?.status ?? 'none');
      const running = typeof useSession === 'function' ? useSession((s) => s && s.running === true) : false;
      const [tab, setTab] = useState('trace');
      const [opened, setOpened] = useState(() => new Map());
      // Turns with a selected row.
      const [reading, setReading] = useState(() => new Set());
      const [copied, copy] = useCopy();
      const [statusNonce, setStatusNonce] = useState(0);
      const [rulesFocus, setRulesFocus] = useState(null);
      const [layersOpen, setLayersOpen] = useState(false);
      const [notice, setNotice] = useNotice(6000);
      const rootRef = useRef(null);
      const stickRef = useRef(true);
      const uid = useId();
      const nowUs = useNow(running || snapshot.status === 'missing');
      const statusFile = useJsonFile(workspaceFiles, sessionId, view.status, useResource, statusNonce);
      // `/relay` actions reply in words; their results are read back from this status file.
      const relay = useMemo(
        () => (commands ? { commands, files: workspaceFiles, statusPath: view.status, sessionId } : undefined),
        [commands, workspaceFiles, view.status, sessionId],
      );
      const status = useMemo(() => (statusFile.status === 'ready' ? readStatus(statusFile.value) : null), [statusFile]);
      const refreshStatus = useCallback(() => setStatusNonce((value) => value + 1), []);
      const rulesApi = useMemo(
        () => ({
          openRule: (id) => {
            // Leave the bottom so arriving events do not pull the view away from the editor.
            stickRef.current = false;
            setRulesFocus({ id, at: Date.now() });
          },
        }),
        [],
      );
      const phoenixChanging =
        status !== null &&
        status.phoenix !== null &&
        (status.phoenix.state === 'starting' || status.phoenix.state === 'waiting');

      // Follow the file: every resource version (or a poll when no resource provider exists) triggers a sync.
      useEffect(() => {
        void source.sync();
      }, [source, version]);
      // Every committed session event changes `seq`; the Host appends its Relay events shortly after.
      useEffect(() => {
        if (view.seq === undefined) return undefined;
        void source.sync();
        const timer = setTimeout(() => void source.sync(), 400);
        return () => clearTimeout(timer);
      }, [source, view.seq]);
      useEffect(() => {
        if (resource && resource.status !== 'none') return undefined;
        const timer = setInterval(() => void source.sync(), running ? 1000 : 4000);
        return () => clearInterval(timer);
      }, [source, resource && resource.status, running]);
      // A file that does not exist yet, or a read that failed, is read again until it works: with a
      // live `file` resource nothing else would, since a failed read changes no file version.
      const retrying = snapshot.status === 'missing' || (snapshot.status === 'error' && snapshot.retry);
      useEffect(() => {
        if (!retrying) return undefined;
        const timer = setInterval(() => void source.sync(), 1500);
        return () => clearInterval(timer);
      }, [source, retrying]);
      // While Phoenix is starting or reconnecting, re-read the status file until it settles.
      useEffect(() => {
        if (!phoenixChanging) return undefined;
        const timer = setInterval(refreshStatus, 1500);
        return () => clearInterval(timer);
      }, [phoenixChanging, refreshStatus]);
      // Stay pinned to the bottom while new Relay events arrive, unless the reader scrolled up.
      useEffect(() => {
        const node = rootRef.current;
        const scroller = node && typeof node.closest === 'function' ? node.closest('[data-conversation-scroll]') : null;
        if (!scroller) return undefined;
        const onScroll = () => {
          stickRef.current = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop < 80;
        };
        scroller.addEventListener('scroll', onScroll, { passive: true });
        return () => scroller.removeEventListener('scroll', onScroll);
      }, []);
      useEffect(() => {
        const node = rootRef.current;
        const scroller = node && typeof node.closest === 'function' ? node.closest('[data-conversation-scroll]') : null;
        if (scroller && stickRef.current) scroller.scrollTop = scroller.scrollHeight;
      }, [snapshot.rev, tab]);

      const model = snapshot.model;
      const timeline = useMemo(() => timelineOf(model), [model, snapshot.rev]);
      const turnCount = model.turns.length;
      // A turn is open or closed as the reader last set it. One the reader has not set is open while
      // a row in it is selected, so a later turn starting does not collapse what is being read;
      // otherwise the last two turns and unfinished turns are open.
      const isOpen = (turn, position) =>
        opened.get(turn.uuid) ?? (reading.has(turn.uuid) || position >= turnCount - 2 || turn.endUs === null);
      const toggle = useCallback((turn, current) => {
        setOpened((prev) => {
          const next = new Map(prev);
          next.set(turn.uuid, !current);
          return next;
        });
      }, []);
      const markReading = useCallback((turn, on) => {
        setReading((prev) => {
          if (prev.has(turn.uuid) === on) return prev;
          const next = new Set(prev);
          if (on) next.add(turn.uuid);
          else next.delete(turn.uuid);
          return next;
        });
      }, []);

      const state = status ? status.state : undefined;
      // A failed read after events were loaded keeps them on screen under a banner.
      const readFailed = snapshot.status === 'error';
      const readError =
        readFailed && model.count > 0
          ? h(
              'div',
              { className: 'nrv-banner err nrv-trace-error', role: 'alert', 'data-nrv': 'trace-error' },
              h(Icon, { name: 'warn' }),
              h(
                'span',
                { className: 'nrv-banner-text' },
                t(snapshot.retry ? 'state.errorRetrying' : 'state.errorKept', {
                  message: snapshot.error ?? t('reply.unknown'),
                }),
              ),
            )
          : null;
      let body;
      if (snapshot.status === 'loading') body = h(StateMessage, { title: t('state.loading') });
      else if (readFailed && model.count === 0)
        body = h(StateMessage, { title: t('state.error'), body: snapshot.error, error: true });
      else if (model.count === 0) {
        body =
          state !== undefined && INACTIVE_STATES.has(state)
            ? h(StateMessage, {
                title: t('state.inactive'),
                body: (status && status.reason) ?? t('state.inactiveBody', { state }),
                error: true,
              })
            : h(StateMessage, { title: t('state.empty'), body: t('state.emptyBody') });
      } else if (tab === 'events') body = h(RawList, { model, t });
      else {
        const liveUs = Math.max(nowUs, model.lastUs);
        let position = -1;
        body = timeline.map((entry) => {
          if (entry.kind === 'loose') return h(LooseItems, { key: entry.key, items: entry.items, nowUs: liveUs, t });
          position += 1;
          const turn = entry.turn;
          return h(TurnCard, {
            key: turn.uuid,
            turn,
            rev: model.turnRev.get(turn.uuid) ?? 0,
            open: isOpen(turn, position),
            onToggle: toggle,
            onReading: markReading,
            nowUs: turn.endUs === null ? liveUs : 0,
            running: turn.endUs === null && running,
            t,
          });
        });
      }

      const layersId = `${uid}-layers`;
      const problem =
        status && status.reason && (state === 'degraded' || state === 'disabled' || state === 'not_initialized')
          ? h(
              'div',
              {
                className: `nrv-banner ${state === 'degraded' ? 'warn' : 'err'}`,
                role: 'status',
                'data-nrv': 'relay-problem',
              },
              h(Icon, { name: 'warn' }),
              h('span', { className: 'nrv-banner-text' }, t(`problem.${state}`, { reason: status.reason })),
            )
          : null;
      const stats = model.stats;
      return h(
        'div',
        { className: 'nrv-root', ref: rootRef, 'data-nemo-relay-view': '' },
        h('style', null, CSS),
        h(
          RulesContext.Provider,
          { value: rulesApi },
          h(
            'div',
            { className: 'nrv-column' },
            h(
              'div',
              { className: 'nrv-top' },
              h(
                Guard,
                { t },
                h(
                  'div',
                  { className: 'nrv-head' },
                  h('span', { className: 'nrv-title' }, h(RelayGlyph), t('title')),
                  h(StatePill, { status, statusFile, statusPath: view.status, t }),
                  status && status.relayVersion
                    ? h(
                        'span',
                        {
                          className: 'nrv-chip mono',
                          'data-nrv': 'relay-version',
                          title: [
                            t('relay.versionTitle', { version: status.relayVersion }),
                            status.nodeVersion ? t('relay.nodeTitle', { version: status.nodeVersion }) : null,
                          ]
                            .filter(Boolean)
                            .join('\n'),
                        },
                        t('relay.version', { version: status.relayVersion }),
                      )
                    : null,
                  running ? h('span', { className: 'nrv-chip live' }, t('chip.live')) : null,
                  h('span', { className: 'nrv-spacer' }),
                  status
                    ? h(PhoenixMenu, {
                        phoenix: status.phoenix,
                        sessionId,
                        relay,
                        onRefresh: refreshStatus,
                        onNotice: setNotice,
                        t,
                      })
                    : null,
                ),
                status
                  ? h(
                      'div',
                      { className: 'nrv-meta-row' },
                      h('span', { className: 'nrv-meta-label' }, t('exporters.label')),
                      h(ExporterChips, { exporters: status.exporters, t }),
                      h(RedactionBadge, { redaction: status.redaction, t }),
                      h(LayersToggle, {
                        paths: status.configPaths,
                        open: layersOpen,
                        listId: layersId,
                        onToggle: () => setLayersOpen(!layersOpen),
                        t,
                      }),
                    )
                  : null,
                status && layersOpen ? h(LayersList, { paths: status.configPaths, listId: layersId, t }) : null,
                problem,
                notice
                  ? h(
                      'div',
                      {
                        className: `nrv-banner ${notice.kind === 'err' ? 'err' : notice.kind === 'ok' ? 'ok' : 'warn'}`,
                        role: notice.kind === 'err' ? 'alert' : 'status',
                        'data-nrv': 'notice',
                      },
                      h(Icon, { name: notice.kind === 'ok' ? 'check' : 'warn' }),
                      h('span', { className: 'nrv-banner-text' }, notice.text),
                      h(
                        'button',
                        {
                          type: 'button',
                          className: 'nrv-btn icon',
                          'aria-label': t('notice.dismiss'),
                          title: t('notice.dismiss'),
                          onClick: () => setNotice(null),
                        },
                        h(Icon, { name: 'close', size: 12 }),
                      ),
                    )
                  : null,
              ),
              h(
                Guard,
                { t },
                h(RulesPanel, {
                  guardrails: status ? status.guardrails : null,
                  statusState: statusFile.status,
                  relayState: state,
                  relay,
                  // The rules belong to the Host process that writes this status file.
                  draftKey: view.status ? `status:${view.status}` : `session:${sessionId}`,
                  focus: rulesFocus,
                  onRefresh: refreshStatus,
                  t,
                }),
              ),
            ),
            h(
              'div',
              { className: 'nrv-toolbar' },
              h(
                'span',
                { className: 'nrv-seg', role: 'group', 'aria-label': t('tabs.label') },
                h(
                  'button',
                  { type: 'button', 'aria-pressed': tab === 'trace', onClick: () => setTab('trace') },
                  t('tabs.trace'),
                ),
                h(
                  'button',
                  { type: 'button', 'aria-pressed': tab === 'events', onClick: () => setTab('events') },
                  t('tabs.events'),
                ),
              ),
              model.count > 0
                ? h(
                    'div',
                    { className: 'nrv-stats' },
                    h('span', null, h('b', null, fmtCount(model.count)), ` ${plural(t, 'stats.events', model.count)}`),
                    h('span', null, h('b', null, String(turnCount)), ` ${plural(t, 'stats.turns', turnCount)}`),
                    h('span', null, h('b', null, String(stats.llm)), ` ${plural(t, 'stats.llm', stats.llm)}`),
                    h('span', null, h('b', null, String(stats.tools)), ` ${plural(t, 'stats.tools', stats.tools)}`),
                    stats.errors > 0
                      ? h(
                          'span',
                          null,
                          h('b', null, String(stats.errors)),
                          ` ${plural(t, 'stats.errors', stats.errors)}`,
                        )
                      : null,
                    stats.blocked > 0
                      ? h(
                          'span',
                          { className: 'nrv-blocked' },
                          h('b', null, String(stats.blocked)),
                          ` ${t('stats.blocked')}`,
                        )
                      : null,
                    stats.prompt + stats.completion > 0
                      ? h(
                          'span',
                          null,
                          t('stats.tokens', {
                            prompt: fmtCount(stats.prompt),
                            cached: fmtCount(stats.cached),
                            completion: fmtCount(stats.completion),
                            reasoning: fmtCount(stats.reasoning),
                          }),
                        )
                      : null,
                  )
                : null,
            ),
            h(
              'div',
              { className: 'nrv-path' },
              h('span', null, t('path.label', { version: model.version ?? '0.1' })),
              h('code', { title: view.file }, view.file),
              h(
                'button',
                { type: 'button', className: 'nrv-btn', onClick: () => copy(view.file) },
                copied ? t('path.copied') : t('path.copy'),
              ),
              h(
                'button',
                {
                  type: 'button',
                  className: 'nrv-btn',
                  onClick: () => {
                    void source.sync();
                    refreshStatus();
                  },
                },
                t('path.reload'),
              ),
            ),
            readError,
            h(Guard, { t }, body),
          ),
        ),
      );
    }

    /**
     * Until the Host projection reaches this page (e.g. a page loaded before the bundle was
     * installed), derive the default trace location from the Host home the Remote reports.
     * Each Host process writes its own status file, so the fallback has no status path and
     * the header shows the status as missing until the projection names it.
     */
    function fallbackView(sessionId, hostHome) {
      if (typeof hostHome !== 'string' || hostHome.length === 0 || typeof sessionId !== 'string') return undefined;
      const sep = hostHome.includes('\\') && !hostHome.includes('/') ? '\\' : '/';
      const dataDir = [hostHome.replace(/[\\/]+$/, ''), '.dsh', 'nemo-relay'].join(sep);
      const stem = sessionId.replace(/[^A-Za-z0-9._-]/g, '_');
      return {
        v: 3,
        sessionId,
        file: `${dataDir}${sep}view${sep}${stem}.atof.jsonl`,
        fallback: true,
      };
    }

    const NO_LOCALE = { subscribe: () => () => undefined, getSnapshot: () => undefined };

    /**
     * The translate function with a new identity whenever the locale service changes (a language
     * switch, or a dictionary registered late). `ctx.locale.bind` returns one function for the life
     * of the page, so a memoized component that receives it would keep the old language.
     */
    function useLocalized(translate, localeSource) {
      const source = localeSource ?? NO_LOCALE;
      const revision = useSyncExternalStore(source.subscribe, source.getSnapshot);
      return useMemo(() => (key, params) => translate(key, params), [translate, revision]);
    }

    function RelayView(props) {
      const { sessionId, useProjection, workspaceFiles, hostHome } = props;
      const t = useLocalized(props.t, props.localeSource);
      const projected = typeof useProjection === 'function' ? useProjection(PROJECTION_KEY) : undefined;
      const view =
        asRecord(projected) && typeof projected.file === 'string' ? projected : fallbackView(sessionId, hostHome);
      const frame = (child) =>
        h('div', { className: 'nrv-root' }, h('style', null, CSS), h('div', { className: 'nrv-column' }, child));
      if (!view || typeof view.file !== 'string')
        return frame(h(StateMessage, { title: t('state.unavailable'), body: t('state.unavailableBody') }));
      if (!workspaceFiles)
        return frame(h(StateMessage, { title: t('state.error'), body: t('state.noRemote'), error: true }));
      return h(
        Guard,
        { t, frame: true },
        h(TracePanel, {
          key: `${sessionId}:${view.file}`,
          sessionId,
          view,
          workspaceFiles,
          commands: props.commands,
          useResource: props.useResource,
          useSession: props.useSession,
          t,
        }),
      );
    }

    /* ------------------------------------------------------------------ */
    /* Locale                                                               */
    /* ------------------------------------------------------------------ */

    const EN = {
      tab: 'NeMo Relay',
      title: 'NeMo Relay',
      'chip.live': 'live',
      'relay.ready': 'Ready',
      'relay.degraded': 'Degraded',
      'relay.not_initialized': 'Not initialized',
      'relay.disabled': 'Disabled',
      'relay.stopping': 'Stopping',
      'relay.stopped': 'Stopped',
      'relay.other': 'State: {state}',
      'relay.unknown': 'State unknown',
      'relay.version': 'Relay {version}',
      'relay.versionTitle': 'nemo-relay-node {version}',
      'relay.nodeTitle': 'Node.js {version}',
      'relay.hostReady': 'Plugin host initialized',
      'relay.hostMissing': 'Plugin host not initialized',
      'relay.counters': 'Turns: {opened} opened, {closed} closed',
      'relay.llmCalls.one': '{n} LLM call',
      'relay.llmCalls.other': '{n} LLM calls',
      'relay.toolCalls.one': '{n} tool call',
      'relay.toolCalls.other': '{n} tool calls',
      'relay.rejections.one': '{n} guardrail rejection',
      'relay.rejections.other': '{n} guardrail rejections',
      'relay.marks.one': '{n} mark',
      'relay.marks.other': '{n} marks',
      'relay.middlewareErrors.one': '{n} middleware error',
      'relay.middlewareErrors.other': '{n} middleware errors',
      'relay.sessions.one': '{n} active session',
      'relay.sessions.other': '{n} active sessions',
      'relay.updated': 'Status written {time}',
      'problem.degraded': 'NeMo Relay is degraded: {reason}',
      'problem.disabled': 'NeMo Relay is disabled: {reason}',
      'problem.not_initialized': 'NeMo Relay is not initialized: {reason}',
      'status.missing': 'No status',
      'status.missingTitle': 'NeMo Relay has not written its status file yet: {path}',
      'status.unreadable': 'Status unreadable',
      'status.unreadableTitle': 'The NeMo Relay status file could not be read: {message}',
      'exporters.label': 'Exports',
      'exporters.none': 'none',
      'exporters.noneTitle':
        'No NeMo Relay exporter is configured. The view’s own ATOF trace below stays on this computer.',
      'exporters.enabled': '{name} exporter enabled',
      'exporters.degraded': '{name} exporter degraded: the NeMo Relay plugin host failed',
      'redaction.on': 'Secrets redacted',
      'redaction.title': 'NeMo Relay’s pii_redaction component removes credentials from every Relay trace and export.',
      'redaction.detectors': 'Detectors: {list}',
      'redaction.byDefault': 'Added by this plugin’s default (redaction.secrets).',
      'redaction.byPlugins': 'Configured in the NeMo Relay plugin configuration.',
      'redaction.off': 'redaction off',
      'redaction.offTitle':
        'No NeMo Relay pii_redaction policy is active, so traces and exports keep credentials as they are.',
      'redaction.partial': 'redaction: no credential detectors',
      'redaction.partialTitle':
        'A NeMo Relay pii_redaction policy is active, but none of its detectors looks for credentials.',
      'layers.one': 'plugins.toml',
      'layers.n': 'plugins.toml ×{n}',
      'layers.none': 'no plugins.toml',
      'layers.title': 'NeMo Relay configuration layers',
      'layers.systemShort': 'operator',
      'layers.system': 'system layer (operator-managed)',
      'layers.user': 'user layer',
      'layers.explain':
        'NeMo Relay merges these plugins.toml layers, system first, with this plugin’s own configuration. Exporters in a system layer were set up by an operator.',
      'layers.noneExplain': 'No plugins.toml layer was found; NeMo Relay uses only this plugin’s own configuration.',
      'notice.dismiss': 'Dismiss',
      'copy.label': 'Copy',
      'copy.done': 'Copied',
      'copy.title': 'Copy to the clipboard',
      'crash.body': 'This part of the NeMo Relay view failed to render: {message}',
      'crash.retry': 'Try again',
      'reply.unavailable':
        'This NeMo Relay plugin did not answer the request. Update or reload the nemo-relay-deepseek-harness plugin and try again.',
      'reply.notJson': 'NeMo Relay sent a reply this view does not understand; update the plugin.',
      'reply.notJsonText': 'NeMo Relay sent a reply this view does not understand: {text}',
      'reply.noResult':
        'NeMo Relay did not report the result of this request in its status file; reload the view and check.',
      'reply.noResultText':
        'NeMo Relay answered “{text}”, but did not report the result in its status file; reload the view and check.',
      'reply.transport': 'The request did not reach NeMo Relay: {message}',
      'reply.timeout':
        'NeMo Relay did not answer within {seconds} seconds. The request may still have taken effect; reload to check.',
      'reply.invalid': 'NeMo Relay rejected the request as invalid.',
      'reply.invalidText': 'NeMo Relay rejected the change: {message}',
      'reply.unknown': 'unknown error',
      'rules.title': 'Guardrail rules',
      'rules.sub':
        'NeMo Relay checks every tool call against these rules before it runs. The first enabled rule that matches blocks the call and tells the agent why. You can edit the rules here or ask the agent; changes that loosen a rule need your approval.',
      'rules.loading': 'Loading rules…',
      'rules.unknown': 'Rules unavailable: NeMo Relay has not written its status file.',
      'rules.notReported': 'This version of the NeMo Relay plugin does not report guardrail rules.',
      'rules.none': 'No rules: every call DeepSeek Harness allows runs.',
      'rules.noneEditor': 'No rules yet.',
      'rules.ofTotal': '/{total} on',
      'rules.onTotalTitle': '{on} of {total} rules are on',
      'rules.failOpen': 'fail-open',
      'rules.failClosed': 'fail-closed',
      'rules.failOpenTitle':
        'If NeMo Relay cannot evaluate the rules for a call, the call runs. Change it with the fail-closed switch below the rules.',
      'rules.failClosedTitle':
        'If NeMo Relay cannot evaluate the rules for a call, the call is blocked. Change it with the fail-closed switch below the rules.',
      'rules.disabledTag': 'not enforced',
      'rules.disabledBody':
        'Guardrails are turned off in the plugin configuration (guardrails.enabled: false), so these rules are not enforced.',
      'rules.notEnforced': 'NeMo Relay is not running ({state}), so these rules are not enforced right now.',
      'rules.safetyOff': 'safety rule off',
      'rules.safetyOffTitle':
        'The shipped safety rule “{id}” is turned off or removed. Open the rules and use Reset to defaults to restore it.',
      'rules.safetyDraftOff':
        'Your changes turn off or remove the shipped safety rule “{id}”, which blocks recursive deletion of / and your home directory. Save only if you mean it; Reset to defaults restores it.',
      'rules.unsaved': 'unsaved changes',
      'rules.rejections': '{n} blocked',
      'rules.rejectionsTitle': 'Tool calls NeMo Relay has blocked since it started',
      'rules.more': '+{n} more',
      'rules.isOff': 'Off',
      'rules.unnamed': 'unnamed',
      'rules.toggle': 'Turn rule {id} on or off',
      'rules.expandRule': 'Edit rule {id}',
      'rules.collapseRule': 'Close rule {id}',
      'rules.hasErrors': 'This rule has errors and cannot be saved',
      'rules.anyTool': 'any tool',
      'rules.anyArgs': 'any arguments',
      'rules.f.name': 'Name',
      'rules.f.tool': 'Tools',
      'rules.f.match': 'Block when the arguments match',
      'rules.f.reason': 'Message to the agent',
      'rules.hint.name': 'Letters, digits, dots, dashes, and underscores',
      'rules.hint.tool': 'Tool names separated by commas, a /regex/, or empty for any tool',
      'rules.hint.match': 'A regular expression, tested against the arguments as JSON and each text argument',
      'rules.hint.reason': 'Shown to the agent when this rule blocks a call. Empty uses “{text}”',
      'rules.reasonPlaceholder': 'Default message',
      'rules.matchCase': 'Match case',
      'rules.blocksAll': 'No pattern: this rule blocks every call to {tool}',
      'rules.delete': 'Delete rule',
      'rules.add': 'Add rule',
      'rules.save': 'Save',
      'rules.saving': 'Saving…',
      'rules.saved': 'Saved. NeMo Relay applies the rules to the next tool call.',
      'rules.failClosedSwitch': 'Block calls the rules cannot check (fail-closed)',
      'rules.failOpenLabel': 'Fail-open: calls run when the rules cannot be checked',
      'rules.failClosedLabel': 'Fail-closed: calls are blocked when the rules cannot be checked',
      'rules.failOpenSaved': 'Fail mode is now open: a call the rules cannot check runs.',
      'rules.failClosedSaved': 'Fail mode is now closed: a call the rules cannot check is blocked.',
      'rules.shortcut': 'Save (⌘S / Ctrl+S)',
      'rules.fixFirst.one': 'Fix the rule with errors first',
      'rules.fixFirst.other': 'Fix the {n} rules with errors first',
      'rules.reloadFirst': 'Reload, or keep your changes, first',
      'rules.discard': 'Discard',
      'rules.reset': 'Reset to defaults',
      'rules.resetTitle': 'Replace all rules with the shipped default rules',
      'rules.resetting': 'Resetting…',
      'rules.resetConfirm': 'Replace all rules with the shipped defaults?',
      'rules.resetDo': 'Reset',
      'rules.cancel': 'Cancel',
      'rules.restored': 'Restored the shipped default rules.',
      'rules.stale': 'Rules changed elsewhere — reload',
      'rules.staleBody': 'Rules changed elsewhere — reload',
      'rules.reload': 'Reload',
      'rules.keepMine': 'Keep my changes',
      'rules.missingRule': 'Rule “{id}” is not in the current rules; it may have been renamed or deleted.',
      'rules.readOnly': 'Read-only: NeMo Relay reports that these rules cannot be edited from this view.',
      'rules.noCommands': 'Editing rules needs the DeepSeek Harness commands service, which this page does not have.',
      'rules.source.config': 'Saved in the plugin configuration',
      'rules.source.file': 'Saved in rules.json in the NeMo Relay data folder',
      'rules.source.default': 'Shipped defaults',
      'rules.revision': 'Revision {rev}',
      'rules.location': 'Stored in {location}',
      'rules.slow': 'Timed out {n}× while checking calls; simplify the pattern',
      'rules.skipped.one': 'NeMo Relay skips 1 rule with problems: {list}',
      'rules.skipped.other': 'NeMo Relay skips {n} rules with problems: {list}',
      'rules.err.long': 'Longer than {n} characters',
      'rules.err.nested': 'Repeats a repeating group, like (a+)+, which can stall the app; simplify it',
      'rules.err.regex': 'Not a valid regular expression: {detail}',
      'rules.err.toolName': '“{name}” is not a tool name; use names like bash, a comma list, or /regex/',
      'rules.err.duplicate': 'Another rule is already named {id}',
      'rules.err.everything': 'This rule would block every tool call; name a tool or add a pattern',
      'rules.err.tooMany': 'At most {n} rules are allowed',
      'rules.try': 'Try a call',
      'rules.try.tool': 'Tool name',
      'rules.try.args': 'Command or JSON arguments',
      'rules.try.placeholder': 'git push origin main --force',
      'rules.try.run': 'Try',
      'rules.try.hint':
        'Enter a command or JSON arguments to ask NeMo Relay whether the saved rules would block the call.',
      'rules.try.offline': 'Trying calls needs the DeepSeek Harness commands service.',
      'rules.try.checking': 'Asking NeMo Relay…',
      'rules.try.blockedBy': 'Blocked by rule',
      'rules.try.blocked': 'Blocked',
      'rules.try.allowed': 'Allowed by the saved rules',
      'rules.try.unknown': 'NeMo Relay answered without a verdict.',
      'rules.try.show': 'Show rule',
      'rules.try.draftBlocked': 'With your unsaved changes, rule {id} would block it.',
      'rules.try.draftAllowed': 'With your unsaved changes, it would be allowed.',
      'rules.try.draftUnsure': 'With your unsaved changes, no rule previewed here would block it.',
      'rules.try.tooLong':
        'Unsaved changes are not previewed here when the arguments are longer than {n} characters as JSON.',
      'rules.try.saveFirst': 'Save the rules, then use Try to ask NeMo Relay.',
      'rules.try.skipped': 'Not previewed here because they can run slowly: {ids}.',
      'rules.try.timedOut': 'Not checked in time, so they did not block: {ids}',
      'phoenix.menu': 'Phoenix options',
      'phoenix.start': 'Start Phoenix',
      'phoenix.retry': 'Retry Phoenix',
      'phoenix.open': 'Open in Phoenix',
      'phoenix.starting': 'Starting Phoenix…',
      'phoenix.stopping': 'Stopping…',
      'phoenix.waiting': 'Waiting for Phoenix…',
      'phoenix.startHint':
        'Connect to Arize Phoenix at {address}, or start a local one, and send this session’s traces to it',
      'phoenix.openHint': 'Open this session in Phoenix at {address}',
      'phoenix.openSession': 'Open this session in Phoenix',
      'phoenix.openPhoenix': 'Open Phoenix',
      'phoenix.copyUrl': 'Copy Phoenix address',
      'phoenix.copied': 'Copied',
      'phoenix.stop': 'Stop Phoenix',
      'phoenix.disconnect': 'Disconnect',
      'phoenix.cancel': 'Stop starting',
      'phoenix.stopWaiting': 'Stop waiting',
      'phoenix.runningAt': 'Phoenix{version} · {address}',
      'phoenix.owned': 'Started by NeMo Relay. Stop shuts it down.',
      'phoenix.external':
        'Already running outside DeepSeek Harness; NeMo Relay only connects to it and never stops it.',
      'phoenix.project': 'Project: {project}',
      'phoenix.startingBody':
        'Launching Arize Phoenix on this computer. The first start downloads it, which can take a minute or two.',
      'phoenix.waitingTitle': 'Waiting for Phoenix',
      'phoenix.waitingBody': 'Phoenix at {address} is not answering. NeMo Relay reconnects as soon as it is back.',
      'phoenix.failedTitle': 'Phoenix could not start',
      'phoenix.notConnected': 'Phoenix is not connected',
      'phoenix.firstRun':
        'Start connects to an Arize Phoenix already running at {address}, or launches a local one (the first launch downloads it). NeMo Relay then exports traces to it with its OpenInference exporter.',
      'phoenix.local': 'Traces go only to this Phoenix on this computer.',
      'phoenix.remote': 'Traces are sent to {address}.',
      'phoenix.failed': 'Phoenix: {message}',
      'phoenix.openFailed': 'Could not open {url}',
      'phoenix.connect': 'Connect Phoenix',
      'phoenix.connecting': 'Connecting…',
      'phoenix.notLinked': 'Running, but NeMo Relay is not sending it traces.',
      'phoenix.notLinkedBody':
        'Phoenix at {address} is running, but NeMo Relay is not sending it traces. Connect to resume.',
      'phoenix.downloading': 'Downloading Phoenix…',
      'phoenix.downloadingBody': 'Downloading Arize Phoenix for its first start; this can take a minute or two.',
      'phoenix.copyLog': 'Copy Phoenix log path',
      'tabs.label': 'Trace presentation',
      'tabs.trace': 'Trace',
      'tabs.events': 'ATOF events',
      'stats.events.one': 'event',
      'stats.events.other': 'events',
      'stats.turns.one': 'turn',
      'stats.turns.other': 'turns',
      'stats.llm.one': 'LLM call',
      'stats.llm.other': 'LLM calls',
      'stats.tools.one': 'tool call',
      'stats.tools.other': 'tool calls',
      'stats.errors.one': 'error',
      'stats.errors.other': 'errors',
      'stats.blocked': 'blocked by NeMo Relay',
      'stats.tokens': 'tokens in {prompt} ({cached} cached) · out {completion} ({reasoning} reasoning)',
      'path.label': 'ATOF {version} file',
      'path.copy': 'Copy path',
      'path.copied': 'Copied',
      'path.reload': 'Reload',
      'turn.title': 'Turn {n}',
      'turn.untitled': 'Turn',
      'turn.llm.one': '{n} LLM',
      'turn.llm.other': '{n} LLM',
      'turn.tools.one': '{n} tool',
      'turn.tools.other': '{n} tools',
      'turn.errors.one': '{n} error',
      'turn.errors.other': '{n} errors',
      'turn.blocked': '{n} blocked',
      'turn.noPrompt': 'No prompt recorded',
      'turn.awaiting': 'Waiting for the next model or tool call…',
      'turn.answer': 'Final assistant text recorded on the turn scope',
      'turn.error': 'Error recorded on the turn scope',
      'turn.moreRows.one': 'Show 1 earlier row',
      'turn.moreRows.other': 'Show {n} earlier rows',
      'axis.scope': 'Relay scope',
      'axis.duration': 'Duration',
      'row.llm': 'LLM call',
      'row.blocked': 'blocked',
      'row.ttft': 'First chunk after {ttft}; the lighter part is the wait',
      'status.ok': 'ok',
      'status.error': 'error',
      'status.open': 'open',
      'status.running': 'running',
      'guard.title.one': 'Passed 1 NeMo Relay guardrail evaluation:',
      'guard.title.other': 'Passed {n} NeMo Relay guardrail evaluations:',
      'guard.label': 'Guardrails passed',
      'block.icon': 'NeMo Relay guardrail',
      'block.tool': 'tool call blocked by NeMo Relay',
      'block.llm': 'model call blocked by NeMo Relay',
      'block.targetTool': 'tool',
      'block.targetLlm': 'model',
      'block.byRule': 'Blocked by rule',
      'block.editRule': 'Edit rule',
      'block.guardrail': 'Guardrail',
      'block.target': 'Target',
      'block.reason': 'Reason',
      'block.evaluated': 'Evaluation time',
      'block.unknown': 'NeMo Relay guardrail',
      'block.noReason': 'no reason given',
      'mark.user_message': 'User message',
      'mark.approval_asked': 'Approval requested',
      'mark.approval_decided': 'Approval decided',
      'mark.subagent': 'Subagent',
      'mark.request_edit_dropped': 'Request edit dropped',
      'mark.compaction': 'Compaction',
      'mark.notCaptured': 'text not captured',
      'subagent.child': 'Child session',
      'subagent.childTitle': 'Child session {id}',
      'subagent.label': 'Label',
      'subagent.mode': 'Mode',
      'loose.title.one': '1 event outside turns',
      'loose.title.other': '{n} events outside turns',
      'raw.more': 'Show earlier events ({n} hidden)',
      'raw.capped.one': 'Only the newest events are kept in this list; 1 older event is in the ATOF file.',
      'raw.capped.other': 'Only the newest events are kept in this list; {count} older events are in the ATOF file.',
      'detail.atof': 'ATOF events',
      'detail.mark': 'Mark',
      'detail.time': 'Time',
      'detail.uuid': 'UUID',
      'detail.data': 'Data',
      'detail.model': 'Model',
      'detail.provider': 'Provider',
      'detail.step': 'Step',
      'detail.purpose': 'Purpose',
      'detail.status': 'Status',
      'detail.duration': 'Duration',
      'detail.ttft': 'Time to first chunk',
      'detail.tokens': 'Tokens',
      'detail.tokenLine': '{prompt} in ({cached} cached) · {completion} out ({reasoning} reasoning)',
      'detail.finish': 'Finish reason',
      'detail.finishTitle': 'Finish reason: {reason}',
      'detail.request': 'Request',
      'detail.messages.one': '{n} message',
      'detail.messages.other': '{n} messages',
      'detail.tools.one': '{n} tool',
      'detail.tools.other': '{n} tools',
      'detail.requestNotCaptured': 'not captured (capture.includePrompts is off)',
      'detail.responseNotCaptured': 'Response not captured (capture.includeResponses is off)',
      'detail.lastUser': 'Last user message',
      'detail.toolResults': 'Tool results sent with this request ({n})',
      'detail.reasoning': 'Reasoning',
      'detail.response': 'Response',
      'detail.toolCalls': 'Tool calls',
      'detail.tool': 'Tool',
      'detail.callId': 'Tool call ID',
      'detail.parentCall': 'Parent tool call',
      'detail.command': 'Command',
      'detail.arguments': 'Arguments',
      'detail.argsStripped': 'Argument values not captured (capture.stripToolArgs); keys: {keys}',
      'detail.result': 'Result',
      'detail.resultError': 'Error result',
      'detail.resultValue': 'Result value',
      'detail.resultStripped': 'Result content not captured (capture.stripToolResults)',
      'state.loading': 'Loading the NeMo Relay trace…',
      'state.empty': 'No NeMo Relay events yet',
      'state.emptyBody':
        'Send a message; NeMo Relay records every turn, model call, tool call, and guardrail decision of this session here.',
      'state.inactive': 'NeMo Relay is not active',
      'state.inactiveBody': 'The NeMo Relay runtime in the Host is {state}; no calls are routed through Relay.',
      'state.error': 'Could not read the NeMo Relay trace',
      'state.errorRetrying':
        'Could not read new NeMo Relay events ({message}). Showing the events loaded so far and trying again…',
      'state.errorKept':
        'Could not read the rest of the NeMo Relay trace ({message}). Showing the events loaded so far.',
      'state.noRemote': 'The workspaceFiles Remote is not available in this client.',
      'state.unavailable': 'The NeMo Relay integration is not active',
      'state.unavailableBody':
        'The Host half of nemo-relay-deepseek-harness is not loaded in this profile, or its view is disabled.',
    };

    const ZH = {
      tab: 'NeMo Relay',
      title: 'NeMo Relay',
      'chip.live': '实时',
      'relay.ready': '就绪',
      'relay.degraded': '已降级',
      'relay.not_initialized': '未初始化',
      'relay.disabled': '已禁用',
      'relay.stopping': '正在停止',
      'relay.stopped': '已停止',
      'relay.other': '状态：{state}',
      'relay.unknown': '状态未知',
      'relay.version': 'Relay {version}',
      'relay.versionTitle': 'nemo-relay-node {version}',
      'relay.nodeTitle': 'Node.js {version}',
      'relay.hostReady': '插件宿主已初始化',
      'relay.hostMissing': '插件宿主未初始化',
      'relay.counters': '轮次：已开始 {opened}，已结束 {closed}',
      'relay.llmCalls.one': '{n} 次 LLM 调用',
      'relay.llmCalls.other': '{n} 次 LLM 调用',
      'relay.toolCalls.one': '{n} 次工具调用',
      'relay.toolCalls.other': '{n} 次工具调用',
      'relay.rejections.one': '{n} 次防护拦截',
      'relay.rejections.other': '{n} 次防护拦截',
      'relay.marks.one': '{n} 个标记',
      'relay.marks.other': '{n} 个标记',
      'relay.middlewareErrors.one': '{n} 个中间件错误',
      'relay.middlewareErrors.other': '{n} 个中间件错误',
      'relay.sessions.one': '{n} 个活动会话',
      'relay.sessions.other': '{n} 个活动会话',
      'relay.updated': '状态写入于 {time}',
      'problem.degraded': 'NeMo Relay 已降级：{reason}',
      'problem.disabled': 'NeMo Relay 已禁用：{reason}',
      'problem.not_initialized': 'NeMo Relay 未初始化：{reason}',
      'status.missing': '无状态',
      'status.missingTitle': 'NeMo Relay 尚未写入状态文件：{path}',
      'status.unreadable': '状态无法读取',
      'status.unreadableTitle': '无法读取 NeMo Relay 状态文件：{message}',
      'exporters.label': '导出',
      'exporters.none': '无',
      'exporters.noneTitle': '未配置 NeMo Relay 导出器。下方视图自己的 ATOF 轨迹只保存在本机。',
      'exporters.enabled': '已启用 {name} 导出器',
      'exporters.degraded': '{name} 导出器已降级：NeMo Relay 插件宿主运行失败',
      'redaction.on': '已脱敏密钥',
      'redaction.title': 'NeMo Relay 的 pii_redaction 组件会从所有 Relay 轨迹和导出数据中移除凭据。',
      'redaction.detectors': '检测器：{list}',
      'redaction.byDefault': '由此插件的默认设置添加（redaction.secrets）。',
      'redaction.byPlugins': '在 NeMo Relay 插件配置中设置。',
      'redaction.off': '未脱敏',
      'redaction.offTitle': '当前没有生效的 NeMo Relay pii_redaction 策略，轨迹和导出数据中的凭据会保持原样。',
      'redaction.partial': '脱敏：无凭据检测器',
      'redaction.partialTitle': '已启用 NeMo Relay pii_redaction 策略，但其检测器都不检测凭据。',
      'layers.one': 'plugins.toml',
      'layers.n': 'plugins.toml ×{n}',
      'layers.none': '无 plugins.toml',
      'layers.title': 'NeMo Relay 配置层',
      'layers.systemShort': '运维',
      'layers.system': '系统层（由运维管理）',
      'layers.user': '用户层',
      'layers.explain':
        'NeMo Relay 会先合并系统层、再合并用户层的 plugins.toml，并叠加此插件自己的配置。系统层中的导出器由运维人员设置。',
      'layers.noneExplain': '未找到 plugins.toml 配置层；NeMo Relay 仅使用此插件自己的配置。',
      'notice.dismiss': '关闭',
      'copy.label': '复制',
      'copy.done': '已复制',
      'copy.title': '复制到剪贴板',
      'crash.body': 'NeMo Relay 视图的这一部分渲染失败：{message}',
      'crash.retry': '重试',
      'reply.unavailable': '此 NeMo Relay 插件未响应该请求。请更新或重新加载 nemo-relay-deepseek-harness 插件后重试。',
      'reply.notJson': 'NeMo Relay 返回了此视图无法识别的回复；请更新插件。',
      'reply.notJsonText': 'NeMo Relay 返回了此视图无法识别的回复：{text}',
      'reply.noResult': 'NeMo Relay 未在状态文件中报告此请求的结果；请重新加载视图后检查。',
      'reply.noResultText': 'NeMo Relay 回复了“{text}”，但未在状态文件中报告结果；请重新加载视图后检查。',
      'reply.transport': '请求未能送达 NeMo Relay：{message}',
      'reply.timeout': 'NeMo Relay 在 {seconds} 秒内没有回应。请求可能已经生效；请重新加载后检查。',
      'reply.invalid': 'NeMo Relay 认为该请求无效，已拒绝。',
      'reply.invalidText': 'NeMo Relay 拒绝了此更改：{message}',
      'reply.unknown': '未知错误',
      'rules.title': '防护规则',
      'rules.sub':
        'NeMo Relay 会在每次工具调用执行前按这些规则检查。第一条匹配的已启用规则会拦截该调用，并告诉智能体原因。你可以在这里编辑规则，也可以让智能体修改；放宽规则的更改需要你批准。',
      'rules.loading': '正在加载规则…',
      'rules.unknown': '规则不可用：NeMo Relay 尚未写入状态文件。',
      'rules.notReported': '此版本的 NeMo Relay 插件不报告防护规则。',
      'rules.none': '暂无规则：DeepSeek Harness 允许的调用都会执行。',
      'rules.noneEditor': '暂无规则。',
      'rules.ofTotal': '/{total} 已启用',
      'rules.onTotalTitle': '已启用 {on} 条规则，共 {total} 条',
      'rules.failOpen': '失败放行',
      'rules.failClosed': '失败拦截',
      'rules.failOpenTitle':
        '如果 NeMo Relay 无法评估某次调用的规则，该调用会继续执行。可使用规则下方的“失败拦截”开关更改。',
      'rules.failClosedTitle':
        '如果 NeMo Relay 无法评估某次调用的规则，该调用会被拦截。可使用规则下方的“失败拦截”开关更改。',
      'rules.disabledTag': '未生效',
      'rules.disabledBody': '插件配置中已关闭防护（guardrails.enabled: false），这些规则不会生效。',
      'rules.notEnforced': 'NeMo Relay 未运行（{state}），这些规则当前不会生效。',
      'rules.safetyOff': '安全规则已关闭',
      'rules.safetyOffTitle': '内置安全规则“{id}”已关闭或被删除。打开规则并使用“恢复默认”即可还原。',
      'rules.safetyDraftOff':
        '你的更改会关闭或删除内置安全规则“{id}”，该规则用于阻止递归删除 / 和主目录。请确认后再保存；“恢复默认”可将其还原。',
      'rules.unsaved': '有未保存的更改',
      'rules.rejections': '已拦截 {n} 次',
      'rules.rejectionsTitle': 'NeMo Relay 启动以来拦截的工具调用次数',
      'rules.more': '还有 {n} 条',
      'rules.isOff': '已关闭',
      'rules.unnamed': '未命名',
      'rules.toggle': '开启或关闭规则 {id}',
      'rules.expandRule': '编辑规则 {id}',
      'rules.collapseRule': '收起规则 {id}',
      'rules.hasErrors': '此规则有错误，无法保存',
      'rules.anyTool': '任意工具',
      'rules.anyArgs': '任意参数',
      'rules.f.name': '名称',
      'rules.f.tool': '工具',
      'rules.f.match': '参数匹配时拦截',
      'rules.f.reason': '告知智能体的信息',
      'rules.hint.name': '字母、数字、点、连字符和下划线',
      'rules.hint.tool': '用逗号分隔的工具名、/正则/，留空表示任意工具',
      'rules.hint.match': '正则表达式，会与 JSON 形式的参数及每个文本参数比对',
      'rules.hint.reason': '此规则拦截调用时显示给智能体。留空则使用“{text}”',
      'rules.reasonPlaceholder': '默认信息',
      'rules.matchCase': '区分大小写',
      'rules.blocksAll': '未设置匹配模式：此规则会拦截对 {tool} 的所有调用',
      'rules.delete': '删除规则',
      'rules.add': '添加规则',
      'rules.save': '保存',
      'rules.saving': '正在保存…',
      'rules.saved': '已保存。NeMo Relay 会从下一次工具调用起应用这些规则。',
      'rules.failClosedSwitch': '拦截规则无法检查的调用（失败拦截）',
      'rules.failOpenLabel': '失败放行：规则无法检查时，调用照常运行',
      'rules.failClosedLabel': '失败拦截：规则无法检查时，调用会被拦截',
      'rules.failOpenSaved': '失败模式已改为放行：规则无法检查的调用会运行。',
      'rules.failClosedSaved': '失败模式已改为拦截：规则无法检查的调用会被拦截。',
      'rules.shortcut': '保存（⌘S / Ctrl+S）',
      'rules.fixFirst.one': '请先修正有错误的规则',
      'rules.fixFirst.other': '请先修正 {n} 条有错误的规则',
      'rules.reloadFirst': '请先重新加载，或保留你的更改',
      'rules.discard': '放弃更改',
      'rules.reset': '恢复默认',
      'rules.resetTitle': '用内置的默认规则替换全部规则',
      'rules.resetting': '正在恢复…',
      'rules.resetConfirm': '用内置的默认规则替换全部规则？',
      'rules.resetDo': '恢复',
      'rules.cancel': '取消',
      'rules.restored': '已恢复内置的默认规则。',
      'rules.stale': '规则已在别处更改——请重新加载',
      'rules.staleBody': '规则已在别处更改——请重新加载',
      'rules.reload': '重新加载',
      'rules.keepMine': '保留我的更改',
      'rules.missingRule': '当前规则中没有“{id}”；它可能已被重命名或删除。',
      'rules.readOnly': '只读：NeMo Relay 报告这些规则不能在此视图中编辑。',
      'rules.noCommands': '编辑规则需要 DeepSeek Harness 的命令服务，但此页面没有该服务。',
      'rules.source.config': '保存在插件配置中',
      'rules.source.file': '保存在 NeMo Relay 数据目录的 rules.json 中',
      'rules.source.default': '内置默认规则',
      'rules.revision': '版本 {rev}',
      'rules.location': '存储位置：{location}',
      'rules.slow': '检查调用时已超时 {n} 次；请简化匹配模式',
      'rules.skipped.one': 'NeMo Relay 跳过了 1 条有问题的规则：{list}',
      'rules.skipped.other': 'NeMo Relay 跳过了 {n} 条有问题的规则：{list}',
      'rules.err.long': '超过 {n} 个字符',
      'rules.err.nested': '重复了一个本身会重复的分组（如 (a+)+），可能导致应用卡顿；请简化',
      'rules.err.regex': '不是有效的正则表达式：{detail}',
      'rules.err.toolName': '“{name}”不是工具名；请使用 bash 这样的名称、逗号列表或 /正则/',
      'rules.err.duplicate': '已有名为 {id} 的规则',
      'rules.err.everything': '此规则会拦截所有工具调用；请指定工具或添加匹配模式',
      'rules.err.tooMany': '最多允许 {n} 条规则',
      'rules.try': '试一试',
      'rules.try.tool': '工具名',
      'rules.try.args': '命令或 JSON 参数',
      'rules.try.placeholder': 'git push origin main --force',
      'rules.try.run': '检查',
      'rules.try.hint': '输入命令或 JSON 参数，询问 NeMo Relay 已保存的规则是否会拦截该调用。',
      'rules.try.offline': '检查调用需要 DeepSeek Harness 的命令服务。',
      'rules.try.checking': '正在询问 NeMo Relay…',
      'rules.try.blockedBy': '拦截规则',
      'rules.try.blocked': '会被拦截',
      'rules.try.allowed': '已保存的规则允许此调用',
      'rules.try.unknown': 'NeMo Relay 的回复中没有结论。',
      'rules.try.show': '查看规则',
      'rules.try.draftBlocked': '按你未保存的更改，规则 {id} 会拦截此调用。',
      'rules.try.draftAllowed': '按你未保存的更改，此调用会被允许。',
      'rules.try.draftUnsure': '按你未保存的更改，此处预览的规则都不会拦截此调用。',
      'rules.try.tooLong': '参数的 JSON 形式超过 {n} 个字符时，此处不预览未保存的更改。',
      'rules.try.saveFirst': '请先保存规则，再点“检查”询问 NeMo Relay。',
      'rules.try.skipped': '以下规则可能运行较慢，此处不预览：{ids}。',
      'rules.try.timedOut': '以下规则未能及时完成检查，因此没有拦截：{ids}',
      'phoenix.menu': 'Phoenix 选项',
      'phoenix.start': '启动 Phoenix',
      'phoenix.retry': '重试 Phoenix',
      'phoenix.open': '在 Phoenix 中打开',
      'phoenix.starting': '正在启动 Phoenix…',
      'phoenix.stopping': '正在停止…',
      'phoenix.waiting': '正在等待 Phoenix…',
      'phoenix.startHint': '连接 {address} 上的 Arize Phoenix（或在本机启动一个），并把此会话的轨迹发送过去',
      'phoenix.openHint': '在 {address} 的 Phoenix 中打开此会话',
      'phoenix.openSession': '在 Phoenix 中打开此会话',
      'phoenix.openPhoenix': '打开 Phoenix',
      'phoenix.copyUrl': '复制 Phoenix 地址',
      'phoenix.copied': '已复制',
      'phoenix.stop': '停止 Phoenix',
      'phoenix.disconnect': '断开连接',
      'phoenix.cancel': '停止启动',
      'phoenix.stopWaiting': '停止等待',
      'phoenix.runningAt': 'Phoenix{version} · {address}',
      'phoenix.owned': '由 NeMo Relay 启动。“停止”会将其关闭。',
      'phoenix.external': '已在 DeepSeek Harness 之外运行；NeMo Relay 只连接它，从不停止它。',
      'phoenix.project': '项目：{project}',
      'phoenix.startingBody': '正在本机启动 Arize Phoenix。首次启动需要下载，可能需要一两分钟。',
      'phoenix.waitingTitle': '正在等待 Phoenix',
      'phoenix.waitingBody': '{address} 上的 Phoenix 暂无响应。它恢复后，NeMo Relay 会立即重新连接。',
      'phoenix.failedTitle': 'Phoenix 无法启动',
      'phoenix.notConnected': 'Phoenix 未连接',
      'phoenix.firstRun':
        '“启动”会连接已在 {address} 运行的 Arize Phoenix，或在本机启动一个（首次启动需要下载）。之后 NeMo Relay 会通过其 OpenInference 导出器把轨迹发送过去。',
      'phoenix.local': '轨迹只会发送到本机上的这个 Phoenix。',
      'phoenix.remote': '轨迹会发送到 {address}。',
      'phoenix.failed': 'Phoenix：{message}',
      'phoenix.openFailed': '无法打开 {url}',
      'phoenix.connect': '连接 Phoenix',
      'phoenix.connecting': '正在连接…',
      'phoenix.notLinked': '正在运行，但 NeMo Relay 未向其发送轨迹。',
      'phoenix.notLinkedBody': '{address} 上的 Phoenix 正在运行，但 NeMo Relay 未向其发送轨迹。连接后即可继续发送。',
      'phoenix.downloading': '正在下载 Phoenix…',
      'phoenix.downloadingBody': '正在为首次启动下载 Arize Phoenix，可能需要一两分钟。',
      'phoenix.copyLog': '复制 Phoenix 日志路径',
      'tabs.label': '轨迹显示方式',
      'tabs.trace': '轨迹',
      'tabs.events': 'ATOF 事件',
      'stats.events.one': '个事件',
      'stats.events.other': '个事件',
      'stats.turns.one': '轮',
      'stats.turns.other': '轮',
      'stats.llm.one': '次 LLM 调用',
      'stats.llm.other': '次 LLM 调用',
      'stats.tools.one': '次工具调用',
      'stats.tools.other': '次工具调用',
      'stats.errors.one': '个错误',
      'stats.errors.other': '个错误',
      'stats.blocked': '次被 NeMo Relay 拦截',
      'stats.tokens': 'token：输入 {prompt}（缓存 {cached}）· 输出 {completion}（推理 {reasoning}）',
      'path.label': 'ATOF {version} 文件',
      'path.copy': '复制路径',
      'path.copied': '已复制',
      'path.reload': '重新加载',
      'turn.title': '第 {n} 轮',
      'turn.untitled': '轮次',
      'turn.llm.one': '{n} 次 LLM',
      'turn.llm.other': '{n} 次 LLM',
      'turn.tools.one': '{n} 次工具',
      'turn.tools.other': '{n} 次工具',
      'turn.errors.one': '{n} 个错误',
      'turn.errors.other': '{n} 个错误',
      'turn.blocked': '{n} 次被拦截',
      'turn.noPrompt': '未记录提示词',
      'turn.awaiting': '正在等待下一次模型或工具调用……',
      'turn.answer': '记录在轮次作用域上的最终回复',
      'turn.error': '记录在轮次作用域上的错误',
      'turn.moreRows.one': '显示更早的 1 行',
      'turn.moreRows.other': '显示更早的 {n} 行',
      'axis.scope': 'Relay 作用域',
      'axis.duration': '耗时',
      'row.llm': 'LLM 调用',
      'row.blocked': '已拦截',
      'row.ttft': '{ttft} 后收到首个数据块；颜色较浅的部分为等待时间',
      'status.ok': '成功',
      'status.error': '错误',
      'status.open': '未结束',
      'status.running': '运行中',
      'guard.title.one': '已通过 1 次 NeMo Relay 防护规则评估：',
      'guard.title.other': '已通过 {n} 次 NeMo Relay 防护规则评估：',
      'guard.label': '已通过的防护规则',
      'block.icon': 'NeMo Relay 防护规则',
      'block.tool': '工具调用被 NeMo Relay 拦截',
      'block.llm': '模型调用被 NeMo Relay 拦截',
      'block.targetTool': '工具',
      'block.targetLlm': '模型',
      'block.byRule': '拦截规则',
      'block.editRule': '编辑规则',
      'block.guardrail': '防护规则',
      'block.target': '目标',
      'block.reason': '原因',
      'block.evaluated': '评估耗时',
      'block.unknown': 'NeMo Relay 防护规则',
      'block.noReason': '未提供原因',
      'mark.user_message': '用户消息',
      'mark.approval_asked': '请求批准',
      'mark.approval_decided': '批准结果',
      'mark.subagent': '子智能体',
      'mark.request_edit_dropped': '请求修改被丢弃',
      'mark.compaction': '上下文压缩',
      'mark.notCaptured': '未记录文本',
      'subagent.child': '子会话',
      'subagent.childTitle': '子会话 {id}',
      'subagent.label': '名称',
      'subagent.mode': '模式',
      'loose.title.one': '1 个轮次外事件',
      'loose.title.other': '{n} 个轮次外事件',
      'raw.more': '显示更早的事件（隐藏 {n} 个）',
      'raw.capped.one': '此列表只保留最新的事件；更早的 1 个事件在 ATOF 文件中。',
      'raw.capped.other': '此列表只保留最新的事件；更早的 {count} 个事件在 ATOF 文件中。',
      'detail.atof': 'ATOF 事件',
      'detail.mark': '标记',
      'detail.time': '时间',
      'detail.uuid': 'UUID',
      'detail.data': '数据',
      'detail.model': '模型',
      'detail.provider': '提供方',
      'detail.step': '步骤',
      'detail.purpose': '用途',
      'detail.status': '状态',
      'detail.duration': '耗时',
      'detail.ttft': '首个数据块时间',
      'detail.tokens': 'Token',
      'detail.tokenLine': '输入 {prompt}（缓存 {cached}）· 输出 {completion}（推理 {reasoning}）',
      'detail.finish': '结束原因',
      'detail.finishTitle': '结束原因：{reason}',
      'detail.request': '请求',
      'detail.messages.one': '{n} 条消息',
      'detail.messages.other': '{n} 条消息',
      'detail.tools.one': '{n} 个工具',
      'detail.tools.other': '{n} 个工具',
      'detail.requestNotCaptured': '未记录（capture.includePrompts 已关闭）',
      'detail.responseNotCaptured': '未记录回复（capture.includeResponses 已关闭）',
      'detail.lastUser': '最后一条用户消息',
      'detail.toolResults': '随此请求发送的工具结果（{n}）',
      'detail.reasoning': '推理',
      'detail.response': '回复',
      'detail.toolCalls': '工具调用',
      'detail.tool': '工具',
      'detail.callId': '工具调用 ID',
      'detail.parentCall': '父工具调用',
      'detail.command': '命令',
      'detail.arguments': '参数',
      'detail.argsStripped': '未记录参数值（capture.stripToolArgs）；参数名：{keys}',
      'detail.result': '结果',
      'detail.resultError': '错误结果',
      'detail.resultValue': '结果值',
      'detail.resultStripped': '未记录结果内容（capture.stripToolResults）',
      'state.loading': '正在加载 NeMo Relay 轨迹……',
      'state.empty': '暂无 NeMo Relay 事件',
      'state.emptyBody':
        '发送一条消息后，此会话的每一轮、每次模型调用、工具调用和防护规则决策都会通过 NeMo Relay 记录在这里。',
      'state.inactive': 'NeMo Relay 未启用',
      'state.inactiveBody': '主机中的 NeMo Relay 运行时状态为 {state}，没有调用经由 Relay。',
      'state.error': '无法读取 NeMo Relay 轨迹',
      'state.errorRetrying': '无法读取新的 NeMo Relay 事件（{message}）。正在显示已加载的事件，并继续重试……',
      'state.errorKept': '无法读取 NeMo Relay 轨迹的其余部分（{message}）。正在显示已加载的事件。',
      'state.noRemote': '此客户端中没有可用的 workspaceFiles Remote。',
      'state.unavailable': 'NeMo Relay 集成未启用',
      'state.unavailableBody': '此配置中未加载 nemo-relay-deepseek-harness 的主机端，或其视图已禁用。',
    };

    const plugin = {
      // `remote.commands` runs the user-invoked `/relay` command; DeepSeek Harness's own command UI
      // declares the same namespace.
      inject: ['slots', 'locale', 'remote', 'remote.workspaceFiles', 'remote.commands'],
      apply(ctx) {
        ctx.effect(() => ctx.locale.register(NS, 'en', EN), 'nemo-relay-deepseek-harness: en');
        ctx.effect(() => ctx.locale.register(NS, 'zh', ZH), 'nemo-relay-deepseek-harness: zh');
        const t = ctx.locale.bind(NS);
        const remote = ctx.remote;
        const safely = (read) => {
          try {
            return read();
          } catch {
            return undefined;
          }
        };
        // The locale service's snapshot changes with the language; the view re-renders its
        // memoized parts when it does. A Host without this pair keeps the bound `t` as it is.
        const service = ctx.locale;
        const localeSource =
          typeof service.subscribe === 'function' && typeof service.getSnapshot === 'function'
            ? {
                subscribe: (listener) => service.subscribe(listener),
                // The revision counts every change; a snapshot without one is compared by identity.
                getSnapshot: () => {
                  const snapshot = safely(() => service.getSnapshot());
                  return asRecord(snapshot)?.revision ?? snapshot;
                },
              }
            : undefined;
        ctx.slots.inject('conversation.view', () =>
          ctx.slots.register(
            {
              name: 'conversation.view',
              id: VIEW_ID,
              order: 30,
              label: () => t('tab'),
              inject: () => ({
                t,
                localeSource,
                workspaceFiles: safely(() => remote.workspaceFiles),
                commands: safely(() => remote.commands),
                hostHome: safely(() => remote.$host.home),
              }),
            },
            RelayView,
          ),
        );
      },
    };
    // Pure functions for offline checks; non-enumerable, so loaders never see it.
    Object.defineProperty(plugin, '__test', {
      value: {
        createModel,
        ingest,
        buildModel,
        flattenItems,
        turnSummary,
        timelineOf,
        turnNumber,
        llmInfo,
        itemName,
        itemSubtitle,
        fallbackView,
        readStatus,
        readGuardrails,
        readPhoenix,
        normalizeRule,
        validateRules,
        previewCall,
        tryArguments,
        ruleRefOf,
        phoenixSessionUrl,
        runRelay,
        fmtDuration,
        fmtCount,
        // Mutable: a check shortens `relayDeadlineMs` and clears the kept drafts between cases.
        TIMING,
        DRAFTS,
        CSS,
        EN,
        ZH,
      },
    });
    return plugin;
  },
});
