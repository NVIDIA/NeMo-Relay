// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NeMo Relay rules: which DeepSeek Harness tool calls the integration blocks.
 *
 * The rules run as one Relay tool conditional-execution guardrail, so every
 * managed tool call evaluates the current rules and a save applies to the next
 * call without re-registering anything. Rule shape:
 *
 * - `tool`: empty or `*` matches any tool; `/regex/flags` tests the tool name;
 *   otherwise a comma-separated list (or array) of tool names, case-insensitive.
 * - `match`: empty matches any arguments; otherwise a regular expression tested
 *   against the arguments as JSON and against each top-level string argument.
 * - The first enabled, valid rule that matches blocks the call with
 *   `<reason> (NeMo Relay rule "<id>")`.
 *
 * Every pattern runs in a `vm` context with a time limit, so a pathological
 * regular expression cannot stall the Host; nested unbounded repetition such
 * as `(a+)+` is rejected when the rule is validated, and a save refuses a new
 * or changed pattern that is slow on ordinary text. A rule whose own check
 * still runs out of time is not run again on texts that long until the rules
 * change, so it cannot use up the time of the rules after it.
 */
import { createHash, randomUUID } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';

import type { FailMode } from './config.js';

export const RULES_GUARDRAIL_NAME = 'nemo-relay-deepseek-harness.rules';
export const RULES_GUARDRAIL_PRIORITY = 1000;
export const RULES_FILE_VERSION = 1;
export const DEFAULT_REASON = 'This call is blocked by NeMo Relay policy.';

export type GuardrailRule = {
  id: string;
  tool: string;
  match: string;
  flags: string;
  reason: string;
  enabled: boolean;
};

/**
 * One step within a shell segment: a line continuation (a backslash before a
 * newline, as the shell reads it or as the JSON text writes it), or one
 * character that is neither a separator nor the start of a JSON-escaped
 * newline or carriage return.
 */
const SEGMENT = String.raw`(?:\\\\(?:\\r)?\\n|\\\r?\n|(?!\\[nr])[^;&|\n])`;

/**
 * The shipped rule set; restored by `/relay rules reset`.
 *
 * The default rule checks DeepSeek Harness's shell tools, `bash` and `pwsh` (where `rm` is an
 * alias of `Remove-Item` and `-r` abbreviates `-Recurse`). It matches an `rm` command, within one
 * shell segment (up to `;`, `&`, `|`, or a newline), that has a recursive flag (`-r`, `-R`,
 * `-rf`, `-fr`, `-rfv`, `--recursive`, ...) and an operand that is `/`, `~`, `~/`, `$HOME`, or
 * `${HOME}` (optionally quoted, optionally with a trailing `/` or `/*`). A narrower path such as
 * `~/project/build` or `$HOME/dev` is allowed. A newline also ends the segment in the JSON text
 * of the arguments, where it is the two characters `\n` (or `\r`), so a later line of a script
 * never completes a deletion on an earlier one; a line continuation (a backslash at the end of
 * the line) does not end it, because the shell reads both lines as one command. Known gap: a
 * literal backslash followed by `n` or `r` inside an earlier argument, as in `"a\nb"`, ends the
 * segment too, so an operand after it is not seen.
 *
 * The rule guards against mistakes; it is not a security boundary (an agent set on deleting these
 * directories has easier routes, such as scripts, encodings, or other tools). Like every rule it
 * runs under a time limit, and a pathological single shell segment (more than about 25 KB of `rm`
 * words without a recursive flag) can exceed it: the fail mode then decides, so `open` allows the
 * call and reports the rule in `timedOut`, and `closed` blocks it. The rule then skips texts that
 * long until the rules change, and still checks every shorter one.
 */
export const DEFAULT_RULES: readonly GuardrailRule[] = Object.freeze([
  Object.freeze({
    id: 'no-recursive-delete-of-root-or-home',
    tool: '/^(?:bash|pwsh)$/i',
    match: String.raw`\brm\b(?=${SEGMENT}*\s(?:-[a-z]*r[a-z]*|--recursive)(?=[\s;&|)'"\\]|$))${SEGMENT}*?\s["']?(?:/|~/?|\$\{?HOME\}?/?)\*?["']?(?=$|[\s;&|)'"\\])`,
    flags: 'i',
    reason:
      'Deleting the filesystem root or the home directory recursively is blocked by NeMo Relay policy. Delete a specific path instead, or ask the user.',
    enabled: true,
  }),
]);

const LIMITS = Object.freeze({ rules: 100, id: 64, tool: 300, pattern: 1000, reason: 500, text: 2_000_000 });
const TEST_TIMEOUT_MS = 50;
const CALL_BUDGET_MS = 250;
/** A save reports this many slow rules at most: each one costs a whole time limit to find. */
const PROBE_MAX_SLOW = 5;
const TOOL_NAME = /^[A-Za-z0-9_.:-]+$/;
/** Real tool names a selector is tried on, to notice a selector that selects every tool. */
const SAMPLE_TOOL_NAMES = ['bash', 'read', 'edit', 'nemo_relay', 'glob'];
const ALLOWED_FLAGS = new Set(['i', 'm', 's', 'u']);
/** Repetition counts above this on a group that itself repeats are treated like `+`. */
const MAX_GROUP_REPEAT = 5;

export type RuleError = { index: number; id?: string; field: string; message: string };

type ToolSelector = { any: true } | { regex: RegExp } | { names: Set<string> };

export type CompiledRule = {
  rule: GuardrailRule;
  tool: ToolSelector;
  match: RegExp | null;
  errors: RuleError[];
};

export type ValidatedRules = { rules: GuardrailRule[]; compiled: CompiledRule[]; errors: RuleError[] };

/**
 * `timeout`: the pattern itself ran for the whole time limit. `stalled`: the
 * limit passed while the machine kept the test from running; the answer is
 * unknown, and nothing is learned about the pattern.
 */
export type RegexOutcome = 'match' | 'none' | 'timeout' | 'stalled';
export type RegexTester = (regex: RegExp, text: string, timeoutMs?: number) => RegexOutcome;

/** `timedOut` names the rules an allowed call could not be checked against: out of time, or arguments too large. */
export type Verdict = { verdict: 'allow'; timedOut?: string[] } | { verdict: 'block'; ruleId: string; reason: string };

/* -------------------------------------------------------------------------- */
/* Validation                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Validate a rule list. Every problem is reported with its rule index; the
 * caller decides whether problems refuse a save or only skip the rule.
 */
export function validateRules(input: unknown): ValidatedRules {
  const errors: RuleError[] = [];
  const raw = Array.isArray(input) ? input : [];
  if (!Array.isArray(input)) {
    errors.push({ index: -1, field: 'rules', message: 'Rules must be a list' });
  }
  if (raw.length > LIMITS.rules) {
    errors.push({ index: -1, field: 'rules', message: `At most ${LIMITS.rules} rules are allowed` });
  }
  const rules = raw.slice(0, LIMITS.rules).map((rule, index) => normalizeRule(rule, index));
  const seen = new Set<string>();
  const compiled = rules.map((rule, index) => {
    const result = compileRule(rule, index);
    errors.push(...result.errors);
    const key = rule.id.toLowerCase();
    if (seen.has(key)) {
      const error = { index, id: rule.id, field: 'id', message: `Another rule is already named "${rule.id}"` };
      result.errors.push(error);
      errors.push(error);
    }
    seen.add(key);
    return result;
  });
  return { rules, compiled, errors };
}

/** One line per problem, for command replies and logs. */
export function describeRuleErrors(errors: RuleError[]): string {
  return errors
    .map((error) =>
      error.index >= 0 ? `Rule ${error.index + 1}${error.id ? ` (${error.id})` : ''}: ${error.message}` : error.message,
    )
    .join('\n');
}

/**
 * True when a pattern repeats a group that itself repeats, either without bound
 * or more than a few times, such as `(a+)+`, `([a-z]+\.)*` or `(.*a){20}`: the
 * shapes of catastrophic backtracking.
 */
export function hasNestedRepetition(source: string): boolean {
  const groups: Array<{ repeats: boolean }> = [];
  let inClass = false;
  let closed: { repeats: boolean } | null = null;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    const previous = closed;
    closed = null;
    if (character === '\\') {
      index += 1;
    } else if (inClass) {
      inClass = character !== ']';
    } else if (character === '[') {
      inClass = true;
    } else if (character === '(') {
      groups.push({ repeats: false });
    } else if (character === ')') {
      closed = groups.pop() ?? { repeats: false };
      const parent = groups.at(-1);
      if (closed.repeats && parent !== undefined) {
        parent.repeats = true;
      }
    } else {
      const quantifier =
        character === '*' || character === '+'
          ? { unbounded: true, max: Infinity, length: 1 }
          : character === '{'
            ? braceQuantifier(source, index)
            : null;
      if (quantifier === null) {
        continue;
      }
      if (previous?.repeats === true && (quantifier.unbounded || quantifier.max > MAX_GROUP_REPEAT)) {
        return true;
      }
      const parent = groups.at(-1);
      if (parent !== undefined) {
        parent.repeats = true;
      }
      index += quantifier.length - 1;
    }
  }
  return false;
}

function braceQuantifier(source: string, at: number): { unbounded: boolean; max: number; length: number } | null {
  const match = /^\{(\d+)(,(\d*))?\}/.exec(source.slice(at));
  if (match === null) {
    return null;
  }
  const unbounded = match[2] !== undefined && match[3] === '';
  const max = unbounded ? Infinity : Number(match[2] === undefined ? match[1] : match[3]);
  return { unbounded, max, length: match[0].length };
}

/** A stable rule id: letters, digits, `.`, `_` and `-`. */
function ruleId(value: unknown, index: number): string {
  const id = String(value ?? '')
    .trim()
    .replace(/\s+/g, '-')
    .replace(/[^A-Za-z0-9._-]/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, LIMITS.id);
  return id.length > 0 ? id : `rule-${index + 1}`;
}

function cleanFlags(flags: unknown): string {
  const raw = typeof flags === 'string' ? flags : 'i';
  return [...new Set([...raw].filter((flag) => ALLOWED_FLAGS.has(flag)))].sort().join('');
}

/** One rule in its stored shape. A list of tool names becomes a comma list. */
function normalizeRule(raw: unknown, index: number): GuardrailRule {
  const rule = raw !== null && typeof raw === 'object' && !Array.isArray(raw) ? (raw as Record<string, unknown>) : {};
  const tool = Array.isArray(rule.tool)
    ? rule.tool
        .map((name) => String(name).trim())
        .filter(Boolean)
        .join(', ')
    : typeof rule.tool === 'string'
      ? rule.tool.trim()
      : '';
  return {
    id: ruleId(rule.id, index),
    tool: tool === '*' ? '' : tool,
    match: typeof rule.match === 'string' ? rule.match : '',
    flags: cleanFlags(rule.flags),
    reason: typeof rule.reason === 'string' && rule.reason.trim().length > 0 ? rule.reason.trim() : DEFAULT_REASON,
    enabled: rule.enabled !== false,
  };
}

function compilePattern(source: string, flags: string, label: string): { regex: RegExp } | { error: string } {
  if (source.length > LIMITS.pattern) {
    return { error: `${label} is longer than ${LIMITS.pattern} characters` };
  }
  if (hasNestedRepetition(source)) {
    return { error: `${label} repeats a repeating group, like (a+)+, which can stall the app; simplify it` };
  }
  try {
    return { regex: new RegExp(source, flags) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      error: `${label} is not a valid regular expression: ${message.replace(/^Invalid regular expression: /, '')}`,
    };
  }
}

function compileRule(rule: GuardrailRule, index: number): CompiledRule {
  const errors: RuleError[] = [];
  const fail = (field: string, message: string) => errors.push({ index, id: rule.id, field, message });
  let tool: ToolSelector = { any: true };
  // A tool field like `/bash|sh/i` is a regular expression literal.
  const slash = rule.tool.startsWith('/') ? rule.tool.lastIndexOf('/') : -1;
  if (rule.tool.length > LIMITS.tool) {
    fail('tool', `The tool selector is longer than ${LIMITS.tool} characters`);
  } else if (slash > 0) {
    const compiled = compilePattern(
      rule.tool.slice(1, slash),
      cleanFlags(rule.tool.slice(slash + 1)),
      'The tool pattern',
    );
    if ('error' in compiled) {
      fail('tool', compiled.error);
    } else {
      tool = { regex: compiled.regex };
    }
  } else if (rule.tool.length > 0) {
    const names = rule.tool
      .split(',')
      .map((name) => name.trim())
      .filter(Boolean);
    const bad = names.find((name) => !TOOL_NAME.test(name));
    if (bad !== undefined) {
      fail('tool', `"${bad}" is not a tool name; use names like bash, a comma list, or /regex/`);
    } else if (names.length > 0) {
      tool = { names: new Set(names.map((name) => name.toLowerCase())) };
    }
  }
  let match: RegExp | null = null;
  if (rule.match.length > 0) {
    const compiled = compilePattern(rule.match, rule.flags, 'The pattern');
    if ('error' in compiled) {
      fail('match', compiled.error);
    } else {
      match = compiled.regex;
    }
  }
  const everyTool = rule.tool.length === 0 || ('regex' in tool && selectsEveryTool(tool.regex));
  // No pattern, or one such as `^` or `.` that also matches a call without arguments, matches any call.
  if (everyTool && (rule.match.length === 0 || (match !== null && matchesAnyArguments(match)))) {
    fail('match', 'This rule would block every tool call; name a tool or add a pattern');
  }
  if (rule.reason.length > LIMITS.reason) {
    fail('reason', `The message is longer than ${LIMITS.reason} characters`);
  }
  return { rule, tool, match, errors };
}

let sampleTester: RegexTester | undefined;

/** True when a tool pattern, such as `//` or `/./`, matches every sample tool name. */
function selectsEveryTool(regex: RegExp): boolean {
  const test = (sampleTester ??= createSafeTester());
  return SAMPLE_TOOL_NAMES.every((name) => test(regex, name) === 'match');
}

/** True when a pattern matches the arguments of a call that has none, so it matches every call. */
function matchesAnyArguments(regex: RegExp): boolean {
  const test = (sampleTester ??= createSafeTester());
  return test(regex, '{}') === 'match';
}

const PROBE_SCRIPT = [
  '#!/usr/bin/env bash',
  'set -euo pipefail',
  'cd "$(dirname "$0")/.."',
  'for file in src/*.ts test/*.ts; do',
  '  grep -n "TODO" "$file" || true',
  'done',
  'git status --short && git diff --stat HEAD~1 | tail -n 5',
  'npm run build -- --mode production 2>&1 | tee build.log; echo "exit: $?"',
  "curl -fsSL https://example.com/install.sh -o /tmp/install.sh && sed -i 's/foo/bar/g' config/*.yml",
].join('\n');

/**
 * Ordinary text a new pattern is tried on before it is saved, about 22 KB:
 * a shell script, a tool call as JSON, long runs of one character, whitespace,
 * and JSON data. No text is longer than about 4 KB, so a pattern that merely
 * rescans its input (quadratic time) passes with a wide margin, and one that
 * backtracks exponentially does not.
 */
const PROBE_TEXTS: readonly string[] = Object.freeze([
  PROBE_SCRIPT.repeat(8),
  JSON.stringify({ command: PROBE_SCRIPT.repeat(6), description: 'Build the project and report the result' }),
  `${'a'.repeat(4096)}!`,
  `${' '.repeat(4096)}x`,
  `${' \t\n'.repeat(1365)}x`,
  JSON.stringify({
    files: Array.from({ length: 96 }, (_, index) => ({ path: `src/module-${index}/index.ts`, lines: index * 7 })),
  }),
]);

/** What makes two rules run the same patterns. */
function patternKey(rule: GuardrailRule): string {
  return JSON.stringify([rule.tool, rule.match, rule.flags]);
}

/**
 * Try each enabled rule's pattern and tool pattern on ordinary text under the
 * time limit of one check, and report the rules that run out of time. A rule
 * that passes here can still be slow on unusual text; `evaluateRules` handles
 * that when it happens.
 *
 * @param compiled - Rules that validated without errors.
 * @param test - The time-limited tester.
 * @param known - `patternKey`s of rules that already run, which are not tried again.
 * @returns One error per slow rule, for at most `PROBE_MAX_SLOW` rules.
 */
function probeRules(
  compiled: readonly CompiledRule[],
  test: RegexTester,
  known: ReadonlySet<string> = new Set(),
): RuleError[] {
  const errors: RuleError[] = [];
  const slow = (regex: RegExp): boolean => PROBE_TEXTS.some((text) => test(regex, text, TEST_TIMEOUT_MS) === 'timeout');
  for (const [index, entry] of compiled.entries()) {
    if (errors.length >= PROBE_MAX_SLOW) {
      break;
    }
    if (!entry.rule.enabled || known.has(patternKey(entry.rule))) {
      continue;
    }
    if ('regex' in entry.tool && slow(entry.tool.regex)) {
      errors.push({
        index,
        id: entry.rule.id,
        field: 'tool',
        message: 'The tool pattern is too slow on ordinary text; simplify it',
      });
    } else if (entry.match !== null && slow(entry.match)) {
      errors.push({
        index,
        id: entry.rule.id,
        field: 'match',
        message: 'The pattern is too slow on ordinary text; simplify it',
      });
    }
  }
  return errors;
}

/* -------------------------------------------------------------------------- */
/* Matching                                                                    */
/* -------------------------------------------------------------------------- */

/** A test that the time limit ends is run this many times at most while the system, not the pattern, used the time. */
const STALLED_ATTEMPTS = 3;

/** CPU time this thread has used, in milliseconds, or `undefined` on a runtime that cannot tell. */
function threadCpuMs(): number | undefined {
  if (typeof process.threadCpuUsage !== 'function') {
    return undefined;
  }
  const usage = process.threadCpuUsage();
  return (usage.user + usage.system) / 1000;
}

/**
 * A regular-expression tester whose pattern cannot run longer than its time limit.
 *
 * The limit is wall-clock time, and a busy machine can keep this thread off
 * the CPU for longer than that. A test that ended at the limit after using
 * little CPU time was stopped by the machine, not by the pattern: it is run
 * again, `STALLED_ATTEMPTS` times in all, and reported as `stalled` when
 * every attempt ended that way, so a rule is never refused or remembered as
 * slow because the machine was.
 *
 * @param cpuMs - Reads the CPU time of this thread (tests replace it).
 */
export function createSafeTester(cpuMs: () => number | undefined = threadCpuMs): RegexTester {
  const sandbox = vm.createContext(Object.create(null) as Record<string, unknown>);
  const script = new vm.Script('__nrRe.lastIndex = 0; __nrRe.test(__nrText)');
  return (regex, text, timeoutMs = TEST_TIMEOUT_MS) => {
    const timeout = Math.max(1, Math.round(timeoutMs));
    for (let attempt = 1; ; attempt += 1) {
      sandbox.__nrRe = regex;
      sandbox.__nrText = text;
      const before = cpuMs();
      try {
        return script.runInContext(sandbox, { timeout }) === true ? 'match' : 'none';
      } catch (error) {
        if ((error as { code?: unknown } | null)?.code !== 'ERR_SCRIPT_EXECUTION_TIMEOUT') {
          throw error;
        }
        const used = cpuMs();
        if (before === undefined || used === undefined || used - before >= timeout / 4) {
          return 'timeout';
        }
        if (attempt >= STALLED_ATTEMPTS) {
          return 'stalled';
        }
      } finally {
        sandbox.__nrRe = undefined;
        sandbox.__nrText = undefined;
      }
    }
  };
}

/**
 * The texts a pattern is tested against: the arguments as JSON and each
 * top-level string, shortest first, so a rule that runs out of time on a long
 * text has already checked the short ones. `cut` is true when a text was
 * longer than the limit and only its start is checked.
 */
function argumentTexts(args: unknown): { list: string[]; cut: boolean } {
  const texts: string[] = [];
  if (typeof args === 'string') {
    texts.push(args);
  } else {
    try {
      texts.push(JSON.stringify(args ?? {}) ?? '');
    } catch {
      texts.push(String(args));
    }
    if (args !== null && typeof args === 'object' && !Array.isArray(args)) {
      for (const value of Object.values(args)) {
        for (const item of Array.isArray(value) ? value : [value]) {
          if (typeof item === 'string') {
            texts.push(item);
          }
        }
      }
    }
  }
  return {
    list: texts.map((text) => text.slice(0, LIMITS.text)).sort((a, b) => a.length - b.length),
    cut: texts.some((text) => text.length > LIMITS.text),
  };
}

export type RulesEvaluation = {
  /** The first enabled valid rule that matches the call. */
  blocked?: GuardrailRule;
  /** Rules that could not be checked: out of time, skipped as slow, or shown arguments that were cut. */
  timedOut: string[];
  /** The members of `timedOut` that ran, but only on arguments cut at the length limit. */
  tooLarge: string[];
  /**
   * Rules whose own check used its whole time limit, with the length of the
   * text it ran on (0 for the tool pattern). Rules the call's time budget cut
   * short or never reached are not listed.
   */
  slow: Array<[id: string, length: number]>;
};

/**
 * Evaluate compiled rules for one tool call within a total time budget: the
 * first enabled valid rule that matches blocks; rules that could not be
 * checked are reported in `timedOut`.
 *
 * @param compiled - The validated rules, in order.
 * @param toolName - The tool being called.
 * @param args - The call's arguments.
 * @param test - The time-limited tester.
 * @param slow - Per rule id, the text length from which the rule is skipped (0 skips it for every call) because
 *   its own check ran out of time on a text that long; a skipped rule counts as not checked.
 */
export function evaluateRules(
  compiled: readonly CompiledRule[],
  toolName: string,
  args: unknown,
  test: RegexTester,
  slow: ReadonlyMap<string, number> = new Map(),
): RulesEvaluation {
  const started = Date.now();
  const remaining = () => CALL_BUDGET_MS - (Date.now() - started);
  const result: RulesEvaluation = { timedOut: [], tooLarge: [], slow: [] };
  let texts: { list: string[]; cut: boolean } | undefined;
  for (const entry of compiled) {
    if (!entry.rule.enabled || entry.errors.length > 0) {
      continue;
    }
    const id = entry.rule.id;
    if (remaining() <= 0) {
      result.timedOut.push(id);
      continue;
    }
    if ('names' in entry.tool && !entry.tool.names.has(toolName.toLowerCase())) {
      continue;
    }
    const skipFrom = slow.get(id) ?? Infinity;
    if ('regex' in entry.tool) {
      const limit = Math.min(TEST_TIMEOUT_MS, remaining());
      const outcome = skipFrom === 0 ? 'stalled' : test(entry.tool.regex, toolName, limit);
      if (outcome === 'timeout' || outcome === 'stalled') {
        result.timedOut.push(id);
        if (outcome === 'timeout' && limit === TEST_TIMEOUT_MS) {
          result.slow.push([id, 0]);
        }
      }
      if (outcome !== 'match') {
        continue;
      }
    }
    if (entry.match === null) {
      return { ...result, blocked: entry.rule };
    }
    texts ??= argumentTexts(args);
    let outcome: RegexOutcome = 'none';
    for (const text of texts.list) {
      if (text.length >= skipFrom) {
        // This text and the longer ones after it are not run: the rule ran out of time on one this long.
        outcome = 'stalled';
        break;
      }
      const limit = Math.min(TEST_TIMEOUT_MS, remaining());
      outcome = limit <= 0 ? 'stalled' : test(entry.match, text, limit);
      if (outcome === 'timeout' && limit === TEST_TIMEOUT_MS) {
        result.slow.push([id, text.length]);
      }
      if (outcome !== 'none') {
        break;
      }
    }
    if (outcome === 'match') {
      return { ...result, blocked: entry.rule };
    }
    if (outcome === 'timeout' || outcome === 'stalled') {
      result.timedOut.push(id);
    } else if (texts.cut) {
      result.timedOut.push(id);
      result.tooLarge.push(id);
    }
  }
  return result;
}

/** The rejection reason Relay reports, with the rule id the Client links to its editor. */
export function rejectionReason(rule: Pick<GuardrailRule, 'id' | 'reason'>): string {
  return `${rule.reason} (NeMo Relay rule "${rule.id}")`;
}

/** Short revision of a normalized rule list: the first 8 hex digits of its canonical JSON's SHA-256. */
export function rulesRevision(rules: readonly GuardrailRule[]): string {
  return createHash('sha256').update(canonicalJson(rules)).digest('hex').slice(0, 8);
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  if (value !== null && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/* -------------------------------------------------------------------------- */
/* Storage                                                                     */
/* -------------------------------------------------------------------------- */

/** Where saved rules live. `read()` returns `undefined` when none are saved, so the defaults apply. */
export interface RulesStore {
  readonly kind: 'config' | 'file';
  /** Human-readable location for status output. */
  readonly location: string;
  read(): readonly unknown[] | undefined;
  write(rules: readonly GuardrailRule[] | undefined): Promise<void>;
  /** The saved fail mode, or `undefined` when none is saved, so the configured one applies. */
  readFailMode?(): FailMode | undefined;
  /** Save the fail mode; a store without it cannot change the fail mode while the plugin runs. */
  writeFailMode?(mode: FailMode): Promise<void>;
  /** Why saved rules could not be read, when they could not. */
  readonly loadError?: string | undefined;
  /** Where the rules `read()` returns come from, for a store that can serve rules from the other place. */
  source?(): 'config' | 'file';
  /** Read again what is saved, for a store another process can change. */
  refresh?(): void;
}

/** The subset of DeepSeek Harness's `configEditor` service this store uses. */
export type ConfigEditorLike = {
  edit(
    entry: unknown,
    change: (current: Record<string, unknown>, inherited: Record<string, unknown>) => Record<string, unknown>,
  ): Promise<void>;
};

/**
 * Rules and the fail mode in the plugin's own Loader row (`guardrails.rules`,
 * `guardrails.failMode`), saved through the DeepSeek Harness config editor. Both
 * fields are volatile, so the running plugin reads a saved value in place.
 *
 * While the row configures no rules, rules that an earlier start saved in
 * `rules.json` (when no config editor was available) apply; the next save or
 * reset moves the rules into the row and removes them from that file.
 */
export class ConfigRulesStore implements RulesStore {
  readonly kind = 'config' as const;
  /** Rules from `rules.json`, until a save moves them into the row. */
  private earlier: readonly unknown[] | undefined;

  constructor(
    private readonly current: () => readonly unknown[] | undefined,
    private readonly editor: ConfigEditorLike,
    private readonly entry: unknown,
    private readonly currentFailMode?: () => FailMode,
    private readonly saved?: Pick<RulesStore, 'location' | 'read' | 'write'>,
  ) {
    this.earlier = saved?.read();
  }

  get location(): string {
    return this.saved !== undefined && this.source() === 'file'
      ? this.saved.location
      : 'plugin configuration (guardrails.rules)';
  }

  read(): readonly unknown[] | undefined {
    return this.current() ?? this.earlier;
  }

  source(): 'config' | 'file' {
    return this.current() === undefined && this.earlier !== undefined ? 'file' : 'config';
  }

  readFailMode(): FailMode | undefined {
    return this.currentFailMode?.();
  }

  async write(rules: readonly GuardrailRule[] | undefined): Promise<void> {
    await this.editGuardrails((guardrails) => {
      if (rules === undefined) {
        delete guardrails.rules;
      } else {
        guardrails.rules = rules.map((rule) => ({ ...rule }));
      }
    });
    if (this.earlier !== undefined) {
      // The row now decides, also after a reset: the rules in rules.json must not come back.
      this.earlier = undefined;
      await this.saved?.write(undefined);
    }
  }

  async writeFailMode(mode: FailMode): Promise<void> {
    await this.editGuardrails((guardrails) => {
      guardrails.failMode = mode;
    });
  }

  private async editGuardrails(change: (guardrails: Record<string, unknown>) => void): Promise<void> {
    await this.editor.edit(this.entry, (current) => {
      const next: Record<string, unknown> = { ...current };
      const guardrails = isRecord(next.guardrails) ? { ...next.guardrails } : {};
      change(guardrails);
      if (Object.keys(guardrails).length === 0) {
        delete next.guardrails;
      } else {
        next.guardrails = guardrails;
      }
      return next;
    });
  }
}

/**
 * Rules and the fail mode in `<dataDir>/rules.json`, for profiles without a
 * config editor. The file's rules apply when it saves any; otherwise the rules
 * of the plugin configuration do, and the shipped rules after those. DeepSeek
 * Harness processes can share the file, so every write first reads it again.
 */
export class FileRulesStore implements RulesStore {
  readonly kind = 'file' as const;
  readonly location: string;
  loadError: string | undefined;
  private rules: readonly unknown[] | undefined;
  private failMode: FailMode | undefined;

  /**
   * @param file - The rules file.
   * @param configured - Reads `guardrails.rules` of the plugin configuration, which apply while the file saves no rules.
   */
  constructor(
    readonly file: string,
    private readonly configured?: () => readonly unknown[] | undefined,
  ) {
    this.location = file;
    this.refresh();
  }

  /** Read the file again: another process sharing the data directory may have saved since. */
  refresh(): void {
    let rules: readonly unknown[] | undefined;
    let failMode: FailMode | undefined;
    let loadError: string | undefined;
    try {
      const value = JSON.parse(readFileSync(this.file, 'utf8')) as unknown;
      if (!isRecord(value)) {
        throw new Error('expected an object');
      }
      if (value.rules !== undefined && !Array.isArray(value.rules)) {
        throw new Error('"rules" must be a list');
      }
      if (value.failMode !== undefined && value.failMode !== 'open' && value.failMode !== 'closed') {
        throw new Error('"failMode" must be "open" or "closed"');
      }
      if (value.rules === undefined && value.failMode === undefined) {
        throw new Error('expected a "rules" list');
      }
      rules = Array.isArray(value.rules) ? Object.freeze([...value.rules]) : undefined;
      failMode = value.failMode as FailMode | undefined;
    } catch (error) {
      if ((error as { code?: unknown } | null)?.code !== 'ENOENT') {
        const fallback = this.configured?.() === undefined ? 'default rules' : 'rules of the plugin configuration';
        loadError = `Saved rules in ${this.file} could not be read (${error instanceof Error ? error.message : String(error)}); the ${fallback} apply.`;
      }
    }
    // An unchanged file keeps the list the policy already compiled.
    if (JSON.stringify(rules) !== JSON.stringify(this.rules)) {
      this.rules = rules;
    }
    this.failMode = failMode;
    this.loadError = loadError;
  }

  read(): readonly unknown[] | undefined {
    return this.rules ?? this.configured?.();
  }

  source(): 'config' | 'file' {
    return this.rules === undefined ? 'config' : 'file';
  }

  readFailMode(): FailMode | undefined {
    return this.failMode;
  }

  async write(rules: readonly GuardrailRule[] | undefined): Promise<void> {
    this.refresh();
    // Saving no rules would bring back the rules of the plugin configuration: a reset saves the shipped rules instead.
    const next = rules ?? (this.configured?.() === undefined ? undefined : DEFAULT_RULES.map((rule) => ({ ...rule })));
    await this.persist(next, this.failMode);
    this.rules = next === undefined ? undefined : Object.freeze([...next]);
    this.loadError = undefined;
  }

  async writeFailMode(mode: FailMode): Promise<void> {
    this.refresh();
    await this.persist(this.rules, mode);
    this.failMode = mode;
  }

  /** Write what is saved, or remove the file when nothing is. */
  private async persist(rules: readonly unknown[] | undefined, failMode: FailMode | undefined): Promise<void> {
    if (rules === undefined && failMode === undefined) {
      await rm(this.file, { force: true });
      return;
    }
    await mkdir(path.dirname(this.file), { recursive: true, mode: 0o700 });
    const temp = `${this.file}.${process.pid}.${randomUUID()}.tmp`;
    try {
      const body = {
        version: RULES_FILE_VERSION,
        updatedAt: new Date().toISOString(),
        ...(rules === undefined ? {} : { rules }),
        ...(failMode === undefined ? {} : { failMode }),
      };
      await writeFile(temp, `${JSON.stringify(body, null, 2)}\n`, { mode: 0o600 });
      await rename(temp, this.file);
    } finally {
      await rm(temp, { force: true }).catch(() => undefined);
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Policy                                                                      */
/* -------------------------------------------------------------------------- */

export type RulesSource = 'config' | 'file' | 'default';

export type GuardrailsSnapshot = {
  enabled: boolean;
  failMode: FailMode;
  source: RulesSource;
  editable: boolean;
  revision: string;
  rules: GuardrailRule[];
  rejections: number;
  location: string;
  errors?: RuleError[];
  loadError?: string;
  /** Per rule id, how many calls the rule could not be checked on since the rules last changed. */
  timeouts?: Record<string, number>;
};

export type RulesChangeResult =
  | { ok: true; revision: string; guardrails: GuardrailsSnapshot }
  | {
      ok: false;
      code: 'stale' | 'invalid' | 'unavailable';
      error: string;
      errors?: RuleError[];
      guardrails: GuardrailsSnapshot;
    };

/** Result of `setFailMode`, shared by `/relay rules failmode` and the agent tool. */
export type FailModeChangeResult =
  | { ok: true; failMode: FailMode }
  | { ok: false; code: 'invalid' | 'unavailable'; error: string };

export type GuardrailPolicyOptions = {
  enabled: boolean;
  /** The configured fail mode (read live), used while the store has none saved. */
  failMode: FailMode | (() => FailMode);
  store: RulesStore;
  warn?: (message: string) => void;
  /** Called after a save or reset changed the rules. */
  onChange?: () => void;
  tester?: RegexTester;
};

/** The compiled rules with what their evaluation learned; replaced whenever the rules change. */
type Active = {
  key: readonly unknown[] | undefined;
  validated: ValidatedRules;
  revision: string;
  /** Per rule id, the text length from which the rule is skipped because its own check ran out of time. */
  slow: Map<string, number>;
  /** Per rule id, the calls the rule could not be checked on. */
  timeouts: Map<string, number>;
};

/** The current rules, their evaluation as a Relay guardrail, and user edits. */
export class GuardrailPolicy {
  readonly enabled: boolean;
  /** True while the `/relay` command can save edits. */
  editable = false;
  private readonly configuredFailMode: () => FailMode;
  private readonly store: RulesStore;
  private readonly warn: (message: string) => void;
  private readonly onChange: () => void;
  private readonly test: RegexTester;
  private active: Active | undefined;
  /** A saved value that the store reports only after the Loader commits it. */
  private pending: { value: readonly unknown[] | undefined; before: readonly unknown[] | undefined } | undefined;
  /** A saved fail mode that the store reports only after the Loader commits it. */
  private pendingFailMode: { value: FailMode; before: FailMode | undefined } | undefined;
  private queue: Promise<unknown> = Promise.resolve();
  private rejections = 0;
  private readonly warned = new Set<string>();

  constructor(options: GuardrailPolicyOptions) {
    this.enabled = options.enabled;
    const configured = options.failMode;
    this.configuredFailMode = typeof configured === 'function' ? configured : () => configured;
    this.store = options.store;
    this.warn = options.warn ?? (() => undefined);
    this.onChange = options.onChange ?? (() => undefined);
    this.test = options.tester ?? createSafeTester();
  }

  /** Whether a call the rules cannot check is allowed (`open`) or blocked (`closed`); changes take effect at once. */
  get failMode(): FailMode {
    const stored = this.store.readFailMode?.();
    if (this.pendingFailMode !== undefined) {
      if (stored === this.pendingFailMode.before) {
        return this.pendingFailMode.value;
      }
      // The store now reports a newer value: the save landed, or the fail mode changed elsewhere.
      this.pendingFailMode = undefined;
    }
    return stored ?? this.configuredFailMode();
  }

  /**
   * Persist and apply the fail mode: in the plugin configuration when DeepSeek
   * Harness offers a config editor, otherwise in `rules.json`. The next call
   * uses it. Never rejects.
   */
  setFailMode(mode: unknown): Promise<FailModeChangeResult> {
    return this.serialize(async (): Promise<FailModeChangeResult> => {
      if (mode !== 'open' && mode !== 'closed') {
        return { ok: false, code: 'invalid', error: 'The fail mode must be "open" or "closed".' };
      }
      this.store.refresh?.();
      if (mode === this.failMode) {
        return { ok: true, failMode: mode };
      }
      const write = this.store.writeFailMode?.bind(this.store);
      if (write === undefined) {
        return {
          ok: false,
          code: 'unavailable',
          error:
            'The fail mode cannot be changed here; set guardrails.failMode in the NeMo Relay plugin configuration.',
        };
      }
      const before = this.store.readFailMode?.();
      try {
        await write(mode);
      } catch (error) {
        return {
          ok: false,
          code: 'unavailable',
          error: `The fail mode could not be saved: ${error instanceof Error ? error.message : String(error)}`,
        };
      }
      if (this.store.readFailMode?.() === before) {
        this.pendingFailMode = { value: mode, before };
      }
      try {
        this.onChange();
      } catch {
        // Status publication is best effort.
      }
      return { ok: true, failMode: mode };
    });
  }

  /** The Relay guardrail callback: a rejection reason to block, or `null` to allow. Never throws. */
  evaluate(toolName: unknown, args: unknown): string | null {
    let outcome: Verdict;
    try {
      outcome = this.verdict(String(toolName ?? ''), args);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.warnOnce('evaluation', `NeMo Relay could not evaluate its rules: ${message}`);
      if (this.failMode === 'closed') {
        this.rejections += 1;
        return 'NeMo Relay could not check this call against its rules, and the fail-closed setting blocks it.';
      }
      return null;
    }
    if (outcome.verdict === 'block') {
      this.rejections += 1;
      return outcome.reason;
    }
    return null;
  }

  /** Dry-run one call against the current rules without counting it. */
  dryRun(toolName: string, args: unknown): Verdict {
    return this.verdict(toolName, args, false);
  }

  /** Persist a new rule list when `base` is the current revision. */
  save(base: unknown, rules: unknown): Promise<RulesChangeResult> {
    return this.serialize(async () => {
      if (typeof base !== 'string' || base.length === 0) {
        return this.failure('invalid', 'The request needs the "base" revision of the rules it edits.');
      }
      if (!Array.isArray(rules)) {
        return this.failure('invalid', 'The request needs a "rules" list.');
      }
      const validated = validateRules(rules);
      if (validated.errors.length > 0) {
        return this.failure(
          'invalid',
          `The rules were not saved.\n${describeRuleErrors(validated.errors)}`,
          validated.errors,
        );
      }
      const current = this.latest();
      if (base !== current.revision) {
        return this.stale();
      }
      // Rules that already run are not tried again, so an earlier slow rule never blocks an unrelated edit.
      const running = new Set(current.validated.rules.filter((rule) => rule.enabled).map(patternKey));
      const slow = probeRules(validated.compiled, this.test, running);
      if (slow.length > 0) {
        return this.failure('invalid', `The rules were not saved.\n${describeRuleErrors(slow)}`, slow);
      }
      return this.commit(validated.rules);
    });
  }

  /** Restore the shipped default rules when `base` is the current revision. */
  reset(base: unknown): Promise<RulesChangeResult> {
    return this.serialize(async () => {
      if (typeof base !== 'string' || base.length === 0) {
        return this.failure('invalid', 'The request needs the "base" revision of the rules it resets.');
      }
      if (base !== this.latest().revision) {
        return this.stale();
      }
      return this.commit(undefined);
    });
  }

  /** The `guardrails` section of the status snapshot. */
  snapshot(): GuardrailsSnapshot {
    const active = this.current();
    const loadError = this.store.loadError;
    return {
      enabled: this.enabled,
      failMode: this.failMode,
      source: active.key === undefined ? 'default' : (this.store.source?.() ?? this.store.kind),
      editable: this.editable,
      revision: active.revision,
      rules: active.validated.rules.map((rule) => ({ ...rule })),
      rejections: this.rejections,
      location: this.store.location,
      ...(active.validated.errors.length === 0 ? {} : { errors: active.validated.errors }),
      ...(loadError === undefined ? {} : { loadError }),
      ...(active.timeouts.size === 0 ? {} : { timeouts: Object.fromEntries(active.timeouts) }),
    };
  }

  private verdict(toolName: string, args: unknown, record = true): Verdict {
    if (!this.enabled) {
      return { verdict: 'allow' };
    }
    const active = this.current();
    const { blocked, timedOut, tooLarge, slow } = evaluateRules(
      active.validated.compiled,
      toolName,
      args,
      this.test,
      active.slow,
    );
    if (record) {
      for (const [id, length] of slow) {
        active.slow.set(id, Math.min(length, active.slow.get(id) ?? Infinity));
      }
    }
    if (blocked !== undefined) {
      return { verdict: 'block', ruleId: blocked.id, reason: rejectionReason(blocked) };
    }
    if (timedOut.length === 0) {
      return { verdict: 'allow' };
    }
    const closed = this.failMode === 'closed';
    if (record) {
      for (const id of timedOut) {
        active.timeouts.set(id, (active.timeouts.get(id) ?? 0) + 1);
        this.warnOnce(
          `${tooLarge.includes(id) ? 'size' : 'timeout'}:${id}`,
          `NeMo Relay rule "${id}" could not be checked ${
            tooLarge.includes(id) ? `on arguments longer than ${LIMITS.text} characters` : 'in time'
          } (${closed ? 'blocking' : 'allowing'} the call)`,
        );
      }
    }
    if (closed) {
      const id = timedOut[0] ?? 'rules';
      return {
        verdict: 'block',
        ruleId: id,
        reason: rejectionReason({
          id,
          reason: tooLarge.includes(id)
            ? 'The arguments of this call are too large to check against this rule, and the fail-closed setting blocks the call.'
            : 'This rule could not be checked in time, and the fail-closed setting blocks the call.',
        }),
      };
    }
    return { verdict: 'allow', timedOut };
  }

  /** The validated active rules, recompiled only when the stored value changes. */
  private current(): Active {
    let key = this.store.read();
    if (this.pending !== undefined) {
      if (key === this.pending.before) {
        key = this.pending.value;
      } else {
        // The store now reports a newer value: the save landed, or the rules changed elsewhere.
        this.pending = undefined;
      }
    }
    if (this.active === undefined || this.active.key !== key) {
      const validated = validateRules(key ?? DEFAULT_RULES);
      this.active = { key, validated, revision: rulesRevision(validated.rules), slow: new Map(), timeouts: new Map() };
      if (validated.errors.length > 0) {
        this.warnOnce(
          `invalid:${this.active.revision}`,
          `NeMo Relay skipped invalid rules:\n${describeRuleErrors(validated.errors)}`,
        );
      }
    }
    return this.active;
  }

  /** The current rules after reading again what is saved: another process may have saved since. */
  private latest(): Active {
    this.store.refresh?.();
    return this.current();
  }

  private stale(): RulesChangeResult {
    return this.failure('stale', 'The rules changed since they were loaded. Reload them and try again.');
  }

  private async commit(rules: GuardrailRule[] | undefined): Promise<RulesChangeResult> {
    try {
      await this.store.write(rules);
    } catch (error) {
      return this.failure(
        'unavailable',
        `The rules could not be saved: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    // Until the store reports another value, the rules are the ones just written: the Loader commits a save
    // to the plugin configuration later.
    this.pending = { value: rules === undefined ? undefined : Object.freeze([...rules]), before: this.store.read() };
    try {
      this.onChange();
    } catch {
      // Status publication is best effort.
    }
    const snapshot = this.snapshot();
    return { ok: true, revision: snapshot.revision, guardrails: snapshot };
  }

  private failure(code: 'stale' | 'invalid' | 'unavailable', error: string, errors?: RuleError[]): RulesChangeResult {
    return { ok: false, code, error, ...(errors === undefined ? {} : { errors }), guardrails: this.snapshot() };
  }

  /** Edits run one at a time, so a stale check and its write cannot interleave with another edit. */
  private serialize<T>(task: () => Promise<T>): Promise<T> {
    const result = this.queue.then(task, task);
    this.queue = result.catch(() => undefined);
    return result;
  }

  private warnOnce(key: string, message: string): void {
    if (this.warned.has(key)) {
      return;
    }
    this.warned.add(key);
    try {
      this.warn(message);
    } catch {
      // Logging is best effort.
    }
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
