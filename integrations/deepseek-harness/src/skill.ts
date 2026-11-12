// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The `nemo-relay` skill: on-demand instructions for the `nemo_relay` tool,
 * served by a DeepSeek Harness skill provider.
 *
 * Until the model loads it, the skill costs one catalog entry. Loading it
 * returns `skills/nemo-relay.md` followed by the live status and rules, so the
 * model starts from the current state. Nothing here throws into DeepSeek
 * Harness: an unreadable file or status becomes a note in the content.
 */
import { readFile } from 'node:fs/promises';

import { formatStatus, type StatusSnapshot } from './status.js';

export const RELAY_SKILL_NAME = 'nemo-relay';
/** Provider name; DeepSeek Harness requires every candidate to name its provider. */
export const RELAY_SKILL_PROVIDER = 'nemo-relay-deepseek-harness';
/** `BUNDLED_SKILL_RANK` of `@deepseek-ai/dsh-skill`: project and user skills of the same name win. */
export const BUNDLED_SKILL_RANK = 600;

/** Catalog text: when the model should load the skill. */
export const RELAY_SKILL_DESCRIPTION =
  "Use when the user asks about NeMo Relay; tracing, observability, or exporting this app's model and tool calls " +
  '(Arize Phoenix, OpenTelemetry, OpenInference); guardrail rules or a tool call NeMo Relay blocked; or secret ' +
  'redaction in traces. Explains the nemo_relay tool, rule syntax, and the current Relay status.';

/** The packaged skill text: `<package>/skills/nemo-relay.md`, beside `dist/` (and `.test-dist/`). */
const DEFAULT_SKILL_FILE = new URL('../../skills/nemo-relay.md', import.meta.url);

export type SkillInvocationPolicy = { readonly modelInvocable: boolean; readonly userInvocable: boolean };

/** `SkillSummary` fields of `@deepseek-ai/dsh-skill` this provider fills. */
export type RelaySkillSummary = {
  readonly name: string;
  readonly description: string;
  readonly invocation: SkillInvocationPolicy;
  readonly source: 'bundled';
  readonly provider: string;
};

export type RelaySkillCandidate = RelaySkillSummary & { readonly rank: number; readonly locator: null };
export type RelaySkillDefinition = RelaySkillSummary & { readonly content: string };
export type SkillLookupOptions = { readonly cwd?: string | undefined; readonly signal?: AbortSignal | undefined };

/** A `SkillProvider` of `@deepseek-ai/dsh-skill`. */
export type RelaySkillProvider = {
  readonly name: string;
  list(options?: SkillLookupOptions): Promise<readonly RelaySkillCandidate[]>;
  get(candidate: unknown, options?: SkillLookupOptions): Promise<RelaySkillDefinition | undefined>;
};

/** The part of a DeepSeek Harness plugin context `registerRelaySkill` uses. */
export type RelaySkillHost = {
  readonly skills?:
    | {
        registerProvider(create: (control: unknown) => RelaySkillProvider): unknown;
      }
    | undefined;
};

export type RelaySkillDeps = {
  /** The live status snapshot, as `/relay json` reports it. */
  snapshot(): StatusSnapshot;
  /** The skill markdown; defaults to the packaged `skills/nemo-relay.md`. */
  skillFile?: string | URL | undefined;
  warn?: ((message: string) => void) | undefined;
};

const SUMMARY: RelaySkillSummary = Object.freeze({
  name: RELAY_SKILL_NAME,
  description: RELAY_SKILL_DESCRIPTION,
  invocation: Object.freeze({ modelInvocable: true, userInvocable: true }),
  source: 'bundled',
  provider: RELAY_SKILL_PROVIDER,
});

const CANDIDATE: RelaySkillCandidate = Object.freeze({ ...SUMMARY, rank: BUNDLED_SKILL_RANK, locator: null });

/**
 * Create the skill provider. `list()` names the one skill; `get()` reads its
 * markdown and appends the live status and rules. Neither rejects.
 *
 * @param deps - The live status and, optionally, where the markdown lives.
 * @returns The provider to register with `ctx.skills.registerProvider`.
 */
export function createRelaySkillProvider(deps: RelaySkillDeps): RelaySkillProvider {
  return {
    name: RELAY_SKILL_PROVIDER,
    list: () => Promise.resolve([CANDIDATE]),
    async get(candidate, options) {
      if ((candidate as { name?: unknown } | null | undefined)?.name !== RELAY_SKILL_NAME) {
        return undefined;
      }
      return { ...SUMMARY, content: await skillContent(deps, options?.signal) };
    },
  };
}

/**
 * Register the `nemo-relay` skill. When `ctx.skills` is unavailable or
 * refuses the provider, nothing is registered and nothing throws.
 *
 * @param ctx - A context that provides the `skills` service.
 * @param deps - The live status and, optionally, where the markdown lives.
 * @returns A disposer that unregisters the provider.
 */
export function registerRelaySkill(ctx: RelaySkillHost, deps: RelaySkillDeps): () => void {
  const warn = warner(deps);
  try {
    const skills = ctx.skills;
    if (typeof skills?.registerProvider !== 'function') {
      return () => undefined;
    }
    const dispose = skills.registerProvider(() => createRelaySkillProvider(deps));
    return () => {
      try {
        if (typeof dispose === 'function') {
          (dispose as () => void)();
        }
      } catch (error) {
        warn(`NeMo Relay could not unregister the nemo-relay skill: ${messageOf(error)}`);
      }
    };
  } catch (error) {
    warn(`NeMo Relay could not register the nemo-relay skill: ${messageOf(error)}`);
    return () => undefined;
  }
}

async function skillContent(deps: RelaySkillDeps, signal: AbortSignal | undefined): Promise<string> {
  let body: string;
  try {
    const text = await readFile(deps.skillFile ?? DEFAULT_SKILL_FILE, {
      encoding: 'utf8',
      ...(signal === undefined ? {} : { signal }),
    });
    // The file's license header is for the repository, not the model.
    body = text.replace(/^\s*<!--[\s\S]*?-->\s*/, '');
  } catch (error) {
    if (signal?.aborted !== true) {
      warner(deps)(`NeMo Relay could not read the nemo-relay skill text: ${messageOf(error)}`);
    }
    body = [
      '# NeMo Relay',
      '',
      `The full skill text could not be read (${messageOf(error)}). Use the \`nemo_relay\` tool: \`status\` and`,
      '`rules.list` show the current state; changes that weaken protection or start Phoenix ask the user first.',
    ].join('\n');
  }
  return `${body.trimEnd()}\n\n${liveState(deps)}\n`;
}

/** The status and rules at load time, so the model starts from the current state. */
function liveState(deps: RelaySkillDeps): string {
  try {
    const snapshot = deps.snapshot();
    const guardrails = snapshot.guardrails;
    return [
      '## Current state',
      '',
      'Captured when this skill loaded; `nemo_relay` `status` and `rules.list` show later changes.',
      '',
      '```text',
      formatStatus(snapshot),
      '```',
      '',
      `Rules, revision ${guardrails.revision}, one JSON object per rule:`,
      '',
      '```json',
      ...(guardrails.rules.length === 0 ? ['(no rules)'] : guardrails.rules.map((rule) => JSON.stringify(rule))),
      '```',
    ].join('\n');
  } catch (error) {
    return [
      '## Current state',
      '',
      `The live NeMo Relay status is unavailable (${messageOf(error)}); call \`nemo_relay\` with action \`status\`.`,
    ].join('\n');
  }
}

function warner(deps: RelaySkillDeps): (message: string) => void {
  return (message) => {
    try {
      deps.warn?.(message);
    } catch {
      // Logging is best effort.
    }
  };
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
