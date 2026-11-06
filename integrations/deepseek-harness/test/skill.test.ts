// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The `nemo-relay` skill: the provider shape DeepSeek Harness accepts, the
 * live state it appends, and a skill text whose approval table, rule
 * examples, and exporter example agree with the code they describe.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { describe, it } from 'node:test';

import {
  AGENT_ACTIONS,
  classifyAgentAction,
  createAgentTool,
  type AgentToolArgs,
  type AgentToolDefinitionOptions,
  type AgentToolResult,
} from '../src/agent-tool.js';
import { parseConfig } from '../src/config.js';
import { createHealthSnapshot } from '../src/health.js';
import { atifComponent, describeExporters, planPluginDocument } from '../src/plugin-doc.js';
import { GuardrailPolicy, type RulesStore } from '../src/policy.js';
import {
  BUNDLED_SKILL_RANK,
  createRelaySkillProvider,
  registerRelaySkill,
  RELAY_SKILL_NAME,
  RELAY_SKILL_PROVIDER,
  type RelaySkillDeps,
  type RelaySkillProvider,
} from '../src/skill.js';
import { buildStatusSnapshot, type StatusSnapshot } from '../src/status.js';

/** The packaged skill text, at the same place relative to the compiled tests as to the compiled sources. */
const SKILL_FILE = new URL('../../skills/nemo-relay.md', import.meta.url);
const markdown = await readFile(SKILL_FILE, 'utf8');
/** DeepSeek Harness's skill-name grammar (`@deepseek-ai/dsh-skill`). */
const SKILL_NAME = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

function memoryStore(): RulesStore {
  let rules: readonly unknown[] | undefined;
  return {
    kind: 'config',
    location: 'plugin configuration (guardrails.rules)',
    read: () => rules,
    write: async (next) => void (rules = next === undefined ? undefined : [...next]),
  };
}

function relay() {
  const config = parseConfig({ dataDir: '/data/relay' });
  const policy = new GuardrailPolicy({ enabled: true, failMode: 'open', store: memoryStore() });
  const snapshot = (): StatusSnapshot =>
    buildStatusSnapshot({
      health: createHealthSnapshot({
        status: { state: 'ready' },
        initializedPluginHost: true,
        pluginHostOutputsHealthy: true,
        config,
      }),
      config,
      plan: undefined,
      resolved: { components: [] },
      relayVersion: '1.0.0',
      guardrails: policy.snapshot(),
      phoenix: {
        enabled: true,
        state: 'stopped',
        connected: false,
        url: config.phoenix.url,
        project: config.phoenix.project,
      },
      viewDirectory: '/data/relay/view',
    });
  const tool = createAgentTool((options) => options, {
    context: (sessionId) => ({
      snapshot,
      publish: () => undefined,
      policy,
      phoenix: undefined,
      sessionId,
      sessionTrace: undefined,
    }),
  }) as AgentToolDefinitionOptions;
  const run = (args: unknown): Promise<AgentToolResult> => tool.execute(args as AgentToolArgs, {});
  return { policy, snapshot, run };
}

function load(provider: RelaySkillProvider, options = {}): Promise<{ content: string } | undefined> {
  return provider.get({ name: RELAY_SKILL_NAME }, options);
}

/** Every fenced `json` block of the skill text, parsed. */
function jsonBlocks(text: string): Array<Record<string, unknown>> {
  return [...text.matchAll(/```json\n([\s\S]*?)```/g)].map(
    (match) => JSON.parse(match[1] ?? 'null') as Record<string, unknown>,
  );
}

/** The body of one `## ` section. */
function section(text: string, title: string): string {
  const start = text.indexOf(`## ${title}\n`);
  assert.ok(start >= 0, `the skill has a "${title}" section`);
  const end = text.indexOf('\n## ', start + 1);
  return text.slice(start, end < 0 ? undefined : end);
}

describe('nemo-relay skill provider', () => {
  it('lists one bundled skill that both the model and the user can invoke', async () => {
    const provider = createRelaySkillProvider({ snapshot: relay().snapshot });
    const candidates = await provider.list();
    assert.equal(candidates.length, 1);
    const [candidate] = candidates;
    assert.equal(candidate?.name, RELAY_SKILL_NAME);
    assert.match(RELAY_SKILL_NAME, SKILL_NAME);
    assert.equal(candidate?.provider, provider.name, 'DeepSeek Harness rejects a candidate naming another provider');
    assert.equal(provider.name, RELAY_SKILL_PROVIDER);
    assert.deepEqual(candidate?.invocation, { modelInvocable: true, userInvocable: true });
    assert.equal(candidate?.source, 'bundled');
    assert.equal(candidate?.rank, BUNDLED_SKILL_RANK);
    assert.equal(candidate?.locator, null);
    assert.match(candidate?.description ?? '', /NeMo Relay/);
    assert.match(candidate?.description ?? '', /Phoenix/);
  });

  it('loads the skill text without its license header, followed by the live status and rules', async () => {
    const { snapshot, policy } = relay();
    const provider = createRelaySkillProvider({ snapshot });
    const skill = await load(provider);
    assert.ok(skill !== undefined);
    const { content, ...summary } = skill as { content: string } & Record<string, unknown>;
    const [candidate] = await provider.list();
    assert.ok(candidate !== undefined);
    const { rank: _rank, locator: _locator, ...expected } = candidate;
    assert.deepEqual(summary, expected, 'the definition repeats the summary without rank and locator');
    assert.ok(content.startsWith('# NeMo Relay\n'));
    assert.doesNotMatch(content, /SPDX/);
    const live = content.slice(content.indexOf('## Current state'));
    assert.match(live, /NeMo Relay: ready — nemo-relay-node 1\.0\.0/);
    assert.ok(live.includes(`Rules, revision ${policy.snapshot().revision}`));
    assert.ok(live.includes(JSON.stringify(policy.snapshot().rules[0])), 'each rule appears as exact JSON');
    assert.equal(await provider.get({ name: 'another-skill' }), undefined);
  });

  it('reports the rules as they are when it loads', async () => {
    const { snapshot, run } = relay();
    const provider = createRelaySkillProvider({ snapshot });
    await run({ action: 'rules.add', id: 'no-e2e', tool: 'bash', match: 'NEMO_RELAY_E2E_BLOCK' });
    const content = (await load(provider))?.content ?? '';
    assert.match(content, /\{"id":"no-e2e","tool":"bash","match":"NEMO_RELAY_E2E_BLOCK"/);
    assert.match(content, /Rules: 2 of 2 on/);
  });

  it('never rejects: unreadable text, failing status, and cancellation become notes', async () => {
    const warnings: string[] = [];
    const deps: RelaySkillDeps = {
      snapshot: () => {
        throw new Error('status unavailable');
      },
      skillFile: '/nonexistent/nemo-relay.md',
      warn: (message) => void warnings.push(message),
    };
    const content = (await load(createRelaySkillProvider(deps)))?.content ?? '';
    assert.match(content, /^# NeMo Relay\n/);
    assert.match(content, /could not be read/);
    assert.match(content, /live NeMo Relay status is unavailable \(status unavailable\)/);
    assert.equal(warnings.length, 1);
    const aborted = await load(createRelaySkillProvider({ ...deps, skillFile: SKILL_FILE }), {
      signal: AbortSignal.abort(),
    });
    assert.match(aborted?.content ?? '', /^# NeMo Relay\n/);
    assert.equal(warnings.length, 1, 'a cancelled load is not reported as a failure');
  });
});

describe('registerRelaySkill', () => {
  it('registers one provider and unregisters it', async () => {
    const factories: Array<(control: unknown) => RelaySkillProvider> = [];
    let disposed = 0;
    const dispose = registerRelaySkill(
      { skills: { registerProvider: (create) => (factories.push(create), () => void (disposed += 1)) } },
      { snapshot: relay().snapshot },
    );
    assert.equal(factories.length, 1);
    const provider = factories[0]?.({ signal: new AbortController().signal, invalidate: () => undefined });
    assert.equal(provider?.name, RELAY_SKILL_PROVIDER);
    assert.equal((await provider?.list())?.[0]?.name, RELAY_SKILL_NAME);
    dispose();
    assert.equal(disposed, 1);
  });

  it('is a no-op, never a failure, without a usable skills service', () => {
    const warnings: string[] = [];
    const deps = { snapshot: relay().snapshot, warn: (message: string) => void warnings.push(message) };
    registerRelaySkill({}, deps)();
    const refusing = {
      skills: {
        registerProvider: (): never => {
          throw new Error('skill provider "nemo-relay-deepseek-harness" is already registered');
        },
      },
    };
    registerRelaySkill(refusing, deps)();
    assert.deepEqual(warnings, [
      'NeMo Relay could not register the nemo-relay skill: skill provider "nemo-relay-deepseek-harness" is already registered',
    ]);
  });
});

describe('nemo-relay skill text', () => {
  it('documents every tool action', () => {
    for (const action of AGENT_ACTIONS) {
      assert.ok(markdown.includes(`\`${action}\``), action);
    }
  });

  it('states the approval policy the tool enforces', () => {
    const paragraphs = section(markdown, 'Approval').split('\n\n');
    const listed = (prefix: string) => {
      const paragraph = paragraphs.find((text) => text.startsWith(prefix)) ?? '';
      return [...paragraph.matchAll(/`([a-zA-Z.]+)`(?: with `(open|closed)`)?/g)].map((match) =>
        match[2] === undefined ? { action: match[1] } : { action: match[1], mode: match[2] },
      );
    };
    const immediate = listed('These run immediately');
    const asking = listed('These first ask the user');
    for (const args of immediate) {
      assert.equal(classifyAgentAction(args).needsApproval, false, JSON.stringify(args));
    }
    for (const args of asking) {
      assert.equal(classifyAgentAction(args).needsApproval, true, JSON.stringify(args));
    }
    const documented = new Set([...immediate, ...asking].map((args) => args.action));
    assert.deepEqual(
      AGENT_ACTIONS.filter((action) => !documented.has(action)),
      [],
    );
  });

  it('gives rule examples that block what they say and nothing more', async () => {
    const { run, policy } = relay();
    const calls = jsonBlocks(markdown).filter((block) => typeof block.action === 'string');
    const adds = calls.filter((call) => call.action === 'rules.add');
    assert.deepEqual(
      adds.map((call) => call.id),
      ['no-force-push', 'no-npm-publish'],
    );
    for (const call of adds) {
      await run(call);
    }
    const documentedTest = calls.find((call) => call.action === 'rules.test');
    assert.deepEqual((await run(documentedTest)).data, {
      verdict: 'block',
      ruleId: 'no-force-push',
      reason:
        'Force-pushing is blocked by NeMo Relay policy. Use --force-with-lease, or ask the user to push. (NeMo Relay rule "no-force-push")',
    });
    const verdict = (command: string) => policy.dryRun('bash', { command, description: 'Run a command' });
    assert.equal(verdict('git push -f origin main').verdict, 'block');
    assert.equal(verdict('git push --force-with-lease origin main').verdict, 'allow');
    assert.equal(verdict('git push origin main').verdict, 'allow');
    assert.equal(verdict('npm publish --access public').verdict, 'block');
    assert.equal(verdict('npm run publish-docs').verdict, 'allow');
  });

  it('gives an exporter example the plugin accepts and reports, keeping per-turn trajectories', () => {
    const example = jsonBlocks(markdown).find((block) => 'plugins' in block);
    assert.ok(example !== undefined);
    const config = parseConfig({ dataDir: '/data/relay', plugins: example.plugins });
    assert.deepEqual(describeExporters(config.plugins), [
      { kind: 'atif', target: '/data/relay/atif' },
      { kind: 'otel', target: 'https://collector.example.com/v1/traces' },
    ]);
    // The example's `atif` section is the default one, so adding the exporter changes nothing else.
    assert.deepEqual(
      (config.plugins.components[0] as { config: { atif: unknown } }).config.atif,
      (atifComponent('/data/relay/atif').config as { atif: unknown }).atif,
    );
    const plan = planPluginDocument(config, config.plugins);
    assert.equal(plan.atif.applied, false, 'the component replaces the default ATIF output, as the skill says');
    assert.equal(plan.redaction.applied, true, 'secret redaction still applies to the new exporter');
  });
});
