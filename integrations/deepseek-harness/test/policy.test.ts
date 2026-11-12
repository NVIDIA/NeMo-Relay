// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * NeMo Relay rules: validation, time-limited matching, revisions, and storage.
 *
 * Commands that a rule must block are only ever strings under test; they are
 * assembled at runtime so the literal text never appears in this file.
 */
import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';

import {
  ConfigRulesStore,
  createSafeTester,
  DEFAULT_RULES,
  FileRulesStore,
  GuardrailPolicy,
  hasNestedRepetition,
  rulesRevision,
  validateRules,
  type GuardrailRule,
  type RegexTester,
  type RulesStore,
} from '../src/policy.js';

const RM_RF = ['rm', '-rf'].join(' ');
const DEFAULT_RULE_ID = 'no-recursive-delete-of-root-or-home';

/**
 * An in-memory store; `read()` returns the same array until a write replaces it. A write takes a
 * moment, like a real one, so edits that are not serialized overlap.
 */
function memoryStore(initial?: unknown[]): RulesStore & { writes: number } {
  let rules: readonly unknown[] | undefined = initial;
  return {
    kind: 'file',
    location: 'memory',
    writes: 0,
    read: () => rules,
    async write(next) {
      this.writes += 1;
      await new Promise((resolve) => setImmediate(resolve));
      rules = next === undefined ? undefined : [...next];
    },
  };
}

/** Never notified: `Atomics.wait` on it sleeps for its timeout. */
const PAUSE = new Int32Array(new SharedArrayBuffer(4));

/**
 * The real tester, recording every test it runs. A pattern whose source is in `stall` takes its
 * whole time limit and reports a timeout, which makes the time a call spends exact.
 */
function recordingTester(stall: string[] = []) {
  const real = createSafeTester();
  const tests: Array<{ source: string; length: number; limit: number | undefined }> = [];
  const test: RegexTester = (regex, text, limit) => {
    tests.push({ source: regex.source, length: text.length, limit });
    if (!stall.includes(regex.source)) {
      return real(regex, text, limit);
    }
    const until = Date.now() + (limit ?? 0);
    while (Date.now() < until) {
      Atomics.wait(PAUSE, 0, 0, until - Date.now());
    }
    return 'timeout';
  };
  return { test, tests };
}

function policy(rules?: unknown[], options: { failMode?: 'open' | 'closed'; enabled?: boolean } = {}) {
  return new GuardrailPolicy({
    enabled: options.enabled ?? true,
    failMode: options.failMode ?? 'open',
    store: memoryStore(rules),
  });
}

/** A raw rule as a user or the Client sends it; fields may be absent or loosely typed. */
function rule(overrides: Record<string, unknown>): Record<string, unknown> {
  return { id: 'r', tool: 'bash', match: 'x', reason: 'Blocked.', enabled: true, ...overrides };
}

/** `rm`, assembled at runtime like every command below. */
const R = ['r', 'm'].join('');

/** Every form of recursively deleting the root or home directory that the default rule must block. */
const MUST_BLOCK = [
  `${R} -rf /`,
  `${R} -rf ~`,
  `${R} -rf $HOME`,
  `${R} -rf /*`,
  `${R} -rf ~/*`,
  `${R} -rf ~/`,
  `${R} -rf "$HOME"`,
  `${R} -rf \${HOME}`,
  `${R} -rf "\${HOME}"`,
  `${R} -r -f ~`,
  `${R} -f -r /`,
  `${R} -rfv ~`,
  `${R} --recursive --force ~`,
  `${R} -rf -- /`,
  `${R} -rf --no-preserve-root /`,
  `sudo ${R} -rf /`,
  `bash -c '${R} -rf ~'`,
  `sh -c "${R} -rf $HOME"`,
  `${R} -rf ~; echo done`,
  `${R} -rf ~ && ls`,
  `(${R} -rf ~)`,
  `${R} -R ~`,
  `${R.toUpperCase()} -RF ~`,
  `echo "${R} -rf ~"`,
  `echo '${R} -rf ~'`,
  `${R} -rf $HOME/`,
  `${R} -rf '/'`,
  `/bin/${R} -rf /`,
  `${R} -rf build ~`,
  `${R} build -rf ~`,
  `${R} -rf ./a ./b /`,
  `${R} -rf $HOME/*`,
  // Scripts of several lines: the deletion is one of the lines.
  `echo start\n${R} -rf ~`,
  `${R} -rf ~\necho done`,
  `cd /tmp\n${R} -rf /\nls`,
  `set -e\r\n${R} -rf ~\r\necho ok`,
  // Line continuations: the shell reads these lines as one command.
  `${R} -rf \\\n ~`,
  `${R} -rf \\\n  /`,
  `${R} \\\n  -rf \\\n  ~`,
  `${R} -rf build \\\n  ~`,
  `${R} -rf \\\r\n ~`,
  `${R} -rf a \\\n b \\\n "$HOME"`,
];

/** Narrower deletions and unrelated commands the default rule must allow. */
const MUST_ALLOW = [
  `${R} -rf ~/project/build`,
  `${R} -rf /tmp/x`,
  `${R} -rf ./dist`,
  `${R} -rf $HOME/dev/x`,
  `${R} -f ~/.cache/file`,
  `${R} -rf ~user`,
  `${R} -rf node_modules`,
  `${R} ~/notes.txt`,
  `${R} -rf "$HOME/dev"`,
  `${R} -rf /var/tmp/build`,
  'grep -r foo /',
  `${R} -f /tmp/a /`,
  `${R} -rf build/ && cd ~`,
  `${R} -rf ~/projects ~/old`,
  'perm -rf ~',
  'farm -rf /',
  'ls -R ~',
  'cp -r ~/a /tmp/b',
  `${R} -i ~/x`,
  `git ${R} -r --cached docs/`,
  // Scripts of several lines: a narrower deletion, then the root or home directory on a later line.
  `${R} -rf build\ncd ~`,
  `${R} -rf dist\nls /`,
  `${R} -rf node_modules\nnpm install\ncd /`,
  `${R} -rf build\r\ncd ~`,
  `${R} -rf ./out\necho "$HOME"`,
  `${R} -r tmp\n# back to ~\ncd ~`,
  // Line continuations that belong to a narrower deletion or to another command.
  `${R} -rf build\nls \\\n  ~`,
  `${R} -rf build \\\n  dist`,
  `${R} -rf build \\\n  dist\ncd ~`,
  `${R} -rf build; printf "\\n" > ~/x`,
  `${R} -rf build && echo "\\n" ~`,
  `${R} -rf a \\\n b\ncd \\\n ~`,
];

describe('default rule', () => {
  it(`blocks all ${MUST_BLOCK.length} forms of recursively deleting the root or home directory`, () => {
    const rules = policy();
    // dryRun applies the policy's own matching: the arguments as JSON (where a newline is the two
    // characters `\n`) and each top-level string.
    const missed = MUST_BLOCK.filter(
      (command) => rules.dryRun('bash', { command, description: 'run it' }).verdict !== 'block',
    );
    assert.deepEqual(missed, []);
    assert.deepEqual(rules.dryRun('bash', { command: `${R} -rf ~` }), {
      verdict: 'block',
      ruleId: DEFAULT_RULE_ID,
      reason: `${DEFAULT_RULES[0]?.reason} (NeMo Relay rule "${DEFAULT_RULE_ID}")`,
    });
  });

  it(`allows all ${MUST_ALLOW.length} narrower or unrelated commands`, () => {
    const rules = policy();
    const blocked = MUST_ALLOW.filter(
      (command) => rules.dryRun('bash', { command, description: 'run it' }).verdict !== 'allow',
    );
    assert.deepEqual(blocked, []);
  });

  it('sees nested arguments through their JSON text', () => {
    const rules = policy();
    assert.equal(rules.dryRun('bash', { script: { body: `${R} -rf ~` } }).verdict, 'block');
    assert.equal(rules.dryRun('bash', { script: { body: `${R} -rf ~/build` } }).verdict, 'allow');
  });

  it('checks both DeepSeek Harness shells, bash and pwsh, and no other tool', () => {
    const rules = policy();
    // PowerShell: `rm` aliases Remove-Item and `-r` abbreviates -Recurse.
    assert.equal(rules.dryRun('pwsh', { command: `${R} -r ~` }).verdict, 'block');
    assert.equal(rules.dryRun('pwsh', { command: 'Get-ChildItem ~' }).verdict, 'allow');
    assert.notEqual(rules.evaluate('Bash', { command: `${RM_RF} ~` }), null, 'tool names are case-insensitive');
    for (const tool of ['write_file', 'bash_history', 'mcp__shell__bash']) {
      assert.equal(rules.evaluate(tool, { command: `${RM_RF} /` }), null, tool);
    }
    assert.equal(rules.snapshot().rejections, 1);
  });

  it('leaves a pathological single-segment command to the fail mode when it cannot be checked in time', () => {
    // Every `rm` word re-scans its shell segment, so 100 KB of them exceeds the per-check time limit.
    const command = `${`${R} a `.repeat(20_000)}; ${R} -rf ~`;
    assert.deepEqual(policy().dryRun('bash', { command }), { verdict: 'allow', timedOut: [DEFAULT_RULE_ID] });
    assert.equal(policy(undefined, { failMode: 'closed' }).dryRun('bash', { command }).verdict, 'block');
    assert.equal(policy().dryRun('bash', { command: `${R} a; ${R} -rf ~` }).verdict, 'block', 'without the padding');
  });

  it('keeps checking ordinary commands after one over-size command, under both fail modes', async () => {
    const oversize = { command: `${`${R} a `.repeat(20_000)}; ${R} -rf ~` };
    const ownReason = `${DEFAULT_RULES[0]?.reason} (NeMo Relay rule "${DEFAULT_RULE_ID}")`;
    for (const failMode of ['open', 'closed'] as const) {
      const { test, tests } = recordingTester();
      const rules = new GuardrailPolicy({ enabled: true, failMode, store: memoryStore(), tester: test });
      const unchecked =
        failMode === 'open'
          ? null
          : `This rule could not be checked in time, and the fail-closed setting blocks the call. (NeMo Relay rule "${DEFAULT_RULE_ID}")`;
      assert.equal(rules.evaluate('bash', oversize), unchecked);
      assert.equal(rules.evaluate('bash', { command: `${R} -rf ~` }), ownReason, 'the rule still runs');
      assert.equal(rules.evaluate('bash', { command: 'ls -la' }), null);

      // The same command again: the rule is not run on a text that long, and the fail mode still decides.
      tests.length = 0;
      assert.equal(rules.evaluate('bash', oversize), unchecked);
      assert.deepEqual(
        tests.map((entry) => entry.length),
        ['bash'.length],
        'only the tool pattern ran',
      );
      assert.deepEqual(rules.snapshot().timeouts, { [DEFAULT_RULE_ID]: 2 }, 'both calls are reported');

      // A dry run never changes what later calls are checked against.
      const fresh = new GuardrailPolicy({ enabled: true, failMode, store: memoryStore(), tester: test });
      fresh.dryRun('bash', oversize);
      tests.length = 0;
      fresh.dryRun('bash', oversize);
      assert.ok(tests.some((entry) => entry.length > 100_000));

      // A change of the rules forgets what was learned about them.
      const saved = await rules.save(rules.snapshot().revision, [...rules.snapshot().rules, rule({ id: 'other' })]);
      assert.equal(saved.ok, true);
      assert.equal(saved.guardrails.timeouts, undefined);
      tests.length = 0;
      rules.evaluate('bash', oversize);
      assert.ok(tests.some((entry) => entry.length > 100_000));
    }
  });

  it('treats arguments longer than the length limit as not fully checked', () => {
    const padding = 'x'.repeat(2_000_100);
    const hidden = { command: `: ${padding}; ${R} -rf ~` };
    assert.deepEqual(policy().dryRun('bash', hidden), { verdict: 'allow', timedOut: [DEFAULT_RULE_ID] });
    const closed = policy(undefined, { failMode: 'closed' });
    assert.equal(
      closed.evaluate('bash', hidden),
      `The arguments of this call are too large to check against this rule, and the fail-closed setting blocks the call. (NeMo Relay rule "${DEFAULT_RULE_ID}")`,
    );
    assert.deepEqual(closed.snapshot().timeouts, { [DEFAULT_RULE_ID]: 1 });
    // What is checked still blocks with the rule's own reason, and a tool no rule applies to is unaffected.
    assert.equal(
      closed.evaluate('bash', { command: `${R} -rf ~; : ${padding}` }),
      `${DEFAULT_RULES[0]?.reason} (NeMo Relay rule "${DEFAULT_RULE_ID}")`,
    );
    assert.equal(closed.evaluate('read', hidden), null);
    assert.equal(closed.dryRun('bash', { command: `: ${'x'.repeat(1_000_000)}; ls` }).verdict, 'allow');
  });

  it('is a valid rule without nested repetition', () => {
    assert.deepEqual(validateRules(DEFAULT_RULES).errors, []);
    assert.equal(hasNestedRepetition(DEFAULT_RULES[0]?.match ?? ''), false);
  });

  it('leaves saved rules, including an earlier copy of the default rule, exactly as saved', () => {
    const saved = [
      {
        id: DEFAULT_RULE_ID,
        tool: '/bash/i',
        match: String.raw`\brm\s+-(?:rf|fr)\s+(?:/|~)`,
        flags: 'i',
        reason: 'Saved earlier.',
        enabled: true,
      },
    ];
    const rules = policy(structuredClone(saved));
    assert.deepEqual(rules.snapshot().rules, saved);
    assert.equal(rules.snapshot().source, 'file');
    assert.equal(rules.snapshot().revision, rulesRevision(saved));
    assert.equal(rules.dryRun('pwsh', { command: `${R} -rf ~` }).verdict, 'allow', 'the saved selector still applies');
  });
});

describe('rule matching', () => {
  it('selects tools by name list, array, wildcard, or regular expression', () => {
    const rules = policy([
      rule({ id: 'by-list', tool: 'Write, edit', match: 'secret' }),
      rule({ id: 'by-array', tool: ['fetch', 'web_fetch'], match: 'internal' }),
      rule({ id: 'by-regex', tool: '/^mcp__github__/', match: 'merge' }),
      rule({ id: 'any-tool', tool: '*', match: 'NEMO_RELAY_E2E_BLOCK' }),
    ]);
    const blockedBy = (tool: string, args: unknown) => rules.dryRun(tool, args);
    assert.deepEqual(blockedBy('write', { path: 'secret.txt' }), {
      verdict: 'block',
      ruleId: 'by-list',
      reason: 'Blocked. (NeMo Relay rule "by-list")',
    });
    assert.equal(blockedBy('EDIT', { path: 'secret.txt' }).verdict, 'block');
    assert.equal(blockedBy('web_fetch', { url: 'http://internal/' }).verdict, 'block');
    assert.equal(blockedBy('mcp__github__merge_pull_request', { method: 'merge' }).verdict, 'block');
    assert.equal(blockedBy('MCP__GITHUB__merge', { method: 'merge' }).verdict, 'allow', 'regex flags are explicit');
    assert.deepEqual(blockedBy('bash', { command: 'echo NEMO_RELAY_E2E_BLOCK' }), {
      verdict: 'block',
      ruleId: 'any-tool',
      reason: 'Blocked. (NeMo Relay rule "any-tool")',
    });
    assert.equal(blockedBy('read', { path: 'notes.md' }).verdict, 'allow');
    // Stored shapes: arrays become comma lists, `*` becomes empty.
    assert.deepEqual(
      rules.snapshot().rules.map((entry) => entry.tool),
      ['Write, edit', 'fetch, web_fetch', '/^mcp__github__/', ''],
    );
  });

  it('tests the pattern against the arguments JSON and each top-level string argument', () => {
    const rules = policy([
      rule({ id: 'anchored', match: '^git push --force$' }),
      rule({ id: 'listed', tool: 'run', match: '^--privileged$' }),
      rule({ id: 'json-shape', tool: 'edit', match: '"path":"/etc/' }),
    ]);
    // Only the string argument itself matches the anchors, not the JSON text around it.
    assert.equal(rules.dryRun('bash', { command: 'git push --force' }).verdict, 'block');
    assert.equal(rules.dryRun('bash', { command: 'echo git push --force' }).verdict, 'allow');
    assert.equal(
      rules.dryRun('run', { args: ['ls', '--privileged'] }).verdict,
      'block',
      'strings inside top-level lists',
    );
    assert.equal(rules.dryRun('edit', { path: '/etc/hosts' }).verdict, 'block', 'the JSON text');
    assert.equal(rules.dryRun('edit', { nested: { path: '/etc/hosts' } }).verdict, 'block', 'nested values via JSON');
    // Nested strings are only seen through the JSON text, so anchors do not match them.
    assert.equal(rules.dryRun('bash', { nested: { command: 'git push --force' } }).verdict, 'allow');
  });

  it('applies flags: case-insensitive by default, case-sensitive when cleared', () => {
    const insensitive = policy([rule({ id: 'i', match: 'DROP TABLE' })]);
    const sensitive = policy([rule({ id: 's', match: 'DROP TABLE', flags: '' })]);
    assert.equal(insensitive.dryRun('bash', { command: 'psql -c "drop table x"' }).verdict, 'block');
    assert.equal(sensitive.dryRun('bash', { command: 'psql -c "drop table x"' }).verdict, 'allow');
    assert.equal(sensitive.dryRun('bash', { command: 'psql -c "DROP TABLE x"' }).verdict, 'block');
    const cleaned = validateRules([rule({ flags: 'gmxi' })]).rules[0];
    assert.equal(cleaned?.flags, 'im', 'unsupported flags are dropped and the rest sorted');
  });

  it('skips disabled rules', () => {
    const rules = policy([rule({ id: 'off', match: 'deploy', enabled: false })]);
    assert.equal(rules.dryRun('bash', { command: 'deploy' }).verdict, 'allow');
  });
});

describe('pathological patterns', () => {
  it('rejects nested unbounded repetition when a rule is validated', () => {
    for (const source of ['(a+)+', '([a-z]+\\.)*', '(.*a){20}', '(x+x+)+y']) {
      assert.equal(hasNestedRepetition(source), true, source);
    }
    for (const source of ['(ab)+', 'a{2,5}', '[(+)]+', '(a){3}', String.raw`\(a+\)+`]) {
      assert.equal(hasNestedRepetition(source), false, source);
    }
    const { errors } = validateRules([rule({ id: 'slow', match: '(x+x+)+y' })]);
    assert.match(errors[0]?.message ?? '', /repeats a repeating group/);
  });

  it('stops a catastrophic pattern at its time limit and applies the fail mode', () => {
    // Not nested repetition, yet exponential: only the time limit catches it.
    const slow = rule({ id: 'slow', tool: '', match: '^(a|aa)+$' });
    const args = { text: `${'a'.repeat(64)}b` };
    for (const [failMode, expected] of [
      ['open', null],
      [
        'closed',
        'This rule could not be checked in time, and the fail-closed setting blocks the call. (NeMo Relay rule "slow")',
      ],
    ] as const) {
      const rules = policy([slow], { failMode });
      const started = Date.now();
      assert.equal(rules.evaluate('bash', args), expected);
      assert.ok(Date.now() - started < 1_000, `evaluation took ${Date.now() - started} ms`);
      assert.deepEqual(rules.snapshot().timeouts, { slow: 1 });
    }
  });
});

describe('slow rules', () => {
  /** Passes `hasNestedRepetition` and compiles, yet takes exponential time on any ordinary text. */
  const SLOW = String.raw`^(?:[\s\S]|[^\u0000])*\u0000$`;

  it('refuses to save a new or changed pattern that is slow on ordinary text', async () => {
    assert.equal(hasNestedRepetition(SLOW), false);
    const rules = policy();
    const base = rules.snapshot();
    const refused = await rules.save(base.revision, [
      ...base.rules,
      rule({ id: 'audit', tool: '*', match: SLOW }),
      rule({ id: 'selector', tool: '/^(a|aa)+$/', match: 'x' }),
      rule({ id: 'fine', match: 'deploy' }),
    ]);
    assert.deepEqual(refused.ok ? undefined : [refused.code, refused.errors], [
      'invalid',
      [
        { index: 1, id: 'audit', field: 'match', message: 'The pattern is too slow on ordinary text; simplify it' },
        {
          index: 2,
          id: 'selector',
          field: 'tool',
          message: 'The tool pattern is too slow on ordinary text; simplify it',
        },
      ],
    ]);
    assert.match(
      refused.ok ? '' : refused.error,
      /^The rules were not saved\.\nRule 2 \(audit\): The pattern is too slow/,
    );
    assert.equal(rules.snapshot().revision, base.revision);
    // The shipped rule and an ordinary pattern pass, also when they are new.
    const emptied = await rules.save(base.revision, []);
    const restored = await rules.save(emptied.guardrails.revision, [
      ...base.rules,
      rule({ id: 'fine', match: 'deploy' }),
    ]);
    assert.equal(restored.ok, true);
  });

  it('probes only what a save adds, changes, or turns on', async () => {
    // A slow rule that is already there, as a rule written into the configuration can be.
    const slow = rule({ id: 'slow', tool: '*', match: SLOW });
    const rules = policy([slow, rule({ id: 'off', tool: '*', match: `${SLOW}|x`, enabled: false })]);
    const before = rules.snapshot();
    const unrelated = await rules.save(before.revision, [...before.rules, rule({ id: 'new', match: 'deploy' })]);
    assert.equal(unrelated.ok, true, 'an earlier slow rule does not block an unrelated edit');
    const turnedOn = await rules.save(
      unrelated.guardrails.revision,
      unrelated.guardrails.rules.map((entry) => ({ ...entry, enabled: true })),
    );
    assert.deepEqual(turnedOn.ok ? undefined : turnedOn.errors?.map((error) => [error.id, error.field]), [
      ['off', 'match'],
    ]);
    const changed = await rules.save(
      unrelated.guardrails.revision,
      unrelated.guardrails.rules.map((entry) => (entry.id === 'slow' ? { ...entry, flags: 's' } : entry)),
    );
    assert.deepEqual(changed.ok ? undefined : changed.errors?.map((error) => [error.id, error.field]), [
      ['slow', 'match'],
    ]);
  });

  it('runs the tool pattern under the time limit too, and skips a rule whose tool pattern timed out', () => {
    const { test, tests } = recordingTester();
    const selected = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: memoryStore([rule({ id: 'by-regex', tool: '/^mcp__/', match: 'merge' })]),
      tester: test,
    });
    assert.equal(selected.dryRun('mcp__github__merge', { method: 'merge' }).verdict, 'block');
    assert.deepEqual(tests[0], { source: '^mcp__', length: 'mcp__github__merge'.length, limit: 50 });

    // Exponential on this tool name: only the time limit ends the test.
    const rules = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: memoryStore([rule({ id: 'selector', tool: '/^(a|aa)+$/', match: 'x' })]),
      tester: test,
    });
    const toolName = `${'a'.repeat(64)}b`;
    const started = Date.now();
    assert.equal(rules.evaluate(toolName, { text: 'x' }), null);
    assert.ok(Date.now() - started < 1_000, `evaluation took ${Date.now() - started} ms`);
    tests.length = 0;
    assert.equal(rules.evaluate('bash', { text: 'x' }), null, 'the rule is not known to apply or not');
    assert.deepEqual(tests, [], 'the tool pattern is not run again');
    assert.deepEqual(rules.snapshot().timeouts, { selector: 2 });
    assert.deepEqual(rules.dryRun('bash', { text: 'x' }), { verdict: 'allow', timedOut: ['selector'] });
  });

  it('shares one time budget among the rules of a call, and skips only the rules that used their own limit', () => {
    const stalling = ['stall-1', 'stall-2', 'stall-3', 'stall-4', 'stall-5', 'stall-6'];
    const { test, tests } = recordingTester(stalling);
    const rules = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: memoryStore([
        ...stalling.map((match) => rule({ id: match, match })),
        rule({ id: 'last', match: 'deploy', reason: 'No deploys.' }),
      ]),
      tester: test,
    });
    const call = { command: 'deploy now' };
    const ran = () => [...new Set(tests.splice(0).map((entry) => entry.source))];

    // Five rules use the whole budget (5 x 50 ms = 250 ms): the sixth and the last are never reached.
    const started = Date.now();
    assert.equal(rules.evaluate('bash', call), null);
    assert.ok(Date.now() - started < 400, `the call took ${Date.now() - started} ms`);
    assert.deepEqual(ran(), stalling.slice(0, 5));
    assert.deepEqual(rules.snapshot().timeouts, Object.fromEntries([...stalling, 'last'].map((id) => [id, 1])));

    // The next call skips the rules that ran out of their own time. The sixth rule, which was only
    // never reached, runs now, and the last rule is checked and blocks.
    assert.equal(rules.evaluate('bash', call), 'No deploys. (NeMo Relay rule "last")');
    const second = ran();
    assert.deepEqual(
      second.filter((source) => stalling.slice(0, 4).includes(source)),
      [],
    );
    assert.deepEqual(second.slice(-2), ['stall-6', 'deploy']);
    // A shorter text is tried again; a text as long as the one that timed out is not.
    assert.equal(rules.dryRun('bash', { command: 'deploy' }).verdict, 'allow');
    assert.ok(ran().includes('stall-1'));
  });
});

describe('time limit', () => {
  // Exponential on this text: only the time limit ends the test.
  const slow = /^(a|aa)+$/;
  const text = `${'a'.repeat(64)}b`;

  it('repeats a test that the machine, not the pattern, kept from finishing, three times at most', () => {
    // A clock on which no test ever uses CPU time: every timeout looks like a stalled machine.
    let readings = 0;
    const stalled = createSafeTester(() => ((readings += 1), 0));
    const started = Date.now();
    assert.equal(stalled(slow, text, 20), 'stalled');
    assert.equal(readings, 6, 'three attempts, each read before and after');
    assert.ok(Date.now() - started < 1_000, `the test took ${Date.now() - started} ms`);
    assert.equal(stalled(/a/, 'a', 20), 'match');
    assert.equal(readings, 7, 'a test that finishes is run once');

    // A clock on which the pattern used the whole limit: the timeout is the pattern's own.
    let cpu = 0;
    const busy = createSafeTester(() => (cpu += 1_000));
    assert.equal(busy(slow, text, 20), 'timeout');
    assert.equal(cpu, 2_000, 'one attempt');
    // A runtime that cannot tell: the limit alone decides.
    assert.equal(createSafeTester(() => undefined)(slow, text, 20), 'timeout');
  });

  it('learns nothing about a rule, and refuses no save, while the machine stalls its tests', async () => {
    const real = createSafeTester();
    let busy = true;
    const rules = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: memoryStore([rule({ id: 'no-deploy', match: 'deploy' })]),
      tester: (regex, value, limit) => (busy && regex.source === 'deploy' ? 'stalled' : real(regex, value, limit)),
    });
    assert.equal(rules.evaluate('bash', { command: 'deploy' }), null, 'fail-open allows what could not be checked');
    assert.deepEqual(rules.snapshot().timeouts, { 'no-deploy': 1 }, 'and reports it');
    const changed = await rules.save(rules.snapshot().revision, [
      rule({ id: 'no-deploy', match: 'deploy', flags: '' }),
    ]);
    assert.equal(changed.ok, true, 'the pattern is not called slow');
    assert.equal(rules.evaluate('bash', { command: 'deploy' }), null);
    busy = false;
    assert.equal(
      rules.evaluate('bash', { command: 'deploy' }),
      'Blocked. (NeMo Relay rule "no-deploy")',
      'not skipped',
    );
  });

  it('ends a slow pattern at its limit with the real clocks', () => {
    const test = createSafeTester();
    const started = Date.now();
    assert.equal(test(slow, text, 20), 'timeout');
    assert.ok(Date.now() - started < 1_000, `the test took ${Date.now() - started} ms`);
    assert.deepEqual([test(/b$/, text), test(/c/, text)], ['match', 'none']);
  });
});

describe('invalid rules', () => {
  it('reports every problem with its rule and keeps valid rules working', () => {
    const configured = [
      rule({ id: 'bad tool', tool: 'rm;ls' }),
      rule({ id: 'bad-regex', match: '(' }),
      rule({ id: 'everything', tool: '', match: '' }),
      rule({ id: 'dup', match: 'one' }),
      rule({ id: 'DUP', match: 'two' }),
      rule({ id: 'ok', match: 'blocked-word' }),
    ];
    const { rules, errors } = validateRules(configured);
    assert.equal(rules[0]?.id, 'bad-tool', 'ids are normalized');
    assert.deepEqual(
      errors.map((error) => [error.index, error.field]),
      [
        [0, 'tool'],
        [1, 'match'],
        [2, 'match'],
        [4, 'id'],
      ],
    );
    const active = policy(configured);
    assert.equal(active.dryRun('bash', { command: 'blocked-word' }).verdict, 'block');
    assert.equal(
      active.dryRun('bash', { command: 'one' }).verdict,
      'block',
      'the first of two duplicate ids still applies',
    );
    assert.equal(active.snapshot().errors?.length, 4);
    assert.equal(
      validateRules(Array.from({ length: 101 }, (_, index) => rule({ id: `r${index}` }))).errors[0]?.field,
      'rules',
    );
  });

  it('refuses a tool pattern that selects every tool when the rule has no pattern', () => {
    const every = 'This rule would block every tool call; name a tool or add a pattern';
    for (const tool of ['', '*', '//', '/./', '/^/', '/[a-z_]+/', '/BASH|.*/i']) {
      assert.deepEqual(
        validateRules([rule({ tool, match: '' })]).errors.map((error) => [error.field, error.message]),
        [['match', every]],
        JSON.stringify(tool),
      );
    }
    // A pattern that also matches a call without arguments blocks every call just the same.
    for (const match of ['^', '.', '.*', '$', '\\{', '(?:)']) {
      for (const tool of ['', '*', '/./']) {
        assert.deepEqual(
          validateRules([rule({ tool, match })]).errors.map((error) => [error.field, error.message]),
          [['match', every]],
          JSON.stringify([tool, match]),
        );
      }
    }
    for (const candidate of [
      rule({ tool: '/^(?:bash|pwsh)$/i', match: '' }),
      rule({ tool: '/./', match: 'deploy' }),
      rule({ tool: '*', match: 'password' }),
      rule({ tool: 'bash', match: '' }),
      rule({ tool: 'bash', match: '.' }),
    ]) {
      assert.deepEqual(validateRules([candidate]).errors, [], JSON.stringify([candidate.tool, candidate.match]));
    }
  });
});

describe('revisions and edits', () => {
  it('derives the revision from the normalized rules only', () => {
    const one = validateRules([{ enabled: true, reason: 'No.', match: 'x', tool: 'bash', id: 'a' }]).rules;
    const two = validateRules([{ id: 'a', tool: ['bash'], match: 'x', reason: ' No. ' }]).rules;
    assert.equal(rulesRevision(one), rulesRevision(two));
    assert.match(rulesRevision(one), /^[0-9a-f]{8}$/);
    assert.notEqual(rulesRevision(one), rulesRevision(validateRules([{ ...one[0], match: 'y' }]).rules));
  });

  it('saves against the current revision and refuses stale or invalid edits', async () => {
    const store = memoryStore();
    const rules = new GuardrailPolicy({ enabled: true, failMode: 'open', store });
    const initial = rules.snapshot();
    assert.equal(initial.source, 'default');
    assert.equal(initial.revision, rulesRevision(validateRules(DEFAULT_RULES).rules));

    const edited = [...initial.rules, rule({ id: 'no-force-push', match: 'push\\s+--force' })];
    const saved = await rules.save(initial.revision, edited);
    assert.equal(saved.ok, true);
    assert.equal(saved.guardrails.source, 'file');
    assert.notEqual(saved.ok && saved.revision, initial.revision);
    // The very next call sees the saved rules; nothing restarts.
    assert.equal(rules.dryRun('bash', { command: 'git push --force' }).verdict, 'block');

    const stale = await rules.save(initial.revision, []);
    assert.deepEqual(stale.ok ? undefined : [stale.code, stale.guardrails.revision], [
      'stale',
      saved.guardrails.revision,
    ]);
    const invalid = await rules.save(saved.guardrails.revision, [rule({ match: '(' })]);
    assert.equal(invalid.ok ? undefined : invalid.code, 'invalid');
    assert.match(invalid.ok ? '' : invalid.error, /Rule 1 \(r\): The pattern is not a valid regular expression/);
    const missingBase = await rules.save(undefined, []);
    assert.equal(missingBase.ok ? undefined : missingBase.code, 'invalid');
    assert.equal(store.writes, 1, 'refused edits never reach storage');

    const reset = await rules.reset(saved.guardrails.revision);
    assert.equal(reset.ok, true);
    assert.equal(reset.guardrails.source, 'default');
    assert.equal(reset.guardrails.revision, initial.revision);
    assert.equal(rules.dryRun('bash', { command: 'git push --force' }).verdict, 'allow');
  });

  it('serializes concurrent edits so only one of two saves on the same base wins', async () => {
    const rules = policy();
    const base = rules.snapshot().revision;
    const [first, second] = await Promise.all([
      rules.save(base, [rule({ id: 'first' })]),
      rules.save(base, [rule({ id: 'second' })]),
    ]);
    assert.equal(first.ok, true);
    assert.equal(second.ok ? undefined : second.code, 'stale');
    assert.deepEqual(
      rules.snapshot().rules.map((entry) => entry.id),
      ['first'],
    );
  });

  it('reports a storage failure as unavailable without changing the rules', async () => {
    const store = memoryStore();
    store.write = async () => {
      throw new Error('profile is read-only');
    };
    const rules = new GuardrailPolicy({ enabled: true, failMode: 'open', store });
    const result = await rules.save(rules.snapshot().revision, [rule({ id: 'x' })]);
    assert.deepEqual(result.ok ? undefined : [result.code, result.error], [
      'unavailable',
      'The rules could not be saved: profile is read-only',
    ]);
    assert.equal(rules.snapshot().source, 'default');
  });

  it('allows everything when guardrails are disabled', () => {
    assert.equal(policy(undefined, { enabled: false }).evaluate('bash', { command: `${RM_RF} /` }), null);
  });
});

describe('rule storage', () => {
  it('keeps rules in <dataDir>/rules.json when no config editor exists', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-rules-'));
    const file = path.join(directory, 'rules.json');
    const first = new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) });
    const saved = await first.save(first.snapshot().revision, [rule({ id: 'saved' })]);
    assert.equal(saved.guardrails.source, 'file');
    assert.equal((await stat(file)).mode & 0o777, 0o600);
    assert.deepEqual(
      JSON.parse(await readFile(file, 'utf8')).rules.map((entry: GuardrailRule) => entry.id),
      ['saved'],
    );

    // A new plugin generation reads the saved rules.
    const second = new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) });
    assert.equal(second.snapshot().revision, saved.guardrails.revision);
    const reset = await second.reset(saved.guardrails.revision);
    assert.equal(reset.guardrails.source, 'default');
    await assert.rejects(stat(file), { code: 'ENOENT' });

    await writeFile(file, '{ not json');
    const broken = new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) });
    assert.equal(broken.snapshot().source, 'default');
    assert.match(broken.snapshot().loadError ?? '', /could not be read/);
  });

  it('uses the rules of the plugin configuration while rules.json saves none', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-configured-'));
    const file = path.join(directory, 'rules.json');
    const configured = [rule({ id: 'no-curl', match: 'curl' })];
    const start = () =>
      new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file, () => configured) });
    const ids = (rules: GuardrailPolicy) => rules.snapshot().rules.map((entry) => entry.id);

    const first = start();
    assert.deepEqual([first.snapshot().source, ids(first)], ['config', ['no-curl']]);
    assert.equal(first.dryRun('bash', { command: 'curl example.com' }).verdict, 'block');

    // A save goes to the file, which then takes precedence over the configuration.
    const saved = await first.save(first.snapshot().revision, [...first.snapshot().rules, rule({ id: 'added' })]);
    assert.deepEqual([saved.guardrails.source, ids(first)], ['file', ['no-curl', 'added']]);
    assert.deepEqual([start().snapshot().source, ids(start())], ['file', ['no-curl', 'added']]);

    // A reset restores the shipped rules, not the configured ones, also for the next start.
    const reset = await first.reset(saved.guardrails.revision);
    assert.equal(reset.ok, true);
    assert.deepEqual(ids(first), [DEFAULT_RULE_ID]);
    assert.deepEqual(ids(start()), [DEFAULT_RULE_ID]);

    // An unreadable file falls back to the configured rules and says so.
    await writeFile(file, '{ not json');
    const broken = start().snapshot();
    assert.deepEqual([broken.source, broken.rules.map((entry) => entry.id)], ['config', ['no-curl']]);
    assert.match(broken.loadError ?? '', /could not be read .*the rules of the plugin configuration apply/);
  });

  it('checks an edit against rules.json as another process left it, and never overwrites that process’s rules', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-shared-'));
    const file = path.join(directory, 'rules.json');
    const start = () => new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) });
    const onDisk = async () => {
      const saved = JSON.parse(await readFile(file, 'utf8'));
      return [saved.rules.map((entry: GuardrailRule) => entry.id), saved.failMode];
    };
    // Two DeepSeek Harness processes on one data directory, both started before any edit.
    const app = start();
    const cli = start();
    const base = cli.snapshot().revision;
    const saved = await app.save(base, [...app.snapshot().rules, rule({ id: 'no-curl', match: 'curl' })]);
    assert.equal(saved.ok, true);

    const stale = await cli.save(base, [rule({ id: 'mine' })]);
    assert.deepEqual(stale.ok ? undefined : [stale.code, stale.guardrails.revision], [
      'stale',
      saved.guardrails.revision,
    ]);
    assert.deepEqual(await onDisk(), [[DEFAULT_RULE_ID, 'no-curl'], undefined]);

    // The fail mode is merged with the rules on disk, not with this process's earlier copy.
    const other = start();
    assert.deepEqual(await other.setFailMode('closed'), { ok: true, failMode: 'closed' });
    const added = await app.save(saved.guardrails.revision, [...saved.guardrails.rules, rule({ id: 'later' })]);
    assert.equal(added.ok, true);
    assert.deepEqual(await cli.setFailMode('open'), { ok: true, failMode: 'open' });
    assert.deepEqual(await onDisk(), [[DEFAULT_RULE_ID, 'no-curl', 'later'], 'open']);
    assert.equal(start().dryRun('bash', { command: 'curl example.com' }).verdict, 'block');
    assert.equal((await stat(file)).mode & 0o777, 0o600);
    assert.deepEqual(await readdir(directory), ['rules.json'], 'the temporary file was renamed into place');
  });

  it('reads rules an earlier start saved in rules.json until a save moves them into the plugin row', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-moved-'));
    const file = path.join(directory, 'rules.json');
    // An earlier start had no config editor and saved to the file.
    const earlier = new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) });
    await earlier.save(earlier.snapshot().revision, [rule({ id: 'from-file', match: 'curl' })]);

    let row: Record<string, unknown> = { dataDir: '/data' };
    const editor = {
      async edit(
        _entry: unknown,
        change: (current: Record<string, unknown>, inherited: Record<string, unknown>) => Record<string, unknown>,
      ) {
        row = change(structuredClone(row), {});
      },
    };
    // A plain row: the running plugin never sees the edit, so the policy keeps what it just saved.
    const start = () =>
      new GuardrailPolicy({
        enabled: true,
        failMode: 'open',
        store: new ConfigRulesStore(() => undefined, editor, 'entry-1', undefined, new FileRulesStore(file)),
      });
    const rules = start();
    const before = rules.snapshot();
    assert.deepEqual(
      [before.source, before.location, before.rules.map((entry) => entry.id)],
      ['file', file, ['from-file']],
    );

    const saved = await rules.save(before.revision, [...before.rules, rule({ id: 'added' })]);
    assert.equal(saved.ok, true);
    assert.deepEqual(
      [saved.guardrails.source, saved.guardrails.location],
      ['config', 'plugin configuration (guardrails.rules)'],
    );
    assert.deepEqual(
      (row.guardrails as { rules: GuardrailRule[] }).rules.map((entry) => entry.id),
      ['from-file', 'added'],
    );
    await assert.rejects(stat(file), { code: 'ENOENT' }, 'the rules moved out of the file');

    // A reset now restores the shipped rules: nothing in the file brings the old ones back.
    const reset = await rules.reset(saved.guardrails.revision);
    assert.deepEqual([reset.ok, reset.guardrails.source], [true, 'default']);
    assert.deepEqual(row, { dataDir: '/data' });
    assert.equal(start().snapshot().source, 'default');
  });

  it('changes the fail mode live and keeps it in rules.json beside the rules', async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-failmode-'));
    const file = path.join(directory, 'rules.json');
    // A check that can never finish in time: the fail mode alone decides the call.
    const unresolvable = `${`${R} a `.repeat(20_000)}; ${R} -rf ~`;
    const first = new GuardrailPolicy({ enabled: true, failMode: () => 'open', store: new FileRulesStore(file) });
    assert.equal(first.dryRun('bash', { command: unresolvable }).verdict, 'allow');

    assert.deepEqual(await first.setFailMode('closed'), { ok: true, failMode: 'closed' });
    assert.equal(first.snapshot().failMode, 'closed');
    assert.equal(first.dryRun('bash', { command: unresolvable }).verdict, 'block', 'the next call uses it');
    assert.equal(JSON.parse(await readFile(file, 'utf8')).failMode, 'closed');
    assert.equal(first.snapshot().source, 'default', 'saving the fail mode leaves the default rules in place');

    // Rule edits and resets keep the saved fail mode; a new plugin generation reads both.
    const saved = await first.save(first.snapshot().revision, [rule({ id: 'kept' })]);
    assert.equal(saved.ok, true);
    const second = new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) });
    assert.deepEqual([second.failMode, second.snapshot().rules.map((entry) => entry.id)], ['closed', ['kept']]);
    await second.reset(second.snapshot().revision);
    assert.deepEqual(JSON.parse(await readFile(file, 'utf8')).failMode, 'closed');
    assert.equal(
      new GuardrailPolicy({ enabled: true, failMode: 'open', store: new FileRulesStore(file) }).failMode,
      'closed',
    );
  });

  it('refuses an unknown fail mode and reports a store that cannot save one', async () => {
    const rules = policy();
    assert.deepEqual(await rules.setFailMode('sometimes'), {
      ok: false,
      code: 'invalid',
      error: 'The fail mode must be "open" or "closed".',
    });
    assert.deepEqual(await rules.setFailMode('open'), { ok: true, failMode: 'open' }, 'no change, nothing to save');
    const unavailable = await rules.setFailMode('closed');
    assert.deepEqual([unavailable.ok, unavailable.ok ? undefined : unavailable.code], [false, 'unavailable']);
    assert.equal(rules.failMode, 'open');

    const failing = memoryStore();
    const broken = new GuardrailPolicy({
      enabled: true,
      failMode: 'open',
      store: Object.assign(failing, {
        writeFailMode: async () => {
          throw new Error('profile is read-only');
        },
      }),
    });
    assert.deepEqual(await broken.setFailMode('closed'), {
      ok: false,
      code: 'unavailable',
      error: 'The fail mode could not be saved: profile is read-only',
    });
    assert.equal(broken.failMode, 'open');
  });

  it('saves into the plugin row through the config editor and applies it before the Loader commits', async () => {
    let row: Record<string, unknown> = { dataDir: '/data', guardrails: { failMode: 'closed' } };
    let volatile: readonly unknown[] | undefined;
    const commits: Array<() => void> = [];
    const editor = {
      async edit(
        entry: unknown,
        change: (current: Record<string, unknown>, inherited: Record<string, unknown>) => Record<string, unknown>,
      ) {
        assert.equal(entry, 'entry-1');
        row = change(structuredClone(row), {});
        // The Loader commits the volatile value later, as part of reconciliation.
        commits.push(() => (volatile = (row.guardrails as { rules?: unknown[] } | undefined)?.rules));
      },
    };
    const rules = new GuardrailPolicy({
      enabled: true,
      failMode: 'closed',
      store: new ConfigRulesStore(() => volatile, editor, 'entry-1'),
    });
    const saved = await rules.save(rules.snapshot().revision, [rule({ id: 'cfg', match: 'terraform destroy' })]);
    assert.equal(saved.ok, true);
    assert.deepEqual(row, {
      dataDir: '/data',
      guardrails: {
        failMode: 'closed',
        rules: [{ id: 'cfg', tool: 'bash', match: 'terraform destroy', flags: 'i', reason: 'Blocked.', enabled: true }],
      },
    });
    assert.equal(rules.dryRun('bash', { command: 'terraform destroy' }).verdict, 'block', 'before the commit');
    commits.shift()?.();
    assert.equal(rules.dryRun('bash', { command: 'terraform destroy' }).verdict, 'block', 'after the commit');
    assert.equal(rules.snapshot().source, 'config');

    const reset = await rules.reset(saved.guardrails.revision);
    assert.equal(reset.ok, true);
    assert.deepEqual(row, { dataDir: '/data', guardrails: { failMode: 'closed' } });
    commits.shift()?.();
    assert.equal(rules.snapshot().source, 'default');
  });

  it('saves the fail mode into the plugin row and applies it before the Loader commits', async () => {
    let row: Record<string, unknown> = { dataDir: '/data' };
    let volatile: 'open' | 'closed' = 'open';
    const commits: Array<() => void> = [];
    const editor = {
      async edit(
        _entry: unknown,
        change: (current: Record<string, unknown>, inherited: Record<string, unknown>) => Record<string, unknown>,
      ) {
        row = change(structuredClone(row), {});
        commits.push(() => (volatile = (row.guardrails as { failMode: 'open' | 'closed' }).failMode));
      },
    };
    const rules = new GuardrailPolicy({
      enabled: true,
      failMode: () => volatile,
      store: new ConfigRulesStore(
        () => undefined,
        editor,
        'entry-1',
        () => volatile,
      ),
    });
    assert.deepEqual(await rules.setFailMode('closed'), { ok: true, failMode: 'closed' });
    assert.deepEqual(row, { dataDir: '/data', guardrails: { failMode: 'closed' } });
    assert.equal(rules.failMode, 'closed', 'before the commit');
    commits.shift()?.();
    assert.equal(rules.failMode, 'closed', 'after the commit');
    // A change made elsewhere (for example in Settings) is read in place.
    volatile = 'open';
    assert.equal(rules.snapshot().failMode, 'open');
  });
});
