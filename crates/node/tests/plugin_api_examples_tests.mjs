// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { it } from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const filename = fileURLToPath(import.meta.url);
const repositoryRoot = path.resolve(path.dirname(filename), '../../..');
const require = createRequire(import.meta.url);
const observability = 'docs/reference/api/observability.mdx';
const adaptive = 'docs/reference/api/adaptive.mdx';
const resourceMetrics = 'docs/configure-plugins/resource-metrics.mdx';
const examples = new Map([
  ['atif', [observability, ['Agent Trajectory Interchange Format (ATIF)', 'Manual API']]],
  ['atof', [observability, ['Agent Trajectory Observability Format (ATOF)', 'Manual API']]],
  ['atof_diagnostics', [observability, ['Agent Trajectory Observability Format (ATOF)', 'Runtime Diagnostics']]],
  ['opentelemetry', [observability, ['OpenTelemetry', 'Direct Subscribers']]],
  [
    'opentelemetry_signals',
    [observability, ['OpenTelemetry', 'Direct Subscribers', 'Log and Metric Subscriber Lifecycle']],
  ],
  ['openinference', [observability, ['OpenInference', 'Direct Subscriber']]],
  ['resource_metrics', [resourceMetrics, ['Resource Metrics', 'Collect One Snapshot']]],
]);

function fencedExamples(relativePath, markdown = readFileSync(path.join(repositoryRoot, relativePath), 'utf8')) {
  const result = [];
  let hierarchy = [];
  let language = null;
  let fenceMarker = '';
  let body = [];
  for (const line of markdown.split(/\r?\n/)) {
    if (language !== null) {
      const closing = /^ {0,3}(`+|~+)[ \t]*$/.exec(line);
      if (closing && closing[1][0] === fenceMarker[0] && closing[1].length >= fenceMarker.length) {
        result.push({ headings: hierarchy.map(([, title]) => title), language, source: `${body.join('\n')}\n` });
        language = null;
      } else body.push(line);
    } else {
      const opening = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
      if (opening) {
        fenceMarker = opening[1];
        if (fenceMarker.startsWith('`') && opening[2].includes('`')) continue;
        language = opening[2].trim().split(/\s+/)[0];
        body = [];
      } else {
        const match = /^(#{1,6}) (.+)$/.exec(line);
        if (match) {
          hierarchy = hierarchy.filter(([depth]) => depth < match[1].length);
          hierarchy.push([match[1].length, match[2]]);
        }
      }
    }
  }
  assert.equal(language, null, `Unclosed code fence in ${relativePath}`);
  return result.filter((example) => ['js', 'javascript'].includes(example.language));
}

function displayedSource(name) {
  const [relativePath, hierarchy] = examples.get(name);
  const matches = fencedExamples(relativePath).filter(
    (example) => JSON.stringify(example.headings) === JSON.stringify(hierarchy),
  );
  assert.equal(matches.length, 1, `${relativePath}: ${hierarchy.join(' / ')}`);
  return matches[0].source;
}

async function exerciseExample(name) {
  const relay = require('nemo-relay-node');
  const before = relay.listRuntimeRegistrations();
  const printed = [];
  const observedConsole = {
    ...console,
    log: (...values) => {
      printed.push(...values);
      console.log(...values);
    },
    dir: (value, options) => {
      printed.push(value);
      console.dir(value, options);
    },
  };
  const context = vm.createContext({ require, console: observedConsole, process });
  let diagnosticExporter;
  let subscriber;
  if (name === 'atof_diagnostics') {
    const { mkdtempSync, rmSync } = require('node:fs');
    const { tmpdir } = require('node:os');
    const directory = mkdtempSync(path.join(tmpdir(), 'relay-doc-diagnostics-'));
    diagnosticExporter = new relay.AtofExporter({ type: 'file', outputDirectory: directory, filename: 'events.jsonl' });
    context.exporter = diagnosticExporter;
    process.once('exit', () => rmSync(directory, { recursive: true, force: true }));
  }
  try {
    // Native promises keep the event loop alive while the original example runs.
    // Waiting for beforeExit also covers examples whose top-level IIFE uses void.
    const completed = name === 'resource_metrics'
      ? new Promise((resolve) => process.once('beforeExit', resolve))
      : null;
    new vm.Script(displayedSource(name), { filename: examples.get(name)[0] }).runInContext(context);
    if (['opentelemetry', 'openinference'].includes(name)) subscriber = vm.runInContext('subscriber', context);
    if (completed) await completed;
    if (name === 'atif') {
      const trajectory = JSON.parse(printed[0]);
      assert.equal(trajectory.session_id, 'session-1');
      assert.equal(trajectory.agent.name, 'agent');
      assert.deepEqual(trajectory.steps, []);
    } else if (name === 'atof' || name === 'atof_diagnostics') {
      const exporter = vm.runInContext('exporter', context);
      assert.equal(exporter.runtimeDiagnostics().length, 0);
      assert.deepEqual(printed, []);
    } else if (['opentelemetry', 'openinference'].includes(name)) {
      assert.equal(vm.runInContext('subscriber.runtimeDiagnostics()', context).length, 0);
      assert.deepEqual(printed, []);
    } else if (name === 'resource_metrics') {
      const snapshot = printed[0];
      assert.ok(Number.isFinite(Date.parse(snapshot.timestamp)));
      assert.equal(snapshot.measurementScope, 'process_tree');
      assert.equal(snapshot.cpu.consumptionRate, null);
      await assert.rejects(relay.collectResourceMetrics(), /requires an active resource_metrics component/);
    } else assert.deepEqual(printed, []);
  } finally {
    if (subscriber) subscriber.shutdown();
    if (diagnosticExporter) diagnosticExporter.shutdown();
  }
  assert.deepEqual(relay.listRuntimeRegistrations(), before);
}

if (process.argv[2] === '--example') {
  await exerciseExample(process.argv[3]);
} else {
  it('includes tilde and longer fenced examples in the displayed inventory', () => {
    for (const fence of ['```', '~~~~', '`````']) {
      for (const lineEnding of ['\n', '\r\n']) {
        const markdown = [
          '# Unregistered example',
          `${fence}javascript`,
          'console.log(1);',
          `${fence}${fence[0]} \t`,
        ].join(lineEnding);
        assert.deepEqual(fencedExamples('example.mdx', markdown), [
          { headings: ['Unregistered example'], language: 'javascript', source: 'console.log(1);\n' },
        ]);
      }
    }
  });
  it('keeps shorter, mismatched, and text-suffixed fences inside the source', () => {
    for (const closing of ['```', '~~~~', '```` trailing text']) {
      const markdown = `# Example\n\`\`\`\`js\n${closing}\n\`\`\`\`\n`;
      assert.deepEqual(fencedExamples('example.mdx', markdown), [
        { headings: ['Example'], language: 'js', source: `${closing}\n` },
      ]);
      assert.throws(() => fencedExamples('example.mdx', `\`\`\`\`js\n${closing}\n`), /Unclosed code fence/);
    }
  });
  it('extracts identical displayed examples with LF and CRLF line endings', () => {
    for (const relativePath of [observability, adaptive, resourceMetrics]) {
      const markdown = readFileSync(path.join(repositoryRoot, relativePath), 'utf8').replace(/\r\n/g, '\n');
      const expected = fencedExamples(relativePath, markdown);
      assert.deepEqual(fencedExamples(relativePath, markdown.replace(/\n/g, '\r\n')), expected);
    }
  });
  it('registers every displayed Node.js snippet for coverage', () => {
    for (const relativePath of [observability, adaptive, resourceMetrics]) {
      const actual = fencedExamples(relativePath)
        .map((example) => JSON.stringify(example.headings))
        .sort();
      const expected = [...examples.values()]
        .filter(([page]) => page === relativePath)
        .map(([, headings]) => JSON.stringify(headings))
        .sort();
      assert.deepEqual(actual, expected);
    }
  });
  for (const name of examples.keys()) {
    it(`displayed ${name} example matches its outcome`, () => {
      execFileSync(process.execPath, [filename, '--example', name], {
        cwd: repositoryRoot,
        env: { ...process.env, OTEL_AUTHORIZATION: 'Bearer documentation-test' },
        timeout: 45000,
      });
    });
  }
  for (const name of ['opentelemetry', 'openinference']) {
    for (const authorization of [undefined, ' invalid ']) {
      it(`displayed ${name} constructor rejects ${authorization === undefined ? 'missing' : 'invalid'} headers`, () => {
        const environment = { ...process.env };
        delete environment.OTEL_AUTHORIZATION;
        if (authorization !== undefined) environment.OTEL_AUTHORIZATION = authorization;
        assert.throws(
          () =>
            execFileSync(process.execPath, [filename, '--example', name], {
              cwd: repositoryRoot,
              env: environment,
              timeout: 45000,
              stdio: 'pipe',
            }),
          (error) => error.status !== 0 && /OTEL_AUTHORIZATION/.test(error.stderr.toString()),
        );
      });
    }
  }
}
