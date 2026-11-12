// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Per-process status files: naming, atomic replacement, removal on stop, and
 * pruning of files that crashed processes left behind.
 */
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readdir, readFile, stat, utimes, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';

import { pruneStatusFiles, STALE_STATUS_MS, StatusFile } from '../src/status-file.js';

async function dataDir(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-status-'));
}

describe('status file', () => {
  it('names one file per process, replaces it whole, and removes it on stop', async () => {
    const root = await dataDir();
    const file = new StatusFile(root, { pid: 4242, suffix: 'beef' });
    assert.equal(file.path, path.join(root, 'status', '4242-beef.json'));
    file.write('{"state":"ready"}\n');
    file.write('{"state":"ready","updated":true}\n');
    assert.deepEqual(JSON.parse(await readFile(file.path, 'utf8')), { state: 'ready', updated: true });
    assert.equal((await stat(file.path)).mode & 0o777, 0o600);
    assert.deepEqual(await readdir(path.join(root, 'status')), ['4242-beef.json'], 'no temporary file is left');

    file.close();
    await assert.rejects(stat(file.path), { code: 'ENOENT' });
    file.write('{"state":"late"}\n');
    await assert.rejects(stat(file.path), { code: 'ENOENT' }, 'a late update does not bring the file back');
  });

  it('gives each plugin instance of one process its own file', async () => {
    const root = await dataDir();
    const first = new StatusFile(root);
    const second = new StatusFile(root);
    assert.notEqual(first.path, second.path);
    assert.match(path.basename(first.path), new RegExp(`^${process.pid}-[0-9a-f]{4}\\.json$`));
  });

  it('prunes day-old files of processes that are gone, and nothing else', async () => {
    const root = await dataDir();
    const directory = path.join(root, 'status');
    await mkdir(directory, { recursive: true });
    const names = [
      '101-aaaa.json', // stale, process gone: pruned
      '102-bbbb.json', // stale, process running: kept
      '103-cccc.json', // recent, process gone: kept
      '104-dddd.json.104.0f3a.tmp', // stale temporary file: pruned
      'notes.json', // not a status file: kept
    ];
    for (const name of names) {
      await writeFile(path.join(directory, name), '{}');
    }
    const now = Date.now();
    const old = new Date(now - STALE_STATUS_MS - 60_000);
    for (const name of ['101-aaaa.json', '102-bbbb.json', '104-dddd.json.104.0f3a.tmp', 'notes.json']) {
      await utimes(path.join(directory, name), old, old);
    }
    const removed = pruneStatusFiles(root, { now, alive: (pid) => pid === 102 });
    assert.deepEqual(removed.map((file) => path.basename(file)).sort(), [
      '101-aaaa.json',
      '104-dddd.json.104.0f3a.tmp',
    ]);
    assert.deepEqual((await readdir(directory)).sort(), ['102-bbbb.json', '103-cccc.json', 'notes.json']);
    assert.deepEqual(pruneStatusFiles(path.join(root, 'missing')), [], 'a missing directory is fine');
  });
});
