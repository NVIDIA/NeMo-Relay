// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Retention tests: only old files that match a target's pattern, in the target
 * directory or one level below, are removed.
 */
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readdir, symlink, utimes, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, it } from 'node:test';

import { ATIF_TRAJECTORY_FILES, pruneOldFiles, VIEW_TRACE_FILES } from '../src/retention.js';

const DAY_MS = 24 * 60 * 60 * 1000;
const NOW = Date.UTC(2026, 10, 10);

async function file(target: string, ageDays: number, content = 'x'): Promise<void> {
  await mkdir(path.dirname(target), { recursive: true });
  await writeFile(target, content);
  const time = new Date(NOW - ageDays * DAY_MS);
  await utimes(target, time, time);
}

async function tree(root: string): Promise<string[]> {
  const out: string[] = [];
  const walk = async (directory: string): Promise<void> => {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const target = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        out.push(`${path.relative(root, target)}/`);
        await walk(target);
      } else {
        out.push(path.relative(root, target));
      }
    }
  };
  await walk(root);
  return out.sort();
}

describe('retention', () => {
  it('removes only old view traces and old default trajectories, and the session directories left empty', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-retention-'));
    const view = path.join(root, 'view');
    const atif = path.join(root, 'atif');
    await file(path.join(view, 'old.atof.jsonl'), 31, 'old trace');
    await file(path.join(view, 'recent.atof.jsonl'), 29);
    await file(path.join(view, 'notes.txt'), 400);
    await file(path.join(atif, 'session-old', 'trajectory-1.json'), 45, '12345');
    await file(path.join(atif, 'session-old', 'trajectory-2.json'), 40);
    await file(path.join(atif, 'session-mixed', 'trajectory-1.json'), 45);
    await file(path.join(atif, 'session-mixed', 'trajectory-2.json'), 1);
    await file(path.join(atif, 'session-other', 'export.json'), 300);
    // Deeper than one level, and outside the targets: never touched.
    await file(path.join(atif, 'session-mixed', 'nested', 'trajectory-9.json'), 300);
    await file(path.join(root, 'rules.json'), 300);
    await file(path.join(root, 'outside.atof.jsonl'), 300);
    // A link to an old file outside the target must not remove that file.
    await symlink(path.join(root, 'outside.atof.jsonl'), path.join(view, 'link.atof.jsonl'));

    const result = await pruneOldFiles(
      [
        { directory: view, match: VIEW_TRACE_FILES },
        { directory: atif, match: ATIF_TRAJECTORY_FILES, nested: true },
        { directory: path.join(root, 'missing'), match: VIEW_TRACE_FILES },
      ],
      30,
      NOW,
    );

    assert.deepEqual(result, { removed: 4, bytes: 'old trace'.length + 5 + 1 + 1 });
    assert.deepEqual(await tree(root), [
      'atif/',
      'atif/session-mixed/',
      'atif/session-mixed/nested/',
      'atif/session-mixed/nested/trajectory-9.json',
      'atif/session-mixed/trajectory-2.json',
      'atif/session-other/',
      'atif/session-other/export.json',
      'outside.atof.jsonl',
      'rules.json',
      'view/',
      'view/link.atof.jsonl',
      'view/notes.txt',
      'view/recent.atof.jsonl',
    ]);
  });

  it('keeps everything when the limit is zero, negative, or not a number', async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'nemo-relay-deepseek-harness-retention-'));
    await file(path.join(root, 'old.atof.jsonl'), 1000);
    for (const days of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      assert.deepEqual(await pruneOldFiles([{ directory: root, match: VIEW_TRACE_FILES }], days, NOW), {
        removed: 0,
        bytes: 0,
      });
    }
    assert.deepEqual(await tree(root), ['old.atof.jsonl']);
  });
});
