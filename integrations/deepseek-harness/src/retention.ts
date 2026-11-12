// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Age-based cleanup of the files this plugin writes for every conversation.
 *
 * The conversation view's trace and the default ATIF trajectories accumulate for
 * as long as DeepSeek Harness is used, so files that no turn has touched for the
 * configured number of days are removed. Only regular files that match a
 * target's name pattern, directly in the target directory or one level below
 * it, are removed; links are never followed, and nothing outside a target
 * directory is touched.
 */
import { lstat, readdir, rmdir, unlink } from 'node:fs/promises';
import path from 'node:path';

const DAY_MS = 24 * 60 * 60 * 1000;

export type RetentionTarget = {
  /** Directory owned by this plugin. */
  directory: string;
  /** File names to remove when they are old enough. */
  match: RegExp;
  /** Also clean the directories directly inside `directory`, and remove the ones left empty. */
  nested?: boolean;
};

export type RetentionResult = { removed: number; bytes: number };

/** The conversation view's per-session trace files. */
export const VIEW_TRACE_FILES = /\.atof\.jsonl$/;
/** Trajectories written by the default ATIF component, one directory per session. */
export const ATIF_TRAJECTORY_FILES = /^trajectory-.+\.json$/;

/**
 * Remove files older than `days` days from each target.
 *
 * @param targets - Directories and the file names to clean in them.
 * @param days - Age limit by last modification; zero or less keeps everything.
 * @param now - Current time in milliseconds.
 * @returns How many files were removed and their total size. Never rejects: a
 * file that cannot be read or removed is left in place.
 */
export async function pruneOldFiles(
  targets: readonly RetentionTarget[],
  days: number,
  now: number = Date.now(),
): Promise<RetentionResult> {
  const result: RetentionResult = { removed: 0, bytes: 0 };
  if (!Number.isFinite(days) || days <= 0) {
    return result;
  }
  const cutoff = now - days * DAY_MS;
  for (const target of targets) {
    await pruneDirectory(target.directory, target.match, cutoff, result);
    if (target.nested !== true) {
      continue;
    }
    for (const name of await entries(target.directory)) {
      const child = path.join(target.directory, name);
      if (!(await isDirectory(child))) {
        continue;
      }
      await pruneDirectory(child, target.match, cutoff, result);
      // Fails, and leaves the directory, unless it is now empty.
      await rmdir(child).catch(() => undefined);
    }
  }
  return result;
}

async function pruneDirectory(
  directory: string,
  match: RegExp,
  cutoff: number,
  result: RetentionResult,
): Promise<void> {
  for (const name of await entries(directory)) {
    if (!match.test(name)) {
      continue;
    }
    const file = path.join(directory, name);
    try {
      const stats = await lstat(file);
      if (!stats.isFile() || stats.mtimeMs >= cutoff) {
        continue;
      }
      await unlink(file);
      result.removed += 1;
      result.bytes += stats.size;
    } catch {
      // Removed meanwhile, or not removable: leave it.
    }
  }
}

async function entries(directory: string): Promise<string[]> {
  try {
    return await readdir(directory);
  } catch {
    return [];
  }
}

async function isDirectory(target: string): Promise<boolean> {
  try {
    return (await lstat(target)).isDirectory();
  } catch {
    return false;
  }
}
