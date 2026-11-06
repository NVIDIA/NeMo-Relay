// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * One status file per DeepSeek Harness process.
 *
 * Several DeepSeek Harness processes can share one data directory (the desktop
 * app and a CLI run on the same `DSH_HOME`), so each plugin instance writes its
 * own `<dataDir>/status/<pid>-<4 hex>.json`, and the conversation-view
 * projection points the Client at the file of the process serving it. A file
 * is removed when its plugin instance stops; files a crashed process left
 * behind are pruned once they are a day old.
 */
import { randomBytes, randomUUID } from 'node:crypto';
import { mkdirSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';

/** Directory under the data directory that holds the status files. */
export const STATUS_DIRECTORY = 'status';
/** Files of processes that are gone are pruned once they are this old. */
export const STALE_STATUS_MS = 24 * 60 * 60 * 1000;

const STATUS_NAME = /^(\d+)-[0-9a-f]{4}\.json$/;
const TEMP_NAME = /^\d+-[0-9a-f]{4}\.json\..+\.tmp$/;

export type StatusFileIdentity = { pid: number; suffix: string };

/** The status file of one plugin instance. */
export class StatusFile {
  readonly path: string;
  private closed = false;

  /**
   * @param dataDir - The plugin data directory.
   * @param identity - Process id and 4-hex-digit suffix; the current process and a random suffix by default.
   */
  constructor(
    dataDir: string,
    identity: StatusFileIdentity = { pid: process.pid, suffix: randomBytes(2).toString('hex') },
  ) {
    this.path = path.join(dataDir, STATUS_DIRECTORY, `${identity.pid}-${identity.suffix}.json`);
  }

  /** Replace the file in one step, so a reader never sees a partial file. Does nothing once closed. */
  write(text: string): void {
    if (this.closed) {
      return;
    }
    writeFileAtomic(this.path, text);
  }

  /** Remove the file; later writes do nothing, so a late update cannot bring it back. */
  close(): void {
    this.closed = true;
    rmSync(this.path, { force: true });
  }
}

/**
 * Remove status files (and leftover temporary files) older than `maxAgeMs`
 * whose process is no longer running. Never throws.
 *
 * @returns The removed paths.
 */
export function pruneStatusFiles(
  dataDir: string,
  options: { now?: number; maxAgeMs?: number; alive?: (pid: number) => boolean } = {},
): string[] {
  const directory = path.join(dataDir, STATUS_DIRECTORY);
  const now = options.now ?? Date.now();
  const maxAgeMs = options.maxAgeMs ?? STALE_STATUS_MS;
  const alive = options.alive ?? processAlive;
  const removed: string[] = [];
  let names: string[];
  try {
    names = readdirSync(directory);
  } catch {
    return removed;
  }
  for (const name of names) {
    const status = STATUS_NAME.exec(name);
    if (status === null && !TEMP_NAME.test(name)) {
      continue;
    }
    const file = path.join(directory, name);
    try {
      if (now - statSync(file).mtimeMs < maxAgeMs) {
        continue;
      }
      // A temporary file is only ever needed for the moment of one write.
      if (status !== null && alive(Number(status[1]))) {
        continue;
      }
      rmSync(file, { force: true });
      removed.push(file);
    } catch {
      // Gone already, or not ours to remove.
    }
  }
  return removed;
}

/** Whether a process with this id runs (one owned by another user counts as running). */
export function processAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as NodeJS.ErrnoException | null)?.code === 'EPERM';
  }
}

function writeFileAtomic(file: string, text: string): void {
  mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
  const temp = `${file}.${process.pid}.${randomUUID()}.tmp`;
  try {
    writeFileSync(temp, text, { mode: 0o600 });
    renameSync(temp, file);
  } finally {
    rmSync(temp, { force: true });
  }
}
