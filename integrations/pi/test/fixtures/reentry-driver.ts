// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Test fixture: forces exactly one agent-run re-entry.
 *
 * pi re-enters the agent run from `_handlePostAgentRun` on three paths --
 * provider retry, compaction, and a queued follow-up. The queued-follow-up path
 * is the only one an extension can trigger deterministically, and pi documents
 * it explicitly: messages queued by an `agent_end` handler "need a
 * continuation", so `agent.continue()` runs and a fresh `agent_start` fires with
 * `turnIndex` reset to 0.
 *
 * This drives the *real* re-entry path rather than simulating it, which is the
 * point -- the colliding turn indices it produces are pi's, not the fixture's.
 *
 * Load it alongside the Relay extension:
 *   pi -e <relay-extension> -e <this file>
 */
import type { ExtensionAPI } from '../../src/pi-hook-types.ts';

/** pi's `sendUserMessage`, which the mirrored type subset does not declare. */
type ReentryCapableApi = ExtensionAPI & {
  sendUserMessage(text: string, options?: { deliverAs?: 'steer' | 'followUp' }): void;
};

export default function reentryDriver(pi: ReentryCapableApi): void {
  let fired = false;

  pi.on('agent_end', async () => {
    // Fire once only: queueing on every agent_end would re-enter forever.
    if (fired) return;
    fired = true;
    pi.sendUserMessage('Now reply with exactly the word: done', { deliverAs: 'followUp' });
  });
}
