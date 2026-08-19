// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The refusal half of the inline-shell gate.
 *
 * pi's bang-prefixed shell (`!cmd`, and `!!cmd` to keep the output out of the
 * model's context) never touches the tool path: it goes to `emitUserBash`, so
 * none of the `tool_call` gating covers it. `user_bash` *is* interceptable --
 * a handler that returns a `BashResult` makes pi skip execution entirely and
 * record that result instead -- but pi gives the hook no block-and-reason
 * contract the way `tool_call` does. There is no `{block, reason}` here.
 *
 * So a refusal has to be a **synthetic failed `BashResult`**, and its shape is
 * a design decision rather than something pi dictates:
 *
 * - `exitCode` is {@link REFUSED_EXIT_CODE}, the shell convention for "found,
 *   but could not be executed". It is not 0 (which would read as success), not
 *   1 (which every failing command already uses, so a policy refusal would be
 *   indistinguishable from the command failing), and not 127 (which means "not
 *   found" and would send a user hunting for a missing binary).
 * - `output` is what the user sees in the terminal, and -- for `!cmd`, though
 *   not for `!!cmd` -- what lands in the model's context, because pi records a
 *   returned result through `recordBashResult` exactly as if the command had
 *   run. It is one attribution line followed by the gateway's reason verbatim,
 *   on the standing rule that the reason string is a prompt: it reaches a model
 *   unframed, so it should read as guidance rather than as an error code.
 * - `cancelled` and `truncated` are false: nothing was started, so nothing was
 *   interrupted, and the message is whole.
 *
 * **Never throw from the handler.** `emitUserBash` wraps handlers in
 * try/catch and moves on, so a thrown error fails *open* and the command runs
 * unchecked -- the opposite of `tool_call`, which has no try/catch and fails
 * closed. Every path here returns an explicit decision.
 */

/**
 * pi's `BashResult`, narrowed to the fields a synthetic refusal sets.
 *
 * Mirrored from pi `v0.84.0`, `core/bash-executor.ts`. `fullOutputPath` is
 * deliberately absent: it points at a temp file holding output too large to
 * inline, and a refusal has no output beyond its reason.
 */
export type BashResult = {
  output: string;
  exitCode: number | undefined;
  cancelled: boolean;
  truncated: boolean;
};

/**
 * Exit code reported for a command the gateway refused.
 *
 * 126 is the POSIX shell convention for a command that was found but could not
 * be executed -- a permission problem rather than a missing binary -- which is
 * the closest existing meaning to "a policy declined to run this".
 */
export const REFUSED_EXIT_CODE = 126;

/**
 * The tool name an inline shell command is gated under.
 *
 * Deliberately **not** `bash`. pi's `bash` tool and the bang prefix are two
 * different things with different provenance: one is proposed by the model, the
 * other is typed by the user. A guardrail only receives the tool name and the
 * arguments, so if both arrived as `bash` a policy could not tell them apart --
 * and "the model may not run shell commands" is a common rule that should not
 * also stop the human from typing `!git status`. Keeping the names distinct
 * makes the policy author choose; the cost is that a policy which wants to
 * cover both has to name both, which the docs state.
 */
export const USER_BASH_TOOL_NAME = 'user_bash';

/** Hook event posted when the gate opens, and its synthesized close. */
export const USER_BASH_HOOK = 'user_bash';
export const USER_BASH_END_HOOK = 'user_bash_end';

/** First line of every refusal, so the user knows what declined the command. */
const ATTRIBUTION = 'NeMo Relay blocked this command.';

/**
 * Build the result pi records in place of running the command.
 *
 * The reason is reproduced verbatim after the attribution line; it is the
 * guardrail's own words, and for `!cmd` the model reads it.
 */
export function refusalResult(reason: string): BashResult {
  return {
    output: `${ATTRIBUTION}\n\n${reason}`,
    exitCode: REFUSED_EXIT_CODE,
    cancelled: false,
    truncated: false,
  };
}

/**
 * The reason used when a request intercept rewrote an inline shell command.
 *
 * pi's `UserBashEventResult` can replace the *result* or supply custom
 * execution `operations`, but it cannot replace the command: both call sites
 * pass the original string on to `executeBash`, and neither reads anything back
 * out of the event. Taking over execution to run the rewrite instead
 * would mean reimplementing pi's shell selection, command prefix and
 * process-tree cancellation, which is a behavior change the sidecar has no
 * business making.
 *
 * So the rewrite cannot be honored, and the command is refused rather than run
 * unmodified -- the same rule the tool path already applies to a transform it
 * cannot apply safely, and for the same reason: running the original would
 * silently discard a policy decision.
 */
export function transformRefusalReason(): string {
  return (
    'A NeMo Relay policy rewrote the arguments for this command, but pi provides no way to ' +
    'execute a rewritten inline shell command -- the bang prefix runs the text you typed. The ' +
    'command was refused rather than run unmodified, because running it would ignore the policy. ' +
    'Re-run it with the change applied by hand, or ask the policy owner to gate the bash tool ' +
    'instead. This is a configuration problem in the policy, not a judgment about the command.'
  );
}
