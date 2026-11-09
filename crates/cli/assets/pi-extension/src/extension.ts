// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  ExtensionAPI,
  ExtensionContext,
  ProviderConfig,
  ToolCallEventResult,
  UserBashEvent,
  UserBashEventResult,
} from '@earendil-works/pi-coding-agent';

import { applyTransform, decideTransform, jsonType, refusalReason } from './argument-transform.ts';
import { type FaultPolicy, type HookOutcome, type Verdict, errorDetail, resolveFault } from './gateway.ts';
import {
  type RedirectConfig,
  type RedirectDecision,
  type RedirectModel,
  decideRedirect,
  isNotable,
} from './provider-redirect.ts';
import { summarizeToolResult } from './tool-result.ts';
import {
  USER_BASH_END_HOOK,
  USER_BASH_HOOK,
  USER_BASH_TOOL_NAME,
  refusalResult,
  transformRefusalReason,
} from './user-bash.ts';

export type HookPayload = Record<string, unknown>;

export type Redirect = Extract<RedirectDecision, { kind: 'redirect' }>;

export type Wiring = {
  redirect: RedirectConfig;
  post: (payload: HookPayload) => Promise<HookOutcome>;
  close: (payload: HookPayload) => Promise<HookOutcome>;
  providerRegistration: (decision: Redirect, sessionId: string) => ProviderConfig;
  ready?: () => Promise<void>;
};

export type ExtensionOptions = {
  onFault: FaultPolicy;
  bind: () => Promise<Wiring>;
};

type UserBashStatus = 'policy-allowed' | 'fault-allowed' | 'error';

export function createExtension(pi: ExtensionAPI, options: ExtensionOptions): void {
  let wiring: Promise<Wiring> | undefined;
  let attemptIndex = 0;
  let turnSeq = 0;
  let userBashSeq = 0;
  let chain: Promise<unknown> = Promise.resolve();
  const redirectedProviders = new Set<string>();
  let registeredSessionKey: string | null = null;
  let pendingSessionStart: HookPayload | undefined;
  const pristineCatalog = new Map<string, RedirectModel[]>();

  const bound = (): Promise<Wiring> => (wiring ??= options.bind());

  function enqueue<T>(job: () => Promise<T>): Promise<T> {
    const result = chain.then(job, job);
    chain = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  const attribution = (): { attempt_index: number; turn_seq: number } => ({
    attempt_index: Math.max(0, attemptIndex - 1),
    turn_seq: Math.max(0, turnSeq - 1),
  });

  const payload = (ctx: ExtensionContext, hookEventName: string, fields: HookPayload): HookPayload => ({
    session_id: ctx.sessionManager.getSessionId(),
    hook_event_name: hookEventName,
    ...fields,
  });

  async function deliver(body: HookPayload): Promise<HookOutcome> {
    let active: Wiring;
    try {
      active = await bound();
    } catch (error) {
      return { kind: 'fault', origin: 'configuration', detail: errorDetail(error) };
    }
    return active.post(body);
  }

  const report = (body: HookPayload, outcome: HookOutcome): void => {
    if (outcome.kind === 'allow') return;
    const detail = outcome.kind === 'block' ? outcome.reason : outcome.detail;
    console.error(`NeMo Relay ${String(body.hook_event_name)} hook was not accepted: ${detail}`);
  };

  const emit = (body: HookPayload): void => {
    void enqueue(() => deliver(body)).then((outcome) => report(body, outcome));
  };

  const emitOrdered = async (body: HookPayload): Promise<HookOutcome> => {
    const outcome = await enqueue(() => deliver(body));
    report(body, outcome);
    return outcome;
  };

  async function gate(body: HookPayload, toolName: string): Promise<{ verdict: Verdict; faulted: boolean }> {
    const outcome = await enqueue(() => deliver(body));
    if (outcome.kind === 'fault') {
      return { verdict: resolveFault(options.onFault, outcome, toolName), faulted: true };
    }
    return { verdict: outcome, faulted: false };
  }

  const siblingsOf = (ctx: ExtensionContext, provider: string): RedirectModel[] =>
    pristineCatalog.get(provider) ?? ctx.modelRegistry.getAll().filter((candidate) => candidate.provider === provider);

  async function applyRedirect(ctx: ExtensionContext, source: 'session_start' | 'model_select'): Promise<void> {
    const active = await bound();
    if (active.ready) {
      try {
        await active.ready();
      } catch (error) {
        console.error(`NeMo Relay route is not ready, so the provider redirect waits: ${errorDetail(error)}`);
        return;
      }
    }
    if (pendingSessionStart !== undefined) {
      const retried = await emitOrdered(pendingSessionStart);
      if (retried.kind === 'fault') return;
      pendingSessionStart = undefined;
    }
    const sessionKey = ctx.sessionManager.getSessionId();
    if (registeredSessionKey !== null && registeredSessionKey !== sessionKey) {
      redirectedProviders.clear();
    }
    registeredSessionKey = sessionKey;
    const model = ctx.model;
    const siblings = model ? siblingsOf(ctx, model.provider) : [];
    const native = model !== undefined && ctx.modelRegistry.getRegisteredNativeProvider(model.provider) !== undefined;
    const decision = decideRedirect(model, active.redirect, redirectedProviders, siblings, native);
    if (decision.kind === 'redirect') {
      pristineCatalog.set(decision.provider, siblings);
      pi.registerProvider(decision.provider, active.providerRegistration(decision, sessionKey));
      redirectedProviders.add(decision.provider);
    }
    if (!isNotable(decision)) return;
    emit(
      payload(ctx, 'model_redirect', {
        source,
        outcome: decision.kind,
        ...(decision.kind === 'skip' ? { code: decision.code } : {}),
        reason: decision.reason,
        ...(decision.provider ? { provider: decision.provider } : {}),
        ...(decision.api ? { model_api: decision.api } : {}),
        ...(decision.kind === 'redirect' ? { upstream: decision.upstream } : {}),
        ...(model ? { model_id: model.id } : {}),
        ...attribution(),
      }),
    );
  }

  const endUserBash = (ctx: ExtensionContext, callId: string, status: UserBashStatus, content: string): void => {
    emit(
      payload(ctx, USER_BASH_END_HOOK, {
        tool_call_id: callId,
        tool_name: USER_BASH_TOOL_NAME,
        result: { content },
        status,
        ...attribution(),
      }),
    );
  };

  pi.on('session_start', async (event, ctx) => {
    const body = payload(ctx, 'session_start', {
      reason: event.reason,
      cwd: ctx.cwd,
      ...(event.previousSessionFile ? { previous_session_file: event.previousSessionFile } : {}),
    });
    const gated = await bound().then(
      (active) => active.ready !== undefined,
      () => false,
    );
    if (!gated) {
      emit(body);
      await applyRedirect(ctx, 'session_start');
      return;
    }
    const outcome = await emitOrdered(body);
    if (outcome.kind === 'fault') {
      pendingSessionStart = body;
      return;
    }
    await applyRedirect(ctx, 'session_start');
  });

  pi.on('model_select', async (_event, ctx) => {
    await applyRedirect(ctx, 'model_select');
  });

  pi.on('session_shutdown', async (event, ctx) => {
    if (event.reason === 'reload') {
      await chain;
      return;
    }
    const body = payload(ctx, 'session_shutdown', {
      reason: event.reason,
      ...(event.targetSessionFile ? { target_session_file: event.targetSessionFile } : {}),
    });
    if (event.reason === 'quit') {
      await chain;
      const active = await bound();
      report(body, await active.close(body));
      return;
    }
    await emitOrdered(body);
  });

  pi.on('agent_start', async (_event, ctx) => {
    emit(payload(ctx, 'agent_start', { attempt_index: attemptIndex }));
    attemptIndex += 1;
  });

  pi.on('agent_end', async (event, ctx) => {
    emit(
      payload(ctx, 'agent_end', {
        attempt_index: Math.max(0, attemptIndex - 1),
        message_count: event.messages.length,
      }),
    );
  });

  pi.on('agent_settled', async (_event, ctx) => {
    emit(payload(ctx, 'agent_settled', { attempts: attemptIndex, ...attribution() }));
    attemptIndex = 0;
  });

  pi.on('turn_start', async (event, ctx) => {
    if (pendingSessionStart !== undefined) await applyRedirect(ctx, 'session_start');
    const seq = turnSeq;
    turnSeq += 1;
    await emitOrdered(
      payload(ctx, 'turn_start', {
        turn_index: event.turnIndex,
        turn_seq: seq,
        attempt_index: Math.max(0, attemptIndex - 1),
      }),
    );
  });

  pi.on('turn_end', async (event, ctx) => {
    await emitOrdered(payload(ctx, 'turn_end', { turn_index: event.turnIndex, ...attribution() }));
  });

  pi.on('session_before_compact', async (event, ctx) => {
    emit(
      payload(ctx, 'session_before_compact', {
        reason: event.reason,
        will_retry: event.willRetry,
        tokens_before: event.preparation.tokensBefore,
        is_split_turn: event.preparation.isSplitTurn,
        ...attribution(),
      }),
    );
  });

  pi.on('session_compact', async (event, ctx) => {
    emit(
      payload(ctx, 'session_compact', {
        reason: event.reason,
        will_retry: event.willRetry,
        from_extension: event.fromExtension,
        tokens_before: event.compactionEntry.tokensBefore,
        ...attribution(),
      }),
    );
  });

  pi.on('tool_call', async (event, ctx): Promise<ToolCallEventResult | undefined> => {
    const { verdict } = await gate(
      payload(ctx, 'tool_call', {
        tool_call_id: event.toolCallId,
        tool_name: event.toolName,
        input: event.input,
        ...attribution(),
      }),
      event.toolName,
    );
    if (verdict.kind === 'block') return { block: true, reason: verdict.reason };
    const transform = decideTransform(verdict.body, event.toolCallId, event.input);
    if (transform.kind === 'refuse') {
      return { block: true, reason: refusalReason(event.toolName, transform.reason) };
    }
    if (transform.kind === 'apply') {
      applyTransform(event.input, transform.input);
      emit(
        payload(ctx, 'tool_arguments_transformed', {
          tool_call_id: event.toolCallId,
          tool_name: event.toolName,
          ...attribution(),
        }),
      );
    }
    return undefined;
  });

  pi.on('tool_execution_end', async (event, ctx) => {
    emit(
      payload(ctx, 'tool_execution_end', {
        tool_call_id: event.toolCallId,
        tool_name: event.toolName,
        result: summarizeToolResult(event.result, event.isError),
        status: event.isError ? 'error' : 'ok',
        ...attribution(),
      }),
    );
  });

  pi.on('user_bash', async (event, ctx): Promise<UserBashEventResult | undefined> => {
    try {
      return await gateUserBash(event, ctx);
    } catch (error) {
      const fault = resolveFault(
        options.onFault,
        { kind: 'fault', origin: 'handler', detail: `the inline-shell gate failed: ${errorDetail(error)}` },
        USER_BASH_TOOL_NAME,
      );
      return fault.kind === 'block' ? { result: refusalResult(fault.reason) } : undefined;
    }
  });

  async function gateUserBash(event: UserBashEvent, ctx: ExtensionContext): Promise<UserBashEventResult | undefined> {
    const callId = `user-bash-${userBashSeq++}`;
    const input = {
      command: event.command,
      cwd: event.cwd,
      exclude_from_context: event.excludeFromContext,
    };
    const { verdict, faulted } = await gate(
      payload(ctx, USER_BASH_HOOK, {
        tool_call_id: callId,
        tool_name: USER_BASH_TOOL_NAME,
        input,
        ...attribution(),
      }),
      USER_BASH_TOOL_NAME,
    );
    if (verdict.kind === 'block') {
      endUserBash(ctx, callId, 'error', verdict.reason);
      return { result: refusalResult(verdict.reason) };
    }
    const envelope = verdict.body.tool_call;
    if (envelope !== undefined && jsonType(envelope) !== 'object') {
      const reason = refusalReason(
        USER_BASH_TOOL_NAME,
        `the tool_call envelope is ${jsonType(envelope)}, not an object`,
      );
      endUserBash(ctx, callId, 'error', reason);
      return { result: refusalResult(reason) };
    }
    if (envelope?.input !== undefined) {
      const reason = transformRefusalReason();
      endUserBash(ctx, callId, 'error', reason);
      return { result: refusalResult(reason) };
    }
    endUserBash(
      ctx,
      callId,
      faulted ? 'fault-allowed' : 'policy-allowed',
      faulted
        ? 'Allowed without a policy decision: the gateway did not answer and the fail-open policy applies.'
        : 'Allowed by policy; pi has not reported the outcome.',
    );
    return undefined;
  }
}
