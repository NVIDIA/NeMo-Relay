// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Codec identity available while a managed LLM event is sanitized. */
export type LlmCodecIdentity =
  | { kind: 'none' }
  | {
      kind: 'builtin';
      id: 'openai_chat' | 'openai_responses' | 'anthropic_messages' | 'oci_genai' | 'gemini_generate_content';
    }
  | { kind: 'runtime'; id: string }
  | { kind: 'opaque' };

/** Codec context available while an LLM request is sanitized. */
export interface LlmSanitizeRequestContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): import('./typed').LlmCodec | null;
}

/** Codec context available while an LLM response is sanitized. */
export interface LlmSanitizeResponseContext {
  codec: LlmCodecIdentity;
  /** Resolve the active codec for this callback. Do not retain the result after the callback returns. */
  resolveCodec(): import('./typed').LlmResponseCodec | null;
}

/** Codec capabilities for one managed LLM execution intercept invocation. */
export interface LlmExecutionContext {
  /** Request codec identity plus optional decode and encode capability. */
  requestCodec: LlmSanitizeRequestContext;
  /** Unary response codec identity plus optional decode capability; `null` for streaming execution. */
  responseCodec: LlmSanitizeResponseContext | null;
}

/** Schema tag attached to an opaque optimization contribution payload. */
export interface LlmOptimizationDataSchema {
  name: string;
  version: string;
}

/** Model identity retained for counterfactual pricing and downstream repricing. */
export interface LlmOptimizationModel {
  model: string;
  provider?: string;
}

/** Baseline and effective model identities for a routing optimization. */
export interface LlmOptimizationModelTransition {
  baseline?: LlmOptimizationModel;
  effective?: LlmOptimizationModel;
}

/** Explicit token evidence, independent from a pricing catalog. */
export interface LlmOptimizationTokens {
  /** Token counts must be non-negative JavaScript safe integers. */
  prompt_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  completion_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  cache_read_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  cache_write_tokens?: number;
  /** Token counts must be non-negative JavaScript safe integers. */
  total_tokens?: number;
}

/** Baseline, effective, and saved token evidence for one optimization. */
export interface LlmOptimizationTokenImpact {
  baseline?: LlmOptimizationTokens;
  effective?: LlmOptimizationTokens;
  saved?: LlmOptimizationTokens;
  quality?: 'observed' | 'estimated';
  estimation_method?: string;
}

/**
 * One plugin's optimization evidence.
 *
 * `kind` is deliberately an open string so new optimizer categories round-trip
 * without a Relay release. Unknown top-level fields are retained by the wire
 * contract and represented by this interface's JSON extension surface.
 */
export interface LlmOptimizationContribution {
  id?: string;
  /** Relay ordering must remain within JavaScript's safe-integer range. */
  sequence?: number;
  producer: string;
  kind: 'input_compression' | 'model_routing' | (string & {});
  applied: boolean;
  model_transition?: LlmOptimizationModelTransition;
  token_impact?: LlmOptimizationTokenImpact;
  payload_schema?: LlmOptimizationDataSchema;
  payload?: Json;
  [key: string]: Json | undefined;
}

/** Canonical result returned by an LLM request intercept. */
export interface LlmRequestInterceptOutcome {
  request: Json;
  annotated?: Json | null;
  pendingMarks?: PendingMarkSpec[];
  optimizationContributions?: LlmOptimizationContribution[];
}

/** Scalar value accepted in event metadata additions. */
export type EventMetadataScalar = string | number | boolean;

/**
 * Flat value accepted in event metadata additions. After JSON conversion,
 * numeric arrays must contain only integer values or only floating-point values.
 */
export type EventMetadataValue = EventMetadataScalar | string[] | number[] | boolean[];

/** Metadata additions returned by an event metadata injector. */
export type EventMetadata = Record<string, EventMetadataValue>;
