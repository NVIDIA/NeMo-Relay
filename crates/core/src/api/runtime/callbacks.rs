// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Callback type aliases used by the runtime middleware pipeline.
//!
//! The public middleware registration APIs accept boxed or shared closures with
//! the signatures defined in this module. These aliases centralize those
//! signatures so the runtime can compose tool and LLM middleware consistently
//! across bindings.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;

use crate::api::event::Event;
use crate::api::llm::LlmRequest;
use crate::codec::request::AnnotatedLlmRequest;
use crate::error::Result;
use crate::json::Json;

/// Sanitize a tool request payload before the runtime records it.
///
/// Tool sanitize callbacks are used only for observability payloads. They can
/// rewrite the JSON arguments recorded on tool-start events without changing
/// the caller-owned request that is passed to the tool implementation.
pub type ToolSanitizeFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
/// Decide whether a tool call is allowed to continue.
///
/// The callback receives the tool name and the current argument payload. It can
/// return `Ok(None)` to allow execution, `Ok(Some(reason))` to reject the call
/// with a guardrail message, or an error to abort evaluation entirely.
pub type ToolConditionalFn = Box<dyn Fn(&str, &Json) -> Result<Option<String>> + Send + Sync>;
pub(crate) type ToolConditionalSharedFn =
    Arc<dyn Fn(&str, &Json) -> Result<Option<String>> + Send + Sync>;
/// Rewrite tool arguments before execution.
///
/// Tool request intercepts run in priority order and can transform the JSON
/// payload that is eventually passed into the tool execution callback.
pub type ToolInterceptFn = Box<dyn Fn(&str, Json) -> Result<Json> + Send + Sync>;
/// Continuation type invoked by tool execution intercepts.
///
/// Execution intercepts receive this callable as their `next` continuation and
/// can call it with modified arguments, wrap it, or skip it entirely.
pub type ToolExecutionNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
/// Wrap or replace tool execution.
///
/// A tool execution intercept receives the tool name, the current argument
/// payload, and the continuation representing the rest of the chain.
pub type ToolExecutionFn = Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;

/// Sanitize an LLM request before the runtime records it.
///
/// LLM request sanitizers affect the serialized request payload emitted on
/// start events. They do not mutate the caller-owned [`LlmRequest`] unless a
/// separate request intercept does so.
pub type LlmSanitizeRequestFn = Box<dyn Fn(LlmRequest) -> LlmRequest + Send + Sync>;
/// Sanitize an LLM response before the runtime records it.
///
/// These callbacks rewrite the JSON response payload captured on LLM-end
/// events, which is useful for redaction or payload normalization.
pub type LlmSanitizeResponseFn = Box<dyn Fn(Json) -> Json + Send + Sync>;
/// Decide whether an LLM call is allowed to continue.
///
/// The callback receives the current [`LlmRequest`] and can allow execution,
/// reject it with a guardrail reason, or return an error.
pub type LlmConditionalFn = Box<dyn Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync>;
pub(crate) type LlmConditionalSharedFn =
    Arc<dyn Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync>;
/// Rewrite or annotate an LLM request before execution.
///
/// Request intercepts can transform the wire request, attach or replace a
/// normalized [`AnnotatedLlmRequest`], or both.
pub type LlmRequestInterceptFn = Box<
    dyn Fn(
            &str,
            LlmRequest,
            Option<AnnotatedLlmRequest>,
        ) -> Result<(LlmRequest, Option<AnnotatedLlmRequest>)>
        + Send
        + Sync,
>;
/// Continuation type invoked by non-streaming LLM execution intercepts.
///
/// Execution intercepts use this callable to continue the non-streaming LLM
/// pipeline after applying their own logic.
pub type LlmExecutionNextFn =
    Arc<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
/// Wrap or replace non-streaming LLM execution.
///
/// A non-streaming execution intercept receives the logical provider name, the
/// current request, and the continuation representing the rest of the chain.
pub type LlmExecutionFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;
/// Stream of JSON chunks produced by the managed streaming LLM pipeline.
pub type LlmJsonStream = Pin<Box<dyn Stream<Item = Result<Json>> + Send>>;
/// Per-chunk collector used by the streaming LLM runtime.
pub type LlmCollectorFn = Box<dyn FnMut(Json) -> Result<()> + Send>;
/// Finalizer used to synthesize the aggregate streaming response payload.
pub type LlmFinalizerFn = Box<dyn FnOnce() -> Json + Send>;
/// Scope-local registry references passed into streaming execution-chain builders.
pub type LlmStreamExecutionRegistryRef<'a> = &'a crate::registry::SortedRegistry<
    crate::api::registry::ExecutionIntercept<LlmStreamExecutionFn>,
>;
/// Slice of scope-local streaming execution registries.
pub type LlmStreamExecutionRegistryRefs<'a> = &'a [LlmStreamExecutionRegistryRef<'a>];

/// Continuation type invoked by streaming LLM execution intercepts.
///
/// This callable represents the remainder of the streaming LLM execution chain
/// and resolves to a stream of JSON response chunks.
pub type LlmStreamExecutionNextFn = Arc<
    dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>> + Send + Sync,
>;
/// Wrap or replace streaming LLM execution.
///
/// A streaming execution intercept can observe or modify the request before
/// invoking the continuation, and it can also replace the returned stream.
pub type LlmStreamExecutionFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>>
        + Send
        + Sync,
>;

/// Consume runtime lifecycle events after they are emitted.
///
/// Event subscribers are invoked for scope, tool, LLM, and mark events after
/// the runtime has built the final event payload.
pub type EventSubscriberFn = Arc<dyn Fn(&Event) + Send + Sync>;
