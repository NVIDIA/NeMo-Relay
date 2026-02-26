// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # NVAgentRT Core
//!
//! The core runtime library for the NVAgentRT multi-language agent framework. This crate
//! provides execution scope management, lifecycle event tracking, and middleware pipelines
//! (guardrails and intercepts) for tool and LLM calls.
//!
//! ## Architecture
//!
//! The runtime is organized around a **global context** ([`NVAgentRTContextState`]) that holds
//! all registered middleware (guardrails, intercepts, subscribers) and a **scope stack**
//! ([`ScopeStack`]) that tracks the hierarchical execution context via task-local or
//! thread-local storage.
//!
//! ### Modules
//!
//! - [`api`] — Public API functions for scope management, tool/LLM lifecycle operations,
//!   and guardrail/intercept/subscriber registration. These are the primary entry points.
//! - [`context`] — Global context singleton, scope stack, task-local/thread-local storage,
//!   callable type aliases, and middleware chain execution logic.
//! - [`error`] — Error types ([`AgentRtError`]) and the [`Result`] type alias.
//! - [`json`] — JSON type alias ([`Json`]) and the [`merge_json`] utility.
//! - [`registry`] — [`SortedRegistry`](registry::SortedRegistry) — a priority-sorted, named collection used for
//!   all guardrail and intercept registries.
//! - [`stream`] — [`LlmStreamWrapper`] — an SSE stream adapter that buffers, parses,
//!   intercepts, and aggregates streaming LLM responses.
//! - [`types`] — Core data types: attribute bitflags, enums ([`ScopeType`], [`EventType`]),
//!   handle structs ([`ScopeHandle`], [`ToolHandle`], [`LLMHandle`]), [`LLMRequest`],
//!   [`SseEvent`], [`Event`], and middleware container types.
//!
//! ## Middleware Pipeline
//!
//! Both tool and LLM calls flow through a configurable middleware pipeline:
//!
//! 1. **Request intercepts** — transform the request before execution
//! 2. **Sanitize request guardrails** — sanitize/normalize the request
//! 3. **Conditional execution guardrails** — gate execution (reject if criteria not met)
//! 4. **Execution intercepts** — optionally replace the execution function entirely
//! 5. **Response intercepts** — transform the response after execution
//! 6. **Sanitize response guardrails** — sanitize/normalize the response
//!
//! All middleware is priority-ordered (ascending) and registered by name for
//! easy addition and removal at runtime.

pub mod api;
pub mod context;
pub mod error;
pub mod json;
pub mod registry;
pub mod stream;
pub mod types;

pub use api::*;
pub use context::{
    create_scope_stack, current_scope_stack, global_context, set_thread_scope_stack,
    task_scope_push, task_scope_remove, task_scope_top, EventSubscriberFn, LlmConditionalFn,
    LlmExecutionConditionalFn, LlmExecutionFn, LlmRequestInterceptFn, LlmResponseInterceptFn,
    LlmSanitizeRequestFn, LlmSanitizeResponseFn, LlmStreamExecutionConditionalFn,
    LlmStreamExecutionFn, LlmStreamResponseInterceptFn, NVAgentRTContextState, ScopeStack,
    ScopeStackHandle, ToolConditionalFn, ToolExecutionConditionalFn, ToolExecutionFn,
    ToolInterceptFn, ToolSanitizeFn, TASK_SCOPE_STACK,
};
pub use error::{AgentRtError, Result};
pub use json::{merge_json, Json};
pub use stream::LlmStreamWrapper;
pub use types::*;
