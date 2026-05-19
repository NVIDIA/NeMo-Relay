// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(rustdoc::broken_intra_doc_links, rustdoc::private_intra_doc_links)]

//! # NeMo Flow Core
//!
//! The core runtime library for the NeMo Flow multi-language agent framework. This crate
//! provides execution scope management, lifecycle event tracking, and middleware pipelines
//! (guardrails and intercepts) for tool and LLM calls.
//!
//! ## Architecture
//!
//! The runtime is organized around a **global context**
//! ([`api::runtime::NemoFlowContextState`]) that holds all registered middleware
//! (guardrails, intercepts, subscribers) and a **scope stack**
//! ([`api::runtime::ScopeStack`]) that tracks the hierarchical execution context
//! via task-local or thread-local storage.
//!
//! ## Primary Entry Points
//!
//! Most integrations start with the high-level lifecycle helpers in [`api`]:
//!
//! - [`api::scope::push_scope`] / [`api::scope::pop_scope`] create nested execution scopes.
//! - [`api::tool::tool_call_execute`] runs a complete tool middleware pipeline.
//! - [`api::llm::llm_call_execute`] and [`api::llm::llm_stream_call_execute`] run non-streaming
//!   and streaming LLM middleware pipelines.
//! - [`api::registry`] exposes global and scope-local middleware registration APIs.
//! - [`api::subscriber`] exposes lifecycle event subscriber registration APIs.
//!
//! ### Modules
//!
//! - [`api`] — Public API functions, handles, lifecycle event types, runtime helpers,
//!   and guardrail/intercept/subscriber registration. These are the primary entry points.
//! - [`error`] — Error types ([`error::FlowError`]) and the [`error::Result`] type alias.
//! - [`json`] — JSON type alias ([`json::Json`]) and the [`json::merge_json`] utility.
//! - [`observability`] — Built-in observability backends including
//!   [`atif::AtifExporter`](observability::atif::AtifExporter),
//!   [`otel::OpenTelemetrySubscriber`](observability::otel::OpenTelemetrySubscriber),
//!   and [`openinference::OpenInferenceSubscriber`](observability::openinference::OpenInferenceSubscriber).
//! - [`registry`] — [`SortedRegistry`](registry::SortedRegistry) — a priority-sorted, named collection used for
//!   all guardrail and intercept registries.
//! - [`stream`] — [`stream::LlmStreamWrapper`] — a stream adapter that applies per-chunk
//!   intercepts and aggregates streaming LLM responses.
//!
//! ## Middleware Pipeline
//!
//! Both tool and LLM calls flow through a configurable middleware pipeline:
//!
//! 1. **Request intercepts** — transform the request before execution
//! 2. **Sanitize request guardrails** — sanitize/normalize the request
//! 3. **Conditional execution guardrails** — gate execution (reject if criteria not met)
//! 4. **Execution intercepts** — optionally replace the execution function entirely
//! 5. **Sanitize response guardrails** — sanitize/normalize the response
//!
//! All middleware is priority-ordered (ascending) and registered by name for
//! easy addition and removal at runtime.
pub mod api;
pub mod codec;
pub mod config_editor;
mod context;
pub mod error;
pub mod json;
pub mod nemoguardrails;
pub mod observability;
pub mod plugin;
pub mod registry;
#[doc(hidden)]
pub mod shared_runtime;
pub mod stream;

#[cfg(test)]
#[path = "../tests/unit/types_tests.rs"]
mod types_tests;
