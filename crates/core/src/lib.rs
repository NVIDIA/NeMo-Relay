// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # NeMo Flow Core
//!
//! The core runtime library for the NeMo Flow multi-language agent framework. This crate
//! provides execution scope management, lifecycle event tracking, and middleware pipelines
//! (guardrails and intercepts) for tool and LLM calls.
//!
//! ## Architecture
//!
//! The runtime is organized around a **global context** ([`NemoFlowContextState`]) that holds
//! all registered middleware (guardrails, intercepts, subscribers) and a **scope stack**
//! ([`ScopeStack`]) that tracks the hierarchical execution context via task-local or
//! thread-local storage.
//!
//! ### Modules
//!
//! - [`api`] — Public API functions for scope management, tool/LLM lifecycle operations,
//!   and guardrail/intercept/subscriber registration. These are the primary entry points.
//! - [`atif`] — ATIF (Agent Trajectory Interchange Format) trajectory exporter.
//!   Provides [`AtifExporter`](atif::AtifExporter) — an event subscriber that collects
//!   lifecycle events and exports them as ATIF v1.6 trajectories. Also defines the
//!   ATIF data types: [`AtifTrajectory`](atif::AtifTrajectory),
//!   [`AtifStep`](atif::AtifStep), [`AtifToolCall`](atif::AtifToolCall),
//!   [`AtifObservation`](atif::AtifObservation), and [`AtifMetrics`](atif::AtifMetrics).
//! - [`context`] — Global context singleton, scope stack, task-local/thread-local storage,
//!   callable type aliases, and middleware chain execution logic.
//! - [`error`] — Error types ([`FlowError`]) and the [`Result`] type alias.
//! - [`json`] — JSON type alias ([`Json`]) and the [`merge_json`] utility.
//! - [`registry`] — [`SortedRegistry`](registry::SortedRegistry) — a priority-sorted, named collection used for
//!   all guardrail and intercept registries.
//! - [`stream`] — [`LlmStreamWrapper`] — a stream adapter that applies per-chunk
//!   intercepts and aggregates streaming LLM responses.
//! - [`types`] — Core data types: attribute bitflags, enums ([`ScopeType`]),
//!   handle structs ([`ScopeHandle`], [`ToolHandle`], [`LLMHandle`]), [`LLMRequest`],
//!   [`Event`] (with typed lifecycle fields: `input`, `output`, `model_name`,
//!   `tool_call_id`) and middleware container types.
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
pub mod atif;
pub mod codec;
pub mod context;
pub mod error;
pub mod json;
pub mod observability;
pub mod plugin;
pub mod registry;
#[doc(hidden)]
pub mod shared_runtime;
pub mod stream;
pub mod types;
