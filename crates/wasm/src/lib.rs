// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WebAssembly bindings for the Nexus agent runtime framework.
//!
//! This crate exposes the core Nexus API to JavaScript/TypeScript via
//! `wasm-bindgen`. It provides scope management, tool and LLM lifecycle
//! operations, guardrail/intercept registration, event subscriptions, and
//! streaming LLM responses.
//!
//! # Modules
//!
//! - [`api`] -- Public `#[wasm_bindgen]` functions that form the top-level API
//!   surface: scope operations, tool/LLM lifecycle, guardrail and intercept
//!   registration, and event subscribers.
//! - [`types`] -- WASM-friendly wrapper types (`WasmScopeHandle`,
//!   `WasmToolHandle`, `WasmLLMHandle`, `WasmLLMRequest`, `WasmEvent`) and
//!   integer constants for scope types and attribute flags. `WasmEvent` exposes
//!   typed lifecycle fields (`input`, `output`, `model_name`, `tool_call_id`,
//!   `root_uuid`).
//! - [`stream`] -- `WasmLlmStream`, an async-iterator-like wrapper for
//!   consuming streaming LLM responses from JavaScript.
//!
//! Tool calls accept an optional `tool_call_id` and LLM calls accept an optional
//! `model_name` for ATIF trajectory correlation. The `WasmAtifExporter` class
//! collects lifecycle events and exports ATIF v1.6 trajectories.
//! - `callable` (internal) -- Adapters that convert JavaScript callback
//!   functions into the Rust closure signatures expected by the core runtime.
//! - `convert` (internal) -- JSON marshalling helpers between `JsValue` and
//!   `serde_json::Value`.
//!
//! # Middleware Pipeline
//!
//! The middleware pipeline for both tool and LLM calls follows this order:
//!
//! 1. **Request intercepts** -- transform the request/arguments.
//! 2. **Sanitize-request guardrails** -- sanitize the request data.
//! 3. **Conditional-execution guardrails** -- gate whether execution proceeds.
//! 4. **Execution intercepts** -- optionally replace the execution function.
//! 5. **Execution** -- the actual tool/LLM function runs.
//! 6. **Response intercepts** -- transform the response/result.
//! 7. **Sanitize-response guardrails** -- sanitize the response data.

#![allow(dead_code)]

/// Public API functions exposed to JavaScript via `wasm_bindgen`.
///
/// Contains all top-level functions for scope management, tool and LLM
/// lifecycle operations, guardrail/intercept registration, and event
/// subscriber management.
pub mod api;
/// Internal adapters that convert JS callback functions into Rust closures
/// matching the core runtime's expected signatures.
mod callable;
/// Internal JSON conversion utilities for marshalling data across the
/// JS/Rust boundary.
mod convert;
/// Streaming LLM response wrapper for async iteration from JavaScript.
pub mod stream;
/// WASM-friendly wrapper types and integer constants exposed to JavaScript.
pub mod types;
