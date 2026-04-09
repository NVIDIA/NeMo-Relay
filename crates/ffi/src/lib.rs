// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C FFI layer for NeMo Flow.
//!
//! This crate exposes the NeMo Flow core runtime as a C-compatible shared library.
//! It is consumed by the Go bindings via CGo and generates a C header file
//! (`nemo_flow.h`) through `cbindgen`. All exported symbols use the `nemo_flow_`
//! prefix.
//!
//! # Middleware Pipeline
//!
//! When a tool or LLM call is executed end-to-end via the `_execute` functions,
//! the runtime applies the following middleware pipeline in order:
//!
//! 1. **Request intercepts** -- transform the request before guardrails.
//! 2. **Sanitize-request guardrails** -- validate/sanitize the request.
//! 3. **Conditional-execution guardrails** -- gate whether the call proceeds.
//! 4. **Execution intercepts** -- optionally replace the call implementation.
//! 5. **Sanitize-response guardrails** -- validate/sanitize the response.
//!
//! # Error Handling
//!
//! Every `extern "C"` function returns an [`error::NemoFlowStatus`] code. On
//! failure, call [`error::nemo_flow_last_error`] on the same thread to retrieve
//! a human-readable error description. The error is stored in thread-local
//! storage and is valid until the next FFI call on that thread.
//!
//! # Memory Ownership
//!
//! All opaque handles (`FfiScopeHandle`, `FfiToolHandle`, `FfiLLMHandle`, etc.)
//! are heap-allocated and must be freed through their corresponding
//! `nemo_flow_*_free` functions. C strings returned by accessor functions must
//! be freed with `nemo_flow_string_free`.
//!
//! # Modules
//!
//! - [`api`] -- Top-level FFI entry points (scope, tool, LLM, guardrail, intercept,
//!   subscriber, ATIF exporter). Tool calls accept an optional `tool_call_id` and
//!   LLM calls accept an optional `model_name` for ATIF trajectory correlation.
//!   ATIF exporter functions (`nemo_flow_atif_exporter_*`) create, register,
//!   export, and clear trajectory data.
//! - [`types`] -- C-compatible struct and enum definitions, plus event accessor
//!   functions (`nemo_flow_event_input`, `_output`, `_model_name`, `_tool_call_id`,
//!   `_parent_uuid`, `_scope_type`) and the `FfiAtifExporter`
//!   opaque handle.
//! - [`error`] -- Status codes and thread-local error storage.
//! - [`callable`] -- C function pointer typedefs and wrapper functions.
//! - [`convert`] -- JSON and C-string conversion utilities.

pub mod api;
pub mod callable;
pub mod convert;
pub mod error;
pub mod types;
