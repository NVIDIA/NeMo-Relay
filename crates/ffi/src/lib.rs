//! C FFI layer for NVAgentRT.
//!
//! This crate exposes the NVAgentRT core runtime as a C-compatible shared library.
//! It is consumed by the Go bindings via CGo and generates a C header file
//! (`nvagentrt.h`) through `cbindgen`. All exported symbols use the `nv_agentrt_`
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
//! 5. **Actual execution** -- invoke the user-provided callback.
//! 6. **Response intercepts** -- transform the response.
//! 7. **Sanitize-response guardrails** -- validate/sanitize the response.
//!
//! # Error Handling
//!
//! Every `extern "C"` function returns an [`error::NvAgentRtStatus`] code. On
//! failure, call [`error::nv_agentrt_last_error`] on the same thread to retrieve
//! a human-readable error description. The error is stored in thread-local
//! storage and is valid until the next FFI call on that thread.
//!
//! # Memory Ownership
//!
//! All opaque handles (`FfiScopeHandle`, `FfiToolHandle`, `FfiLLMHandle`, etc.)
//! are heap-allocated and must be freed through their corresponding
//! `nv_agentrt_*_free` functions. C strings returned by accessor functions must
//! be freed with `nv_agentrt_string_free`.
//!
//! # Modules
//!
//! - [`api`] -- Top-level FFI entry points (scope, tool, LLM, guardrail, intercept, subscriber).
//! - [`types`] -- C-compatible struct and enum definitions.
//! - [`error`] -- Status codes and thread-local error storage.
//! - [`callable`] -- C function pointer typedefs and wrapper functions.
//! - [`convert`] -- JSON and C-string conversion utilities.

pub mod api;
pub mod callable;
pub mod convert;
pub mod error;
pub mod types;
