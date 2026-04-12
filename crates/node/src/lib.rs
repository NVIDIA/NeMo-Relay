// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NAPI-RS bindings for NeMo Flow, exposing the agent runtime framework to Node.js.
//!
//! This crate provides JavaScript/TypeScript access to scope management, tool and LLM
//! lifecycle operations, guardrails, intercepts, event subscriptions, and ATIF trajectory
//! export via NAPI-RS. Doc comments on `#[napi]` items are emitted into the generated
//! `index.d.ts` TypeScript definitions.
//!
//! Tool calls accept an optional `toolCallId` and LLM calls accept an optional `modelName`
//! for ATIF trajectory correlation. The `JsAtifExporter` class collects lifecycle events
//! and exports ATIF v1.6 trajectories.

#![allow(dead_code)]

mod api;
mod callable;
mod convert;
mod promise_call;
mod stream;
mod types;

#[cfg(test)]
#[path = "../tests/integration/api_tests.rs"]
mod integration_tests;

#[cfg(test)]
#[unsafe(no_mangle)]
extern "C" fn napi_release_threadsafe_function(
    _func: napi::sys::napi_threadsafe_function,
    _mode: napi::sys::napi_threadsafe_function_release_mode,
) -> napi::sys::napi_status {
    0
}

#[cfg(test)]
#[unsafe(no_mangle)]
extern "C" fn napi_call_threadsafe_function(
    _func: napi::sys::napi_threadsafe_function,
    _data: *mut std::ffi::c_void,
    _mode: napi::sys::napi_threadsafe_function_call_mode,
) -> napi::sys::napi_status {
    0
}
