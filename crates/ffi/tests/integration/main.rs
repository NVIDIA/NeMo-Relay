// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test support for the NeMo Relay FFI crate.

use libc::c_char;
use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmAttributes, LlmHandle, LlmRequest};
use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use nemo_relay::api::scope::{ScopeAttributes, ScopeHandle, ScopeType};
use nemo_relay::api::tool::{ToolAttributes, ToolHandle};
use nemo_relay::codec::request::AnnotatedLlmRequest as AnnotatedLLMRequest;
use nemo_relay::error::FlowError;
use nemo_relay_ffi::api::*;
use nemo_relay_ffi::callable::*;
use nemo_relay_ffi::convert::*;
use nemo_relay_ffi::error::*;
use nemo_relay_ffi::types::*;
use nemo_relay_ffi::{api, convert, error};
use serde_json::{Value as Json, json};
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

unsafe fn nemo_relay_string_free_internal(ptr: *mut c_char) {
    unsafe { nemo_relay_string_free(ptr) };
}

mod api_tests;
mod callable_extra_tests;
#[path = "../unit/callable_tests.rs"]
mod callable_tests;
#[path = "../coverage/convert_tests.rs"]
mod convert_coverage_tests;
#[path = "../coverage/error_tests.rs"]
mod error_coverage_tests;
mod plugin_activation_tests;
#[path = "../unit/types_tests.rs"]
mod types_tests;
