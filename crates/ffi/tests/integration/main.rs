// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod api {
    pub use nemo_flow_ffi::api::*;
}

pub mod callable {
    pub use nemo_flow_ffi::callable::*;
}

pub mod convert {
    pub use nemo_flow_ffi::convert::*;
}

pub mod error {
    pub use nemo_flow_ffi::error::*;
}

pub mod types {
    pub use nemo_flow_ffi::types::*;
}

pub use api::*;
pub use callable::*;
pub use convert::*;
pub use error::*;
pub use libc::c_char;
pub use nemo_flow::codec::request::AnnotatedLLMRequest;
pub use nemo_flow::codec::response::AnnotatedLLMResponse;
pub use nemo_flow::context::callbacks::{
    LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn,
};
pub use nemo_flow::error::{FlowError, Result};
pub use nemo_flow::types::event::Event;
pub use nemo_flow::types::llm::{LLMAttributes, LLMHandle, LLMRequest};
pub use nemo_flow::types::scope::{ScopeAttributes, ScopeHandle, ScopeType};
pub use nemo_flow::types::tool::{ToolAttributes, ToolHandle};
pub use serde_json::{Value as Json, json};
pub use std::ffi::{CStr, CString};
pub use std::pin::Pin;
pub use std::sync::Arc;
pub use tokio_stream::Stream;
pub use types::*;

unsafe fn nemo_flow_string_free_internal(ptr: *mut c_char) {
    unsafe { convert::nemo_flow_string_free(ptr) };
}

mod api_tests;
mod callable_extra_tests;
#[path = "../unit/callable_tests.rs"]
mod callable_tests;
#[path = "../coverage/convert_tests.rs"]
mod convert_coverage_tests;
#[path = "../coverage/error_tests.rs"]
mod error_coverage_tests;
#[path = "../unit/types_tests.rs"]
mod types_tests;
