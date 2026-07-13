// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-facing type wrappers for NeMo Relay core types.
//!
//! Each type wraps its corresponding `nemo_relay::types` struct and exposes
//! properties via `#[getter]`. Doc comments on `#[pyclass]` and `#[pymethods]`
//! become Python `help()` output.

use std::collections::HashMap;
use std::sync::Arc;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use nemo_relay::api::event::{MarkEvent, ScopeEvent};
use nemo_relay::api::llm::{LlmAttributes, LlmHandle, LlmRequest};
use nemo_relay::api::runtime::ScopeStackHandle;
use nemo_relay::api::scope::{ScopeAttributes, ScopeHandle, ScopeType as CoreScopeType};
use nemo_relay::api::tool::{ToolAttributes, ToolHandle};
use nemo_relay::codec::request::{
    AnnotatedLlmRequest as AnnotatedLLMRequest, GenerationParams, Message, ToolChoice,
    ToolDefinition,
};
use nemo_relay::codec::response::AnnotatedLlmResponse as AnnotatedLLMResponse;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::Result as FlowResult;
use pyo3::prelude::*;
use serde::Serialize;

use crate::convert::{json_to_py, opt_json_to_py, py_to_json};

#[cfg(test)]
static FORCED_SERIALIZATION_MASK: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(crate) const FORCE_ATIF_EXPORT_VALUE_SERIALIZATION_ERROR: u64 = 1 << 0;
#[cfg(test)]
pub(crate) const FORCE_ATIF_EXPORT_JSON_SERIALIZATION_ERROR: u64 = 1 << 1;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_REQUEST_MESSAGES_SERIALIZATION_ERROR: u64 = 1 << 2;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_REQUEST_PARAMS_SERIALIZATION_ERROR: u64 = 1 << 3;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_REQUEST_TOOLS_SERIALIZATION_ERROR: u64 = 1 << 4;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_REQUEST_TOOL_CHOICE_SERIALIZATION_ERROR: u64 = 1 << 5;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_RESPONSE_MESSAGE_SERIALIZATION_ERROR: u64 = 1 << 6;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_RESPONSE_TOOL_CALLS_SERIALIZATION_ERROR: u64 = 1 << 7;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_RESPONSE_USAGE_SERIALIZATION_ERROR: u64 = 1 << 8;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_RESPONSE_API_SPECIFIC_SERIALIZATION_ERROR: u64 = 1 << 9;
#[cfg(test)]
pub(crate) const FORCE_ANNOTATED_RESPONSE_OPTIMIZATION_SUMMARY_SERIALIZATION_ERROR: u64 = 1 << 10;

#[cfg(test)]
pub(crate) fn set_forced_serialization_mask_for_tests(mask: u64) {
    FORCED_SERIALIZATION_MASK.store(mask, Ordering::SeqCst);
}

fn to_python_json_value<T: Serialize>(
    value: &T,
    error_prefix: &'static str,
    #[cfg(test)] forced_mask_bit: u64,
) -> PyResult<serde_json::Value> {
    #[cfg(test)]
    if FORCED_SERIALIZATION_MASK.load(Ordering::SeqCst) & forced_mask_bit != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{error_prefix}: forced serialization failure"
        )));
    }

    serde_json::to_value(value)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{error_prefix}: {e}")))
}

fn to_python_json_string<T: Serialize>(
    value: &T,
    error_prefix: &'static str,
    #[cfg(test)] forced_mask_bit: u64,
) -> PyResult<String> {
    #[cfg(test)]
    if FORCED_SERIALIZATION_MASK.load(Ordering::SeqCst) & forced_mask_bit != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{error_prefix}: forced serialization failure"
        )));
    }

    serde_json::to_string(value)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{error_prefix}: {e}")))
}

fn py_string_map(obj: &Bound<'_, PyAny>, field_name: &str) -> PyResult<HashMap<String, String>> {
    let json = py_to_json(obj)?;
    let serde_json::Value::Object(map) = json else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{field_name} must be a dict[str, str]"
        )));
    };

    let mut out = HashMap::with_capacity(map.len());
    for (key, value) in map {
        let serde_json::Value::String(value) = value else {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "{field_name} must be a dict[str, str]"
            )));
        };
        out.insert(key, value);
    }
    Ok(out)
}

mod codecs;
mod core;
mod events;
mod observability;

pub use codecs::*;
pub use core::*;
pub use events::*;
pub use observability::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScopeStack>()?;
    m.add_class::<PyLlmStream>()?;
    m.add_class::<PyScopeAttributes>()?;
    m.add_class::<PyToolAttributes>()?;
    m.add_class::<PyLLMAttributes>()?;
    m.add_class::<PyScopeType>()?;
    m.add_class::<PyScopeHandle>()?;
    m.add_class::<PyToolHandle>()?;
    m.add_class::<PyLLMHandle>()?;
    m.add_class::<PyLLMRequest>()?;
    m.add_class::<PyPendingMarkSpec>()?;
    m.add_class::<PyLLMRequestInterceptOutcome>()?;
    m.add_class::<PyToolExecutionInterceptOutcome>()?;
    m.add_class::<PyAnnotatedLLMRequest>()?;
    m.add_class::<PyAnnotatedLLMResponse>()?;
    m.add_class::<PyScopeEvent>()?;
    m.add_class::<PyMarkEvent>()?;
    m.add_class::<PyAtifExporter>()?;
    m.add_class::<PyAtofExporterMode>()?;
    m.add_class::<PyAtofEndpointConfig>()?;
    m.add_class::<PyAtofExporterConfig>()?;
    m.add_class::<PyAtofExporter>()?;
    m.add_class::<PyOpenTelemetryConfig>()?;
    m.add_class::<PyOpenTelemetrySubscriber>()?;
    m.add_class::<PyOpenInferenceConfig>()?;
    m.add_class::<PyOpenInferenceSubscriber>()?;
    m.add_class::<PyOpenAIChatCodec>()?;
    m.add_class::<PyOpenAIResponsesCodec>()?;
    m.add_class::<PyAnthropicMessagesCodec>()?;
    Ok(())
}

#[cfg(test)]
#[path = "../../tests/coverage/py_types_coverage_tests.rs"]
mod coverage_tests;
