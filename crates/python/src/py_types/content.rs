// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python snapshots share selected Rust content; Python values are copied on access.

use std::sync::Arc;

use nemo_relay_types::codec::content::{
    ContentEntry, ContentField, ContentKind, ContentRoot, MessageView, PayloadRef,
};
use nemo_relay_types::codec::request::{Message, MessageContent};
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
use pyo3::prelude::*;

use super::{json_to_py, py_to_json};

fn python_value(py: Python<'_>, value: &impl serde::Serialize) -> PyResult<Py<PyAny>> {
    pythonize::pythonize(py, value)
        .map(Bound::unbind)
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

enum Source {
    Message {
        message: Message,
        request_index: Option<usize>,
    },
    Content(MessageContent),
    Response(Box<AnnotatedLlmResponse>),
}

impl Source {
    fn view(&self) -> MessageView<'_> {
        match self {
            Self::Message { message, .. } => message.content_view(),
            Self::Content(content) => content.content_view(),
            Self::Response(response) => response.output(),
        }
    }

    fn entry(&self, index: usize) -> Option<ContentEntry<'_>> {
        let mut entry = self.view().parts().nth(index)?;
        if let Self::Message {
            request_index: Some(index),
            ..
        } = self
        {
            entry.location.root = ContentRoot::RequestMessage(*index);
        }
        Some(entry)
    }
}

/// Read-only snapshot of selected normalized content. Entries retain its storage.
#[pyclass(name = "MessageView", frozen)]
pub struct PyMessageView {
    source: Arc<Source>,
}

impl PyMessageView {
    pub(super) fn message(message: &Message, request_index: Option<usize>) -> Self {
        Self {
            source: Arc::new(Source::Message {
                message: message.clone(),
                request_index,
            }),
        }
    }

    pub(super) fn content(content: &MessageContent) -> Self {
        Self {
            source: Arc::new(Source::Content(content.clone())),
        }
    }

    pub(super) fn response(response: &AnnotatedLlmResponse) -> Self {
        Self {
            source: Arc::new(Source::Response(Box::new(AnnotatedLlmResponse {
                message: response.message.clone(),
                tool_calls: response.tool_calls.clone(),
                ..Default::default()
            }))),
        }
    }
}

#[pymethods]
impl PyMessageView {
    /// Snapshot a normalized message dictionary, including tool-result semantics.
    #[staticmethod]
    fn from_message(message: &Bound<'_, PyAny>) -> PyResult<Self> {
        let message = serde_json::from_value(py_to_json(message)?)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            source: Arc::new(Source::Message {
                message,
                request_index: None,
            }),
        })
    }

    /// Snapshot normalized standalone content (a string or content-part list).
    #[staticmethod]
    fn from_content(content: &Bound<'_, PyAny>) -> PyResult<Self> {
        let content = serde_json::from_value(py_to_json(content)?)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            source: Arc::new(Source::Content(content)),
        })
    }

    /// Iterate typed entries in normalized order; the iterator retains this snapshot.
    fn parts(&self) -> PyContentIterator {
        PyContentIterator {
            source: Arc::clone(&self.source),
            index: 0,
        }
    }

    /// Iterate Python-owned text/refusal strings, including supported native fallbacks.
    fn text_parts(&self) -> PyTextIterator {
        PyTextIterator {
            source: Arc::clone(&self.source),
            index: 0,
        }
    }

    /// Join eligible text/refusals with newlines, preserving whitespace and empty parts.
    fn text(&self, py: Python<'_>) -> Option<Py<pyo3::types::PyString>> {
        self.source
            .view()
            .text()
            .map(|text| pyo3::types::PyString::new(py, &text).unbind())
    }
}

/// Single-pass iterator materializing one Python text string at a time.
#[pyclass(name = "_TextIterator")]
pub struct PyTextIterator {
    source: Arc<Source>,
    index: usize,
}

#[pymethods]
impl PyTextIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<&str> {
        let text = self.source.view().text_parts().nth(self.index)?;
        self.index += 1;
        Some(text)
    }
}

/// Iterator whose entries share the snapshot rather than copying payloads.
#[pyclass(name = "ContentIterator")]
pub struct PyContentIterator {
    source: Arc<Source>,
    index: usize,
}

#[pymethods]
impl PyContentIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyContentEntry> {
        self.source.entry(self.index)?;
        let entry = PyContentEntry {
            source: Arc::clone(&self.source),
            index: self.index,
        };
        self.index += 1;
        Some(entry)
    }
}

/// Read-time normalized source location, not an edit handle or wire offset.
#[pyclass(name = "ContentLocation", frozen, get_all)]
pub struct PyContentLocation {
    root: &'static str,
    message_index: Option<usize>,
    field: &'static str,
    index: Option<usize>,
}

/// Typed entry in a snapshot. Payload dictionaries/lists are independent copies.
#[pyclass(name = "ContentEntry", frozen)]
pub struct PyContentEntry {
    source: Arc<Source>,
    index: usize,
}

impl PyContentEntry {
    fn entry(&self) -> ContentEntry<'_> {
        self.source
            .entry(self.index)
            .expect("entry belongs to immutable snapshot")
    }
}

#[pymethods]
impl PyContentEntry {
    /// Discriminator: text, refusal, tool_call, tool_result, image_url, image, audio, file, or provider_native.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.entry().kind {
            ContentKind::Text(_) => "text",
            ContentKind::Refusal(_) => "refusal",
            ContentKind::ToolCall(_) => "tool_call",
            ContentKind::ToolResult(_) => "tool_result",
            ContentKind::ImageUrl(_) => "image_url",
            ContentKind::Image(_) => "image",
            ContentKind::Audio(_) => "audio",
            ContentKind::File(_) => "file",
            ContentKind::ProviderNative { .. } => "provider_native",
            _ => "unknown",
        }
    }

    /// Original field and array position in the selected normalized source.
    #[getter]
    fn location(&self) -> PyContentLocation {
        let location = self.entry().location;
        let (root, message_index) = match location.root {
            ContentRoot::RequestMessage(index) => ("request_message", Some(index)),
            ContentRoot::Message => ("message", None),
            ContentRoot::Response => ("response", None),
            ContentRoot::Content => ("content", None),
        };
        let field = match location.field {
            ContentField::Content => "content",
            ContentField::Message => "message",
            ContentField::ToolCalls => "tool_calls",
            ContentField::ToolCallItem => "tool_call_item",
            ContentField::Output => "output",
            ContentField::Value => "value",
            ContentField::NativeContent => "native_content",
        };
        PyContentLocation {
            root,
            message_index,
            field,
            index: location.index,
        }
    }

    /// Explicit text or refusal string; None for every other kind.
    #[getter]
    fn text(&self) -> Option<&str> {
        match self.entry().kind {
            ContentKind::Text(text) | ContentKind::Refusal(text) => Some(text),
            _ => None,
        }
    }

    /// Typed tool invocation, if this entry is a tool call.
    #[getter]
    fn tool_call(&self) -> Option<PyToolCallView> {
        matches!(self.entry().kind, ContentKind::ToolCall(_)).then(|| PyToolCallView {
            source: Arc::clone(&self.source),
            index: self.index,
        })
    }

    /// Typed tool result, whose payload is excluded from default text extraction.
    #[getter]
    fn tool_result(&self) -> Option<PyToolResultView> {
        matches!(self.entry().kind, ContentKind::ToolResult(_)).then(|| PyToolResultView {
            source: Arc::clone(&self.source),
            index: self.index,
        })
    }

    /// Independent Python value for media/native content; None for text and tools.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.entry().kind {
            ContentKind::ImageUrl(value) => python_value(py, value),
            ContentKind::Image(value)
            | ContentKind::Audio(value)
            | ContentKind::File(value)
            | ContentKind::ProviderNative { value, .. } => json_to_py(py, value),
            _ => Ok(py.None()),
        }
    }

    /// Owning provider for native content, otherwise None.
    #[getter]
    fn provider(&self) -> Option<&str> {
        match self.entry().kind {
            ContentKind::ProviderNative { provider, .. } => Some(provider),
            _ => None,
        }
    }

    /// Original native discriminator, otherwise None.
    #[getter]
    fn native_kind(&self) -> Option<&str> {
        match self.entry().kind {
            ContentKind::ProviderNative { kind, .. } => Some(kind),
            _ => None,
        }
    }
}

/// A tool invocation backed by shared snapshot storage.
#[pyclass(name = "ToolCallView", frozen)]
pub struct PyToolCallView {
    source: Arc<Source>,
    index: usize,
}

impl PyToolCallView {
    fn call(&self) -> nemo_relay_types::codec::content::ToolCallView<'_> {
        match self.source.entry(self.index).expect("snapshot entry").kind {
            ContentKind::ToolCall(call) => call,
            _ => unreachable!("tool-call view is constructed only for calls"),
        }
    }
}

#[pymethods]
impl PyToolCallView {
    /// Source item/tool-call identifier, when present.
    #[getter]
    fn id(&self) -> Option<&str> {
        self.call().id
    }
    /// Separate correlation identifier, when present.
    #[getter]
    fn call_id(&self) -> Option<&str> {
        self.call().call_id
    }
    /// Function/tool name.
    #[getter]
    fn name(&self) -> &str {
        self.call().name
    }
    /// Arguments retaining their string versus JSON representation.
    #[getter]
    fn arguments(&self) -> PyPayloadView {
        PyPayloadView {
            source: Arc::clone(&self.source),
            index: self.index,
        }
    }
}

/// A tool result backed by shared snapshot storage.
#[pyclass(name = "ToolResultView", frozen)]
pub struct PyToolResultView {
    source: Arc<Source>,
    index: usize,
}

impl PyToolResultView {
    fn result(&self) -> nemo_relay_types::codec::content::ToolResultView<'_> {
        match self.source.entry(self.index).expect("snapshot entry").kind {
            ContentKind::ToolResult(result) => result,
            _ => unreachable!("tool-result view is constructed only for results"),
        }
    }
}

#[pymethods]
impl PyToolResultView {
    /// Source item identifier, when present.
    #[getter]
    fn id(&self) -> Option<&str> {
        self.result().id
    }
    /// Corresponding tool-call identifier, when present.
    #[getter]
    fn call_id(&self) -> Option<&str> {
        self.result().call_id
    }
    /// Legacy function-result name, when present.
    #[getter]
    fn name(&self) -> Option<&str> {
        self.result().name
    }
    /// Explicit error flag, when supplied.
    #[getter]
    fn is_error(&self) -> Option<bool> {
        self.result().is_error
    }
    /// Complete result payload without recursively extracting its text.
    #[getter]
    fn payload(&self) -> PyPayloadView {
        PyPayloadView {
            source: Arc::clone(&self.source),
            index: self.index,
        }
    }
}

/// Original tool payload representation. Accessing value creates a Python-owned value.
#[pyclass(name = "PayloadView", frozen)]
pub struct PyPayloadView {
    source: Arc<Source>,
    index: usize,
}

impl PyPayloadView {
    fn payload(&self) -> PayloadRef<'_> {
        match self.source.entry(self.index).expect("snapshot entry").kind {
            ContentKind::ToolCall(call) => call.arguments,
            ContentKind::ToolResult(result) => result.payload,
            _ => unreachable!("payload views belong to tools"),
        }
    }
}

#[pymethods]
impl PyPayloadView {
    /// One of null, string, json, or content; JSON strings remain kind json.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.payload() {
            PayloadRef::Null => "null",
            PayloadRef::String(_) => "string",
            PayloadRef::Json(_) => "json",
            PayloadRef::Content(_) => "content",
        }
    }

    /// Independent Python value. No JSON parsing or string rendering is performed.
    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.payload() {
            PayloadRef::Null => Ok(py.None()),
            PayloadRef::String(text) => Ok(text.into_pyobject(py)?.into_any().unbind()),
            PayloadRef::Json(value) => json_to_py(py, value),
            PayloadRef::Content(content) => python_value(py, content),
        }
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTextIterator>()?;
    m.add_class::<PyMessageView>()?;
    m.add_class::<PyContentIterator>()?;
    m.add_class::<PyContentEntry>()?;
    m.add_class::<PyContentLocation>()?;
    m.add_class::<PyToolCallView>()?;
    m.add_class::<PyToolResultView>()?;
    m.add_class::<PyPayloadView>()
}
