// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use pyo3::prelude::*;

use super::core::PyLLMRequest;
use super::{
    AnnotatedLLMRequest, AnnotatedLLMResponse, Arc, Bound, GenerationParams, LlmCodec,
    LlmResponseCodec, Message, PyAny, PyResult, Python, ToolChoice, ToolDefinition, json_to_py,
    py_to_json, to_python_json_value,
};
#[cfg(test)]
use super::{
    FORCE_ANNOTATED_REQUEST_MESSAGES_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_REQUEST_PARAMS_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_REQUEST_TOOL_CHOICE_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_REQUEST_TOOLS_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_RESPONSE_API_SPECIFIC_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_RESPONSE_MESSAGE_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_RESPONSE_TOOL_CALLS_SERIALIZATION_ERROR,
    FORCE_ANNOTATED_RESPONSE_USAGE_SERIALIZATION_ERROR,
};

// ---------------------------------------------------------------------------
// AnnotatedLLMRequest
// ---------------------------------------------------------------------------

/// A structured view of an LLM request produced by a Codec.
///
/// Provides typed access to conversation messages, model name, generation
/// parameters, tool definitions, tool choice, and extensible extra fields.
///
/// Properties:
///     messages (list): Parsed conversation messages (list of dicts with a ``role`` key).
///     model (str | None): Model identifier (e.g., ``"gpt-4"``).
///     params (dict | None): Normalized generation parameters.
///     tools (list | None): Tool definitions (function schemas).
///     tool_choice (Any | None): Tool choice control.
///     extra (dict): Provider-specific extra fields.
///
/// Helper methods:
///     system_prompt() -> str | None: Text of the first system message.
///     last_user_message() -> str | None: Text of the last user message.
///     has_tool_calls() -> bool: Whether any assistant message has tool calls.
#[pyclass(name = "AnnotatedLLMRequest", from_py_object)]
#[derive(Clone)]
pub struct PyAnnotatedLLMRequest {
    pub inner: AnnotatedLLMRequest,
}

fn optional_json_getter(py: Python<'_>, value: &Option<serde_json::Value>) -> PyResult<Py<PyAny>> {
    match value {
        Some(value) => json_to_py(py, value),
        None => Ok(py.None()),
    }
}

fn optional_json_setter(
    target: &mut Option<serde_json::Value>,
    value: &Bound<'_, PyAny>,
    field: &str,
) -> PyResult<()> {
    if value.is_none() {
        *target = None;
    } else {
        *target = Some(pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("invalid {field}: {e}"))
        })?);
    }
    Ok(())
}

#[pymethods]
impl PyAnnotatedLLMRequest {
    /// Create a new AnnotatedLLMRequest.
    ///
    /// Args:
    ///     messages: A list of message dicts, each with a ``role`` key.
    ///     model: Optional model identifier.
    ///     params: Optional generation parameters dict.
    ///     tools: Optional list of tool definition dicts.
    ///     tool_choice: Optional tool choice control.
    ///     extra: Optional dict of provider-specific extra fields.
    #[new]
    #[pyo3(signature = (messages, *, model=None, params=None, tools=None, tool_choice=None, extra=None))]
    pub(crate) fn new(
        messages: &Bound<'_, PyAny>,
        model: Option<String>,
        params: Option<&Bound<'_, PyAny>>,
        tools: Option<&Bound<'_, PyAny>>,
        tool_choice: Option<&Bound<'_, PyAny>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let msgs: Vec<Message> = pythonize::depythonize(messages).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid messages: each dict must include a 'role' key (user/system/assistant/tool): {e}"
            ))
        })?;
        let gen_params: Option<GenerationParams> = match params {
            Some(p) if !p.is_none() => Some(pythonize::depythonize(p).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid params: {e}"))
            })?),
            _ => None,
        };
        let tool_defs: Option<Vec<ToolDefinition>> = match tools {
            Some(t) if !t.is_none() => Some(pythonize::depythonize(t).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tools: {e}"))
            })?),
            _ => None,
        };
        let tc: Option<ToolChoice> = match tool_choice {
            Some(tc_val) if !tc_val.is_none() => {
                Some(pythonize::depythonize(tc_val).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("invalid tool_choice: {e}"))
                })?)
            }
            _ => None,
        };
        let extra_map: serde_json::Map<String, serde_json::Value> = match extra {
            Some(e) if !e.is_none() => pythonize::depythonize(e).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid extra: {e}"))
            })?,
            _ => serde_json::Map::new(),
        };
        Ok(Self {
            inner: AnnotatedLLMRequest {
                messages: msgs,
                model,
                params: gen_params,
                tools: tool_defs,
                tool_choice: tc,
                store: None,
                previous_response_id: None,
                truncation: None,
                reasoning: None,
                include: None,
                user: None,
                metadata: None,
                service_tier: None,
                parallel_tool_calls: None,
                max_output_tokens: None,
                max_tool_calls: None,
                top_logprobs: None,
                stream: None,
                extra: extra_map,
            },
        })
    }

    #[getter]
    pub(crate) fn messages(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = to_python_json_value(
            &self.inner.messages,
            "serialization error",
            #[cfg(test)]
            FORCE_ANNOTATED_REQUEST_MESSAGES_SERIALIZATION_ERROR,
        )?;
        json_to_py(py, &value)
    }

    #[setter]
    pub(crate) fn set_messages(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.messages = pythonize::depythonize(value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid messages: each dict must include a 'role' key (user/system/assistant/tool): {e}"
            ))
        })?;
        Ok(())
    }

    #[getter]
    pub(crate) fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[setter]
    pub(crate) fn set_model(&mut self, value: Option<String>) {
        self.inner.model = value;
    }

    #[getter]
    pub(crate) fn params(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.params {
            Some(p) => {
                let value = to_python_json_value(
                    p,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_REQUEST_PARAMS_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[setter]
    pub(crate) fn set_params(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.params = None;
        } else {
            self.inner.params = Some(pythonize::depythonize(value).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid params: {e}"))
            })?);
        }
        Ok(())
    }

    #[getter]
    pub(crate) fn tools(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.tools {
            Some(t) => {
                let value = to_python_json_value(
                    t,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_REQUEST_TOOLS_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[setter]
    pub(crate) fn set_tools(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.tools = None;
        } else {
            self.inner.tools = Some(pythonize::depythonize(value).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tools: {e}"))
            })?);
        }
        Ok(())
    }

    #[getter]
    pub(crate) fn tool_choice(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.tool_choice {
            Some(tc) => {
                let value = to_python_json_value(
                    tc,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_REQUEST_TOOL_CHOICE_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[setter]
    pub(crate) fn set_tool_choice(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if value.is_none() {
            self.inner.tool_choice = None;
        } else {
            self.inner.tool_choice = Some(pythonize::depythonize(value).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid tool_choice: {e}"))
            })?);
        }
        Ok(())
    }

    #[getter]
    pub(crate) fn store(&self) -> Option<bool> {
        self.inner.store
    }

    #[setter]
    pub(crate) fn set_store(&mut self, value: Option<bool>) {
        self.inner.store = value;
    }

    #[getter]
    pub(crate) fn previous_response_id(&self) -> Option<String> {
        self.inner.previous_response_id.clone()
    }

    #[setter]
    pub(crate) fn set_previous_response_id(&mut self, value: Option<String>) {
        self.inner.previous_response_id = value;
    }

    #[getter]
    pub(crate) fn truncation(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_json_getter(py, &self.inner.truncation)
    }

    #[setter]
    pub(crate) fn set_truncation(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        optional_json_setter(&mut self.inner.truncation, value, "truncation")
    }

    #[getter]
    pub(crate) fn reasoning(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_json_getter(py, &self.inner.reasoning)
    }

    #[setter]
    pub(crate) fn set_reasoning(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        optional_json_setter(&mut self.inner.reasoning, value, "reasoning")
    }

    #[getter]
    pub(crate) fn include(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_json_getter(py, &self.inner.include)
    }

    #[setter]
    pub(crate) fn set_include(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        optional_json_setter(&mut self.inner.include, value, "include")
    }

    #[getter]
    pub(crate) fn user(&self) -> Option<String> {
        self.inner.user.clone()
    }

    #[setter]
    pub(crate) fn set_user(&mut self, value: Option<String>) {
        self.inner.user = value;
    }

    #[getter]
    pub(crate) fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        optional_json_getter(py, &self.inner.metadata)
    }

    #[setter]
    pub(crate) fn set_metadata(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        optional_json_setter(&mut self.inner.metadata, value, "metadata")
    }

    #[getter]
    pub(crate) fn service_tier(&self) -> Option<String> {
        self.inner.service_tier.clone()
    }

    #[setter]
    pub(crate) fn set_service_tier(&mut self, value: Option<String>) {
        self.inner.service_tier = value;
    }

    #[getter]
    pub(crate) fn parallel_tool_calls(&self) -> Option<bool> {
        self.inner.parallel_tool_calls
    }

    #[setter]
    pub(crate) fn set_parallel_tool_calls(&mut self, value: Option<bool>) {
        self.inner.parallel_tool_calls = value;
    }

    #[getter]
    pub(crate) fn max_output_tokens(&self) -> Option<u64> {
        self.inner.max_output_tokens
    }

    #[setter]
    pub(crate) fn set_max_output_tokens(&mut self, value: Option<u64>) {
        self.inner.max_output_tokens = value;
    }

    #[getter]
    pub(crate) fn max_tool_calls(&self) -> Option<u64> {
        self.inner.max_tool_calls
    }

    #[setter]
    pub(crate) fn set_max_tool_calls(&mut self, value: Option<u64>) {
        self.inner.max_tool_calls = value;
    }

    #[getter]
    pub(crate) fn top_logprobs(&self) -> Option<u64> {
        self.inner.top_logprobs
    }

    #[setter]
    pub(crate) fn set_top_logprobs(&mut self, value: Option<u64>) {
        self.inner.top_logprobs = value;
    }

    #[getter]
    pub(crate) fn stream(&self) -> Option<bool> {
        self.inner.stream
    }

    #[setter]
    pub(crate) fn set_stream(&mut self, value: Option<bool>) {
        self.inner.stream = value;
    }

    #[getter]
    pub(crate) fn extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::Value::Object(self.inner.extra.clone());
        json_to_py(py, &value)
    }

    #[setter]
    pub(crate) fn set_extra(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.extra = pythonize::depythonize(value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid extra: {e}")))?;
        Ok(())
    }

    /// Extract the text content of the first system message, if any.
    pub(crate) fn system_prompt(&self) -> Option<String> {
        self.inner.system_prompt().map(|s| s.to_string())
    }

    /// Get the text content of the last user message, if any.
    pub(crate) fn last_user_message(&self) -> Option<String> {
        self.inner.last_user_message().map(|s| s.to_string())
    }

    /// Check if any assistant message contains tool calls.
    pub(crate) fn has_tool_calls(&self) -> bool {
        self.inner.has_tool_calls()
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "<AnnotatedLLMRequest messages={} model={:?}>",
            self.inner.messages.len(),
            self.inner.model
        )
    }
}

// ---------------------------------------------------------------------------
// AnnotatedLLMResponse (read-only wrapper)
// ---------------------------------------------------------------------------

/// Structured view of an LLM response produced by a response codec.
///
/// Read-only: fields are accessed via properties. Complex fields
/// (message, tool_calls, usage, api_specific) return Python dicts/lists.
///
/// Properties:
///     id -> str | None: Response ID from the API.
///     model -> str | None: The model that served the request.
///     message -> Any | None: The assistant's response content.
///     tool_calls -> list | None: Tool calls requested by the model.
///     finish_reason -> str | None: Why generation stopped.
///     usage -> dict | None: Token usage statistics.
///     api_specific -> dict | None: API-specific response data.
///     extra -> dict: Unmodeled top-level fields (catch-all).
///
/// Helper methods:
///     response_text() -> str | None: Text content of the response message.
///     has_tool_calls() -> bool: Whether the response contains tool calls.
#[pyclass(name = "AnnotatedLLMResponse", skip_from_py_object)]
#[derive(Clone)]
pub struct PyAnnotatedLLMResponse {
    pub inner: AnnotatedLLMResponse,
}

#[pymethods]
impl PyAnnotatedLLMResponse {
    #[getter]
    pub(crate) fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }

    #[getter]
    pub(crate) fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }

    #[getter]
    pub(crate) fn message(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.message {
            Some(m) => {
                let value = to_python_json_value(
                    m,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_RESPONSE_MESSAGE_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    pub(crate) fn tool_calls(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.tool_calls {
            Some(tc) => {
                let value = to_python_json_value(
                    tc,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_RESPONSE_TOOL_CALLS_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    pub(crate) fn finish_reason(&self) -> Option<String> {
        self.inner
            .finish_reason
            .as_ref()
            .and_then(|fr| serde_json::to_value(fr).ok())
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    #[getter]
    pub(crate) fn usage(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.usage {
            Some(u) => {
                let value = to_python_json_value(
                    u,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_RESPONSE_USAGE_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    pub(crate) fn api_specific(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.api_specific {
            Some(a) => {
                let value = to_python_json_value(
                    a,
                    "serialization error",
                    #[cfg(test)]
                    FORCE_ANNOTATED_RESPONSE_API_SPECIFIC_SERIALIZATION_ERROR,
                )?;
                json_to_py(py, &value)
            }
            None => Ok(py.None()),
        }
    }

    #[getter]
    pub(crate) fn extra(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &serde_json::Value::Object(self.inner.extra.clone()))
    }

    /// Extract the text content of the response message.
    pub(crate) fn response_text(&self) -> Option<String> {
        self.inner.response_text().map(|s| s.to_string())
    }

    /// Check if the response contains any tool calls.
    pub(crate) fn has_tool_calls(&self) -> bool {
        self.inner.has_tool_calls()
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "<AnnotatedLLMResponse id={:?} model={:?}>",
            self.inner.id, self.inner.model
        )
    }
}

// ---------------------------------------------------------------------------
// Built-in LLM Codec pyclasses
// ---------------------------------------------------------------------------

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both ``LlmCodec`` (decode/encode for requests) and
/// ``LlmResponseCodec`` (decode_response for responses).
///
/// Example:
/// ```python
/// from nemo_relay.codecs import OpenAIChatCodec
/// codec = OpenAIChatCodec()
/// annotated_req = codec.decode(request)
/// annotated_resp = codec.decode_response(response)
/// ```
#[pyclass(name = "OpenAIChatCodec")]
pub struct PyOpenAIChatCodec {
    pub(crate) inner_codec: Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: Arc<dyn LlmResponseCodec>,
}

#[pymethods]
impl PyOpenAIChatCodec {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            inner_codec: Arc::new(nemo_relay::codec::openai_chat::OpenAIChatCodec),
            inner_response_codec: Arc::new(nemo_relay::codec::openai_chat::OpenAIChatCodec),
        }
    }

    /// Parse an opaque ``LlmRequest`` into a structured ``AnnotatedLLMRequest``.
    pub(crate) fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        self.inner_codec
            .decode(&request.inner)
            .map(|r| PyAnnotatedLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge structured changes back into the opaque request.
    pub(crate) fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        self.inner_codec
            .encode(&annotated.inner, &original.inner)
            .map(|r| PyLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a raw JSON response into a structured ``AnnotatedLLMResponse``.
    pub(crate) fn decode_response(
        &self,
        response: &Bound<'_, PyAny>,
    ) -> PyResult<PyAnnotatedLLMResponse> {
        let json = py_to_json(response)?;
        self.inner_response_codec
            .decode_response(&json)
            .map(|r| PyAnnotatedLLMResponse { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub(crate) fn __repr__(&self) -> &'static str {
        "<OpenAIChatCodec>"
    }
}

/// Built-in codec for the OpenAI Responses API.
///
/// Implements both ``LlmCodec`` (decode/encode for requests) and
/// ``LlmResponseCodec`` (decode_response for responses).
///
/// Example:
/// ```python
/// from nemo_relay.codecs import OpenAIResponsesCodec
/// codec = OpenAIResponsesCodec()
/// annotated_req = codec.decode(request)
/// annotated_resp = codec.decode_response(response)
/// ```
#[pyclass(name = "OpenAIResponsesCodec")]
pub struct PyOpenAIResponsesCodec {
    pub(crate) inner_codec: Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: Arc<dyn LlmResponseCodec>,
}

#[pymethods]
impl PyOpenAIResponsesCodec {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            inner_codec: Arc::new(nemo_relay::codec::openai_responses::OpenAIResponsesCodec),
            inner_response_codec: Arc::new(
                nemo_relay::codec::openai_responses::OpenAIResponsesCodec,
            ),
        }
    }

    /// Parse an opaque ``LlmRequest`` into a structured ``AnnotatedLLMRequest``.
    pub(crate) fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        self.inner_codec
            .decode(&request.inner)
            .map(|r| PyAnnotatedLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge structured changes back into the opaque request.
    pub(crate) fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        self.inner_codec
            .encode(&annotated.inner, &original.inner)
            .map(|r| PyLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a raw JSON response into a structured ``AnnotatedLLMResponse``.
    pub(crate) fn decode_response(
        &self,
        response: &Bound<'_, PyAny>,
    ) -> PyResult<PyAnnotatedLLMResponse> {
        let json = py_to_json(response)?;
        self.inner_response_codec
            .decode_response(&json)
            .map(|r| PyAnnotatedLLMResponse { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub(crate) fn __repr__(&self) -> &'static str {
        "<OpenAIResponsesCodec>"
    }
}

/// Built-in codec for the Anthropic Messages API.
///
/// Implements both ``LlmCodec`` (decode/encode for requests) and
/// ``LlmResponseCodec`` (decode_response for responses).
///
/// Example:
/// ```python
/// from nemo_relay.codecs import AnthropicMessagesCodec
/// codec = AnthropicMessagesCodec()
/// annotated_req = codec.decode(request)
/// annotated_resp = codec.decode_response(response)
/// ```
#[pyclass(name = "AnthropicMessagesCodec")]
pub struct PyAnthropicMessagesCodec {
    pub(crate) inner_codec: Arc<dyn LlmCodec>,
    pub(crate) inner_response_codec: Arc<dyn LlmResponseCodec>,
}

#[pymethods]
impl PyAnthropicMessagesCodec {
    #[new]
    pub(crate) fn new() -> Self {
        Self {
            inner_codec: Arc::new(nemo_relay::codec::anthropic::AnthropicMessagesCodec),
            inner_response_codec: Arc::new(nemo_relay::codec::anthropic::AnthropicMessagesCodec),
        }
    }

    /// Parse an opaque ``LlmRequest`` into a structured ``AnnotatedLLMRequest``.
    pub(crate) fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        self.inner_codec
            .decode(&request.inner)
            .map(|r| PyAnnotatedLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Merge structured changes back into the opaque request.
    pub(crate) fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        self.inner_codec
            .encode(&annotated.inner, &original.inner)
            .map(|r| PyLLMRequest { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Parse a raw JSON response into a structured ``AnnotatedLLMResponse``.
    pub(crate) fn decode_response(
        &self,
        response: &Bound<'_, PyAny>,
    ) -> PyResult<PyAnnotatedLLMResponse> {
        let json = py_to_json(response)?;
        self.inner_response_codec
            .decode_response(&json)
            .map(|r| PyAnnotatedLLMResponse { inner: r })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub(crate) fn __repr__(&self) -> &'static str {
        "<AnthropicMessagesCodec>"
    }
}
