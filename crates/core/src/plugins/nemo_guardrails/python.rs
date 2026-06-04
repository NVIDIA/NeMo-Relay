// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::api::llm::LlmRequest;
use crate::api::registry::{
    deregister_llm_execution_intercept, deregister_llm_stream_execution_intercept,
    deregister_tool_execution_intercept, register_llm_execution_intercept,
    register_llm_stream_execution_intercept, register_tool_execution_intercept,
};
use crate::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn, ToolExecutionNextFn};
use crate::codec::anthropic::AnthropicMessagesCodec;
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::request::{AnnotatedLlmRequest, Message};
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result as FlowResult};
use crate::json::Json;
use crate::plugin::{
    PluginError, PluginRegistration, PluginRegistrationContext, Result as PluginResult,
    rollback_registrations,
};

use super::NeMoGuardrailsConfig;

const SUPPORT_MODULE_NAME: &str = "_nemo_guardrails_local_runtime";
const HELPER_MODULE_NAME: &str = "_nemo_guardrails_local";
const HELPER_FILENAME: &str = "_nemo_guardrails_local.py";
const HELPER_SOURCE: &str = include_str!("embedded_python/_guardrails_local.py");

pub(super) fn register_local_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Python::initialize();

    let plugin_config = match serde_json::to_value(config) {
        Ok(Json::Object(config)) => config,
        Ok(_) => {
            return Err(PluginError::Internal(
                "NeMo Guardrails local config did not serialize to a JSON object".to_string(),
            ));
        }
        Err(err) => {
            return Err(PluginError::Internal(format!(
                "failed to serialize NeMo Guardrails local config: {err}"
            )));
        }
    };

    let registrations = Python::attach(|py| {
        let register_fn = load_guardrails_local_register_fn(py)?;
        invoke_embedded_plugin_register(py, &register_fn, &plugin_config, ctx.qualify_name(""))
    })
    .map_err(|err| PluginError::RegistrationFailed(err.to_string()))?;

    ctx.extend_registrations(registrations);
    Ok(())
}

fn invoke_embedded_plugin_register(
    py: Python<'_>,
    register_fn: &Bound<'_, PyAny>,
    plugin_config: &serde_json::Map<String, Json>,
    namespace_prefix: String,
) -> PyResult<Vec<PluginRegistration>> {
    let context = Py::new(
        py,
        PyLocalPluginContext {
            registrations: Arc::new(Mutex::new(vec![])),
            namespace_prefix,
        },
    )?;
    let plugin_config_py = json_to_py(py, &Json::Object(plugin_config.clone()))?;

    match register_fn.call1((plugin_config_py, context.clone_ref(py))) {
        Ok(_) => context.bind(py).borrow().drain_registrations(),
        Err(err) => {
            if let Ok(mut registrations) = context.bind(py).borrow().drain_registrations() {
                rollback_registrations(&mut registrations);
            }
            Err(err)
        }
    }
}

fn load_guardrails_local_register_fn(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    install_support_module(py)?;
    let module = load_guardrails_local_module(py)?;
    module.getattr("register_local_backend")
}

fn install_support_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;
    if let Some(existing) = modules.get_item(SUPPORT_MODULE_NAME)? {
        return Ok(existing.cast_into::<PyModule>()?);
    }

    let module = PyModule::new(py, SUPPORT_MODULE_NAME)?;
    module.add_class::<PyLLMRequest>()?;
    module.add_class::<PyAnnotatedLLMRequest>()?;
    module.add_class::<PyAnnotatedLLMResponse>()?;
    module.add_class::<PyOpenAIChatCodec>()?;
    module.add_class::<PyOpenAIResponsesCodec>()?;
    module.add_class::<PyAnthropicMessagesCodec>()?;
    module.add_class::<PyLocalPluginContext>()?;
    modules.set_item(SUPPORT_MODULE_NAME, &module)?;
    Ok(module)
}

fn load_guardrails_local_module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?.cast_into::<PyDict>()?;
    if let Some(existing) = modules.get_item(HELPER_MODULE_NAME)? {
        return Ok(existing.cast_into::<PyModule>()?);
    }

    let source = CString::new(HELPER_SOURCE).unwrap();
    let filename = CString::new(HELPER_FILENAME).unwrap();
    let module_name = CString::new(HELPER_MODULE_NAME).unwrap();
    let module = PyModule::from_code(py, &source, &filename, &module_name)?;
    modules.set_item(HELPER_MODULE_NAME, &module)?;
    Ok(module)
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Json> {
    pythonize::depythonize(obj).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert to JSON: {e}"))
    })
}

fn json_to_py(py: Python<'_>, value: &Json) -> PyResult<Py<PyAny>> {
    let obj: Bound<'_, PyAny> = pythonize::pythonize(py, value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert from JSON: {e}"))
    })?;
    Ok(obj.unbind())
}

fn messages_to_json(messages: &[Message]) -> PyResult<Json> {
    serde_json::to_value(messages).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to serialize messages: {e}"
        ))
    })
}

#[pyclass(name = "LLMRequest", from_py_object)]
#[derive(Clone)]
struct PyLLMRequest {
    inner: LlmRequest,
}

#[pymethods]
impl PyLLMRequest {
    #[new]
    #[pyo3(signature = (headers, content), text_signature = "(headers: dict[str, str], content: object)")]
    fn new(headers: &Bound<'_, PyAny>, content: &Bound<'_, PyAny>) -> PyResult<Self> {
        let headers_json = py_to_json(headers)?;
        let Json::Object(headers_map) = headers_json else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "headers must be a dict",
            ));
        };
        let content_json = py_to_json(content)?;
        Ok(Self {
            inner: LlmRequest {
                headers: headers_map,
                content: content_json,
            },
        })
    }

    #[getter]
    fn headers(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &Json::Object(self.inner.headers.clone()))
    }

    #[getter]
    fn content(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &self.inner.content)
    }

    fn __repr__(&self) -> String {
        "LLMRequest(...)".to_string()
    }
}

#[pyclass(name = "AnnotatedLLMRequest", from_py_object)]
#[derive(Clone)]
struct PyAnnotatedLLMRequest {
    inner: AnnotatedLlmRequest,
}

#[pymethods]
impl PyAnnotatedLLMRequest {
    #[getter]
    fn messages(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_py(py, &messages_to_json(&self.inner.messages)?)
    }

    #[setter]
    fn set_messages(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.messages = pythonize::depythonize(value).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid messages: {e}"))
        })?;
        Ok(())
    }
}

#[pyclass(name = "AnnotatedLLMResponse", skip_from_py_object)]
#[derive(Clone)]
struct PyAnnotatedLLMResponse {
    inner: AnnotatedLlmResponse,
}

#[pymethods]
impl PyAnnotatedLLMResponse {
    fn response_text(&self) -> Option<String> {
        self.inner.response_text().map(str::to_string)
    }
}

#[pyclass(name = "OpenAIChatCodec")]
struct PyOpenAIChatCodec;

#[pymethods]
impl PyOpenAIChatCodec {
    #[new]
    fn new() -> Self {
        Self
    }

    fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        OpenAIChatCodec
            .decode(&request.inner)
            .map(|inner| PyAnnotatedLLMRequest { inner })
            .map_err(flow_to_py_err)
    }

    fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        OpenAIChatCodec
            .encode(&annotated.inner, &original.inner)
            .map(|inner| PyLLMRequest { inner })
            .map_err(flow_to_py_err)
    }

    fn decode_response(&self, response: &Bound<'_, PyAny>) -> PyResult<PyAnnotatedLLMResponse> {
        let response = py_to_json(response)?;
        OpenAIChatCodec
            .decode_response(&response)
            .map(|inner| PyAnnotatedLLMResponse { inner })
            .map_err(flow_to_py_err)
    }
}

#[pyclass(name = "OpenAIResponsesCodec")]
struct PyOpenAIResponsesCodec;

#[pymethods]
impl PyOpenAIResponsesCodec {
    #[new]
    fn new() -> Self {
        Self
    }

    fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        OpenAIResponsesCodec
            .decode(&request.inner)
            .map(|inner| PyAnnotatedLLMRequest { inner })
            .map_err(flow_to_py_err)
    }

    fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        OpenAIResponsesCodec
            .encode(&annotated.inner, &original.inner)
            .map(|inner| PyLLMRequest { inner })
            .map_err(flow_to_py_err)
    }

    fn decode_response(&self, response: &Bound<'_, PyAny>) -> PyResult<PyAnnotatedLLMResponse> {
        let response = py_to_json(response)?;
        OpenAIResponsesCodec
            .decode_response(&response)
            .map(|inner| PyAnnotatedLLMResponse { inner })
            .map_err(flow_to_py_err)
    }
}

#[pyclass(name = "AnthropicMessagesCodec")]
struct PyAnthropicMessagesCodec;

#[pymethods]
impl PyAnthropicMessagesCodec {
    #[new]
    fn new() -> Self {
        Self
    }

    fn decode(&self, request: &PyLLMRequest) -> PyResult<PyAnnotatedLLMRequest> {
        AnthropicMessagesCodec
            .decode(&request.inner)
            .map(|inner| PyAnnotatedLLMRequest { inner })
            .map_err(flow_to_py_err)
    }

    fn encode(
        &self,
        annotated: &PyAnnotatedLLMRequest,
        original: &PyLLMRequest,
    ) -> PyResult<PyLLMRequest> {
        AnthropicMessagesCodec
            .encode(&annotated.inner, &original.inner)
            .map(|inner| PyLLMRequest { inner })
            .map_err(flow_to_py_err)
    }

    fn decode_response(&self, response: &Bound<'_, PyAny>) -> PyResult<PyAnnotatedLLMResponse> {
        let response = py_to_json(response)?;
        AnthropicMessagesCodec
            .decode_response(&response)
            .map(|inner| PyAnnotatedLLMResponse { inner })
            .map_err(flow_to_py_err)
    }
}

#[pyclass(name = "PluginContext")]
struct PyLocalPluginContext {
    registrations: Arc<Mutex<Vec<PluginRegistration>>>,
    namespace_prefix: String,
}

impl PyLocalPluginContext {
    fn qualify_name(&self, name: &str) -> String {
        format!("{}{}", self.namespace_prefix, name)
    }

    fn drain_registrations(&self) -> PyResult<Vec<PluginRegistration>> {
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        Ok(std::mem::take(&mut *guard))
    }
}

#[pymethods]
impl PyLocalPluginContext {
    #[pyo3(signature = (name, priority, callback), text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_execution_intercept(
            &qualified_name,
            priority,
            wrap_py_llm_exec_intercept_fn(callback),
        )
        .map_err(plugin_to_py_err)?;
        self.push_registration(qualified_name, RegistrationKind::Llm)
    }

    #[pyo3(signature = (name, priority, callback), text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_llm_stream_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        let qualified_name = self.qualify_name(name);
        register_llm_stream_execution_intercept(
            &qualified_name,
            priority,
            wrap_py_llm_stream_exec_intercept_fn(callback),
        )
        .map_err(plugin_to_py_err)?;
        self.push_registration(qualified_name, RegistrationKind::LlmStream)
    }

    #[pyo3(signature = (name, priority, callback), text_signature = "(name: str, priority: int, callback: object) -> None")]
    fn register_tool_execution_intercept(
        &self,
        name: &str,
        priority: i32,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        let qualified_name = self.qualify_name(name);
        register_tool_execution_intercept(
            &qualified_name,
            priority,
            wrap_py_tool_exec_intercept_fn(callback),
        )
        .map_err(plugin_to_py_err)?;
        self.push_registration(qualified_name, RegistrationKind::Tool)
    }

    fn __repr__(&self) -> String {
        "<PluginContext>".to_string()
    }
}

impl PyLocalPluginContext {
    fn push_registration(&self, name: String, kind: RegistrationKind) -> PyResult<()> {
        let mut guard = self.registrations.lock().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("plugin context lock poisoned: {e}"))
        })?;
        guard.push(PluginRegistration::new(
            "plugin",
            name.clone(),
            Box::new(move || match kind {
                RegistrationKind::Llm => deregister_llm_execution_intercept(&name)
                    .map(|_| ())
                    .map_err(registration_failure),
                RegistrationKind::LlmStream => deregister_llm_stream_execution_intercept(&name)
                    .map(|_| ())
                    .map_err(registration_failure),
                RegistrationKind::Tool => deregister_tool_execution_intercept(&name)
                    .map(|_| ())
                    .map_err(registration_failure),
            }),
        ));
        Ok(())
    }
}

enum RegistrationKind {
    Llm,
    LlmStream,
    Tool,
}

fn registration_failure(err: FlowError) -> PluginError {
    PluginError::RegistrationFailed(err.to_string())
}

fn plugin_to_py_err(err: FlowError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

fn flow_to_py_err(err: FlowError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
}

type PyValueFuture = Pin<Box<dyn Future<Output = PyResult<Py<PyAny>>> + Send>>;
type ToolExecIntercept = Arc<
    dyn Fn(
            &str,
            Json,
            ToolExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>>
        + Send
        + Sync,
>;
type LlmExecIntercept = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = FlowResult<Json>> + Send>>
        + Send
        + Sync,
>;
type LlmStreamIntercept = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<
                        Output = FlowResult<
                            Pin<Box<dyn tokio_stream::Stream<Item = FlowResult<Json>> + Send>>,
                        >,
                    > + Send,
            >,
        > + Send
        + Sync,
>;

fn split_py_object_or_future(
    py: Python<'_>,
    result: Py<PyAny>,
) -> FlowResult<Result<Py<PyAny>, PyValueFuture>> {
    let bound = result.bind(py);
    if bound.getattr("__await__").is_ok() {
        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
            .map_err(|e| FlowError::Internal(e.to_string()))?;
        Ok(Err(Box::pin(future) as PyValueFuture))
    } else {
        Ok(Ok(result))
    }
}

async fn resolve_py_object_or_future(
    outcome: FlowResult<Result<Py<PyAny>, PyValueFuture>>,
) -> FlowResult<Py<PyAny>> {
    match outcome? {
        Ok(value) => Ok(value),
        Err(future) => future.await.map_err(|e| FlowError::Internal(e.to_string())),
    }
}

fn next_async_iter_coro(async_iter: &Arc<Py<PyAny>>) -> FlowResult<Option<Py<PyAny>>> {
    Python::attach(|py| {
        let iter = async_iter.bind(py);
        match iter.call_method0("__anext__") {
            Ok(coro) => Ok(Some(coro.unbind())),
            Err(error) => {
                if error.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                    Ok(None)
                } else {
                    Err(FlowError::Internal(error.to_string()))
                }
            }
        }
    })
}

async fn await_async_iter_value(coro: Py<PyAny>) -> FlowResult<Option<Json>> {
    let future = Python::attach(|py| {
        pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
            .map_err(|e| FlowError::Internal(e.to_string()))
    })?;

    match future.await {
        Ok(result) => Python::attach(|py| {
            py_to_json(result.bind(py))
                .map(Some)
                .map_err(|e| FlowError::Internal(e.to_string()))
        }),
        Err(error) => Python::attach(|py| {
            if error.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                Ok(None)
            } else {
                Err(FlowError::Internal(error.to_string()))
            }
        }),
    }
}

async fn forward_async_iter(
    async_iter: Arc<Py<PyAny>>,
    tx: tokio::sync::mpsc::Sender<FlowResult<Json>>,
) {
    loop {
        let next_value = match next_async_iter_coro(&async_iter) {
            Ok(None) => break,
            Ok(Some(coro)) => await_async_iter_value(coro).await,
            Err(error) => Err(error),
        };

        match next_value {
            Ok(Some(value)) => {
                if tx.send(Ok(value)).await.is_err() {
                    break;
                }
            }
            Ok(None) => break,
            Err(error) => {
                let _ = tx.send(Err(error)).await;
                break;
            }
        }
    }
}

fn stream_from_async_iter(
    async_iter: Py<PyAny>,
) -> FlowResult<Pin<Box<dyn tokio_stream::Stream<Item = FlowResult<Json>> + Send>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(32);
    let task_locals = Python::attach(|py| {
        pyo3_async_runtimes::tokio::get_current_locals(py)
            .map_err(|e: pyo3::PyErr| FlowError::Internal(e.to_string()))
    })?;

    let async_iter = Arc::new(async_iter);
    tokio::spawn(pyo3_async_runtimes::tokio::scope(task_locals, async move {
        forward_async_iter(async_iter, tx).await;
    }));

    Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
}

#[pyclass]
struct PyToolNextFn {
    inner: ToolExecutionNextFn,
}

#[pymethods]
impl PyToolNextFn {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let args = py_to_json(&args)?;
        let next = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = next(args)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| json_to_py(py, &result))
        })
    }
}

#[pyclass]
struct PyLlmNextFn {
    inner: LlmExecutionNextFn,
}

#[pymethods]
impl PyLlmNextFn {
    fn __call__<'py>(&self, py: Python<'py>, request: PyLLMRequest) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = next(request.inner)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Python::attach(|py| json_to_py(py, &result))
        })
    }
}

#[pyclass(name = "LlmStream")]
struct PyLlmStream {
    receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<Json>>>,
}

#[pymethods]
impl PyLlmStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver_ptr = &self.receiver
            as *const tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<Json>>>;
        let receiver_ref = unsafe { &*receiver_ptr };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = receiver_ref.lock().await;
            match guard.recv().await {
                None => Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>(
                    "stream exhausted",
                )),
                Some(Ok(value)) => Python::attach(|py| json_to_py(py, &value)),
                Some(Err(err)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    err.to_string(),
                )),
            }
        })
    }
}

#[pyclass]
struct PyLlmStreamNextFn {
    inner: LlmStreamExecutionNextFn,
}

#[pymethods]
impl PyLlmStreamNextFn {
    fn __call__<'py>(&self, py: Python<'py>, request: PyLLMRequest) -> PyResult<Bound<'py, PyAny>> {
        let next = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rust_stream = next(request.inner)
                .await
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let (tx, rx) = tokio::sync::mpsc::channel::<FlowResult<Json>>(32);
            tokio::spawn(async move {
                use tokio_stream::StreamExt;
                let mut stream = rust_stream;
                while let Some(item) = stream.next().await {
                    if tx.send(item).await.is_err() {
                        break;
                    }
                }
            });
            Ok(PyLlmStream {
                receiver: tokio::sync::Mutex::new(rx),
            })
        })
    }
}

fn wrap_py_tool_exec_intercept_fn(py_fn: Py<PyAny>) -> ToolExecIntercept {
    let py_fn = Arc::new(py_fn);
    Arc::new(move |name: &str, args: Json, next: ToolExecutionNextFn| {
        let py_fn = py_fn.clone();
        let name = name.to_string();
        Box::pin(async move {
            let outcome: FlowResult<Result<Json, PyValueFuture>> = Python::attach(|py| {
                let py_args =
                    json_to_py(py, &args).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                let py_next = PyToolNextFn { inner: next };
                let result = py_fn
                    .call1(
                        py,
                        (
                            &name,
                            py_args,
                            py_next
                                .into_pyobject(py)
                                .map_err(|e| FlowError::Internal(e.to_string()))?
                                .into_any(),
                        ),
                    )
                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                let bound = result.bind(py);
                if bound.getattr("__await__").is_ok() {
                    let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                        .map_err(|e| FlowError::Internal(e.to_string()))?;
                    Ok(Err(Box::pin(future) as PyValueFuture))
                } else {
                    let json =
                        py_to_json(bound).map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    Ok(Ok(json))
                }
            });

            match outcome? {
                Ok(json) => Ok(json),
                Err(future) => {
                    let py_result = future
                        .await
                        .map_err(|e| FlowError::Internal(e.to_string()))?;
                    Python::attach(|py| {
                        py_to_json(py_result.bind(py))
                            .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
                    })
                }
            }
        })
    })
}

fn wrap_py_llm_exec_intercept_fn(py_fn: Py<PyAny>) -> LlmExecIntercept {
    let py_fn = Arc::new(py_fn);
    Arc::new(
        move |name: &str, request: LlmRequest, next: LlmExecutionNextFn| {
            let py_fn = py_fn.clone();
            let name = name.to_string();
            Box::pin(async move {
                let outcome: FlowResult<Result<Json, PyValueFuture>> = Python::attach(|py| {
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmNextFn { inner: next };
                    let result = py_fn
                        .call1(
                            py,
                            (
                                &name,
                                py_req
                                    .into_pyobject(py)
                                    .map_err(|e| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                                py_next
                                    .into_pyobject(py)
                                    .map_err(|e| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                            ),
                        )
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    let bound = result.bind(py);
                    if bound.getattr("__await__").is_ok() {
                        let future = pyo3_async_runtimes::tokio::into_future(result.into_bound(py))
                            .map_err(|e| FlowError::Internal(e.to_string()))?;
                        Ok(Err(Box::pin(future) as PyValueFuture))
                    } else {
                        let json = py_to_json(bound)
                            .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                        Ok(Ok(json))
                    }
                });

                match outcome? {
                    Ok(json) => Ok(json),
                    Err(future) => {
                        let py_result = future
                            .await
                            .map_err(|e| FlowError::Internal(e.to_string()))?;
                        Python::attach(|py| {
                            py_to_json(py_result.bind(py))
                                .map_err(|e: PyErr| FlowError::Internal(e.to_string()))
                        })
                    }
                }
            })
        },
    )
}

fn wrap_py_llm_stream_exec_intercept_fn(py_fn: Py<PyAny>) -> LlmStreamIntercept {
    let py_fn = Arc::new(py_fn);
    Arc::new(
        move |_name: &str, request: LlmRequest, next: LlmStreamExecutionNextFn| {
            let py_fn = py_fn.clone();
            Box::pin(async move {
                let async_iter = resolve_py_object_or_future(Python::attach(|py| {
                    let py_req = PyLLMRequest { inner: request };
                    let py_next = PyLlmStreamNextFn { inner: next };
                    let result = py_fn
                        .call1(
                            py,
                            (
                                py_req
                                    .into_pyobject(py)
                                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                                py_next
                                    .into_pyobject(py)
                                    .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?
                                    .into_any(),
                            ),
                        )
                        .map_err(|e: PyErr| FlowError::Internal(e.to_string()))?;
                    split_py_object_or_future(py, result)
                }))
                .await?;

                stream_from_async_iter(async_iter)
            })
        },
    )
}
