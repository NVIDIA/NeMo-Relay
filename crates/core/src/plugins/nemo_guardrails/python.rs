// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::llm::LlmRequest;
use crate::api::runtime::{LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn, ToolExecutionFn};
use crate::codec::anthropic::AnthropicMessagesCodec;
use crate::codec::openai_chat::OpenAIChatCodec;
use crate::codec::openai_responses::OpenAIResponsesCodec;
use crate::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::{FlowError, Result as FlowResult};
use crate::json::Json;
use crate::plugin::{PluginError, PluginRegistrationContext, Result as PluginResult};

use super::NeMoGuardrailsConfig;

const DEFAULT_MODULE_NAME: &str = "nemoguardrails";
const SUPPORTED_NEMOGUARDRAILS_VERSION: &str = "0.22.0";

pub(super) fn register_local_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let runtime = Arc::new(LocalGuardrailsRuntime::new(&config)?);

    if config.input || config.output {
        let llm_runtime = Arc::clone(&runtime);
        let enable_input = config.input;
        let enable_output = config.output;
        let llm_execution: LlmExecutionFn = Arc::new(move |_name, request, next| {
            let runtime = Arc::clone(&llm_runtime);
            Box::pin(async move {
                runtime
                    .execute_llm(request, next, enable_input, enable_output)
                    .await
            })
        });
        ctx.register_llm_execution_intercept(
            "nemo_guardrails_local",
            config.priority,
            llm_execution,
        )?;

        let stream_runtime = Arc::clone(&runtime);
        let enable_input = config.input;
        let enable_output = config.output;
        let llm_stream_execution: LlmStreamExecutionFn = Arc::new(move |_name, request, next| {
            let runtime = Arc::clone(&stream_runtime);
            Box::pin(async move {
                runtime
                    .execute_llm_stream(request, next, enable_input, enable_output)
                    .await
            })
        });
        ctx.register_llm_stream_execution_intercept(
            "nemo_guardrails_local_stream",
            config.priority,
            llm_stream_execution,
        )?;
    }

    if config.tool_input || config.tool_output {
        let tool_runtime = Arc::clone(&runtime);
        let enable_tool_input = config.tool_input;
        let enable_tool_output = config.tool_output;
        let tool_execution: ToolExecutionFn = Arc::new(move |tool_name, args, next| {
            let runtime = Arc::clone(&tool_runtime);
            let tool_name = tool_name.to_string();
            Box::pin(async move {
                let current_args = if enable_tool_input {
                    runtime.check_tool_input(&tool_name, &args).await?
                } else {
                    args
                };

                let tool_result = next(current_args.clone()).await?;
                if !enable_tool_output {
                    return Ok(tool_result);
                }

                runtime
                    .check_tool_output(&tool_name, &current_args, &tool_result)
                    .await
            })
        });
        ctx.register_tool_execution_intercept(
            "nemo_guardrails_local",
            config.priority,
            tool_execution,
        )?;
    }

    Ok(())
}

struct LocalGuardrailsRuntime {
    bridge: LocalGuardrailsBridge,
    codec: Option<LocalGuardrailsCodec>,
}

impl LocalGuardrailsRuntime {
    fn new(config: &NeMoGuardrailsConfig) -> PluginResult<Self> {
        Python::initialize();
        Ok(Self {
            bridge: LocalGuardrailsBridge::new(config)?,
            codec: resolve_codec(config)?,
        })
    }

    async fn execute_llm(
        &self,
        request: LlmRequest,
        next: crate::api::runtime::LlmExecutionNextFn,
        enable_input: bool,
        enable_output: bool,
    ) -> FlowResult<Json> {
        let (request, messages) = self.prepare_llm_request(request, enable_input).await?;
        let response = next(request).await?;

        if enable_output {
            let annotated_response = self.codec()?.decode_response(&response)?;
            if let Some(response_text) = annotated_response.response_text() {
                self.check_output_rails(&messages, response_text).await?;
            }
        }

        Ok(response)
    }

    async fn execute_llm_stream(
        &self,
        request: LlmRequest,
        next: crate::api::runtime::LlmStreamExecutionNextFn,
        enable_input: bool,
        enable_output: bool,
    ) -> FlowResult<LlmJsonStream> {
        let (request, messages) = self.prepare_llm_request(request, enable_input).await?;
        let provider_stream = next(request).await?;

        if !enable_output || !self.bridge.has_streaming_output_rails()? {
            return Ok(provider_stream);
        }

        self.bridge.ensure_streaming_output_supported()?;
        self.guard_provider_stream(messages, provider_stream)
    }

    async fn prepare_llm_request(
        &self,
        request: LlmRequest,
        enable_input: bool,
    ) -> FlowResult<(LlmRequest, Vec<Json>)> {
        let codec = self.codec()?;
        let mut current_request = request;
        let mut annotated = codec.decode(&current_request)?;
        let mut messages = messages_from_annotated(&annotated)?;

        if enable_input {
            match self
                .bridge
                .check(messages.clone(), LocalRailKind::Input)
                .await?
            {
                LocalCheckOutcome::Passed => {}
                LocalCheckOutcome::Blocked { rail, .. } => {
                    return Err(blocked_error("input", rail.as_deref()));
                }
                LocalCheckOutcome::Modified { content, .. } => {
                    replace_last_role_content(&mut annotated, "user", content)?;
                    current_request = codec.encode(&annotated, &current_request)?;
                    messages = messages_from_annotated(&annotated)?;
                }
            }
        }

        Ok((current_request, messages))
    }

    async fn check_output_rails(&self, messages: &[Json], response_text: &str) -> FlowResult<()> {
        let mut output_messages = messages.to_vec();
        output_messages.push(json!({
            "role": "assistant",
            "content": response_text,
        }));

        match self
            .bridge
            .check(output_messages, LocalRailKind::Output)
            .await?
        {
            LocalCheckOutcome::Passed => Ok(()),
            LocalCheckOutcome::Blocked { rail, .. } => {
                Err(blocked_error("output", rail.as_deref()))
            }
            LocalCheckOutcome::Modified { .. } => Err(local_violation(
                "NeMo Guardrails output rail returned modified content, but the local backend \
                 does not rewrite provider responses yet.",
            )),
        }
    }

    async fn check_tool_input(&self, tool_name: &str, args: &Json) -> FlowResult<Json> {
        let messages = vec![json!({
            "role": "user",
            "content": tool_input_content(tool_name, args)?,
        })];

        match self.bridge.check(messages, LocalRailKind::Input).await? {
            LocalCheckOutcome::Passed => Ok(args.clone()),
            LocalCheckOutcome::Blocked { rail, .. } => {
                Err(blocked_error("tool_input", rail.as_deref()))
            }
            LocalCheckOutcome::Modified { content, .. } => {
                modified_tool_payload(&content, "arguments")
            }
        }
    }

    async fn check_tool_output(
        &self,
        tool_name: &str,
        args: &Json,
        result: &Json,
    ) -> FlowResult<Json> {
        let messages = vec![
            json!({
                "role": "user",
                "content": tool_input_content(tool_name, args)?,
            }),
            json!({
                "role": "assistant",
                "content": tool_output_content(tool_name, args, result)?,
            }),
        ];

        match self.bridge.check(messages, LocalRailKind::Output).await? {
            LocalCheckOutcome::Passed => Ok(result.clone()),
            LocalCheckOutcome::Blocked { rail, .. } => {
                Err(blocked_error("tool_output", rail.as_deref()))
            }
            LocalCheckOutcome::Modified { content, .. } => {
                modified_tool_payload(&content, "result")
            }
        }
    }

    fn guard_provider_stream(
        &self,
        messages: Vec<Json>,
        provider_stream: LlmJsonStream,
    ) -> FlowResult<LlmJsonStream> {
        let (text_tx, text_rx) = mpsc::channel::<Option<String>>(32);
        let (chunk_tx, chunk_rx) = mpsc::channel::<FlowResult<Json>>(32);
        let blocked = Arc::new(Mutex::new(None));
        let monitor = self
            .bridge
            .spawn_stream_monitor(messages, text_rx, Arc::clone(&blocked))?;
        let codec = *self.codec()?;

        tokio::spawn(async move {
            forward_guarded_provider_stream(
                provider_stream,
                codec,
                text_tx,
                chunk_tx,
                monitor,
                blocked,
            )
            .await;
        });

        Ok(Box::pin(ReceiverStream::new(chunk_rx)) as LlmJsonStream)
    }

    fn codec(&self) -> FlowResult<&LocalGuardrailsCodec> {
        self.codec.as_ref().ok_or_else(|| {
            FlowError::Internal(
                "local NeMo Guardrails backend requires a supported codec".to_string(),
            )
        })
    }
}

struct LocalGuardrailsBridge {
    rails: Py<PyAny>,
    input_rail: Py<PyAny>,
    output_rail: Py<PyAny>,
    blocked_status: String,
    modified_status: String,
}

impl LocalGuardrailsBridge {
    fn new(config: &NeMoGuardrailsConfig) -> PluginResult<Self> {
        Python::attach(|py| {
            let imports = load_nemoguardrails(
                py,
                config.local.as_ref().and_then(|l| {
                    l.python_module
                        .as_deref()
                        .filter(|module| !module.trim().is_empty())
                }),
            )?;
            let guardrails_config = build_guardrails_config(py, config, &imports.rails_config_cls)?;
            let rails = imports.llm_rails_cls.call1(py, (guardrails_config,))?;
            let input_rail = imports.rail_type.getattr(py, "INPUT")?;
            let output_rail = imports.rail_type.getattr(py, "OUTPUT")?;
            let blocked = imports.rail_status.getattr(py, "BLOCKED")?;
            let modified = imports.rail_status.getattr(py, "MODIFIED")?;
            let blocked_status = py_status_value(blocked.bind(py))?;
            let modified_status = py_status_value(modified.bind(py))?;

            Ok::<Self, PyErr>(Self {
                rails,
                input_rail,
                output_rail,
                blocked_status,
                modified_status,
            })
        })
        .map_err(|err| PluginError::RegistrationFailed(err.to_string()))
    }

    async fn check(
        &self,
        messages: Vec<Json>,
        kind: LocalRailKind,
    ) -> FlowResult<LocalCheckOutcome> {
        let future = Python::attach(|py| {
            let messages = json_to_py(py, &Json::Array(messages))
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            let rail_type = match kind {
                LocalRailKind::Input => self.input_rail.clone_ref(py),
                LocalRailKind::Output => self.output_rail.clone_ref(py),
            };
            let rail_types =
                PyList::new(py, [rail_type]).map_err(|err| FlowError::Internal(err.to_string()))?;
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("rail_types", rail_types)
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            let result = self
                .rails
                .bind(py)
                .call_method("check_async", (messages,), Some(&kwargs))
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            pyo3_async_runtimes::tokio::into_future(result.unbind().into_bound(py))
                .map_err(|err| FlowError::Internal(err.to_string()))
        })?;

        let result = future
            .await
            .map_err(|err| FlowError::Internal(err.to_string()))?;

        Python::attach(|py| {
            self.parse_check_result(result.bind(py))
                .map_err(|err| FlowError::Internal(err.to_string()))
        })
    }

    fn has_streaming_output_rails(&self) -> FlowResult<bool> {
        Python::attach(|py| {
            let Some(output) = self.output_rails_config(py)? else {
                return Ok(false);
            };
            match output.getattr("flows") {
                Ok(flows) => flows
                    .is_truthy()
                    .map_err(|err| FlowError::Internal(err.to_string())),
                Err(_) => Ok(false),
            }
        })
    }

    fn ensure_streaming_output_supported(&self) -> FlowResult<()> {
        Python::attach(|py| {
            let Some(output) = self.output_rails_config(py)? else {
                return Ok(());
            };
            let streaming = output.getattr("streaming").map_err(|_| {
                FlowError::Internal(
                    "local NeMo Guardrails streaming output rails require \
                     rails.output.streaming.enabled = true in the Guardrails config."
                        .to_string(),
                )
            })?;
            let enabled = streaming
                .getattr("enabled")
                .and_then(|value| value.is_truthy())
                .unwrap_or(false);
            if !enabled {
                return Err(FlowError::Internal(
                    "local NeMo Guardrails streaming output rails require \
                     rails.output.streaming.enabled = true in the Guardrails config."
                        .to_string(),
                ));
            }

            let stream_first = streaming
                .getattr("stream_first")
                .and_then(|value| value.is_truthy())
                .unwrap_or(true);
            if !stream_first {
                return Err(FlowError::Internal(
                    "local NeMo Guardrails streaming output rails currently require \
                     rails.output.streaming.stream_first = true."
                        .to_string(),
                ));
            }

            Ok(())
        })
    }

    fn spawn_stream_monitor(
        &self,
        messages: Vec<Json>,
        text_rx: mpsc::Receiver<Option<String>>,
        blocked: Arc<Mutex<Option<String>>>,
    ) -> FlowResult<JoinHandle<FlowResult<()>>> {
        let (async_iter, task_locals) = Python::attach(|py| {
            let generator = Py::new(
                py,
                PyStringStream {
                    receiver: Arc::new(tokio::sync::Mutex::new(text_rx)),
                },
            )
            .map_err(|err| FlowError::Internal(err.to_string()))?;
            let messages = json_to_py(py, &Json::Array(messages))
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("messages", messages)
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            kwargs
                .set_item("generator", generator)
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            kwargs
                .set_item("include_metadata", false)
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            let async_iter = self
                .rails
                .bind(py)
                .call_method("stream_async", (), Some(&kwargs))
                .map_err(|err| FlowError::Internal(err.to_string()))?
                .unbind();
            let task_locals = pyo3_async_runtimes::tokio::get_current_locals(py)
                .map_err(|err| FlowError::Internal(err.to_string()))?;
            Ok((async_iter, task_locals))
        })?;

        let async_iter = Arc::new(async_iter);
        Ok(tokio::spawn(pyo3_async_runtimes::tokio::scope(
            task_locals,
            async move { monitor_guardrails_stream(async_iter, blocked).await },
        )))
    }

    fn output_rails_config<'py>(&self, py: Python<'py>) -> FlowResult<Option<Bound<'py, PyAny>>> {
        let rails = self.rails.bind(py);
        let config = match rails.getattr("config") {
            Ok(config) => config,
            Err(_) => return Ok(None),
        };
        let rails_config = match config.getattr("rails") {
            Ok(rails_config) => rails_config,
            Err(_) => return Ok(None),
        };
        match rails_config.getattr("output") {
            Ok(output) => Ok(Some(output)),
            Err(_) => Ok(None),
        }
    }

    fn parse_check_result(&self, result: &Bound<'_, PyAny>) -> PyResult<LocalCheckOutcome> {
        let status = py_status_value(&result.getattr("status")?)?;
        let rail = optional_string_attr(result, "rail")?;
        let content = string_attr_or_empty(result, "content")?;

        if status == self.blocked_status {
            return Ok(LocalCheckOutcome::Blocked { rail });
        }
        if status == self.modified_status {
            return Ok(LocalCheckOutcome::Modified { content });
        }
        Ok(LocalCheckOutcome::Passed)
    }
}

struct GuardrailsRuntimeImports {
    rails_config_cls: Py<PyAny>,
    llm_rails_cls: Py<PyAny>,
    rail_type: Py<PyAny>,
    rail_status: Py<PyAny>,
}

fn load_nemoguardrails(
    py: Python<'_>,
    module_name: Option<&str>,
) -> PyResult<GuardrailsRuntimeImports> {
    let root_module = module_name.unwrap_or(DEFAULT_MODULE_NAME);
    let importlib = py.import("importlib")?;
    let import_module = importlib.getattr("import_module")?;
    let guardrails = import_module
        .call1((root_module,))
        .map_err(|err| import_dependency_error(py, err, root_module))?;
    let options_module_name = format!("{root_module}.rails.llm.options");
    let options = import_module
        .call1((options_module_name.as_str(),))
        .map_err(|err| import_dependency_error(py, err, root_module))?;

    let version = guardrails
        .getattr("__version__")
        .ok()
        .and_then(|value| value.extract::<String>().ok());
    if version.as_deref() != Some(SUPPORTED_NEMOGUARDRAILS_VERSION) {
        let found = version
            .map(|version| format!("{version:?}"))
            .unwrap_or_else(|| "None".to_string());
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "NeMo Guardrails local backend requires nemoguardrails==\
             {SUPPORTED_NEMOGUARDRAILS_VERSION}, but found {found}. \
             Install it with: pip install nemoguardrails=={SUPPORTED_NEMOGUARDRAILS_VERSION}"
        )));
    }

    Ok(GuardrailsRuntimeImports {
        rails_config_cls: guardrails.getattr("RailsConfig")?.unbind(),
        llm_rails_cls: guardrails.getattr("LLMRails")?.unbind(),
        rail_type: options.getattr("RailType")?.unbind(),
        rail_status: options.getattr("RailStatus")?.unbind(),
    })
}

fn import_dependency_error(py: Python<'_>, err: PyErr, root_module: &str) -> PyErr {
    if !err.is_instance_of::<pyo3::exceptions::PyImportError>(py) {
        return err;
    }

    let name = err.value(py).getattr("name").ok().and_then(|name| {
        if name.is_none() {
            None
        } else {
            name.extract::<String>().ok()
        }
    });

    if name.as_deref() == Some(root_module) {
        return pyo3::exceptions::PyRuntimeError::new_err(format!(
            "NeMo Guardrails is required for the built-in NeMo Guardrails local backend. \
             Install it with: pip install nemoguardrails=={SUPPORTED_NEMOGUARDRAILS_VERSION}"
        ));
    }

    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "NeMo Guardrails local backend could not import a required dependency: {}. \
         Install the full NeMo Guardrails runtime dependencies.",
        name.unwrap_or_else(|| err.to_string())
    ))
}

fn build_guardrails_config(
    py: Python<'_>,
    config: &NeMoGuardrailsConfig,
    rails_config_cls: &Py<PyAny>,
) -> PyResult<Py<PyAny>> {
    let rails_config_cls = rails_config_cls.bind(py);
    if let Some(config_path) = config.config_path.as_deref() {
        return rails_config_cls
            .call_method1("from_path", (config_path,))
            .map(Bound::unbind);
    }

    let config_yaml = config.config_yaml.as_deref().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "config_yaml is required when config_path is not provided",
        )
    })?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("colang_content", config.colang_content.as_deref())?;
    kwargs.set_item("yaml_content", config_yaml)?;
    rails_config_cls
        .call_method("from_content", (), Some(&kwargs))
        .map(Bound::unbind)
}

fn py_status_value(status: &Bound<'_, PyAny>) -> PyResult<String> {
    let value = status.getattr("value").unwrap_or_else(|_| status.clone());
    Ok(value.str()?.extract::<String>()?.to_lowercase())
}

fn optional_string_attr(obj: &Bound<'_, PyAny>, attr: &str) -> PyResult<Option<String>> {
    match obj.getattr(attr) {
        Ok(value) if !value.is_none() => Ok(Some(value.str()?.extract::<String>()?)),
        Ok(_) | Err(_) => Ok(None),
    }
}

fn string_attr_or_empty(obj: &Bound<'_, PyAny>, attr: &str) -> PyResult<String> {
    match optional_string_attr(obj, attr)? {
        Some(value) => Ok(value),
        None => Ok(String::new()),
    }
}

fn json_to_py(py: Python<'_>, value: &Json) -> PyResult<Py<PyAny>> {
    let obj: Bound<'_, PyAny> = pythonize::pythonize(py, value).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to convert from JSON: {e}"))
    })?;
    Ok(obj.unbind())
}

#[derive(Clone, Copy)]
enum LocalGuardrailsCodec {
    OpenAIChat,
    OpenAIResponses,
    AnthropicMessages,
}

impl LocalGuardrailsCodec {
    fn decode(&self, request: &LlmRequest) -> FlowResult<AnnotatedLlmRequest> {
        match self {
            Self::OpenAIChat => OpenAIChatCodec.decode(request),
            Self::OpenAIResponses => OpenAIResponsesCodec.decode(request),
            Self::AnthropicMessages => AnthropicMessagesCodec.decode(request),
        }
    }

    fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> FlowResult<LlmRequest> {
        match self {
            Self::OpenAIChat => OpenAIChatCodec.encode(annotated, original),
            Self::OpenAIResponses => OpenAIResponsesCodec.encode(annotated, original),
            Self::AnthropicMessages => AnthropicMessagesCodec.encode(annotated, original),
        }
    }

    fn decode_response(
        &self,
        response: &Json,
    ) -> FlowResult<crate::codec::response::AnnotatedLlmResponse> {
        match self {
            Self::OpenAIChat => OpenAIChatCodec.decode_response(response),
            Self::OpenAIResponses => OpenAIResponsesCodec.decode_response(response),
            Self::AnthropicMessages => AnthropicMessagesCodec.decode_response(response),
        }
    }
}

fn resolve_codec(config: &NeMoGuardrailsConfig) -> PluginResult<Option<LocalGuardrailsCodec>> {
    if !(config.input || config.output) {
        return Ok(None);
    }

    match config.codec.as_deref() {
        Some("openai_chat") => Ok(Some(LocalGuardrailsCodec::OpenAIChat)),
        Some("openai_responses") => Ok(Some(LocalGuardrailsCodec::OpenAIResponses)),
        Some("anthropic_messages") => Ok(Some(LocalGuardrailsCodec::AnthropicMessages)),
        Some(other) => Err(PluginError::InvalidConfig(format!(
            "unsupported local NeMo Guardrails codec '{other}'"
        ))),
        None => Err(PluginError::InvalidConfig(
            "local NeMo Guardrails backend requires a supported codec".to_string(),
        )),
    }
}

enum LocalCheckOutcome {
    Passed,
    Blocked { rail: Option<String> },
    Modified { content: String },
}

#[derive(Clone, Copy)]
enum LocalRailKind {
    Input,
    Output,
}

fn messages_from_annotated(annotated: &AnnotatedLlmRequest) -> FlowResult<Vec<Json>> {
    match serde_json::to_value(&annotated.messages)
        .map_err(|err| FlowError::Internal(format!("failed to serialize messages: {err}")))?
    {
        Json::Array(messages) => Ok(messages),
        _ => Err(FlowError::Internal(
            "serialized messages were not a JSON array".to_string(),
        )),
    }
}

fn replace_last_role_content(
    annotated: &mut AnnotatedLlmRequest,
    role: &str,
    content: String,
) -> FlowResult<()> {
    for message in annotated.messages.iter_mut().rev() {
        match (role, message) {
            (
                "user",
                Message::User {
                    content: target, ..
                },
            ) => {
                *target = MessageContent::Text(content);
                return Ok(());
            }
            (
                "assistant",
                Message::Assistant {
                    content: target, ..
                },
            ) => {
                *target = Some(MessageContent::Text(content));
                return Ok(());
            }
            _ => {}
        }
    }

    Err(local_violation(format!(
        "NeMo Guardrails returned modified {role} content but no {role} message was present."
    )))
}

fn tool_input_content(name: &str, args: &Json) -> FlowResult<String> {
    serde_json::to_string(&json!({
        "tool_name": name,
        "arguments": args,
    }))
    .map_err(|err| FlowError::Internal(format!("failed to serialize tool input: {err}")))
}

fn tool_output_content(name: &str, args: &Json, result: &Json) -> FlowResult<String> {
    serde_json::to_string(&json!({
        "tool_name": name,
        "arguments": args,
        "result": result,
    }))
    .map_err(|err| FlowError::Internal(format!("failed to serialize tool output: {err}")))
}

fn modified_tool_payload(content: &str, field: &str) -> FlowResult<Json> {
    let value: Json = serde_json::from_str(content).map_err(|_| {
        local_violation(format!(
            "NeMo Guardrails returned modified tool {field} content that is not valid JSON."
        ))
    })?;

    let Json::Object(object) = value else {
        return Err(local_violation(format!(
            "NeMo Guardrails returned modified tool {field} content without a '{field}' field."
        )));
    };
    object.get(field).cloned().ok_or_else(|| {
        local_violation(format!(
            "NeMo Guardrails returned modified tool {field} content without a '{field}' field."
        ))
    })
}

fn blocked_error(rail_type: &str, rail: Option<&str>) -> FlowError {
    let detail = rail
        .filter(|rail| !rail.is_empty())
        .map(|rail| format!(" by rail '{rail}'"))
        .unwrap_or_default();
    let subject = if matches!(rail_type, "input" | "output") {
        "LLM call"
    } else {
        "tool call"
    };
    local_violation(format!(
        "NeMo Guardrails {rail_type} rail blocked the {subject}{detail}."
    ))
}

fn local_violation(message: impl Into<String>) -> FlowError {
    FlowError::Internal(message.into())
}

async fn forward_guarded_provider_stream(
    mut provider_stream: LlmJsonStream,
    codec: LocalGuardrailsCodec,
    text_tx: mpsc::Sender<Option<String>>,
    chunk_tx: mpsc::Sender<FlowResult<Json>>,
    monitor: JoinHandle<FlowResult<()>>,
    blocked: Arc<Mutex<Option<String>>>,
) {
    while let Some(item) = provider_stream.next().await {
        let chunk = match item {
            Ok(chunk) => chunk,
            Err(err) => {
                let _ = chunk_tx.send(Err(err)).await;
                let _ = text_tx.send(None).await;
                let _ = monitor.await;
                return;
            }
        };

        if let Some(message) = blocked_message(&blocked) {
            let _ = chunk_tx.send(Err(streaming_output_blocked(message))).await;
            let _ = text_tx.send(None).await;
            let _ = monitor.await;
            return;
        }

        let text = extract_stream_text(codec, &chunk);

        if chunk_tx.send(Ok(chunk)).await.is_err() {
            let _ = text_tx.send(None).await;
            let _ = monitor.await;
            return;
        }
        tokio::task::yield_now().await;

        if let Some(text) = text {
            let _ = text_tx.send(Some(text)).await;
        }

        if let Some(message) = blocked_message(&blocked) {
            let _ = chunk_tx.send(Err(streaming_output_blocked(message))).await;
            let _ = text_tx.send(None).await;
            let _ = monitor.await;
            return;
        }
    }

    let _ = text_tx.send(None).await;
    match monitor.await {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            let _ = chunk_tx.send(Err(err)).await;
            return;
        }
        Err(err) => {
            let _ = chunk_tx
                .send(Err(FlowError::Internal(format!(
                    "nemo_guardrails stream monitor task failed: {err}"
                ))))
                .await;
            return;
        }
    }

    if let Some(message) = blocked_message(&blocked) {
        let _ = chunk_tx.send(Err(streaming_output_blocked(message))).await;
    }
}

fn blocked_message(blocked: &Arc<Mutex<Option<String>>>) -> Option<String> {
    blocked.lock().ok().and_then(|guard| guard.clone())
}

fn streaming_output_blocked(message: String) -> FlowError {
    local_violation(format!(
        "NeMo Guardrails output rail blocked the LLM call: {message}"
    ))
}

fn extract_stream_text(codec: LocalGuardrailsCodec, chunk: &Json) -> Option<String> {
    let chunk = chunk.as_object()?;
    match codec {
        LocalGuardrailsCodec::OpenAIChat => {
            let choices = chunk.get("choices")?.as_array()?;
            let mut parts = vec![];
            for choice in choices {
                let content = choice
                    .get("delta")
                    .and_then(Json::as_object)
                    .and_then(|delta| delta.get("content"))
                    .and_then(Json::as_str);
                if let Some(content) = content
                    && !content.is_empty()
                {
                    parts.push(content);
                }
            }
            (!parts.is_empty()).then(|| parts.join(""))
        }
        LocalGuardrailsCodec::OpenAIResponses => {
            if chunk.get("type").and_then(Json::as_str) == Some("response.output_text.delta") {
                chunk
                    .get("delta")
                    .and_then(Json::as_str)
                    .filter(|delta| !delta.is_empty())
                    .map(str::to_string)
            } else {
                None
            }
        }
        LocalGuardrailsCodec::AnthropicMessages => {
            if chunk.get("type").and_then(Json::as_str) != Some("content_block_delta") {
                return None;
            }
            let delta = chunk.get("delta")?.as_object()?;
            if delta.get("type").and_then(Json::as_str) != Some("text_delta") {
                return None;
            }
            delta
                .get("text")
                .and_then(Json::as_str)
                .filter(|text| !text.is_empty())
                .map(str::to_string)
        }
    }
}

async fn monitor_guardrails_stream(
    async_iter: Arc<Py<PyAny>>,
    blocked: Arc<Mutex<Option<String>>>,
) -> FlowResult<()> {
    loop {
        let Some(coro) = next_async_iter_coro(&async_iter)? else {
            break;
        };
        let Some(value) = await_async_iter_value(coro).await? else {
            break;
        };
        Python::attach(|py| {
            if let Ok(chunk) = value.bind(py).extract::<String>()
                && let Some(message) = guardrails_stream_error_message(&chunk)
            {
                let mut guard = blocked.lock().map_err(|err| {
                    FlowError::Internal(format!("stream block state lock poisoned: {err}"))
                })?;
                *guard = Some(message);
            }
            Ok::<(), FlowError>(())
        })?;
        if blocked_message(&blocked).is_some() {
            break;
        }
    }
    Ok(())
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

async fn await_async_iter_value(coro: Py<PyAny>) -> FlowResult<Option<Py<PyAny>>> {
    let future = Python::attach(|py| {
        pyo3_async_runtimes::tokio::into_future(coro.into_bound(py))
            .map_err(|err| FlowError::Internal(err.to_string()))
    })?;

    match future.await {
        Ok(result) => Ok(Some(result)),
        Err(error) => Python::attach(|py| {
            if error.is_instance_of::<pyo3::exceptions::PyStopAsyncIteration>(py) {
                Ok(None)
            } else {
                Err(FlowError::Internal(error.to_string()))
            }
        }),
    }
}

fn guardrails_stream_error_message(chunk: &str) -> Option<String> {
    let payload: Json = serde_json::from_str(chunk).ok()?;
    let error = payload.get("error")?.as_object()?;
    if error.get("type").and_then(Json::as_str) != Some("guardrails_violation") {
        return None;
    }
    error
        .get("message")
        .and_then(Json::as_str)
        .filter(|message| !message.is_empty())
        .map(str::to_string)
        .or_else(|| Some("Blocked by output rails.".to_string()))
}

#[pyclass(name = "StringStream")]
struct PyStringStream {
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<Option<String>>>>,
}

#[pymethods]
impl PyStringStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let receiver = Arc::clone(&self.receiver);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = receiver.lock().await;
            match guard.recv().await {
                Some(Some(value)) => Ok(value),
                Some(None) | None => Err(PyErr::new::<pyo3::exceptions::PyStopAsyncIteration, _>(
                    "stream exhausted",
                )),
            }
        })
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/nemo_guardrails/local_python_tests.rs"]
mod tests;
