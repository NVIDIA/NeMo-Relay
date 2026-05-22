// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use std::sync::Arc;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use std::time::Duration;

use serde_json::{Map, Value as Json, json};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use tokio::sync::mpsc;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use tokio_stream::wrappers::ReceiverStream;

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::api::llm::LlmRequest;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::api::runtime::{LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn, ToolExecutionFn};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::api::scope::{EmitMarkEventParams, ScopeHandle, event, get_handle};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::codec::openai_chat::OpenAIChatCodec;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::codec::streaming::SseEventDecoder;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::codec::traits::LlmCodec;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use crate::error::FlowError;
use crate::plugin::{PluginError, PluginRegistrationContext, Result as PluginResult};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use rustls::crypto::ring;

use super::{NeMoGuardrailsConfig, RemoteBackendConfig, RequestDefaultsConfig};

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
#[derive(Clone)]
struct RemoteBackendRuntime {
    endpoint: String,
    client: reqwest::Client,
    config_id: Option<String>,
    config_ids: Vec<String>,
    request_defaults: Option<RequestDefaultsConfig>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
#[derive(Clone, Copy)]
enum RemoteCheckKind {
    Input,
    Output,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
impl RemoteBackendRuntime {
    fn new(config: &NeMoGuardrailsConfig, remote: &RemoteBackendConfig) -> PluginResult<Self> {
        let endpoint = remote.endpoint.clone().ok_or_else(|| {
            PluginError::InvalidConfig("remote.endpoint is required in remote mode".to_string())
        })?;
        let mut default_headers = HeaderMap::new();
        for (name, value) in &remote.headers {
            let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|err| {
                PluginError::InvalidConfig(format!(
                    "remote.headers contains invalid header name '{name}': {err}"
                ))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|err| {
                PluginError::InvalidConfig(format!(
                    "remote.headers[{name}] has an invalid value: {err}"
                ))
            })?;
            default_headers.insert(header_name, header_value);
        }

        let _ = ring::default_provider().install_default();

        let client = reqwest::Client::builder()
            .default_headers(default_headers)
            .timeout(Duration::from_millis(remote.timeout_millis))
            .build()
            .map_err(|err| {
                PluginError::RegistrationFailed(format!(
                    "failed to construct NeMo Guardrails remote client: {err}"
                ))
            })?;

        Ok(Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            client,
            config_id: remote.config_id.clone(),
            config_ids: remote.config_ids.clone(),
            request_defaults: config.request_defaults.clone(),
        })
    }

    async fn execute(&self, request: LlmRequest, stream: bool) -> crate::error::Result<Json> {
        let parent = get_handle().ok();
        self.emit_mark(
            "nemo_guardrails.remote.start",
            &parent,
            remote_mark_data(stream, &self.config_id, &self.config_ids, None, None),
        );
        let body = self
            .build_request_body(&request, stream)
            .inspect_err(|err| {
                self.emit_mark(
                    "nemo_guardrails.remote.error",
                    &parent,
                    remote_mark_data(
                        stream,
                        &self.config_id,
                        &self.config_ids,
                        None,
                        Some(err.to_string()),
                    ),
                );
            })?;
        let serialized = serde_json::to_vec(&body).map_err(|err| {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    stream,
                    &self.config_id,
                    &self.config_ids,
                    None,
                    Some(format!("failed to serialize remote request body: {err}")),
                ),
            );
            FlowError::Internal(format!(
                "nemo_guardrails failed to serialize remote request body: {err}"
            ))
        })?;
        let response = self
            .client
            .post(self.chat_completions_url())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serialized)
            .send()
            .await
            .map_err(|err| {
                self.emit_mark(
                    "nemo_guardrails.remote.error",
                    &parent,
                    remote_mark_data(
                        stream,
                        &self.config_id,
                        &self.config_ids,
                        None,
                        Some(format!("remote request failed: {err}")),
                    ),
                );
                FlowError::Internal(format!("nemo_guardrails remote request failed: {err}"))
            })?;
        let status = response.status();
        let payload = response.text().await.map_err(|err| {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    stream,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(format!("failed to read remote response body: {err}")),
                ),
            );
            FlowError::Internal(format!(
                "nemo_guardrails failed to read remote response body: {err}"
            ))
        })?;
        if !status.is_success() {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    stream,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(redact_remote_error_payload(status.as_u16(), &payload)),
                ),
            );
            return Err(FlowError::Internal(format!(
                "nemo_guardrails remote request failed with status {status}: {payload}"
            )));
        }
        let response_json = serde_json::from_str(&payload).map_err(|err| {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    stream,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(format!("failed to parse remote response JSON: {err}")),
                ),
            );
            FlowError::Internal(format!(
                "nemo_guardrails failed to parse remote response JSON: {err}"
            ))
        })?;
        self.emit_mark(
            "nemo_guardrails.remote.end",
            &parent,
            remote_mark_data(
                stream,
                &self.config_id,
                &self.config_ids,
                Some(status.as_u16()),
                None,
            ),
        );
        Ok(response_json)
    }

    async fn execute_stream(&self, request: LlmRequest) -> crate::error::Result<LlmJsonStream> {
        let parent = get_handle().ok();
        self.emit_mark(
            "nemo_guardrails.remote.start",
            &parent,
            remote_mark_data(true, &self.config_id, &self.config_ids, None, None),
        );
        let body = self.build_request_body(&request, true).inspect_err(|err| {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    true,
                    &self.config_id,
                    &self.config_ids,
                    None,
                    Some(err.to_string()),
                ),
            );
        })?;
        let serialized = serde_json::to_vec(&body).map_err(|err| {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    true,
                    &self.config_id,
                    &self.config_ids,
                    None,
                    Some(format!(
                        "failed to serialize remote stream request body: {err}"
                    )),
                ),
            );
            FlowError::Internal(format!(
                "nemo_guardrails failed to serialize remote stream request body: {err}"
            ))
        })?;
        let mut response = self
            .client
            .post(self.chat_completions_url())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serialized)
            .send()
            .await
            .map_err(|err| {
                self.emit_mark(
                    "nemo_guardrails.remote.error",
                    &parent,
                    remote_mark_data(
                        true,
                        &self.config_id,
                        &self.config_ids,
                        None,
                        Some(format!("remote stream request failed: {err}")),
                    ),
                );
                FlowError::Internal(format!(
                    "nemo_guardrails remote stream request failed: {err}"
                ))
            })?;
        let status = response.status();
        if !status.is_success() {
            let payload = response.text().await.map_err(|err| {
                self.emit_mark(
                    "nemo_guardrails.remote.error",
                    &parent,
                    remote_mark_data(
                        true,
                        &self.config_id,
                        &self.config_ids,
                        Some(status.as_u16()),
                        Some(format!("failed to read remote stream error body: {err}")),
                    ),
                );
                FlowError::Internal(format!(
                    "nemo_guardrails failed to read remote stream error body: {err}"
                ))
            })?;
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                remote_mark_data(
                    true,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(redact_remote_error_payload(status.as_u16(), &payload)),
                ),
            );
            return Err(FlowError::Internal(format!(
                "nemo_guardrails remote stream request failed with status {status}: {payload}"
            )));
        }

        let (tx, rx) = mpsc::channel(16);
        let parent_for_task = parent.clone();
        let config_id = self.config_id.clone();
        let config_ids = self.config_ids.clone();
        tokio::spawn(async move {
            let mut decoder = SseEventDecoder::new();
            loop {
                let bytes = match response.chunk().await {
                    Ok(Some(bytes)) => bytes,
                    Ok(None) => break,
                    Err(err) => {
                        emit_remote_mark(
                            "nemo_guardrails.remote.error",
                            &parent_for_task,
                            remote_mark_data(
                                true,
                                &config_id,
                                &config_ids,
                                Some(status.as_u16()),
                                Some(format!("failed to read remote stream chunk: {err}")),
                            ),
                        );
                        let _ = tx
                            .send(Err(FlowError::Internal(format!(
                                "nemo_guardrails failed to read remote stream chunk: {err}"
                            ))))
                            .await;
                        return;
                    }
                };
                let events = match decoder.push_bytes(&bytes) {
                    Ok(events) => events,
                    Err(err) => {
                        emit_remote_mark(
                            "nemo_guardrails.remote.error",
                            &parent_for_task,
                            remote_mark_data(
                                true,
                                &config_id,
                                &config_ids,
                                Some(status.as_u16()),
                                Some(err.to_string()),
                            ),
                        );
                        let _ = tx.send(Err(err)).await;
                        return;
                    }
                };
                for event in events {
                    if tx.send(Ok(event.data)).await.is_err() {
                        return;
                    }
                }
            }

            match decoder.finish() {
                Ok(Some(event)) => {
                    let _ = tx.send(Ok(event.data)).await;
                }
                Ok(None) => {}
                Err(err) => {
                    emit_remote_mark(
                        "nemo_guardrails.remote.error",
                        &parent_for_task,
                        remote_mark_data(
                            true,
                            &config_id,
                            &config_ids,
                            Some(status.as_u16()),
                            Some(err.to_string()),
                        ),
                    );
                    let _ = tx.send(Err(err)).await;
                    return;
                }
            }

            emit_remote_mark(
                "nemo_guardrails.remote.end",
                &parent_for_task,
                remote_mark_data(true, &config_id, &config_ids, Some(status.as_u16()), None),
            );
        });

        Ok(Box::pin(ReceiverStream::new(rx)) as LlmJsonStream)
    }

    async fn check_tool_input(&self, tool_name: &str, args: &Json) -> crate::error::Result<Json> {
        let original_content = tool_input_content(tool_name, args);
        let messages = vec![json!({"role": "user", "content": original_content.clone()})];
        let response = self
            .execute_remote_check(messages, RemoteCheckKind::Input, tool_name)
            .await?;
        if let Some(blocking_rail) = blocking_rail_name(&response) {
            return Err(FlowError::GuardrailRejected(format!(
                "nemo_guardrails tool_input rail blocked tool call by rail '{blocking_rail}'"
            )));
        }

        let result_content = chat_completion_content(&response)?;
        if result_content == original_content {
            return Ok(args.clone());
        }

        modified_tool_payload(&result_content, tool_name, "arguments")
    }

    async fn check_tool_output(
        &self,
        tool_name: &str,
        args: &Json,
        result: &Json,
    ) -> crate::error::Result<Json> {
        let input_content = tool_input_content(tool_name, args);
        let original_content = tool_output_content(tool_name, args, result);
        let messages = vec![
            json!({"role": "user", "content": input_content}),
            json!({"role": "assistant", "content": original_content.clone()}),
        ];
        let response = self
            .execute_remote_check(messages, RemoteCheckKind::Output, tool_name)
            .await?;
        if let Some(blocking_rail) = blocking_rail_name(&response) {
            return Err(FlowError::GuardrailRejected(format!(
                "nemo_guardrails tool_output rail blocked tool call by rail '{blocking_rail}'"
            )));
        }

        let result_content = chat_completion_content(&response)?;
        if result_content == original_content {
            return Ok(result.clone());
        }

        modified_tool_payload(&result_content, tool_name, "result")
    }

    fn build_request_body(&self, request: &LlmRequest, stream: bool) -> crate::error::Result<Json> {
        let annotated = OpenAIChatCodec.decode(request)?;
        if annotated.tools.is_some() || annotated.tool_choice.is_some() {
            return Err(FlowError::Internal(
                "nemo_guardrails remote backend does not support OpenAI tool definitions or tool_choice yet"
                    .to_string(),
            ));
        }

        let mut body = request.content.as_object().cloned().ok_or_else(|| {
            FlowError::Internal("LLM request content is not a JSON object".to_string())
        })?;
        body.insert("stream".to_string(), Json::Bool(stream));
        if let Some(guardrails) = self.build_guardrails_config() {
            body.insert("guardrails".to_string(), Json::Object(guardrails));
        }
        Ok(Json::Object(body))
    }

    fn build_guardrails_config(&self) -> Option<Map<String, Json>> {
        let mut guardrails = Map::new();
        if let Some(config_id) = &self.config_id {
            guardrails.insert("config_id".to_string(), Json::String(config_id.clone()));
        }
        if !self.config_ids.is_empty() {
            guardrails.insert(
                "config_ids".to_string(),
                Json::Array(self.config_ids.iter().cloned().map(Json::String).collect()),
            );
        }
        if let Some(request_defaults) = &self.request_defaults {
            if let Some(context) = &request_defaults.context {
                guardrails.insert("context".to_string(), context.clone());
            }
            if let Some(thread_id) = &request_defaults.thread_id {
                guardrails.insert("thread_id".to_string(), Json::String(thread_id.clone()));
            }
            if let Some(state) = &request_defaults.state {
                guardrails.insert("state".to_string(), state.clone());
            }
            let mut options = Map::new();
            if let Some(rails) = &request_defaults.rails {
                options.insert(
                    "rails".to_string(),
                    serde_json::to_value(rails)
                        .expect("request rails config should serialize to JSON"),
                );
            }
            if let Some(llm_params) = &request_defaults.llm_params {
                options.insert("llm_params".to_string(), llm_params.clone());
            }
            if let Some(llm_output) = request_defaults.llm_output {
                options.insert("llm_output".to_string(), Json::Bool(llm_output));
            }
            if let Some(output_vars) = &request_defaults.output_vars {
                options.insert("output_vars".to_string(), output_vars.clone());
            }
            if let Some(log) = &request_defaults.log {
                options.insert("log".to_string(), log.clone());
            }
            if !options.is_empty() {
                guardrails.insert("options".to_string(), Json::Object(options));
            }
        }

        (!guardrails.is_empty()).then_some(guardrails)
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/v1/chat/completions", self.endpoint)
    }

    fn emit_mark(&self, name: &str, parent: &Option<ScopeHandle>, data: Json) {
        emit_remote_mark(name, parent, data);
    }

    async fn execute_remote_check(
        &self,
        messages: Vec<Json>,
        kind: RemoteCheckKind,
        tool_name: &str,
    ) -> crate::error::Result<Json> {
        let parent = get_handle().ok();
        self.emit_mark(
            "nemo_guardrails.remote.start",
            &parent,
            tool_remote_mark_data(
                kind,
                tool_name,
                &self.config_id,
                &self.config_ids,
                None,
                None,
            ),
        );
        let mut body = Map::new();
        body.insert("model".to_string(), Json::String(String::new()));
        body.insert("messages".to_string(), Json::Array(messages));
        body.insert("stream".to_string(), Json::Bool(false));
        body.insert(
            "guardrails".to_string(),
            Json::Object(self.build_tool_check_guardrails(kind)),
        );
        let serialized = serde_json::to_vec(&Json::Object(body)).map_err(|err| {
            let message = format!("nemo_guardrails failed to serialize remote request body: {err}");
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                tool_remote_mark_data(
                    kind,
                    tool_name,
                    &self.config_id,
                    &self.config_ids,
                    None,
                    Some(message.clone()),
                ),
            );
            FlowError::Internal(message)
        })?;
        let response = self
            .client
            .post(self.chat_completions_url())
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(serialized)
            .send()
            .await
            .map_err(|err| {
                let message = format!("nemo_guardrails remote request failed: {err}");
                self.emit_mark(
                    "nemo_guardrails.remote.error",
                    &parent,
                    tool_remote_mark_data(
                        kind,
                        tool_name,
                        &self.config_id,
                        &self.config_ids,
                        None,
                        Some(message.clone()),
                    ),
                );
                FlowError::Internal(message)
            })?;
        let status = response.status();
        let payload = response.text().await.map_err(|err| {
            let message = format!("nemo_guardrails failed to read remote response body: {err}");
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                tool_remote_mark_data(
                    kind,
                    tool_name,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(message.clone()),
                ),
            );
            FlowError::Internal(message)
        })?;
        if !status.is_success() {
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                tool_remote_mark_data(
                    kind,
                    tool_name,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(redact_remote_error_payload(status.as_u16(), &payload)),
                ),
            );
            return Err(FlowError::Internal(format!(
                "nemo_guardrails remote request failed with status {status}: {payload}"
            )));
        }
        let response_json = serde_json::from_str(&payload).map_err(|err| {
            let message = format!("nemo_guardrails failed to parse remote response JSON: {err}");
            self.emit_mark(
                "nemo_guardrails.remote.error",
                &parent,
                tool_remote_mark_data(
                    kind,
                    tool_name,
                    &self.config_id,
                    &self.config_ids,
                    Some(status.as_u16()),
                    Some(message.clone()),
                ),
            );
            FlowError::Internal(message)
        })?;
        self.emit_mark(
            "nemo_guardrails.remote.end",
            &parent,
            tool_remote_mark_data(
                kind,
                tool_name,
                &self.config_id,
                &self.config_ids,
                Some(status.as_u16()),
                None,
            ),
        );
        Ok(response_json)
    }

    fn build_tool_check_guardrails(&self, kind: RemoteCheckKind) -> Map<String, Json> {
        let mut guardrails = Map::new();
        if let Some(config_id) = &self.config_id {
            guardrails.insert("config_id".to_string(), Json::String(config_id.clone()));
        }
        if !self.config_ids.is_empty() {
            guardrails.insert(
                "config_ids".to_string(),
                Json::Array(self.config_ids.iter().cloned().map(Json::String).collect()),
            );
        }
        if let Some(request_defaults) = &self.request_defaults {
            if let Some(context) = &request_defaults.context {
                guardrails.insert("context".to_string(), context.clone());
            }
            if let Some(thread_id) = &request_defaults.thread_id {
                guardrails.insert("thread_id".to_string(), Json::String(thread_id.clone()));
            }
            if let Some(state) = &request_defaults.state {
                guardrails.insert("state".to_string(), state.clone());
            }
        }

        let mut options = Map::new();
        let rails = match kind {
            RemoteCheckKind::Input => json!({
                "input": false,
                "output": false,
                "dialog": false,
                "retrieval": false,
                "tool_input": true,
                "tool_output": false,
            }),
            RemoteCheckKind::Output => json!({
                "input": false,
                "output": false,
                "dialog": false,
                "retrieval": false,
                "tool_input": false,
                "tool_output": true,
            }),
        };
        options.insert("rails".to_string(), rails);
        let mut log = self
            .request_defaults
            .as_ref()
            .and_then(|defaults| defaults.log.as_ref())
            .and_then(Json::as_object)
            .cloned()
            .unwrap_or_default();
        log.insert("activated_rails".to_string(), Json::Bool(true));
        options.insert("log".to_string(), Json::Object(log));
        guardrails.insert("options".to_string(), Json::Object(options));
        guardrails
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn emit_remote_mark(name: &str, parent: &Option<ScopeHandle>, data: Json) {
    let _ = event(
        EmitMarkEventParams::builder()
            .name(name)
            .parent_opt(parent.as_ref())
            .data(data)
            .build(),
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn tool_input_content(tool_name: &str, args: &Json) -> String {
    serde_json::to_string(&json!({
        "tool_name": tool_name,
        "arguments": args,
    }))
    .expect("tool input payload should serialize to JSON")
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn redact_remote_error_payload(status: u16, payload: &str) -> String {
    format!(
        "remote request failed with status {status}; error body omitted from marks ({} bytes)",
        payload.len()
    )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn tool_output_content(tool_name: &str, args: &Json, result: &Json) -> String {
    serde_json::to_string(&json!({
        "tool_name": tool_name,
        "arguments": args,
        "result": result,
    }))
    .expect("tool output payload should serialize to JSON")
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn modified_tool_payload(
    content: &str,
    expected_tool_name: &str,
    field: &str,
) -> crate::error::Result<Json> {
    let value: Json = serde_json::from_str(content).map_err(|err| {
        FlowError::Internal(format!(
            "nemo_guardrails returned modified tool {field} content that is not valid JSON: {err}"
        ))
    })?;
    let Json::Object(object) = value else {
        return Err(FlowError::Internal(format!(
            "nemo_guardrails returned modified tool {field} content without a '{field}' field"
        )));
    };
    if let Some(tool_name) = object.get("tool_name").and_then(Json::as_str)
        && tool_name != expected_tool_name
    {
        return Err(FlowError::Internal(format!(
            "nemo_guardrails returned modified tool {field} content for unexpected tool '{tool_name}'"
        )));
    }
    object.get(field).cloned().ok_or_else(|| {
        FlowError::Internal(format!(
            "nemo_guardrails returned modified tool {field} content without a '{field}' field"
        ))
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn chat_completion_content(response: &Json) -> crate::error::Result<String> {
    response
        .get("choices")
        .and_then(Json::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Json::as_str)
        .map(str::to_string)
        .ok_or_else(|| {
            FlowError::Internal(
                "nemo_guardrails remote response did not contain choices[0].message.content"
                    .to_string(),
            )
        })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn blocking_rail_name(response: &Json) -> Option<String> {
    response
        .get("guardrails")
        .and_then(|guardrails| guardrails.get("log"))
        .and_then(|log| log.get("activated_rails"))
        .and_then(Json::as_array)
        .and_then(|activated| {
            activated.iter().find_map(|rail| {
                if rail.get("stop").and_then(Json::as_bool) == Some(true) {
                    rail.get("name").and_then(Json::as_str).map(str::to_string)
                } else {
                    None
                }
            })
        })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn remote_mark_data(
    stream: bool,
    config_id: &Option<String>,
    config_ids: &[String],
    status: Option<u16>,
    error: Option<String>,
) -> Json {
    let mut data = Map::new();
    data.insert("stream".to_string(), Json::Bool(stream));
    if let Some(config_id) = config_id {
        data.insert("config_id".to_string(), Json::String(config_id.clone()));
    }
    if !config_ids.is_empty() {
        data.insert(
            "config_ids".to_string(),
            Json::Array(config_ids.iter().cloned().map(Json::String).collect()),
        );
    }
    if let Some(status) = status {
        data.insert(
            "http_status".to_string(),
            Json::Number(serde_json::Number::from(status)),
        );
    }
    if let Some(error) = error {
        data.insert("error".to_string(), Json::String(error));
    }
    Json::Object(data)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
fn tool_remote_mark_data(
    kind: RemoteCheckKind,
    tool_name: &str,
    config_id: &Option<String>,
    config_ids: &[String],
    status: Option<u16>,
    error: Option<String>,
) -> Json {
    let mut data = match remote_mark_data(false, config_id, config_ids, status, error) {
        Json::Object(data) => data,
        _ => unreachable!("remote_mark_data always returns an object"),
    };
    data.insert(
        "surface".to_string(),
        Json::String(match kind {
            RemoteCheckKind::Input => "tool_input".to_string(),
            RemoteCheckKind::Output => "tool_output".to_string(),
        }),
    );
    data.insert("tool_name".to_string(), Json::String(tool_name.to_string()));
    Json::Object(data)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
pub(super) fn register_remote_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    let remote = config.remote.clone().ok_or_else(|| {
        PluginError::InvalidConfig("remote config is required when mode is 'remote'".to_string())
    })?;
    let runtime = Arc::new(RemoteBackendRuntime::new(&config, &remote)?);

    if config.input || config.output {
        let llm_execution_runtime = Arc::clone(&runtime);
        let llm_execution: LlmExecutionFn = Arc::new(move |_name, request, _next| {
            let runtime = Arc::clone(&llm_execution_runtime);
            Box::pin(async move { runtime.execute(request, false).await })
        });
        ctx.register_llm_execution_intercept("llm_remote_backend", config.priority, llm_execution)?;

        let llm_stream_runtime = Arc::clone(&runtime);
        let llm_stream_execution: LlmStreamExecutionFn = Arc::new(move |_name, request, _next| {
            let runtime = Arc::clone(&llm_stream_runtime);
            Box::pin(async move { runtime.execute_stream(request).await })
        });
        ctx.register_llm_stream_execution_intercept(
            "llm_stream_remote_backend",
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
            "tool_remote_backend",
            config.priority,
            tool_execution,
        )?;
    }

    Ok(())
}

#[cfg(any(target_arch = "wasm32", not(feature = "guardrails-remote")))]
pub(super) fn register_remote_backend(
    _config: NeMoGuardrailsConfig,
    _ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Err(PluginError::RegistrationFailed(
        "built-in NeMo Guardrails remote backend is unavailable in this build".to_string(),
    ))
}
