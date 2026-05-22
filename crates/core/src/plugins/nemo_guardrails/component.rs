// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NeMo Guardrails plugin component contract.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use std::time::Duration;

use serde::{Deserialize, Serialize};
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
use crate::plugin::{
    ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, UnsupportedBehavior, deregister_plugin,
    lookup_plugin, register_plugin,
};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
use rustls::crypto::ring;

/// The plugin kind reserved for the planned first-party component.
pub const NEMO_GUARDRAILS_PLUGIN_KIND: &str = "nemo_guardrails";

/// Top-level NeMo Guardrails component wrapper.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the component should be activated.
    pub enabled: bool,
    /// Component-local NeMo Guardrails config.
    pub config: NeMoGuardrailsConfig,
}

impl ComponentSpec {
    /// Creates an enabled NeMo Guardrails component spec.
    pub fn new(config: NeMoGuardrailsConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(value: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(value.config)
            .expect("NeMo Guardrails config should serialize to an object")
        else {
            unreachable!("NeMo Guardrails config must serialize to an object");
        };

        PluginComponentSpec {
            kind: NEMO_GUARDRAILS_PLUGIN_KIND.to_string(),
            enabled: value.enabled,
            config,
        }
    }
}

/// Canonical config document for the planned NeMo Guardrails component.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct NeMoGuardrailsConfig {
    /// NeMo Guardrails config schema version.
    #[serde(default = "default_nemo_guardrails_config_version")]
    pub version: u32,
    /// Backend mode: `remote` or `local`.
    #[serde(default = "default_mode")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "mode_schema"))]
    pub mode: String,
    /// Path to a native NeMo Guardrails config directory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_path: Option<String>,
    /// Inline native NeMo Guardrails YAML config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_yaml: Option<String>,
    /// Optional inline Colang content. Valid only with `config_yaml`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub colang_content: Option<String>,
    /// Provider request/response codec for LLM-managed surfaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(schema_with = "codec_schema"))]
    pub codec: Option<String>,
    /// Whether to run input rails around managed LLM execution.
    #[serde(default = "default_true")]
    pub input: bool,
    /// Whether to run output rails around managed LLM execution.
    #[serde(default = "default_true")]
    pub output: bool,
    /// Whether to run tool-input rails around managed tool execution.
    #[serde(default)]
    pub tool_input: bool,
    /// Whether to run tool-output rails around managed tool execution.
    #[serde(default)]
    pub tool_output: bool,
    /// Intercept priority. Lower values run earlier.
    #[serde(default = "default_priority")]
    pub priority: i32,
    /// Remote-backend settings used when `mode = "remote"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote: Option<RemoteBackendConfig>,
    /// Local-backend settings used when `mode = "local"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local: Option<LocalBackendConfig>,
    /// Default request semantics passed through to the selected Guardrails backend.
    ///
    /// This models request-time concepts such as rail selection and generation
    /// options without claiming backend parity for every Guardrails feature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_defaults: Option<RequestDefaultsConfig>,
    /// Component-local unsupported-config policy.
    #[serde(default)]
    pub policy: ConfigPolicy,
}

impl Default for NeMoGuardrailsConfig {
    fn default() -> Self {
        Self {
            version: default_nemo_guardrails_config_version(),
            mode: default_mode(),
            config_path: None,
            config_yaml: None,
            colang_content: None,
            codec: None,
            input: true,
            output: true,
            tool_input: false,
            tool_output: false,
            priority: default_priority(),
            remote: None,
            local: None,
            request_defaults: None,
            policy: ConfigPolicy::default(),
        }
    }
}

/// Remote-backend settings for a hosted NeMo Guardrails service.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RemoteBackendConfig {
    /// Base URL for the remote Guardrails service.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<String>,
    /// One remote Guardrails config identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_id: Option<String>,
    /// Multiple remote Guardrails config identifiers to combine.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config_ids: Vec<String>,
    /// Static request headers sent to the remote service.
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_millis")]
    pub timeout_millis: u64,
}

impl Default for RemoteBackendConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            config_id: None,
            config_ids: vec![],
            headers: HashMap::new(),
            timeout_millis: default_timeout_millis(),
        }
    }
}

/// Local-backend settings for the Python `nemoguardrails` runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LocalBackendConfig {
    /// Optional import path for the Python runtime module.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub python_module: Option<String>,
}

/// Default request semantics applied by the selected Guardrails backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RequestDefaultsConfig {
    /// Default context object passed into Guardrails requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Json>,
    /// Default remote thread identifier for continuation-aware requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Default remote Guardrails state payload for continuation-aware requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<Json>,
    /// Default request-time rail selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rails: Option<RequestRailsConfig>,
    /// Default model parameters applied to Guardrails-backed LLM calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_params: Option<Json>,
    /// Whether to include raw LLM output in Guardrails responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llm_output: Option<bool>,
    /// Default output variables selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_vars: Option<Json>,
    /// Default generation-log selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub log: Option<Json>,
}

/// Request-time rail selection for Guardrails generation.
///
/// These are backend request options, not top-level NeMo Relay interception
/// surfaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RequestRailsConfig {
    /// Input rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<RailSelector>,
    /// Output rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<RailSelector>,
    /// Retrieval rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval: Option<RailSelector>,
    /// Dialog rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialog: Option<bool>,
    /// Tool-output rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_output: Option<RailSelector>,
    /// Tool-input rails selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_input: Option<RailSelector>,
}

/// Rail-selection shape used by Guardrails generation options.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub enum RailSelector {
    /// Enable or disable the whole rail family.
    Enabled(bool),
    /// Enable only named rails within a family.
    Named(Vec<String>),
}

crate::editor_config! {
    impl NeMoGuardrailsConfig {
        mode => {
            label: "mode",
            kind: Enum,
            values: ["remote", "local"],
        },
        config_path => { label: "config_path", kind: String, optional: true },
        config_yaml => { label: "config_yaml", kind: String, optional: true },
        colang_content => { label: "colang_content", kind: String, optional: true },
        codec => {
            label: "codec",
            kind: Enum,
            values: ["openai_chat", "openai_responses", "anthropic_messages"],
            optional: true,
        },
        input => { label: "input", kind: Boolean },
        output => { label: "output", kind: Boolean },
        tool_input => { label: "tool_input", kind: Boolean },
        tool_output => { label: "tool_output", kind: Boolean },
        priority => { label: "priority", kind: Integer },
        remote => {
            label: "remote",
            kind: Section,
            optional: true,
            nested: RemoteBackendConfig,
            default: RemoteBackendConfig,
        },
        local => {
            label: "local",
            kind: Section,
            optional: true,
            nested: LocalBackendConfig,
            default: LocalBackendConfig,
        },
        request_defaults => {
            label: "request_defaults",
            kind: Section,
            optional: true,
            nested: RequestDefaultsConfig,
            default: RequestDefaultsConfig,
        },
        policy => {
            label: "policy",
            kind: Section,
            nested: ConfigPolicy,
            default: ConfigPolicy,
        },
    }
}

crate::editor_config! {
    impl RemoteBackendConfig {
        endpoint => { label: "endpoint", kind: String, optional: true },
        config_id => { label: "config_id", kind: String, optional: true },
        config_ids => { label: "config_ids", kind: Json },
        headers => { label: "headers", kind: StringMap },
        timeout_millis => { label: "timeout_millis", kind: Integer },
    }
}

crate::editor_config! {
    impl LocalBackendConfig {
        python_module => { label: "python_module", kind: String, optional: true },
    }
}

crate::editor_config! {
    impl RequestDefaultsConfig {
        context => { label: "context", kind: Json, optional: true },
        thread_id => { label: "thread_id", kind: String, optional: true },
        state => { label: "state", kind: Json, optional: true },
        rails => {
            label: "rails",
            kind: Section,
            optional: true,
            nested: RequestRailsConfig,
            default: RequestRailsConfig,
        },
        llm_params => { label: "llm_params", kind: Json, optional: true },
        llm_output => { label: "llm_output", kind: Boolean, optional: true },
        output_vars => { label: "output_vars", kind: Json, optional: true },
        log => { label: "log", kind: Json, optional: true },
    }
}

crate::editor_config! {
    impl RequestRailsConfig {
        input => { label: "input", kind: Json, optional: true },
        output => { label: "output", kind: Json, optional: true },
        retrieval => { label: "retrieval", kind: Json, optional: true },
        dialog => { label: "dialog", kind: Boolean, optional: true },
        tool_output => { label: "tool_output", kind: Json, optional: true },
        tool_input => { label: "tool_input", kind: Json, optional: true },
    }
}

struct NeMoGuardrailsPlugin;

impl Plugin for NeMoGuardrailsPlugin {
    fn plugin_kind(&self) -> &str {
        NEMO_GUARDRAILS_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_nemo_guardrails_plugin_config(plugin_config)
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let parsed = parse_nemo_guardrails_config(plugin_config);
        Box::pin(async move {
            let config = parsed?;
            register_nemo_guardrails_backend(config, ctx)
        })
    }
}

/// Registers the `nemo_guardrails` component kind in the plugin registry.
pub fn register_nemo_guardrails_component() -> PluginResult<()> {
    match register_plugin(Arc::new(NeMoGuardrailsPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(_))
            if lookup_plugin(NEMO_GUARDRAILS_PLUGIN_KIND).is_some() =>
        {
            Ok(())
        }
        Err(err) => Err(err),
    }
}

/// Deregisters the `nemo_guardrails` component kind from the plugin registry.
pub fn deregister_nemo_guardrails_component() -> bool {
    deregister_plugin(NEMO_GUARDRAILS_PLUGIN_KIND)
}

/// Returns the JSON Schema for the NeMo Guardrails component configuration.
#[cfg(feature = "schema")]
pub fn nemo_guardrails_config_schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(NeMoGuardrailsConfig))
        .expect("NeMo Guardrails config schema should serialize")
}

#[cfg(feature = "schema")]
fn mode_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    string_enum_schema(generator, &["remote", "local"], Some("remote"))
}

#[cfg(feature = "schema")]
fn codec_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
    string_enum_schema(
        generator,
        &["openai_chat", "openai_responses", "anthropic_messages"],
        None,
    )
}

#[cfg(feature = "schema")]
fn string_enum_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
    values: &[&str],
    default: Option<&str>,
) -> schemars::schema::Schema {
    let mut schema: schemars::schema::SchemaObject =
        <String as schemars::JsonSchema>::json_schema(generator).into();
    schema.enum_values = Some(
        values
            .iter()
            .map(|value| Json::String((*value).into()))
            .collect(),
    );
    if let Some(default) = default {
        schema.metadata().default = Some(Json::String(default.into()));
    }
    schema.into()
}

fn register_nemo_guardrails_backend(
    config: NeMoGuardrailsConfig,
    ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    match config.mode.as_str() {
        "remote" => register_remote_backend(config, ctx),
        "local" => Err(PluginError::RegistrationFailed(
            "built-in NeMo Guardrails local backend is not implemented yet".to_string(),
        )),
        other => Err(PluginError::InvalidConfig(format!(
            "unsupported NeMo Guardrails mode '{other}'"
        ))),
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "guardrails-remote"))]
#[derive(Clone)]
// PR 2 intentionally implements the first honest remote slice:
// OpenAI chat requests, non-streaming + streaming execution, managed tool
// input/output checks, broad request-defaults pass-through, and response
// passthrough from the Guardrails server. The local backend remains out of
// scope.
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
                    Some(payload.clone()),
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
        // Preserve the server's OpenAI-compatible chat response and nested
        // guardrails payload verbatim in the first remote slice.
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
                    Some(payload.clone()),
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
        // Remote mode currently accepts only NeMo Flow's OpenAI chat request
        // shape and forwards it to the Guardrails server with a nested
        // `guardrails` envelope.
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
                    Some(payload.clone()),
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
                "input": true,
                "output": false,
                "dialog": false,
                "retrieval": false,
                "tool_input": false,
                "tool_output": false,
            }),
            RemoteCheckKind::Output => json!({
                "input": false,
                "output": true,
                "dialog": false,
                "retrieval": false,
                "tool_input": false,
                "tool_output": false,
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
fn register_remote_backend(
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
fn register_remote_backend(
    _config: NeMoGuardrailsConfig,
    _ctx: &mut PluginRegistrationContext,
) -> PluginResult<()> {
    Err(PluginError::RegistrationFailed(
        "built-in NeMo Guardrails remote backend is unavailable in this build".to_string(),
    ))
}

fn parse_nemo_guardrails_config(
    plugin_config: &Map<String, Json>,
) -> PluginResult<NeMoGuardrailsConfig> {
    serde_json::from_value(Json::Object(plugin_config.clone())).map_err(|err| {
        PluginError::InvalidConfig(format!("invalid NeMo Guardrails plugin config: {err}"))
    })
}

fn validate_nemo_guardrails_plugin_config(
    plugin_config: &Map<String, Json>,
) -> Vec<ConfigDiagnostic> {
    let config = match parse_nemo_guardrails_config(plugin_config) {
        Ok(config) => config,
        Err(err) => {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "nemo_guardrails.invalid_plugin_config".to_string(),
                component: Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                field: None,
                message: err.to_string(),
            }];
        }
    };

    let mut diagnostics = vec![];

    validate_unknown_fields(
        &mut diagnostics,
        &config.policy,
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        plugin_config,
        &[
            "version",
            "mode",
            "config_path",
            "config_yaml",
            "colang_content",
            "codec",
            "input",
            "output",
            "tool_input",
            "tool_output",
            "priority",
            "remote",
            "local",
            "request_defaults",
            "policy",
        ],
    );

    validate_policy_fields(&mut diagnostics, &config.policy, plugin_config);
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "remote",
        &[
            "endpoint",
            "config_id",
            "config_ids",
            "headers",
            "timeout_millis",
        ],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "local",
        &["python_module"],
    );
    validate_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "request_defaults",
        &[
            "context",
            "thread_id",
            "state",
            "rails",
            "llm_params",
            "llm_output",
            "output_vars",
            "log",
        ],
    );
    validate_nested_section_fields(
        &mut diagnostics,
        &config.policy,
        plugin_config,
        "request_defaults",
        "rails",
        &[
            "input",
            "output",
            "retrieval",
            "dialog",
            "tool_output",
            "tool_input",
        ],
    );

    validate_version(&mut diagnostics, &config.policy, config.version);
    validate_mode(&mut diagnostics, &config.policy, &config.mode);
    validate_non_empty_strings(&mut diagnostics, &config.policy, &config);
    validate_config_shape(&mut diagnostics, &config.policy, &config);
    validate_codec_requirements(&mut diagnostics, &config.policy, &config);
    validate_surface_selection(&mut diagnostics, &config.policy, &config);
    validate_remote_backend_support(&mut diagnostics, &config.policy, &config);
    validate_request_defaults(&mut diagnostics, &config.policy, &config);

    diagnostics
}

fn validate_version(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, version: u32) {
    if version != 1 {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_config_version",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("version".to_string()),
            format!("NeMo Guardrails config version {version} is unsupported"),
        );
    }
}

fn validate_mode(diagnostics: &mut Vec<ConfigDiagnostic>, policy: &ConfigPolicy, mode: &str) {
    if !matches!(mode, "remote" | "local") {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("mode".to_string()),
            "mode must be 'remote' or 'local'".to_string(),
        );
    }
}

fn validate_non_empty_strings(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if let Some(config_path) = &config.config_path
        && config_path.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("config_path".to_string()),
            "config_path must not be empty".to_string(),
        );
    }

    if let Some(config_yaml) = &config.config_yaml
        && config_yaml.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("config_yaml".to_string()),
            "config_yaml must not be empty".to_string(),
        );
    }

    if let Some(colang_content) = &config.colang_content
        && colang_content.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("colang_content".to_string()),
            "colang_content must not be empty".to_string(),
        );
    }

    if let Some(remote) = &config.remote {
        if let Some(endpoint) = &remote.endpoint
            && endpoint.trim().is_empty()
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote.endpoint".to_string()),
                "remote.endpoint must not be empty".to_string(),
            );
        }
        if let Some(config_id) = &remote.config_id
            && config_id.trim().is_empty()
        {
            push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("remote.config_id".to_string()),
                "remote.config_id must not be empty".to_string(),
            );
        }
        for (index, config_id) in remote.config_ids.iter().enumerate() {
            if config_id.trim().is_empty() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemo_guardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some(format!("remote.config_ids[{index}]")),
                    "remote.config_ids entries must not be empty".to_string(),
                );
            }
        }
    }

    if let Some(local) = &config.local
        && let Some(python_module) = &local.python_module
        && python_module.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("local.python_module".to_string()),
            "local.python_module must not be empty".to_string(),
        );
    }
}

fn validate_config_shape(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let flags = ConfigShapeFlags::from(config);

    match config.mode.as_str() {
        "local" => validate_local_config_shape(diagnostics, policy, config, &flags),
        "remote" => validate_remote_config_shape(diagnostics, policy, config, &flags),
        _ => {}
    }
}

struct ConfigShapeFlags {
    has_config_path: bool,
    has_config_yaml: bool,
    has_colang_content: bool,
    has_remote_config_id: bool,
    has_remote_config_ids: bool,
}

impl From<&NeMoGuardrailsConfig> for ConfigShapeFlags {
    fn from(config: &NeMoGuardrailsConfig) -> Self {
        Self {
            has_config_path: config.config_path.is_some(),
            has_config_yaml: config.config_yaml.is_some(),
            has_colang_content: config.colang_content.is_some(),
            has_remote_config_id: config
                .remote
                .as_ref()
                .and_then(|remote| remote.config_id.as_ref())
                .is_some(),
            has_remote_config_ids: config
                .remote
                .as_ref()
                .map(|remote| !remote.config_ids.is_empty())
                .unwrap_or(false),
        }
    }
}

fn validate_local_config_shape(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
    flags: &ConfigShapeFlags,
) {
    if flags.has_config_path == flags.has_config_yaml {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.invalid_config_source",
            None,
            "exactly one of config_path or config_yaml is required in local mode",
        );
    }

    if flags.has_colang_content && !flags.has_config_yaml {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("colang_content"),
            "colang_content can only be used with config_yaml",
        );
    }

    if config.remote.is_some() {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("remote"),
            "remote backend settings cannot be used when mode is 'local'",
        );
    }
}

fn validate_remote_config_shape(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
    flags: &ConfigShapeFlags,
) {
    if flags.has_config_path || flags.has_config_yaml || flags.has_colang_content {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.invalid_config_source",
            None,
            "remote mode uses remote config identity and cannot include config_path, config_yaml, or colang_content",
        );
    }

    if config.local.is_some() {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("local"),
            "local backend settings cannot be used when mode is 'remote'",
        );
    }

    match &config.remote {
        Some(remote)
            if remote
                .endpoint
                .as_ref()
                .is_some_and(|value| !value.trim().is_empty()) => {}
        _ => push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("remote.endpoint"),
            "remote.endpoint is required when mode is 'remote'",
        ),
    }

    if flags.has_remote_config_id && flags.has_remote_config_ids {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some("remote"),
            "remote.config_id and remote.config_ids cannot be used together",
        );
    }

    if !(flags.has_remote_config_id || flags.has_remote_config_ids) {
        push_config_shape_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.invalid_config_source",
            None,
            "remote mode requires remote.config_id or remote.config_ids",
        );
    }
}

fn push_config_shape_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    field: Option<&str>,
    message: &str,
) {
    push_policy_diag(
        diagnostics,
        behavior,
        code,
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        field.map(str::to_string),
        message.to_string(),
    );
}

fn validate_codec_requirements(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let llm_surface_enabled = config.input || config.output;
    if !llm_surface_enabled {
        return;
    }

    let Some(codec) = config.codec.as_deref() else {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec is required when any LLM surface is enabled".to_string(),
        );
        return;
    };

    if !matches!(
        codec,
        "openai_chat" | "openai_responses" | "anthropic_messages"
    ) {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "codec must be 'openai_chat', 'openai_responses', or 'anthropic_messages'".to_string(),
        );
    }
}

fn validate_surface_selection(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if config.input || config.output || config.tool_input || config.tool_output {
        return;
    }

    push_policy_diag(
        diagnostics,
        policy.unsupported_value,
        "nemo_guardrails.unsupported_value",
        Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
        None,
        "at least one Guardrails surface must be enabled".to_string(),
    );
}

fn validate_remote_backend_support(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    if config.mode != "remote" {
        return;
    }

    if (config.input || config.output)
        && config
            .codec
            .as_deref()
            .is_some_and(|codec| codec != "openai_chat")
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("codec".to_string()),
            "remote mode currently supports only codec = 'openai_chat'".to_string(),
        );
    }
}

fn validate_request_defaults(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    config: &NeMoGuardrailsConfig,
) {
    let Some(request_defaults) = &config.request_defaults else {
        return;
    };

    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.context.as_ref(),
        "request_defaults.context",
        "request_defaults.context must be a JSON object",
    );
    if let Some(thread_id) = &request_defaults.thread_id
        && thread_id.trim().is_empty()
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("request_defaults.thread_id".to_string()),
            "request_defaults.thread_id must not be empty".to_string(),
        );
    }
    if let Some(thread_id) = &request_defaults.thread_id
        && !thread_id.trim().is_empty()
        && thread_id.len() < 16
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("request_defaults.thread_id".to_string()),
            "request_defaults.thread_id must be at least 16 characters long".to_string(),
        );
    }
    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.state.as_ref(),
        "request_defaults.state",
        "request_defaults.state must be a JSON object",
    );
    if let Some(state) = request_defaults
        .state
        .as_ref()
        .and_then(|value| value.as_object())
        && !state.is_empty()
        && !state.contains_key("events")
        && !state.contains_key("state")
    {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some("request_defaults.state".to_string()),
            "request_defaults.state must be empty or contain 'events' or 'state'".to_string(),
        );
    }
    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.llm_params.as_ref(),
        "request_defaults.llm_params",
        "request_defaults.llm_params must be a JSON object",
    );
    validate_json_object_field(
        diagnostics,
        policy,
        request_defaults.log.as_ref(),
        "request_defaults.log",
        "request_defaults.log must be a JSON object",
    );

    if let Some(output_vars) = &request_defaults.output_vars {
        match output_vars {
            Json::Bool(_) => {}
            Json::Array(values) => {
                for (index, value) in values.iter().enumerate() {
                    if !value.is_string()
                        || value.as_str().is_some_and(|entry| entry.trim().is_empty())
                    {
                        push_policy_diag(
                            diagnostics,
                            policy.unsupported_value,
                            "nemo_guardrails.unsupported_value",
                            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                            Some(format!("request_defaults.output_vars[{index}]")),
                            "request_defaults.output_vars array entries must be non-empty strings"
                                .to_string(),
                        );
                    }
                }
            }
            _ => push_policy_diag(
                diagnostics,
                policy.unsupported_value,
                "nemo_guardrails.unsupported_value",
                Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                Some("request_defaults.output_vars".to_string()),
                "request_defaults.output_vars must be a boolean or an array of strings".to_string(),
            ),
        }
    }

    if let Some(rails) = &request_defaults.rails {
        validate_rail_selector(
            diagnostics,
            policy,
            rails.input.as_ref(),
            "request_defaults.rails.input",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.output.as_ref(),
            "request_defaults.rails.output",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.retrieval.as_ref(),
            "request_defaults.rails.retrieval",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.tool_output.as_ref(),
            "request_defaults.rails.tool_output",
        );
        validate_rail_selector(
            diagnostics,
            policy,
            rails.tool_input.as_ref(),
            "request_defaults.rails.tool_input",
        );
    }
}

fn validate_json_object_field(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    value: Option<&Json>,
    field: &str,
    message: &str,
) {
    let Some(value) = value else {
        return;
    };

    if !value.is_object() {
        push_policy_diag(
            diagnostics,
            policy.unsupported_value,
            "nemo_guardrails.unsupported_value",
            Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
            Some(field.to_string()),
            message.to_string(),
        );
    }
}

fn validate_rail_selector(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    value: Option<&RailSelector>,
    field: &str,
) {
    let Some(value) = value else {
        return;
    };

    if let RailSelector::Named(names) = value {
        for (index, name) in names.iter().enumerate() {
            if name.trim().is_empty() {
                push_policy_diag(
                    diagnostics,
                    policy.unsupported_value,
                    "nemo_guardrails.unsupported_value",
                    Some(NEMO_GUARDRAILS_PLUGIN_KIND.to_string()),
                    Some(format!("{field}[{index}]")),
                    "named rail selections must not contain empty strings".to_string(),
                );
            }
        }
    }
}

fn validate_policy_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
) {
    if let Some(policy_json) = plugin_config.get("policy").and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some("policy".to_string()),
            policy_json,
            &["unknown_component", "unknown_field", "unsupported_value"],
        );
    }
}

fn validate_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section: &str,
    known_fields: &[&str],
) {
    if let Some(section_json) = plugin_config.get(section).and_then(Json::as_object) {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(section.to_string()),
            section_json,
            known_fields,
        );
    }
}

fn validate_nested_section_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    plugin_config: &Map<String, Json>,
    section: &str,
    nested_section: &str,
    known_fields: &[&str],
) {
    if let Some(section_json) = plugin_config.get(section).and_then(Json::as_object)
        && let Some(nested_json) = section_json.get(nested_section).and_then(Json::as_object)
    {
        validate_unknown_fields(
            diagnostics,
            policy,
            Some(format!("{section}.{nested_section}")),
            nested_json,
            known_fields,
        );
    }
}

fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    component: Option<String>,
    config: &Map<String, Json>,
    known_fields: &[&str],
) {
    for field in config.keys() {
        if !known_fields.contains(&field.as_str()) {
            push_policy_diag(
                diagnostics,
                policy.unknown_field,
                "nemo_guardrails.unknown_field",
                component.clone(),
                Some(field.clone()),
                format!(
                    "field '{}' is not recognized for '{}'",
                    field,
                    component.as_deref().unwrap_or("unknown")
                ),
            );
        }
    }
}

fn push_policy_diag(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    component: Option<String>,
    field: Option<String>,
    message: String,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };

    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component,
        field,
        message,
    });
}

fn default_nemo_guardrails_config_version() -> u32 {
    1
}

fn default_mode() -> String {
    "remote".to_string()
}

fn default_true() -> bool {
    true
}

fn default_priority() -> i32 {
    100
}

fn default_timeout_millis() -> u64 {
    3_000
}

#[cfg(test)]
#[path = "../../../tests/unit/plugins/nemo_guardrails/component_tests.rs"]
mod tests;
