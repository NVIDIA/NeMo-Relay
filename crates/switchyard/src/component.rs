// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchyard plugin configuration and Relay execution integration.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_stream::stream;
use futures_util::{StreamExt, stream as futures_stream};
use nemo_relay::api::event::{CategoryProfile, DataSchema, EventCategory};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn};
use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginComponentSpec, PluginConfig, PluginError,
    PluginRegistrationContext, Result as PluginResult, deregister_plugin, register_plugin,
};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json, json};
use uuid::Uuid;

use crate::contract::{
    DecisionAttempt, DecisionProfile, ROUTING_DECISION_SCHEMA_VERSION,
    ROUTING_REQUEST_SCHEMA_VERSION, RequestIdentity, RequestMaterialization, RequestProtocol,
    RequestSummary, RoutingDecision, RoutingRequest,
};
use crate::stream_translation::StreamTranscoder;
use crate::translation::{
    decode_request, encode_request, latest_user_prompt, recent_message_window, translate_response,
    validate_portable_request,
};

/// Plugin kind used in Relay plugin configuration.
pub const SWITCHYARD_PLUGIN_KIND: &str = "switchyard";

const INTERNAL_DISPATCH_URL_HEADER: &str = "x-nemo-relay-internal-dispatch-url";
const INTERNAL_DISPATCH_ROUTE_HEADER: &str = "x-nemo-relay-internal-dispatch-route";
const INTERNAL_RETRY_AWARE_HEADER: &str = "x-nemo-relay-internal-retry-aware";
const ROUTING_MARK_SCHEMA: &str = "switchyard.routing_mark";

/// Supported provider wire protocols.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WireProtocol {
    /// OpenAI Chat Completions.
    OpenaiChat,
    /// OpenAI Responses.
    OpenaiResponses,
    /// Anthropic Messages.
    AnthropicMessages,
}

impl WireProtocol {
    fn label(self) -> &'static str {
        match self {
            Self::OpenaiChat => "openai_chat",
            Self::OpenaiResponses => "openai_responses",
            Self::AnthropicMessages => "anthropic_messages",
        }
    }

    fn endpoint(self) -> &'static str {
        match self {
            Self::OpenaiChat => "/v1/chat/completions",
            Self::OpenaiResponses => "/v1/responses",
            Self::AnthropicMessages => "/v1/messages",
        }
    }

    fn from_call(name: &str, request: &LlmRequest) -> Option<Self> {
        match name {
            "openai.chat_completions" | "openai_chat" | "openai_chat_completions" => {
                Some(Self::OpenaiChat)
            }
            "openai.responses" | "openai_responses" => Some(Self::OpenaiResponses),
            "anthropic.messages" | "anthropic" | "anthropic_messages" => {
                Some(Self::AnthropicMessages)
            }
            _ if request.content.get("input").is_some() => Some(Self::OpenaiResponses),
            _ if request.content.get("system").is_some() => Some(Self::AnthropicMessages),
            _ if request.content.get("messages").is_some() => Some(Self::OpenaiChat),
            _ => None,
        }
    }
}

/// Routing rollout mode.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Apply Switchyard decisions.
    #[default]
    Enforce,
    /// Record decisions but dispatch trusted defaults.
    ObserveOnly,
}

impl RoutingMode {
    fn label(self) -> &'static str {
        match self {
            Self::Enforce => "enforce",
            Self::ObserveOnly => "observe_only",
        }
    }
}

/// Whether the selected Switchyard profile depends on ATOF-derived history.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ContextMode {
    /// The router uses only current request material.
    PayloadOnly,
    /// Stable identity and a configured ATOF endpoint are required.
    AtofRequired,
}

/// Exact Relay-owned backend binding for one Switchyard backend ID.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TargetBinding {
    /// Exact model expected in the Switchyard decision.
    pub model: String,
    /// Exact protocol expected in the Switchyard decision.
    pub protocol: WireProtocol,
    /// Exact endpoint expected in the Switchyard decision.
    pub endpoint: String,
    /// Relay-owned backend base URL.
    pub base_url: String,
    /// Static non-sensitive backend headers.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Backend headers resolved from environment variables.
    #[serde(default)]
    pub header_env: BTreeMap<String, String>,
}

/// Trusted fallback target IDs for each inbound protocol.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ProtocolDefaults {
    /// OpenAI Chat fallback target.
    pub openai_chat: String,
    /// OpenAI Responses fallback target.
    pub openai_responses: String,
    /// Anthropic Messages fallback target.
    pub anthropic_messages: String,
}

impl ProtocolDefaults {
    fn target(&self, protocol: WireProtocol) -> &str {
        match protocol {
            WireProtocol::OpenaiChat => &self.openai_chat,
            WireProtocol::OpenaiResponses => &self.openai_responses,
            WireProtocol::AnthropicMessages => &self.anthropic_messages,
        }
    }
}

/// Versioned Switchyard plugin configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SwitchyardConfig {
    /// Config schema version.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Enforce or observe-only rollout mode.
    #[serde(default)]
    pub mode: RoutingMode,
    /// Execution-intercept priority.
    #[serde(default)]
    pub priority: i32,
    /// Switchyard Decision API URL.
    pub decision_api_url: String,
    /// Switchyard profile ID.
    pub decision_profile_id: String,
    /// Current-request materialization.
    pub request_materialization: RequestMaterialization,
    /// Profile context requirement.
    pub context_mode: ContextMode,
    /// Decision call timeout.
    #[serde(default = "default_decision_timeout_millis")]
    pub decision_timeout_millis: u64,
    /// Provider retries after the initial attempt.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Number of messages in recent-message materialization.
    #[serde(default = "default_recent_message_count")]
    pub recent_message_count: usize,
    /// Static non-sensitive Decision API headers.
    #[serde(default)]
    pub decision_headers: BTreeMap<String, String>,
    /// Decision API headers resolved from environment variables.
    #[serde(default)]
    pub decision_header_env: BTreeMap<String, String>,
    /// Enabled inbound protocols.
    #[serde(default = "default_enabled_protocols")]
    pub enabled_inbound_profiles: BTreeSet<WireProtocol>,
    /// Exact backend bindings keyed by Switchyard backend ID.
    pub targets: BTreeMap<String, TargetBinding>,
    /// Trusted per-protocol fallbacks.
    pub default_targets: ProtocolDefaults,
    /// Optional explicit ATOF endpoint used by CLI cross-validation.
    #[serde(default)]
    pub atof_endpoint_url: Option<String>,
}

impl Default for SwitchyardConfig {
    fn default() -> Self {
        Self {
            version: default_version(),
            mode: RoutingMode::default(),
            priority: 0,
            decision_api_url: "http://127.0.0.1:8080/v1/routing/decision".into(),
            decision_profile_id: String::new(),
            request_materialization: RequestMaterialization::SummaryOnly,
            context_mode: ContextMode::PayloadOnly,
            decision_timeout_millis: default_decision_timeout_millis(),
            max_retries: default_max_retries(),
            recent_message_count: default_recent_message_count(),
            decision_headers: BTreeMap::new(),
            decision_header_env: BTreeMap::new(),
            enabled_inbound_profiles: default_enabled_protocols(),
            targets: BTreeMap::new(),
            default_targets: ProtocolDefaults {
                openai_chat: String::new(),
                openai_responses: String::new(),
                anthropic_messages: String::new(),
            },
            atof_endpoint_url: None,
        }
    }
}

nemo_relay::editor_config! {
    impl SwitchyardConfig {
        mode => { label: "Rollout mode", kind: Enum, values: ["enforce", "observe_only"] },
        priority => { label: "Intercept priority", kind: Integer },
        decision_api_url => { label: "Decision API URL", kind: String },
        decision_profile_id => { label: "Decision profile ID", kind: String },
        request_materialization => {
            label: "Request materialization",
            kind: Enum,
            values: ["none", "summary_only", "latest_user_prompt", "recent_message_window", "annotated_request", "full_body"]
        },
        context_mode => { label: "Context mode", kind: Enum, values: ["payload_only", "atof_required"] },
        decision_timeout_millis => { label: "Decision timeout (ms)", kind: Integer },
        max_retries => { label: "Maximum provider retries", kind: Integer },
        recent_message_count => { label: "Recent message count", kind: Integer },
        decision_headers => { label: "Decision API static headers", kind: StringMap },
        decision_header_env => { label: "Decision API environment headers", kind: StringMap },
        enabled_inbound_profiles => { label: "Enabled inbound profiles", kind: Json },
        targets => { label: "Backend target bindings", kind: Json },
        default_targets => { label: "Trusted protocol defaults", kind: Json },
        atof_endpoint_url => { label: "ATOF endpoint URL", kind: String, optional: true }
    }
}

impl From<SwitchyardConfig> for PluginComponentSpec {
    fn from(value: SwitchyardConfig) -> Self {
        let Json::Object(config) =
            serde_json::to_value(value).expect("Switchyard config should serialize to an object")
        else {
            unreachable!("Switchyard config must serialize to an object")
        };
        Self {
            kind: SWITCHYARD_PLUGIN_KIND.into(),
            enabled: true,
            config,
        }
    }
}

fn default_version() -> u32 {
    1
}
fn default_decision_timeout_millis() -> u64 {
    25
}
fn default_max_retries() -> u32 {
    3
}
fn default_recent_message_count() -> usize {
    8
}
fn default_enabled_protocols() -> BTreeSet<WireProtocol> {
    BTreeSet::from([
        WireProtocol::OpenaiChat,
        WireProtocol::OpenaiResponses,
        WireProtocol::AnthropicMessages,
    ])
}

struct SwitchyardPlugin;

impl Plugin for SwitchyardPlugin {
    fn plugin_kind(&self) -> &str {
        SWITCHYARD_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        match parse_config(plugin_config).and_then(SwitchyardRuntime::new) {
            Ok(_) => Vec::new(),
            Err(error) => vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "switchyard.invalid_config".into(),
                component: Some(SWITCHYARD_PLUGIN_KIND.into()),
                field: None,
                message: error,
            }],
        }
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let parsed = parse_config(plugin_config);
        Box::pin(async move {
            let runtime = Arc::new(
                parsed
                    .and_then(SwitchyardRuntime::new)
                    .map_err(PluginError::InvalidConfig)?,
            );
            let buffered = Arc::clone(&runtime);
            let buffered_intercept: LlmExecutionFn = Arc::new(move |name, request, next| {
                let runtime = Arc::clone(&buffered);
                let name = name.to_string();
                Box::pin(async move { runtime.execute_buffered(&name, request, next).await })
            });
            ctx.register_llm_execution_intercept(
                "decision",
                runtime.config.priority,
                buffered_intercept,
            )?;

            let streaming = Arc::clone(&runtime);
            let stream_intercept: LlmStreamExecutionFn = Arc::new(move |name, request, next| {
                let runtime = Arc::clone(&streaming);
                let name = name.to_string();
                Box::pin(async move { runtime.execute_stream(&name, request, next).await })
            });
            ctx.register_llm_stream_execution_intercept(
                "decision_stream",
                runtime.config.priority,
                stream_intercept,
            )?;
            Ok(())
        })
    }
}

/// Register the first-party Switchyard component kind.
pub fn register_switchyard_component() -> PluginResult<()> {
    match register_plugin(Arc::new(SwitchyardPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message)) if message.contains("already registered") => {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

/// Deregister the first-party Switchyard component kind.
pub fn deregister_switchyard_component() -> bool {
    deregister_plugin(SWITCHYARD_PLUGIN_KIND)
}

/// Validate the cross-component ATOF requirement for enabled history-backed profiles.
pub fn validate_switchyard_atof_configuration(config: &PluginConfig) -> Result<(), String> {
    let Some(component) = config
        .components
        .iter()
        .find(|component| component.enabled && component.kind == SWITCHYARD_PLUGIN_KIND)
    else {
        return Ok(());
    };
    let switchyard = parse_config(&component.config)?;
    if switchyard.context_mode != ContextMode::AtofRequired {
        return Ok(());
    }
    let required_url = match &switchyard.atof_endpoint_url {
        Some(url) => url.clone(),
        None => derived_atof_url(&switchyard.decision_api_url)?,
    };
    let observability = config
        .components
        .iter()
        .find(|component| component.enabled && component.kind == "observability")
        .ok_or_else(|| "atof_required Switchyard profiles require observability".to_string())?;
    let endpoints = observability
        .config
        .get("atof")
        .filter(|atof| atof.get("enabled").and_then(Json::as_bool) == Some(true))
        .and_then(|atof| atof.get("endpoints"))
        .and_then(Json::as_array)
        .ok_or_else(|| {
            "atof_required Switchyard profiles require an enabled ATOF endpoint".to_string()
        })?;
    let endpoint = endpoints
        .iter()
        .find(|endpoint| {
            endpoint.get("url").and_then(Json::as_str) == Some(required_url.as_str())
                && endpoint
                    .get("transport")
                    .and_then(Json::as_str)
                    .unwrap_or("http_post")
                    == "http_post"
        })
        .ok_or_else(|| {
            format!("atof_required Switchyard profile requires HTTP ATOF endpoint {required_url}")
        })?;
    if endpoint
        .get("field_name_policy")
        .and_then(Json::as_str)
        .unwrap_or("preserve")
        != "preserve"
    {
        return Err("Switchyard ATOF endpoint must use field_name_policy = preserve".into());
    }
    if endpoint
        .get("header_env")
        .and_then(Json::as_object)
        .is_none_or(Map::is_empty)
    {
        return Err(
            "Switchyard ATOF endpoint authentication must use at least one environment-referenced header"
                .into(),
        );
    }
    Ok(())
}

fn derived_atof_url(decision_api_url: &str) -> Result<String, String> {
    let mut url = reqwest::Url::parse(decision_api_url)
        .map_err(|error| format!("decision_api_url is invalid: {error}"))?;
    url.set_path("/v1/atof/events");
    url.set_query(None);
    url.set_fragment(None);
    Ok(url.to_string().trim_end_matches('/').to_string())
}

fn parse_config(config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    serde_json::from_value(Json::Object(config.clone()))
        .map_err(|error| format!("invalid Switchyard plugin config: {error}"))
}

struct SwitchyardRuntime {
    config: SwitchyardConfig,
    client: reqwest::Client,
    target_headers: BTreeMap<String, Map<String, Json>>,
}

impl SwitchyardRuntime {
    fn new(config: SwitchyardConfig) -> Result<Self, String> {
        validate_config(&config)?;
        let headers = resolve_headers(&config.decision_headers, &config.decision_header_env)?;
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_millis(config.decision_timeout_millis))
            .build()
            .map_err(|error| format!("failed to build Decision API client: {error}"))?;
        let target_headers = config
            .targets
            .iter()
            .map(|(id, target)| {
                let headers = resolve_json_headers(&target.headers, &target.header_env)?;
                Ok((id.clone(), headers))
            })
            .collect::<Result<_, String>>()?;
        Ok(Self {
            config,
            client,
            target_headers,
        })
    }

    async fn execute_buffered(
        &self,
        name: &str,
        original: LlmRequest,
        next: nemo_relay::api::runtime::LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let Some(inbound) = WireProtocol::from_call(name, &original) else {
            return next(original).await;
        };
        if !self.config.enabled_inbound_profiles.contains(&inbound) {
            return next(original).await;
        }
        if let Err(error) = validate_portable_request(inbound, &original) {
            self.emit_error(
                None,
                0,
                "unsupported_provider_extension",
                &error.to_string(),
            );
            return self
                .dispatch_fallback_buffered(
                    inbound,
                    original,
                    next,
                    "unsupported_provider_extension",
                )
                .await;
        }

        if self.config.mode == RoutingMode::ObserveOnly {
            if let Err(error) = self.decided_request(inbound, &original, 1, None).await {
                self.emit_error(None, 1, "decision_api", &error);
            }
            return self
                .dispatch_fallback_buffered(inbound, original, next, "observe_only")
                .await;
        }

        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut previous = None;
        for attempt in 1..=max_attempts {
            let decided = self
                .decided_request(inbound, &original, attempt, previous.clone())
                .await;
            let (routing_request, decision, routed) = match decided {
                Ok(value) => value,
                Err(error) => {
                    self.emit_error(None, attempt, "decision_api", &error);
                    return self
                        .dispatch_fallback_buffered(inbound, original, next, "decision_error")
                        .await;
                }
            };
            let target_protocol = protocol_from_label(&decision.route.target_protocol_profile)?;
            match next(routed).await {
                Ok(response) => match translate_response(target_protocol, inbound, &response) {
                    Ok(response) => return Ok(response),
                    Err(error) => {
                        self.emit_error(
                            Some(&routing_request),
                            attempt,
                            "response_translation",
                            &error.to_string(),
                        );
                        return self
                            .dispatch_fallback_buffered(
                                inbound,
                                original,
                                next,
                                "translation_error",
                            )
                            .await;
                    }
                },
                Err(error) if error_is_retryable(&error) && attempt < max_attempts => {
                    let retry_reason = provider_error_summary(&error);
                    self.emit_error(Some(&routing_request), attempt, "provider", &retry_reason);
                    self.emit_retry(&routing_request, &decision, attempt, &retry_reason);
                    previous = Some((decision.route.backend_id, retry_reason));
                }
                Err(error) => {
                    let summary = provider_error_summary(&error);
                    self.emit_error(Some(&routing_request), attempt, "provider", &summary);
                    return self
                        .dispatch_fallback_buffered(
                            inbound,
                            original,
                            next,
                            if error_is_retryable(&error) {
                                "retry_exhausted"
                            } else {
                                "non_retryable_provider_error"
                            },
                        )
                        .await;
                }
            }
        }
        unreachable!("routing attempt loop always returns")
    }

    async fn execute_stream(
        &self,
        name: &str,
        original: LlmRequest,
        next: nemo_relay::api::runtime::LlmStreamExecutionNextFn,
    ) -> FlowResult<LlmJsonStream> {
        let Some(inbound) = WireProtocol::from_call(name, &original) else {
            return next(original).await;
        };
        if !self.config.enabled_inbound_profiles.contains(&inbound) {
            return next(original).await;
        }
        if let Err(error) = validate_portable_request(inbound, &original) {
            self.emit_error(
                None,
                0,
                "unsupported_provider_extension",
                &error.to_string(),
            );
            return self
                .dispatch_fallback_stream(inbound, original, next, "unsupported_provider_extension")
                .await;
        }
        if self.config.mode == RoutingMode::ObserveOnly {
            if let Err(error) = self.decided_request(inbound, &original, 1, None).await {
                self.emit_error(None, 1, "decision_api", &error);
            }
            return self
                .dispatch_fallback_stream(inbound, original, next, "observe_only")
                .await;
        }

        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut previous = None;
        for attempt in 1..=max_attempts {
            let (routing_request, decision, routed) = match self
                .decided_request(inbound, &original, attempt, previous.clone())
                .await
            {
                Ok(value) => value,
                Err(error) => {
                    self.emit_error(None, attempt, "decision_api", &error);
                    return self
                        .dispatch_fallback_stream(inbound, original, next, "decision_error")
                        .await;
                }
            };
            let target_protocol = protocol_from_label(&decision.route.target_protocol_profile)?;
            match next(routed).await {
                Ok(mut upstream) => match upstream.next().await {
                    Some(Ok(first)) => {
                        let committed = Box::pin(
                            futures_stream::once(async move { Ok(first) }).chain(upstream),
                        ) as LlmJsonStream;
                        let output = if target_protocol == inbound {
                            committed
                        } else {
                            translated_stream(
                                target_protocol,
                                inbound,
                                decision.route.target_model.clone(),
                                committed,
                            )
                        };
                        return Ok(mark_terminal_stream(
                            output,
                            "provider_stream_committed",
                            self.config.mode.label(),
                            identity_metadata(&routing_request),
                        ));
                    }
                    Some(Err(error)) if error_is_retryable(&error) && attempt < max_attempts => {
                        let retry_reason = provider_error_summary(&error);
                        self.emit_error(
                            Some(&routing_request),
                            attempt,
                            "provider_stream_open",
                            &retry_reason,
                        );
                        self.emit_retry(&routing_request, &decision, attempt, &retry_reason);
                        previous = Some((decision.route.backend_id, retry_reason));
                    }
                    None if attempt < max_attempts => {
                        self.emit_retry(&routing_request, &decision, attempt, "empty_stream");
                        previous = Some((decision.route.backend_id, "empty_stream".into()));
                    }
                    Some(Err(error)) => {
                        let summary = provider_error_summary(&error);
                        self.emit_error(
                            Some(&routing_request),
                            attempt,
                            "provider_stream_open",
                            &summary,
                        );
                        return self
                            .dispatch_fallback_stream(
                                inbound,
                                original,
                                next,
                                if error_is_retryable(&error) {
                                    "retry_exhausted"
                                } else {
                                    "non_retryable_provider_error"
                                },
                            )
                            .await;
                    }
                    None => {
                        return self
                            .dispatch_fallback_stream(inbound, original, next, "empty_stream")
                            .await;
                    }
                },
                Err(error) if error_is_retryable(&error) && attempt < max_attempts => {
                    let retry_reason = provider_error_summary(&error);
                    self.emit_error(
                        Some(&routing_request),
                        attempt,
                        "provider_stream_setup",
                        &retry_reason,
                    );
                    self.emit_retry(&routing_request, &decision, attempt, &retry_reason);
                    previous = Some((decision.route.backend_id, retry_reason));
                }
                Err(error) => {
                    let summary = provider_error_summary(&error);
                    self.emit_error(
                        Some(&routing_request),
                        attempt,
                        "provider_stream_setup",
                        &summary,
                    );
                    return self
                        .dispatch_fallback_stream(
                            inbound,
                            original,
                            next,
                            if error_is_retryable(&error) {
                                "retry_exhausted"
                            } else {
                                "non_retryable_provider_error"
                            },
                        )
                        .await;
                }
            }
        }
        unreachable!("stream routing attempt loop always returns")
    }

    async fn decided_request(
        &self,
        inbound: WireProtocol,
        original: &LlmRequest,
        attempt: u32,
        previous: Option<(String, String)>,
    ) -> Result<(RoutingRequest, RoutingDecision, LlmRequest), String> {
        let request = self.routing_request(inbound, original, attempt, previous)?;
        self.emit_requested(&request);
        let started = Instant::now();
        let response = self
            .client
            .post(&self.config.decision_api_url)
            .header("x-nemo-relay-session-id", &request.identity.session_id)
            .json(&request)
            .send()
            .await
            .map_err(|error| format!("Decision API request failed: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Decision API returned HTTP {status}: {body}"));
        }
        let decision = response
            .json::<RoutingDecision>()
            .await
            .map_err(|error| format!("Decision API returned invalid JSON: {error}"))?;
        self.validate_decision(&decision)?;
        let routed = self.apply_target(inbound, original.clone(), &decision)?;
        let latency = started.elapsed().as_millis() as u64;
        self.emit_decision(
            &request,
            &decision,
            attempt,
            self.config.mode == RoutingMode::ObserveOnly,
            latency,
        );
        Ok((request, decision, routed))
    }

    fn routing_request(
        &self,
        inbound: WireProtocol,
        request: &LlmRequest,
        attempt: u32,
        previous: Option<(String, String)>,
    ) -> Result<RoutingRequest, String> {
        let session = header(request, "x-nemo-relay-session-id");
        let stable_request_id = header(request, "x-nemo-relay-request-id");
        if self.config.context_mode == ContextMode::AtofRequired
            && (session.is_none() || stable_request_id.is_none())
        {
            return Err("stable session and request identity are required for this profile".into());
        }
        let identity_is_stable = session.is_some() && stable_request_id.is_some();
        let synthetic_session = format!("request-{}", Uuid::now_v7());
        let session_id = session.unwrap_or_else(|| synthetic_session.clone());
        let request_id = stable_request_id.unwrap_or_else(|| format!("request-{}", Uuid::now_v7()));
        let annotated = decode_request(inbound, request)
            .map_err(|error| format!("request codec failed: {error}"))?;
        let current_request = self.materialize(inbound, request, &annotated)?;
        let (previous_route, retry_reason) = previous.unzip();
        Ok(RoutingRequest {
            schema_version: ROUTING_REQUEST_SCHEMA_VERSION.into(),
            decision_profile: DecisionProfile {
                profile_id: self.config.decision_profile_id.clone(),
                request_materialization: self.config.request_materialization,
            },
            identity: RequestIdentity {
                session_id,
                request_id,
                turn_id: header(request, "x-nemo-relay-turn-id"),
                parent_scope_id: header(request, "x-nemo-relay-parent-scope-id"),
                root_scope_id: header(request, "x-nemo-relay-root-scope-id"),
                harness: header(request, "x-nemo-relay-agent-kind")
                    .unwrap_or_else(|| "unknown".into()),
                source: header(request, "x-nemo-relay-source")
                    .unwrap_or_else(|| "nemo-relay".into()),
                owner_id: header(request, "x-nemo-relay-owner-id"),
                quality: header(request, "x-nemo-relay-identity-quality").unwrap_or_else(|| {
                    if identity_is_stable {
                        "explicit".into()
                    } else {
                        "synthetic".into()
                    }
                }),
            },
            protocol: RequestProtocol {
                inbound_profile: inbound.label().into(),
                inbound_endpoint: inbound.endpoint().into(),
                desired_response_profile: inbound.label().into(),
            },
            request_summary: RequestSummary {
                client_requested_model: request
                    .content
                    .get("model")
                    .and_then(Json::as_str)
                    .map(ToOwned::to_owned),
                prompt_token_estimate: None,
                tool_count_in_payload: request
                    .content
                    .get("tools")
                    .and_then(Json::as_array)
                    .map(|tools| tools.len() as u64),
                has_system_prompt: Some(annotated.messages.iter().any(|message| {
                    matches!(message, nemo_relay::codec::request::Message::System { .. })
                })),
            },
            current_request,
            attempt: DecisionAttempt {
                routing_attempt: attempt,
                max_routing_attempts: self.config.max_retries.saturating_add(1),
                previous_route,
                retry_reason,
            },
        })
    }

    fn materialize(
        &self,
        inbound: WireProtocol,
        request: &LlmRequest,
        annotated: &nemo_relay::codec::request::AnnotatedLlmRequest,
    ) -> Result<Option<Json>, String> {
        match self.config.request_materialization {
            RequestMaterialization::None | RequestMaterialization::SummaryOnly => Ok(None),
            RequestMaterialization::FullBody => Ok(Some(json!({"body": request.content}))),
            RequestMaterialization::AnnotatedRequest => Ok(Some(json!({
                "body": request.content,
                "annotated_request": annotated,
            }))),
            RequestMaterialization::LatestUserPrompt => {
                let prompt = latest_user_prompt(annotated)
                    .ok_or_else(|| "latest_user_prompt requires a user message".to_string())?;
                let latest = recent_message_window(annotated, 1);
                let body = encode_request(inbound, &latest, Map::new())
                    .map_err(|error| format!("latest user prompt encode failed: {error}"))?
                    .content;
                Ok(Some(json!({"body": body, "latest_user_prompt": prompt})))
            }
            RequestMaterialization::RecentMessageWindow => {
                let window = recent_message_window(annotated, self.config.recent_message_count);
                let body = encode_request(inbound, &window, Map::new())
                    .map_err(|error| format!("recent window encode failed: {error}"))?
                    .content;
                Ok(Some(json!({"body": body, "annotated_request": window})))
            }
        }
    }

    fn validate_decision(&self, decision: &RoutingDecision) -> Result<(), String> {
        if decision.schema_version != ROUTING_DECISION_SCHEMA_VERSION {
            return Err(format!(
                "unsupported decision schema {:?}",
                decision.schema_version
            ));
        }
        let binding = self
            .config
            .targets
            .get(&decision.route.backend_id)
            .ok_or_else(|| format!("unknown backend_id {:?}", decision.route.backend_id))?;
        if binding.model != decision.route.target_model
            || binding.protocol.label() != decision.route.target_protocol_profile
            || binding.endpoint != decision.route.target_endpoint
        {
            return Err(format!(
                "decision target {:?} does not match its exact Relay binding",
                decision.route.backend_id
            ));
        }
        Ok(())
    }

    fn apply_target(
        &self,
        inbound: WireProtocol,
        request: LlmRequest,
        decision: &RoutingDecision,
    ) -> Result<LlmRequest, String> {
        let binding = self
            .config
            .targets
            .get(&decision.route.backend_id)
            .ok_or_else(|| format!("unknown backend_id {:?}", decision.route.backend_id))?;
        let annotated = decode_request(inbound, &request)
            .map_err(|error| format!("request decode failed: {error}"))?;
        let mut routed = if inbound == binding.protocol {
            request
        } else {
            encode_request(binding.protocol, &annotated, request.headers)
                .map_err(|error| format!("request translation failed: {error}"))?
        };
        let object = routed
            .content
            .as_object_mut()
            .ok_or_else(|| "translated request body is not an object".to_string())?;
        object.insert("model".into(), Json::String(binding.model.clone()));
        if let Some(headers) = self.target_headers.get(&decision.route.backend_id) {
            routed.headers.extend(headers.clone());
        }
        routed.headers.insert(
            INTERNAL_DISPATCH_ROUTE_HEADER.into(),
            Json::String(binding.protocol.label().into()),
        );
        routed.headers.insert(
            INTERNAL_DISPATCH_URL_HEADER.into(),
            Json::String(dispatch_url(&binding.base_url, &binding.endpoint)),
        );
        routed.headers.insert(
            INTERNAL_RETRY_AWARE_HEADER.into(),
            Json::String("true".into()),
        );
        Ok(routed)
    }

    fn fallback_request(
        &self,
        inbound: WireProtocol,
        request: LlmRequest,
    ) -> Result<LlmRequest, String> {
        let id = self.config.default_targets.target(inbound);
        let binding = self
            .config
            .targets
            .get(id)
            .ok_or_else(|| format!("unknown fallback target {id:?}"))?;
        let decision = RoutingDecision {
            schema_version: ROUTING_DECISION_SCHEMA_VERSION.into(),
            decision_id: "relay-fallback".into(),
            router: crate::contract::DecisionProvider {
                name: "relay-fallback".into(),
                version: "1".into(),
            },
            route: crate::contract::RoutingTarget {
                tier: "fallback".into(),
                target_model: binding.model.clone(),
                backend_id: id.to_string(),
                target_protocol_profile: binding.protocol.label().into(),
                target_endpoint: binding.endpoint.clone(),
            },
            confidence: None,
            reason_code: Some("relay_trusted_fallback".into()),
            reason_summary: None,
            metadata: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        self.apply_target(inbound, request, &decision)
    }

    async fn dispatch_fallback_buffered(
        &self,
        inbound: WireProtocol,
        original: LlmRequest,
        next: nemo_relay::api::runtime::LlmExecutionNextFn,
        reason: &str,
    ) -> FlowResult<Json> {
        self.emit_fallback(inbound, reason, &original);
        let metadata = identity_metadata_from_request(&original);
        let request = self
            .fallback_request(inbound, original)
            .map_err(FlowError::Internal)?;
        match next(request).await {
            Ok(response) => Ok(response),
            Err(error) => {
                emit_terminal_error(
                    &error,
                    "fallback_buffered",
                    self.config.mode.label(),
                    metadata,
                );
                Err(error)
            }
        }
    }

    async fn dispatch_fallback_stream(
        &self,
        inbound: WireProtocol,
        original: LlmRequest,
        next: nemo_relay::api::runtime::LlmStreamExecutionNextFn,
        reason: &str,
    ) -> FlowResult<LlmJsonStream> {
        self.emit_fallback(inbound, reason, &original);
        let metadata = identity_metadata_from_request(&original);
        let request = self
            .fallback_request(inbound, original)
            .map_err(FlowError::Internal)?;
        match next(request).await {
            Ok(stream) => Ok(mark_terminal_stream(
                stream,
                "fallback_stream",
                self.config.mode.label(),
                metadata.clone(),
            )),
            Err(error) => {
                emit_terminal_error(
                    &error,
                    "fallback_stream_setup",
                    self.config.mode.label(),
                    metadata,
                );
                Err(error)
            }
        }
    }

    fn emit_requested(&self, request: &RoutingRequest) {
        emit_mark(
            "switchyard.routing.requested",
            json!({
                "session_id": request.identity.session_id,
                "request_id": request.identity.request_id,
                "routing_attempt": request.attempt.routing_attempt,
                "profile_id": request.decision_profile.profile_id,
                "rollout_mode": self.config.mode.label(),
            }),
            identity_metadata(request),
        );
    }

    fn emit_decision(
        &self,
        request: &RoutingRequest,
        decision: &RoutingDecision,
        attempt: u32,
        observe_only: bool,
        latency_ms: u64,
    ) {
        emit_mark(
            "switchyard.routing.decision",
            json!({
                "decision_id": decision.decision_id,
                "profile_id": request.decision_profile.profile_id,
                "router": decision.router.name,
                "router_version": decision.router.version,
                "routing_attempt": attempt,
                "backend_id": decision.route.backend_id,
                "selected_tier": decision.route.tier,
                "selected_model": decision.route.target_model,
                "target_protocol_profile": decision.route.target_protocol_profile,
                "target_endpoint": decision.route.target_endpoint,
                "confidence": decision.confidence,
                "reason_code": decision.reason_code,
                "reason_summary": decision.reason_summary,
                "latency_ms": latency_ms,
                "observe_only": observe_only,
                "rollout_mode": self.config.mode.label(),
            }),
            identity_metadata(request),
        );
    }

    fn emit_retry(
        &self,
        request: &RoutingRequest,
        decision: &RoutingDecision,
        attempt: u32,
        reason: &str,
    ) {
        emit_mark(
            "switchyard.routing.retry",
            json!({"routing_attempt": attempt, "previous_route": decision.route.backend_id, "retry_reason": reason, "rollout_mode": self.config.mode.label()}),
            identity_metadata(request),
        );
    }

    fn emit_error(&self, request: Option<&RoutingRequest>, attempt: u32, class: &str, error: &str) {
        emit_mark(
            "switchyard.routing.error",
            json!({"routing_attempt": attempt, "error_class": class, "error": error, "rollout_mode": self.config.mode.label()}),
            request.map(identity_metadata).unwrap_or_else(|| json!({})),
        );
    }

    fn emit_fallback(&self, inbound: WireProtocol, reason: &str, request: &LlmRequest) {
        emit_mark(
            "switchyard.routing.fallback",
            json!({
                "fallback_reason": reason,
                "fallback_route": self.config.default_targets.target(inbound),
                "inbound_profile": inbound.label(),
                "rollout_mode": self.config.mode.label(),
            }),
            identity_metadata_from_request(request),
        );
    }
}

fn validate_config(config: &SwitchyardConfig) -> Result<(), String> {
    if config.version != 1 {
        return Err(format!(
            "unsupported Switchyard config version {}",
            config.version
        ));
    }
    if config.decision_profile_id.trim().is_empty() {
        return Err("decision_profile_id must be non-empty".into());
    }
    if config.decision_timeout_millis == 0 {
        return Err("decision_timeout_millis must be greater than zero".into());
    }
    if config.max_retries > 10 {
        return Err("max_retries must not exceed 10".into());
    }
    if config.recent_message_count == 0 {
        return Err("recent_message_count must be greater than zero".into());
    }
    let url = reqwest::Url::parse(&config.decision_api_url)
        .map_err(|error| format!("decision_api_url is invalid: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("decision_api_url must use http or https".into());
    }
    if config.targets.is_empty() {
        return Err("targets must not be empty".into());
    }
    if config.enabled_inbound_profiles.is_empty() {
        return Err("enabled_inbound_profiles must not be empty".into());
    }
    let mut exact_bindings = BTreeSet::new();
    for (id, target) in &config.targets {
        if id.trim().is_empty()
            || target.model.trim().is_empty()
            || target.endpoint.trim().is_empty()
        {
            return Err("target IDs, models, and endpoints must be non-empty".into());
        }
        let base_url = reqwest::Url::parse(&target.base_url)
            .map_err(|error| format!("target {id:?} base_url is invalid: {error}"))?;
        if !matches!(base_url.scheme(), "http" | "https") {
            return Err(format!("target {id:?} base_url must use http or https"));
        }
        if target.endpoint != target.protocol.endpoint() {
            return Err(format!(
                "target {id:?} endpoint must be {:?} for {}",
                target.protocol.endpoint(),
                target.protocol.label()
            ));
        }
        if !exact_bindings.insert((
            target.model.clone(),
            target.protocol,
            target.endpoint.clone(),
            target.base_url.trim_end_matches('/').to_string(),
        )) {
            return Err(format!(
                "target {id:?} conflicts with another exact backend binding"
            ));
        }
    }
    for protocol in default_enabled_protocols() {
        let id = config.default_targets.target(protocol);
        let target = config
            .targets
            .get(id)
            .ok_or_else(|| format!("default target {id:?} is not configured"))?;
        if target.protocol != protocol {
            return Err(format!(
                "default target {id:?} must use protocol {}",
                protocol.label()
            ));
        }
    }
    Ok(())
}

fn resolve_headers(
    static_headers: &BTreeMap<String, String>,
    environment_headers: &BTreeMap<String, String>,
) -> Result<HeaderMap, String> {
    let mut headers = HeaderMap::new();
    for (name, value) in static_headers {
        insert_http_header(&mut headers, name, value)?;
    }
    for (name, variable) in environment_headers {
        if static_headers.contains_key(name) {
            return Err(format!(
                "header {name:?} cannot appear in both headers and header_env"
            ));
        }
        let value = std::env::var(variable)
            .map_err(|_| format!("environment variable {variable:?} is not set"))?;
        if value.trim().is_empty() {
            return Err(format!("environment variable {variable:?} is blank"));
        }
        insert_http_header(&mut headers, name, &value)?;
    }
    Ok(headers)
}

fn resolve_json_headers(
    static_headers: &BTreeMap<String, String>,
    environment_headers: &BTreeMap<String, String>,
) -> Result<Map<String, Json>, String> {
    let mut headers = Map::new();
    for (name, value) in static_headers {
        headers.insert(name.clone(), Json::String(value.clone()));
    }
    for (name, variable) in environment_headers {
        if static_headers.contains_key(name) {
            return Err(format!(
                "target header {name:?} cannot appear in both headers and header_env"
            ));
        }
        let value = std::env::var(variable)
            .map_err(|_| format!("environment variable {variable:?} is not set"))?;
        if value.trim().is_empty() {
            return Err(format!("environment variable {variable:?} is blank"));
        }
        headers.insert(name.clone(), Json::String(value));
    }
    Ok(headers)
}

fn insert_http_header(headers: &mut HeaderMap, name: &str, value: &str) -> Result<(), String> {
    let name = HeaderName::from_bytes(name.as_bytes())
        .map_err(|error| format!("invalid header name: {error}"))?;
    let value =
        HeaderValue::from_str(value).map_err(|error| format!("invalid header value: {error}"))?;
    headers.insert(name, value);
    Ok(())
}

fn protocol_from_label(label: &str) -> FlowResult<WireProtocol> {
    match label {
        "openai_chat" | "openai_chat_completions" | "openai_chat_completions.v1" => {
            Ok(WireProtocol::OpenaiChat)
        }
        "openai_responses" | "openai_responses.v1" => Ok(WireProtocol::OpenaiResponses),
        "anthropic_messages" | "anthropic_messages.v1" => Ok(WireProtocol::AnthropicMessages),
        value => Err(FlowError::InvalidArgument(format!(
            "unsupported Switchyard target protocol {value:?}"
        ))),
    }
}

fn header(request: &LlmRequest, name: &str) -> Option<String> {
    request
        .headers
        .get(name)
        .and_then(Json::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn dispatch_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let endpoint = if base.ends_with("/v1") && endpoint.starts_with("/v1/") {
        &endpoint[3..]
    } else {
        endpoint
    };
    format!("{base}{endpoint}")
}

fn identity_metadata(request: &RoutingRequest) -> Json {
    json!({
        "session_id": request.identity.session_id,
        "request_id": request.identity.request_id,
        "turn_id": request.identity.turn_id,
        "owner_id": request.identity.owner_id,
    })
}

fn identity_metadata_from_request(request: &LlmRequest) -> Json {
    json!({
        "session_id": header(request, "x-nemo-relay-session-id"),
        "request_id": header(request, "x-nemo-relay-request-id"),
        "turn_id": header(request, "x-nemo-relay-turn-id"),
        "owner_id": header(request, "x-nemo-relay-owner-id"),
    })
}

fn error_is_retryable(error: &FlowError) -> bool {
    matches!(error, FlowError::Upstream(failure) if failure.is_retryable())
}

fn emit_mark(name: &str, data: Json, metadata: Json) {
    if let Err(error) = event(
        EmitMarkEventParams::builder()
            .name(name)
            .data(data)
            .data_schema(
                DataSchema::builder()
                    .name(ROUTING_MARK_SCHEMA)
                    .version("1")
                    .build(),
            )
            .metadata(metadata)
            .category(EventCategory::custom())
            .category_profile(CategoryProfile::builder().subtype(name).build())
            .build(),
    ) {
        eprintln!("nemo-relay switchyard: failed to emit {name}: {error}");
    }
}

fn emit_terminal_error(error: &FlowError, phase: &str, rollout_mode: &str, metadata: Json) {
    emit_mark(
        "switchyard.routing.terminal_error",
        json!({"error_class": provider_error_class(error), "error": provider_error_summary(error), "phase": phase, "rollout_mode": rollout_mode}),
        metadata,
    );
}

fn provider_error_class(error: &FlowError) -> &'static str {
    match error {
        FlowError::Upstream(failure) => match failure.class {
            nemo_relay::error::UpstreamFailureClass::Connection => "connection",
            nemo_relay::error::UpstreamFailureClass::Timeout => "timeout",
            nemo_relay::error::UpstreamFailureClass::RetryableStatus => "retryable_status",
            nemo_relay::error::UpstreamFailureClass::ContextWindow => "context_window",
            nemo_relay::error::UpstreamFailureClass::ModelUnavailable => "model_unavailable",
            nemo_relay::error::UpstreamFailureClass::Authentication => "authentication",
            nemo_relay::error::UpstreamFailureClass::InvalidRequest => "invalid_request",
            nemo_relay::error::UpstreamFailureClass::Other => "other",
        },
        _ => "relay",
    }
}

fn provider_error_summary(error: &FlowError) -> String {
    match error {
        FlowError::Upstream(failure) => match failure.status {
            Some(status) => format!("{}:http_{status}", provider_error_class(error)),
            None => provider_error_class(error).to_string(),
        },
        _ => error.to_string(),
    }
}

fn mark_terminal_stream(
    mut upstream: LlmJsonStream,
    phase: &'static str,
    rollout_mode: &'static str,
    metadata: Json,
) -> LlmJsonStream {
    Box::pin(stream! {
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => yield Ok(chunk),
                Err(error) => {
                    emit_terminal_error(&error, phase, rollout_mode, metadata.clone());
                    yield Err(error);
                    return;
                }
            }
        }
    })
}

fn translated_stream(
    source: WireProtocol,
    target: WireProtocol,
    effective_model: String,
    mut upstream: LlmJsonStream,
) -> LlmJsonStream {
    let mut transcoder = StreamTranscoder::new(source, target, effective_model);
    Box::pin(stream! {
        while let Some(item) = upstream.next().await {
            match item {
                Ok(chunk) => {
                    match transcoder.transcode(&chunk) {
                        Ok(chunks) => {
                            for chunk in chunks {
                                yield Ok(chunk);
                            }
                        }
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    }
                }
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::{Json as AxumJson, Router, extract::State, routing::post};
    use nemo_relay::api::event::Event;
    use nemo_relay::api::runtime::{LlmExecutionNextFn, LlmStreamExecutionNextFn};
    use nemo_relay::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    use nemo_relay::error::{UpstreamFailure, UpstreamFailureClass};

    use super::*;

    fn binding(protocol: WireProtocol, model: &str) -> TargetBinding {
        TargetBinding {
            model: model.into(),
            protocol,
            endpoint: protocol.endpoint().into(),
            base_url: "http://127.0.0.1:9999".into(),
            headers: BTreeMap::new(),
            header_env: BTreeMap::new(),
        }
    }

    fn config(decision_api_url: String) -> SwitchyardConfig {
        SwitchyardConfig {
            decision_api_url,
            decision_profile_id: "cascade".into(),
            request_materialization: RequestMaterialization::SummaryOnly,
            context_mode: ContextMode::PayloadOnly,
            targets: BTreeMap::from([
                (
                    "selected-chat".into(),
                    binding(WireProtocol::OpenaiChat, "selected"),
                ),
                (
                    "fallback-chat".into(),
                    binding(WireProtocol::OpenaiChat, "fallback"),
                ),
                (
                    "fallback-responses".into(),
                    binding(WireProtocol::OpenaiResponses, "fallback"),
                ),
                (
                    "fallback-anthropic".into(),
                    binding(WireProtocol::AnthropicMessages, "fallback"),
                ),
            ]),
            default_targets: ProtocolDefaults {
                openai_chat: "fallback-chat".into(),
                openai_responses: "fallback-responses".into(),
                anthropic_messages: "fallback-anthropic".into(),
            },
            ..SwitchyardConfig::default()
        }
    }

    fn decision() -> RoutingDecision {
        RoutingDecision {
            schema_version: ROUTING_DECISION_SCHEMA_VERSION.into(),
            decision_id: "decision-1".into(),
            router: crate::contract::DecisionProvider {
                name: "cascade".into(),
                version: "1".into(),
            },
            route: crate::contract::RoutingTarget {
                tier: "strong".into(),
                target_model: "selected".into(),
                backend_id: "selected-chat".into(),
                target_protocol_profile: "openai_chat".into(),
                target_endpoint: "/v1/chat/completions".into(),
            },
            confidence: Some(0.9),
            reason_code: Some("test".into()),
            reason_summary: None,
            metadata: BTreeMap::new(),
            extra: BTreeMap::new(),
        }
    }

    fn chat_request() -> LlmRequest {
        LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "inbound",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": "latest"}
                ]
            }),
        }
    }

    fn request(protocol: WireProtocol) -> LlmRequest {
        let content = match protocol {
            WireProtocol::OpenaiChat => chat_request().content,
            WireProtocol::OpenaiResponses => {
                json!({"model": "inbound", "instructions": "system", "input": "latest"})
            }
            WireProtocol::AnthropicMessages => {
                json!({"model": "inbound", "system": "system", "max_tokens": 32, "messages": [{"role": "user", "content": "latest"}]})
            }
        };
        LlmRequest {
            headers: Map::new(),
            content,
        }
    }

    fn chat_response() -> Json {
        json!({
            "id": "chat-1", "object": "chat.completion", "model": "selected",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    }

    fn chat_chunk(text: &str, finish_reason: Json) -> Json {
        json!({
            "id": "chat-1", "object": "chat.completion.chunk", "model": "selected",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish_reason}]
        })
    }

    #[test]
    fn all_materialization_modes_are_bounded_and_provider_valid() {
        for protocol in [
            WireProtocol::OpenaiChat,
            WireProtocol::OpenaiResponses,
            WireProtocol::AnthropicMessages,
        ] {
            for mode in [
                RequestMaterialization::None,
                RequestMaterialization::SummaryOnly,
                RequestMaterialization::LatestUserPrompt,
                RequestMaterialization::RecentMessageWindow,
                RequestMaterialization::AnnotatedRequest,
                RequestMaterialization::FullBody,
            ] {
                let mut config = config("http://127.0.0.1:1/v1/routing/decision".into());
                config.request_materialization = mode;
                config.recent_message_count = 2;
                let runtime = SwitchyardRuntime::new(config).unwrap();
                let routing = runtime
                    .routing_request(protocol, &request(protocol), 1, None)
                    .unwrap();
                match mode {
                    RequestMaterialization::None | RequestMaterialization::SummaryOnly => {
                        assert!(routing.current_request.is_none())
                    }
                    RequestMaterialization::LatestUserPrompt => {
                        let current = routing.current_request.unwrap();
                        assert_eq!(current["latest_user_prompt"], "latest");
                        let body = current["body"].clone();
                        decode_request(
                            protocol,
                            &LlmRequest {
                                headers: Map::new(),
                                content: body,
                            },
                        )
                        .unwrap();
                    }
                    RequestMaterialization::RecentMessageWindow => {
                        let current = routing.current_request.unwrap();
                        decode_request(
                            protocol,
                            &LlmRequest {
                                headers: Map::new(),
                                content: current["body"].clone(),
                            },
                        )
                        .unwrap();
                    }
                    RequestMaterialization::AnnotatedRequest | RequestMaterialization::FullBody => {
                        assert!(routing.current_request.is_some())
                    }
                }
            }
        }
    }

    #[test]
    fn identity_policy_requires_stable_request_scope_only_for_atof_profiles() {
        let mut config = config("http://127.0.0.1:1/v1/routing/decision".into());
        let payload_runtime = SwitchyardRuntime::new(config.clone()).unwrap();
        let synthetic = payload_runtime
            .routing_request(WireProtocol::OpenaiChat, &chat_request(), 1, None)
            .unwrap();
        assert_eq!(synthetic.identity.quality, "synthetic");

        config.context_mode = ContextMode::AtofRequired;
        let atof_runtime = SwitchyardRuntime::new(config).unwrap();
        assert!(
            atof_runtime
                .routing_request(WireProtocol::OpenaiChat, &chat_request(), 1, None)
                .is_err()
        );
        let mut stable = chat_request();
        stable
            .headers
            .insert("x-nemo-relay-session-id".into(), json!("session-1"));
        stable
            .headers
            .insert("x-nemo-relay-request-id".into(), json!("request-1"));
        let routed = atof_runtime
            .routing_request(WireProtocol::OpenaiChat, &stable, 1, None)
            .unwrap();
        assert_eq!(routed.identity.quality, "explicit");
    }

    #[test]
    fn exact_target_validation_rejects_any_switchyard_drift() {
        let runtime =
            SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into()))
                .unwrap();
        assert!(runtime.validate_decision(&decision()).is_ok());
        let mut drifted = decision();
        drifted.route.target_model = "unbound-model".into();
        assert!(runtime.validate_decision(&drifted).is_err());
        drifted = decision();
        drifted.route.backend_id = "unknown".into();
        assert!(runtime.validate_decision(&drifted).is_err());
    }

    #[test]
    fn routing_decision_mark_has_canonical_shape_and_mirrored_identity() {
        let subscriber_name = format!("switchyard-mark-shape-{}", uuid::Uuid::now_v7());
        let events = Arc::new(Mutex::new(Vec::<Event>::new()));
        let captured = Arc::clone(&events);
        register_subscriber(
            &subscriber_name,
            Arc::new(move |event| captured.lock().unwrap().push(event.clone())),
        )
        .unwrap();

        let runtime =
            SwitchyardRuntime::new(config("http://127.0.0.1:1/v1/routing/decision".into()))
                .unwrap();
        let routing_request = runtime
            .routing_request(WireProtocol::OpenaiChat, &chat_request(), 1, None)
            .unwrap();
        runtime.emit_decision(&routing_request, &decision(), 1, false, 17);
        flush_subscribers().unwrap();
        deregister_subscriber(&subscriber_name).unwrap();

        let event = events
            .lock()
            .unwrap()
            .iter()
            .map(Event::to_json_value)
            .find(|event| {
                event["name"] == "switchyard.routing.decision"
                    && event["metadata"]["session_id"] == routing_request.identity.session_id
                    && event["metadata"]["request_id"] == routing_request.identity.request_id
            })
            .expect("decision mark should be captured");
        assert_eq!(event["kind"], "mark");
        assert_eq!(event["category"], "custom");
        assert_eq!(
            event["category_profile"]["subtype"],
            "switchyard.routing.decision"
        );
        assert_eq!(event["data_schema"]["name"], ROUTING_MARK_SCHEMA);
        assert_eq!(event["data_schema"]["version"], "1");
        assert_eq!(event["data"]["profile_id"], "cascade");
        assert_eq!(event["data"]["selected_model"], "selected");
        assert_eq!(event["data"]["latency_ms"], 17);
        assert_eq!(
            event["metadata"]["session_id"],
            routing_request.identity.session_id
        );
        assert_eq!(
            event["metadata"]["request_id"],
            routing_request.identity.request_id
        );
    }

    #[test]
    fn atof_required_cross_component_validation_is_context_sensitive() {
        let mut switchyard = config("http://switchyard.test:8080/v1/routing/decision".into());
        switchyard.context_mode = ContextMode::AtofRequired;
        let mut plugin_config = PluginConfig {
            components: vec![switchyard.into()],
            ..PluginConfig::default()
        };
        assert!(validate_switchyard_atof_configuration(&plugin_config).is_err());
        plugin_config.components.push(PluginComponentSpec {
            kind: "observability".into(),
            enabled: true,
            config: json!({"atof": {
                "enabled": true,
                "endpoints": [{
                    "url": "http://switchyard.test:8080/v1/atof/events",
                    "transport": "http_post",
                    "field_name_policy": "preserve",
                    "header_env": {"authorization": "SWITCHYARD_TOKEN"}
                }]
            }})
            .as_object()
            .unwrap()
            .clone(),
        });
        assert!(validate_switchyard_atof_configuration(&plugin_config).is_ok());
    }

    #[derive(Clone)]
    struct DecisionState {
        requests: Arc<Mutex<Vec<RoutingRequest>>>,
        decision: RoutingDecision,
    }

    async fn decision_handler(
        State(state): State<DecisionState>,
        AxumJson(request): AxumJson<RoutingRequest>,
    ) -> AxumJson<RoutingDecision> {
        state.requests.lock().unwrap().push(request);
        AxumJson(state.decision)
    }

    async fn decision_server() -> (String, Arc<Mutex<Vec<RoutingRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let state = DecisionState {
            requests: Arc::clone(&requests),
            decision: decision(),
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new()
                    .route("/v1/routing/decision", post(decision_handler))
                    .with_state(state),
            )
            .await
            .unwrap();
        });
        (format!("http://{address}/v1/routing/decision"), requests)
    }

    #[tokio::test]
    async fn retry_exhaustion_redecides_four_times_then_dispatches_fallback_once() {
        let (url, decisions) = decision_server().await;
        let runtime = SwitchyardRuntime::new(config(url)).unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&dispatches);
        let next: LlmExecutionNextFn = Arc::new(move |request| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                let attempt = seen.fetch_add(1, Ordering::SeqCst) + 1;
                if attempt <= 4 {
                    return Err(FlowError::Upstream(UpstreamFailure {
                        status: Some(503),
                        body: "temporarily unavailable".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::RetryableStatus,
                    }));
                }
                assert_eq!(request.content["model"], "fallback");
                Ok(chat_response())
            })
        });
        let response = runtime
            .execute_buffered("openai.chat_completions", chat_request(), next)
            .await
            .unwrap();
        assert_eq!(response["choices"][0]["message"]["content"], "ok");
        assert_eq!(dispatches.load(Ordering::SeqCst), 5);
        let requests = decisions.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[3].attempt.routing_attempt, 4);
        assert_eq!(
            requests[3].attempt.previous_route.as_deref(),
            Some("selected-chat")
        );
    }

    #[tokio::test]
    async fn non_retryable_provider_failure_bypasses_retry_loop() {
        let (url, decisions) = decision_server().await;
        let runtime = SwitchyardRuntime::new(config(url)).unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&dispatches);
        let next: LlmExecutionNextFn = Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                let attempt = seen.fetch_add(1, Ordering::SeqCst);
                if attempt == 0 {
                    return Err(FlowError::Upstream(UpstreamFailure {
                        status: Some(401),
                        body: "unauthorized".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::Authentication,
                    }));
                }
                Ok(chat_response())
            })
        });
        runtime
            .execute_buffered("openai.chat_completions", chat_request(), next)
            .await
            .unwrap();
        assert_eq!(dispatches.load(Ordering::SeqCst), 2);
        assert_eq!(decisions.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn observe_only_records_one_decision_and_dispatches_only_the_trusted_default() {
        let (url, decisions) = decision_server().await;
        let mut config = config(url);
        config.mode = RoutingMode::ObserveOnly;
        let runtime = SwitchyardRuntime::new(config).unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&dispatches);
        let next: LlmExecutionNextFn = Arc::new(move |request| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.fetch_add(1, Ordering::SeqCst);
                assert_eq!(request.content["model"], "fallback");
                Ok(chat_response())
            })
        });
        runtime
            .execute_buffered("openai.chat_completions", chat_request(), next)
            .await
            .unwrap();
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(decisions.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn streaming_retries_before_first_item() {
        let (url, decisions) = decision_server().await;
        let runtime = SwitchyardRuntime::new(config(url)).unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&dispatches);
        let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                let attempt = seen.fetch_add(1, Ordering::SeqCst);
                let items = if attempt == 0 {
                    vec![Err(FlowError::Upstream(UpstreamFailure {
                        status: Some(503),
                        body: "retry".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::RetryableStatus,
                    }))]
                } else {
                    vec![Ok(chat_chunk("ok", json!("stop")))]
                };
                Ok(Box::pin(futures_stream::iter(items)) as LlmJsonStream)
            })
        });
        let stream = runtime
            .execute_stream("openai.chat_completions", chat_request(), next)
            .await
            .unwrap();
        let output = stream.collect::<Vec<_>>().await;
        assert_eq!(output.len(), 1);
        assert!(output[0].is_ok());
        assert_eq!(dispatches.load(Ordering::SeqCst), 2);
        assert_eq!(decisions.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn streaming_never_retries_after_first_item() {
        let (url, decisions) = decision_server().await;
        let runtime = SwitchyardRuntime::new(config(url)).unwrap();
        let dispatches = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&dispatches);
        let next: LlmStreamExecutionNextFn = Arc::new(move |_| {
            let seen = Arc::clone(&seen);
            Box::pin(async move {
                seen.fetch_add(1, Ordering::SeqCst);
                let items = vec![
                    Ok(chat_chunk("partial", Json::Null)),
                    Err(FlowError::Upstream(UpstreamFailure {
                        status: None,
                        body: "connection closed".into(),
                        headers: BTreeMap::new(),
                        class: UpstreamFailureClass::Connection,
                    })),
                ];
                Ok(Box::pin(futures_stream::iter(items)) as LlmJsonStream)
            })
        });
        let stream = runtime
            .execute_stream("openai.chat_completions", chat_request(), next)
            .await
            .unwrap();
        let output = stream.collect::<Vec<_>>().await;
        assert_eq!(output.len(), 2);
        assert!(output[0].is_ok());
        assert!(output[1].is_err());
        assert_eq!(dispatches.load(Ordering::SeqCst), 1);
        assert_eq!(decisions.lock().unwrap().len(), 1);
    }
}
