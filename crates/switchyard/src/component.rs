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
use nemo_relay::api::optimization::record_llm_optimization_contribution;
use nemo_relay::api::runtime::{LlmExecutionFn, LlmJsonStream, LlmStreamExecutionFn};
use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::codec::optimization::{
    LlmOptimizationContribution, LlmOptimizationKind, LlmOptimizationModel,
    LlmOptimizationModelTransition,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::observability::atof::{AtofEndpointFieldNamePolicy, AtofEndpointTransport};
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
    RequestSummary, RoutingDecision, RoutingRequest, RoutingTarget,
};
use crate::stream_translation::StreamTranscoder;
use crate::translation::{
    decode_request, encode_request, latest_user_prompt, recent_message_window, translate_response,
    translation_engine, validate_portable_request,
};

/// Plugin kind used in Relay plugin configuration.
pub const SWITCHYARD_PLUGIN_KIND: &str = "switchyard";

const SWITCHYARD_HEALTH_PATH: &str = "/health";
const SWITCHYARD_HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
const SWITCHYARD_HEALTH_MAX_ATTEMPTS: usize = 3;
const SWITCHYARD_HEALTH_INITIAL_BACKOFF: Duration = Duration::from_millis(100);
const INTERNAL_DISPATCH_URL_HEADER: &str = "x-nemo-relay-internal-dispatch-url";
const INTERNAL_DISPATCH_ROUTE_HEADER: &str = "x-nemo-relay-internal-dispatch-route";
const INTERNAL_RETRY_AWARE_HEADER: &str = "x-nemo-relay-internal-retry-aware";
const ROUTING_MARK_SCHEMA: &str = "switchyard.routing_mark";
const ROUTING_CONTRIBUTION_SCHEMA: &str = "nvidia.switchyard.routing_optimization";

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
    #[serde(default)]
    pub openai_chat: String,
    /// OpenAI Responses fallback target.
    #[serde(default)]
    pub openai_responses: String,
    /// Anthropic Messages fallback target.
    #[serde(default)]
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
    /// Named observability ATOF endpoint used by history-backed profiles.
    #[serde(default)]
    pub atof_endpoint_name: Option<String>,
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
            atof_endpoint_name: None,
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
        atof_endpoint_name => { label: "ATOF endpoint name", kind: String, optional: true }
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
            runtime
                .require_healthy_sidecar()
                .await
                .map_err(PluginError::RegistrationFailed)?;
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
    let required_name = validate_atof_endpoint_name(switchyard.atof_endpoint_name.as_deref())?
        .ok_or_else(|| {
            "atof_required Switchyard profiles require atof_endpoint_name".to_string()
        })?;
    let observability = config
        .components
        .iter()
        .find(|component| component.enabled && component.kind == "observability")
        .ok_or_else(|| "atof_required Switchyard profiles require observability".to_string())?;
    let sinks = observability
        .config
        .get("atof")
        .filter(|atof| atof.get("enabled").and_then(Json::as_bool) == Some(true))
        .and_then(|atof| atof.get("sinks"))
        .and_then(Json::as_array)
        .ok_or_else(|| {
            "atof_required Switchyard profiles require an enabled ATOF endpoint".to_string()
        })?;
    let matching_sinks = sinks
        .iter()
        .filter(|sink| {
            sink.get("type").and_then(Json::as_str) == Some("stream")
                && sink.get("name").and_then(Json::as_str) == Some(required_name)
        })
        .collect::<Vec<_>>();
    let endpoint = match matching_sinks.as_slice() {
        [sink] => *sink,
        [] => {
            return Err(format!(
                "atof_required Switchyard profile requires named ATOF endpoint {required_name:?}"
            ));
        }
        _ => {
            return Err(format!(
                "ATOF endpoint name {required_name:?} must resolve to exactly one endpoint"
            ));
        }
    };
    let transport = endpoint.get("transport").map_or_else(
        || Some(AtofEndpointTransport::default()),
        |value| value.as_str().and_then(AtofEndpointTransport::parse),
    );
    if transport != Some(AtofEndpointTransport::HttpPost) {
        return Err(format!(
            "Switchyard ATOF endpoint {required_name:?} must use transport = http_post"
        ));
    }
    let field_name_policy = endpoint.get("field_name_policy").map_or_else(
        || Some(AtofEndpointFieldNamePolicy::default()),
        |value| value.as_str().and_then(AtofEndpointFieldNamePolicy::parse),
    );
    if field_name_policy != Some(AtofEndpointFieldNamePolicy::Preserve) {
        return Err(format!(
            "Switchyard ATOF endpoint {required_name:?} must use field_name_policy = preserve"
        ));
    }
    if endpoint
        .get("header_env")
        .and_then(Json::as_object)
        .is_none_or(Map::is_empty)
    {
        return Err(format!(
            "Switchyard ATOF endpoint {required_name:?} authentication must use at least one environment-referenced header"
        ));
    }
    Ok(())
}

fn parse_config(config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    serde_json::from_value(Json::Object(config.clone()))
        .map_err(|error| format!("invalid Switchyard plugin config: {error}"))
}

struct SwitchyardRuntime {
    config: SwitchyardConfig,
    client: reqwest::Client,
    target_headers: BTreeMap<String, Map<String, Json>>,
    translation: switchyard_translation::TranslationEngine,
}

enum BufferedAttempt {
    Complete(Json),
    Retry((String, String)),
    Fallback(&'static str),
}

enum StreamAttempt {
    Committed(LlmJsonStream),
    Retry((String, String)),
    Fallback(&'static str),
}

struct StreamAttemptContext {
    routing_request: RoutingRequest,
    decision: RoutingDecision,
    attempt: u32,
    max_attempts: u32,
}

fn provider_fallback_reason(error: &FlowError) -> &'static str {
    if error_is_retryable(error) {
        "retry_exhausted"
    } else {
        "non_retryable_provider_error"
    }
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
            translation: translation_engine(),
        })
    }

    // Skip the portability guard when no configured target uses a different protocol: with no
    // possible cross-protocol translation, provider-specific fields never need to be portable.
    fn may_translate_protocol(&self, inbound: WireProtocol) -> bool {
        self.config
            .targets
            .values()
            .any(|target| target.protocol != inbound)
    }

    async fn require_healthy_sidecar(&self) -> Result<(), String> {
        let health_url = switchyard_health_url(&self.config.decision_api_url)?;
        let client = reqwest::Client::builder()
            .timeout(SWITCHYARD_HEALTH_TIMEOUT)
            .build()
            .map_err(|error| format!("failed to build Switchyard health client: {error}"))?;
        let mut backoff = SWITCHYARD_HEALTH_INITIAL_BACKOFF;
        let mut final_error = None;
        for attempt in 1..=SWITCHYARD_HEALTH_MAX_ATTEMPTS {
            match check_switchyard_health(&client, &health_url).await {
                Ok(()) => return Ok(()),
                Err(error) => final_error = Some(error),
            }
            if attempt < SWITCHYARD_HEALTH_MAX_ATTEMPTS {
                tokio::time::sleep(backoff).await;
                backoff *= 2;
            }
        }
        Err(final_error.expect("at least one Switchyard health attempt is configured"))
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
        if self.may_translate_protocol(inbound)
            && let Err(error) = validate_portable_request(&self.translation, inbound, &original)
        {
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
            match self.decided_request(inbound, &original, 1, None).await {
                Ok((_, decision, _)) => {
                    self.record_routing_contribution(&decision, 1, false);
                }
                Err(error) => self.emit_error(None, 1, "decision_api", &error),
            }
            return self
                .dispatch_fallback_buffered(inbound, original, next, "observe_only")
                .await;
        }

        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut previous = None;
        for attempt in 1..=max_attempts {
            match self
                .buffered_attempt(inbound, &original, &next, attempt, previous, max_attempts)
                .await?
            {
                BufferedAttempt::Complete(response) => return Ok(response),
                BufferedAttempt::Retry(retry) => previous = Some(retry),
                BufferedAttempt::Fallback(reason) => {
                    return self
                        .dispatch_fallback_buffered(inbound, original, next, reason)
                        .await;
                }
            }
        }
        unreachable!("routing attempt loop always returns")
    }

    async fn buffered_attempt(
        &self,
        inbound: WireProtocol,
        original: &LlmRequest,
        next: &nemo_relay::api::runtime::LlmExecutionNextFn,
        attempt: u32,
        previous: Option<(String, String)>,
        max_attempts: u32,
    ) -> FlowResult<BufferedAttempt> {
        let (routing_request, decision, routed) = match self
            .decided_request(inbound, original, attempt, previous)
            .await
        {
            Ok(value) => value,
            Err(error) => {
                self.emit_error(None, attempt, "decision_api", &error);
                return Ok(BufferedAttempt::Fallback("decision_error"));
            }
        };
        let target_protocol = protocol_from_label(&decision.route.target_protocol_profile)?;
        match next(routed).await {
            Ok(response) => {
                match translate_response(&self.translation, target_protocol, inbound, &response) {
                    Ok(response) => {
                        self.record_routing_contribution(&decision, attempt, true);
                        Ok(BufferedAttempt::Complete(response))
                    }
                    Err(error) => {
                        self.emit_error(
                            Some(&routing_request),
                            attempt,
                            "response_translation",
                            &error.to_string(),
                        );
                        Ok(BufferedAttempt::Fallback("translation_error"))
                    }
                }
            }
            Err(error) if error_is_retryable(&error) && attempt < max_attempts => {
                let retry_reason = provider_error_summary(&error);
                self.emit_error(Some(&routing_request), attempt, "provider", &retry_reason);
                self.emit_retry(&routing_request, &decision, attempt, &retry_reason);
                Ok(BufferedAttempt::Retry((
                    decision.route.backend_id,
                    retry_reason,
                )))
            }
            Err(error) => {
                let summary = provider_error_summary(&error);
                self.emit_error(Some(&routing_request), attempt, "provider", &summary);
                Ok(BufferedAttempt::Fallback(provider_fallback_reason(&error)))
            }
        }
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
        if self.may_translate_protocol(inbound)
            && let Err(error) = validate_portable_request(&self.translation, inbound, &original)
        {
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
            match self.decided_request(inbound, &original, 1, None).await {
                Ok((_, decision, _)) => {
                    self.record_routing_contribution(&decision, 1, false);
                }
                Err(error) => self.emit_error(None, 1, "decision_api", &error),
            }
            return self
                .dispatch_fallback_stream(inbound, original, next, "observe_only")
                .await;
        }

        let max_attempts = self.config.max_retries.saturating_add(1);
        let mut previous = None;
        for attempt in 1..=max_attempts {
            match self
                .stream_attempt(inbound, &original, &next, attempt, previous, max_attempts)
                .await?
            {
                StreamAttempt::Committed(stream) => return Ok(stream),
                StreamAttempt::Retry(retry) => previous = Some(retry),
                StreamAttempt::Fallback(reason) => {
                    return self
                        .dispatch_fallback_stream(inbound, original, next, reason)
                        .await;
                }
            }
        }
        unreachable!("stream routing attempt loop always returns")
    }

    async fn stream_attempt(
        &self,
        inbound: WireProtocol,
        original: &LlmRequest,
        next: &nemo_relay::api::runtime::LlmStreamExecutionNextFn,
        attempt: u32,
        previous: Option<(String, String)>,
        max_attempts: u32,
    ) -> FlowResult<StreamAttempt> {
        let (routing_request, decision, routed) = match self
            .decided_request(inbound, original, attempt, previous)
            .await
        {
            Ok(value) => value,
            Err(error) => {
                self.emit_error(None, attempt, "decision_api", &error);
                return Ok(StreamAttempt::Fallback("decision_error"));
            }
        };
        let target_protocol = protocol_from_label(&decision.route.target_protocol_profile)?;
        let context = StreamAttemptContext {
            routing_request,
            decision,
            attempt,
            max_attempts,
        };
        match next(routed).await {
            Ok(mut upstream) => {
                let first = upstream.next().await;
                Ok(self.classify_open_stream(inbound, target_protocol, context, upstream, first))
            }
            Err(error) => Ok(self.classify_stream_setup_error(context, error)),
        }
    }

    fn classify_open_stream(
        &self,
        inbound: WireProtocol,
        target_protocol: WireProtocol,
        context: StreamAttemptContext,
        upstream: LlmJsonStream,
        first: Option<FlowResult<Json>>,
    ) -> StreamAttempt {
        let StreamAttemptContext {
            routing_request,
            decision,
            attempt,
            max_attempts,
        } = context;
        match first {
            Some(Ok(first)) => {
                self.record_routing_contribution(&decision, attempt, true);
                let committed = LlmJsonStream::new(Box::pin(
                    futures_stream::once(async move { Ok(first) }).chain(upstream),
                ));
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
                StreamAttempt::Committed(mark_terminal_stream(
                    output,
                    "provider_stream_committed",
                    self.config.mode.label(),
                    identity_metadata(&routing_request),
                ))
            }
            Some(Err(error)) if error_is_retryable(&error) && attempt < max_attempts => self
                .retry_stream_attempt(
                    &routing_request,
                    decision,
                    attempt,
                    "provider_stream_open",
                    provider_error_summary(&error),
                ),
            None if attempt < max_attempts => self.retry_stream_attempt(
                &routing_request,
                decision,
                attempt,
                "provider_stream_open",
                "empty_stream".into(),
            ),
            Some(Err(error)) => {
                let summary = provider_error_summary(&error);
                self.emit_error(
                    Some(&routing_request),
                    attempt,
                    "provider_stream_open",
                    &summary,
                );
                StreamAttempt::Fallback(provider_fallback_reason(&error))
            }
            None => StreamAttempt::Fallback("empty_stream"),
        }
    }

    fn classify_stream_setup_error(
        &self,
        context: StreamAttemptContext,
        error: FlowError,
    ) -> StreamAttempt {
        let StreamAttemptContext {
            routing_request,
            decision,
            attempt,
            max_attempts,
        } = context;
        let summary = provider_error_summary(&error);
        if error_is_retryable(&error) && attempt < max_attempts {
            return self.retry_stream_attempt(
                &routing_request,
                decision,
                attempt,
                "provider_stream_setup",
                summary,
            );
        }
        self.emit_error(
            Some(&routing_request),
            attempt,
            "provider_stream_setup",
            &summary,
        );
        StreamAttempt::Fallback(provider_fallback_reason(&error))
    }

    fn retry_stream_attempt(
        &self,
        routing_request: &RoutingRequest,
        decision: RoutingDecision,
        attempt: u32,
        error_class: &str,
        reason: String,
    ) -> StreamAttempt {
        if reason != "empty_stream" {
            self.emit_error(Some(routing_request), attempt, error_class, &reason);
        }
        self.emit_retry(routing_request, &decision, attempt, &reason);
        StreamAttempt::Retry((decision.route.backend_id, reason))
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
        if let Some(baseline) = decision.baseline_route.as_ref()
            && let Err(error) = self.validate_target(baseline)
        {
            self.emit_error(Some(&request), attempt, "baseline_binding", &error);
        }
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
        let annotated = decode_request(&self.translation, inbound, request)
            .map_err(|error| format!("request translation decode failed: {error}"))?;
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
                has_system_prompt: Some(
                    annotated.instructions.iter().any(|instruction| {
                        instruction.role == switchyard_translation::Role::System
                    }) || annotated
                        .messages
                        .iter()
                        .any(|message| message.role == switchyard_translation::Role::System),
                ),
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
        annotated: &switchyard_translation::LlmRequest,
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
                let body = encode_request(&self.translation, inbound, &latest, Map::new())
                    .map_err(|error| format!("latest user prompt encode failed: {error}"))?
                    .content;
                Ok(Some(json!({"body": body, "latest_user_prompt": prompt})))
            }
            RequestMaterialization::RecentMessageWindow => {
                let window = recent_message_window(annotated, self.config.recent_message_count);
                let body = encode_request(&self.translation, inbound, &window, Map::new())
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
        self.validate_target(&decision.route).map(|_| ())
    }

    fn validate_target(&self, target: &RoutingTarget) -> Result<&TargetBinding, String> {
        let binding = self
            .config
            .targets
            .get(&target.backend_id)
            .ok_or_else(|| format!("unknown backend_id {:?}", target.backend_id))?;
        if binding.model != target.target_model
            || binding.protocol.label() != target.target_protocol_profile
            || binding.endpoint != target.target_endpoint
        {
            return Err(format!(
                "decision target {:?} does not match its exact Relay binding",
                target.backend_id
            ));
        }
        Ok(binding)
    }

    fn record_routing_contribution(&self, decision: &RoutingDecision, attempt: u32, applied: bool) {
        let Some(contribution) = self.routing_contribution(decision, attempt, applied) else {
            return;
        };
        let _ = record_llm_optimization_contribution(contribution);
    }

    fn routing_contribution(
        &self,
        decision: &RoutingDecision,
        attempt: u32,
        applied: bool,
    ) -> Option<LlmOptimizationContribution> {
        let baseline = decision
            .baseline_route
            .as_ref()
            .filter(|baseline| self.validate_target(baseline).is_ok())?;
        let mut contribution = LlmOptimizationContribution::new(
            SWITCHYARD_PLUGIN_KIND,
            LlmOptimizationKind::model_routing(),
        );
        contribution.applied = applied;
        contribution.model_transition = Some(LlmOptimizationModelTransition {
            baseline: Some(LlmOptimizationModel::new(&baseline.target_model)),
            effective: Some(LlmOptimizationModel::new(&decision.route.target_model)),
        });
        contribution.payload_schema = Some(DataSchema {
            name: ROUTING_CONTRIBUTION_SCHEMA.to_string(),
            version: "1".to_string(),
        });
        contribution.payload = Some(json!({
            "decision_id": decision.decision_id,
            "selected_backend_id": decision.route.backend_id,
            "selected_tier": decision.route.tier,
            "baseline_backend_id": baseline.backend_id,
            "baseline_tier": baseline.tier,
            "routing_attempt": attempt,
            "rollout_mode": self.config.mode.label(),
            "reason_code": decision.reason_code,
            "reason_summary": decision.reason_summary,
            "router_metadata": decision.metadata,
        }));
        Some(contribution)
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
        let annotated = decode_request(&self.translation, inbound, &request)
            .map_err(|error| format!("request decode failed: {error}"))?;
        let mut routed = if inbound == binding.protocol {
            request
        } else {
            encode_request(
                &self.translation,
                binding.protocol,
                &annotated,
                request.headers,
            )
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
            baseline_route: None,
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
                "router_metadata": decision.metadata,
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

async fn check_switchyard_health(
    client: &reqwest::Client,
    health_url: &reqwest::Url,
) -> Result<(), String> {
    let response = client
        .get(health_url.clone())
        .send()
        .await
        .map_err(|error| {
            format!("Switchyard service is required but health check {health_url} failed: {error}")
        })?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!(
            "Switchyard service is required but health check {health_url} returned HTTP {status}"
        ));
    }
    let body = response.json::<Json>().await.map_err(|error| {
        format!("Switchyard health check {health_url} returned invalid JSON: {error}")
    })?;
    if body.get("status").and_then(Json::as_str) != Some("ok") {
        return Err(format!(
            "Switchyard health check {health_url} did not report status=ok"
        ));
    }
    Ok(())
}

fn switchyard_health_url(decision_api_url: &str) -> Result<reqwest::Url, String> {
    let mut url = reqwest::Url::parse(decision_api_url)
        .map_err(|error| format!("decision_api_url is invalid: {error}"))?;
    url.set_path(SWITCHYARD_HEALTH_PATH);
    url.set_query(None);
    url.set_fragment(None);
    Ok(url)
}

fn validate_atof_endpoint_name(name: Option<&str>) -> Result<Option<&str>, String> {
    if let Some(name) = name {
        if name.trim().is_empty() {
            return Err("atof_endpoint_name must be non-empty when configured".into());
        }
        if name != name.trim() {
            return Err("atof_endpoint_name must not have leading or trailing whitespace".into());
        }
    }
    Ok(name)
}

fn validate_config(config: &SwitchyardConfig) -> Result<(), String> {
    validate_scalar_config(config)?;
    validate_decision_api_url(&config.decision_api_url)?;
    validate_target_bindings(config)?;
    validate_default_targets(config)
}

fn validate_scalar_config(config: &SwitchyardConfig) -> Result<(), String> {
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
    let atof_endpoint_name = validate_atof_endpoint_name(config.atof_endpoint_name.as_deref())?;
    if config.context_mode == ContextMode::AtofRequired && atof_endpoint_name.is_none() {
        return Err("atof_required Switchyard profiles require atof_endpoint_name".into());
    }
    Ok(())
}

fn validate_decision_api_url(decision_api_url: &str) -> Result<(), String> {
    let url = reqwest::Url::parse(decision_api_url)
        .map_err(|error| format!("decision_api_url is invalid: {error}"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err("decision_api_url must use http or https".into());
    }
    Ok(())
}

fn validate_target_bindings(config: &SwitchyardConfig) -> Result<(), String> {
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
    Ok(())
}

fn validate_default_targets(config: &SwitchyardConfig) -> Result<(), String> {
    for &protocol in &config.enabled_inbound_profiles {
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
    LlmJsonStream::new(Box::pin(stream! {
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
    }))
}

fn translated_stream(
    source: WireProtocol,
    target: WireProtocol,
    effective_model: String,
    mut upstream: LlmJsonStream,
) -> LlmJsonStream {
    let mut transcoder = StreamTranscoder::new(source, target, effective_model);
    LlmJsonStream::new(Box::pin(stream! {
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
        match transcoder.finish() {
            Ok(chunks) => {
                for chunk in chunks {
                    yield Ok(chunk);
                }
            }
            Err(error) => yield Err(error),
        }
    }))
}

#[cfg(test)]
#[path = "../tests/unit/component_tests.rs"]
mod tests;
