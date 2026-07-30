// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Switchyard libsy plugin configuration and Relay execution integration.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures_util::StreamExt;
use http::Uri;
use http::header::{HeaderName, HeaderValue};
use nemo_relay::api::event::{CategoryProfile, DataSchema, EventCategory};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    LlmExecutionFn, LlmExecutionNextFn, LlmJsonStream, LlmStreamExecutionFn,
    LlmStreamExecutionNextFn, task_scope_top,
};
use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result as PluginResult, deregister_plugin, register_plugin,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json, json};
use switchyard_libsy::algorithms::{LlmTaskClassifier, Random, TaskClassifierConfig};
use switchyard_libsy::{
    Algorithm, CallLlmRequest, Context as LibsyContext, Decision, LibsyError, LlmResponse,
    LlmTarget, LlmTargetSet, Request as LibsyRequest, Response as LibsyResponse, Step,
};
use switchyard_protocol::{LlmClientError, Metadata};

use crate::stream_translation::{
    StreamCloseTracker, flow_to_client_error, provider_response_stream, relay_response_stream,
};
use crate::translation::{
    decode_request, decode_response, encode_request, encode_response, translation_engine,
    wire_format,
};

/// Plugin kind used in Relay plugin configuration.
pub const SWITCHYARD_PLUGIN_KIND: &str = "switchyard";

const INTERNAL_DISPATCH_URL_HEADER: &str = "x-nemo-relay-internal-dispatch-url";
const INTERNAL_DISPATCH_ROUTE_HEADER: &str = "x-nemo-relay-internal-dispatch-route";
const INTERNAL_RETRY_AWARE_HEADER: &str = "x-nemo-relay-internal-retry-aware";
const RELAY_REQUEST_ID_HEADER: &str = "x-nemo-relay-request-id";
const RELAY_TURN_ID_HEADER: &str = "x-nemo-relay-turn-id";
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
    pub(crate) fn label(self) -> &'static str {
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

    fn from_call(name: &str) -> Option<Self> {
        match name {
            "openai.chat_completions" | "openai_chat" | "openai_chat_completions" => {
                Some(Self::OpenaiChat)
            }
            "openai.responses" | "openai_responses" => Some(Self::OpenaiResponses),
            "anthropic.messages" | "anthropic" | "anthropic_messages" => {
                Some(Self::AnthropicMessages)
            }
            _ => None,
        }
    }
}

/// Exact Relay-owned provider binding for one libsy semantic target.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TargetBinding {
    /// Provider model sent on the physical request.
    pub model: String,
    /// Provider wire protocol.
    pub protocol: WireProtocol,
    /// Provider endpoint.
    pub endpoint: String,
    /// Relay-owned provider base URL.
    pub base_url: String,
    /// Relative random-routing weight.
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Static non-sensitive provider headers.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Provider headers resolved from environment variables.
    #[serde(default)]
    pub header_env: BTreeMap<String, String>,
}

/// Trusted fallback target names for each inbound protocol.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
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

/// In-process libsy algorithm configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AlgorithmConfig {
    /// Uniform or weighted random routing.
    Random {
        /// Optional deterministic random seed.
        #[serde(default)]
        seed: Option<u64>,
    },
    /// Judge-backed routing between efficient and capable targets.
    LlmClassifier {
        /// Semantic target used for the classifier consultation.
        classifier_target: String,
        /// Semantic target used for tasks the classifier considers efficient.
        weak_target: String,
        /// Semantic target used for tasks requiring the capable tier.
        strong_target: String,
        /// Lowest solve probability that selects the weak target.
        base_threshold: f64,
        /// Lowest classifier confidence that permits weak-target routing.
        #[serde(default)]
        min_confidence: f64,
        /// Higher solve-probability floor for uncertain capability verdicts.
        #[serde(default)]
        capability_elevated_floor: Option<f64>,
        /// Reuse the first routing decision for subsequent requests in a session.
        #[serde(default)]
        session_affinity: bool,
        /// Key affinity from the first user message when session metadata is absent.
        #[serde(default)]
        message_hash_fallback: bool,
    },
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self::Random { seed: None }
    }
}

/// Versioned Switchyard plugin configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct SwitchyardConfig {
    /// Config schema version. The libsy-only contract is version 2.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Execution-intercept priority.
    #[serde(default)]
    pub priority: i32,
    /// Provider retries after the initial libsy run.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// In-process libsy algorithm.
    #[serde(default)]
    pub algorithm: AlgorithmConfig,
    /// Semantic libsy targets mapped to Relay-owned provider bindings.
    pub targets: BTreeMap<String, TargetBinding>,
    /// Trusted per-protocol fallbacks.
    #[serde(default)]
    pub default_targets: ProtocolDefaults,
    /// Inbound protocols handled by this component.
    #[serde(default = "default_enabled_protocols")]
    pub enabled_inbound_profiles: BTreeSet<WireProtocol>,
}

impl Default for SwitchyardConfig {
    fn default() -> Self {
        Self {
            version: default_version(),
            priority: 0,
            max_retries: default_max_retries(),
            algorithm: AlgorithmConfig::default(),
            targets: BTreeMap::new(),
            default_targets: ProtocolDefaults::default(),
            enabled_inbound_profiles: default_enabled_protocols(),
        }
    }
}

nemo_relay::editor_config! {
    impl SwitchyardConfig {
        priority => { label: "Intercept priority", kind: Integer },
        max_retries => { label: "Maximum provider retries", kind: Integer },
        algorithm => { label: "libsy algorithm", kind: Json },
        targets => { label: "Provider target bindings", kind: Json },
        default_targets => { label: "Trusted protocol defaults", kind: Json },
        enabled_inbound_profiles => { label: "Enabled inbound profiles", kind: Json }
    }
}

impl From<SwitchyardConfig> for PluginComponentSpec {
    fn from(value: SwitchyardConfig) -> Self {
        let config = match serde_json::to_value(value) {
            Ok(Json::Object(config)) => config,
            _ => Map::new(),
        };
        Self {
            kind: SWITCHYARD_PLUGIN_KIND.into(),
            enabled: true,
            config,
        }
    }
}

fn default_version() -> u32 {
    2
}

fn default_max_retries() -> u32 {
    3
}

fn default_weight() -> f64 {
    1.0
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
                "libsy",
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
                "libsy_stream",
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

fn parse_config(config: &Map<String, Json>) -> Result<SwitchyardConfig, String> {
    if config.get("version").and_then(Json::as_u64) == Some(1)
        || config.contains_key("decision_api_url")
    {
        return Err(
            "Switchyard config version 1 used the removed switchyard-server Decision API; migrate to version = 2 with [components.config.algorithm]".into(),
        );
    }
    serde_json::from_value(Json::Object(config.clone()))
        .map_err(|error| format!("invalid Switchyard plugin config: {error}"))
}

struct SwitchyardRuntime {
    config: SwitchyardConfig,
    algorithm: Arc<dyn Algorithm>,
    target_headers: BTreeMap<String, Map<String, Json>>,
    translation: switchyard_translation::TranslationEngine,
}

impl SwitchyardRuntime {
    fn new(config: SwitchyardConfig) -> Result<Self, String> {
        validate_config(&config)?;
        let target_headers = config
            .targets
            .iter()
            .map(|(name, target)| {
                resolve_json_headers(&target.headers, &target.header_env)
                    .map(|headers| (name.clone(), headers))
            })
            .collect::<Result<_, _>>()?;
        let algorithm = build_algorithm(&config)?;
        Ok(Self {
            config,
            algorithm,
            target_headers,
            translation: translation_engine(),
        })
    }

    async fn execute_buffered(
        &self,
        name: &str,
        original: LlmRequest,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let Some(inbound) = WireProtocol::from_call(name) else {
            return next(original).await;
        };
        if !self.config.enabled_inbound_profiles.contains(&inbound) {
            return next(original).await;
        }

        let max_attempts = self.config.max_retries.saturating_add(1);
        for attempt in 1..=max_attempts {
            self.emit_requested(&original, inbound, attempt);
            let request = match self.libsy_request(inbound, &original, false) {
                Ok(request) => request,
                Err(error) => {
                    self.emit_error(
                        attempt,
                        "request_translation",
                        &error.to_string(),
                        &original,
                    );
                    return self
                        .dispatch_fallback_buffered(inbound, original, next, "translation_error")
                        .await;
                }
            };
            match self
                .drive_buffered_run(
                    request,
                    dispatch_headers(&original.headers),
                    next.clone(),
                    attempt,
                )
                .await
            {
                Ok(response) => match self.finish_buffered(inbound, response) {
                    Ok(response) => return Ok(response),
                    Err(error) => {
                        self.emit_error(
                            attempt,
                            "response_translation",
                            &error.to_string(),
                            &original,
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
                Err(failure) if failure.is_retryable() && attempt < max_attempts => {
                    self.emit_retry(attempt, &failure.error.to_string(), &original);
                }
                Err(failure) => {
                    let reason = if failure.is_retryable() {
                        "retry_exhausted"
                    } else {
                        "non_retryable_libsy_error"
                    };
                    self.emit_error(attempt, "libsy_run", &failure.error.to_string(), &original);
                    return self
                        .dispatch_fallback_buffered(inbound, original, next, reason)
                        .await;
                }
            }
        }
        Err(FlowError::Internal(
            "Switchyard retry loop ended without a result".into(),
        ))
    }

    async fn execute_stream(
        &self,
        name: &str,
        original: LlmRequest,
        next: LlmStreamExecutionNextFn,
    ) -> FlowResult<LlmJsonStream> {
        let Some(inbound) = WireProtocol::from_call(name) else {
            return next(original).await;
        };
        if !self.config.enabled_inbound_profiles.contains(&inbound) {
            return next(original).await;
        }

        let max_attempts = self.config.max_retries.saturating_add(1);
        for attempt in 1..=max_attempts {
            self.emit_requested(&original, inbound, attempt);
            let request = match self.libsy_request(inbound, &original, true) {
                Ok(request) => request,
                Err(error) => {
                    self.emit_error(
                        attempt,
                        "request_translation",
                        &error.to_string(),
                        &original,
                    );
                    return self
                        .dispatch_fallback_stream(inbound, original, next, "translation_error")
                        .await;
                }
            };
            let tracker = StreamCloseTracker::default();
            match self
                .drive_stream_run(
                    request,
                    dispatch_headers(&original.headers),
                    next.clone(),
                    tracker.clone(),
                    attempt,
                )
                .await
            {
                Ok(response) => match self.finish_stream(inbound, response, tracker) {
                    Ok(response) => return Ok(response),
                    Err(error) => {
                        self.emit_error(
                            attempt,
                            "response_translation",
                            &error.to_string(),
                            &original,
                        );
                        return self
                            .dispatch_fallback_stream(inbound, original, next, "translation_error")
                            .await;
                    }
                },
                Err(failure) if failure.is_retryable() && attempt < max_attempts => {
                    self.emit_retry(attempt, &failure.error.to_string(), &original);
                }
                Err(failure) => {
                    let reason = if failure.is_retryable() {
                        "retry_exhausted"
                    } else {
                        "non_retryable_libsy_error"
                    };
                    self.emit_error(
                        attempt,
                        "libsy_stream_run",
                        &failure.error.to_string(),
                        &original,
                    );
                    return self
                        .dispatch_fallback_stream(inbound, original, next, reason)
                        .await;
                }
            }
        }
        Err(FlowError::Internal(
            "Switchyard stream retry loop ended without a result".into(),
        ))
    }

    async fn drive_buffered_run(
        &self,
        request: LibsyRequest,
        headers: Map<String, Json>,
        next: LlmExecutionNextFn,
        attempt: u32,
    ) -> Result<LibsyResponse, RunFailure> {
        let values = context_values(request.metadata.as_ref(), attempt);
        let mark_metadata = libsy_identity_metadata(request.metadata.as_ref());
        let mut context = LibsyContext::default();
        context.values = values.into_iter().collect();
        let mut steps = self.algorithm.clone().run_stream(context, request);
        let provider_error = Arc::new(Mutex::new(None));
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
                    self.emit_decision(decision.as_ref(), attempt, mark_metadata.clone());
                }
                Ok(Step::CallLlm(call)) => {
                    self.emit_call(call.get_decision(), attempt, mark_metadata.clone());
                    self.serve_buffered_call(
                        *call,
                        headers.clone(),
                        next.clone(),
                        Arc::clone(&provider_error),
                    )
                    .await
                    .map_err(|error| RunFailure::new(error, &provider_error))?;
                }
                Ok(Step::ReturnToAgent(response)) => return Ok(*response),
                Err(error) => return Err(RunFailure::new(error, &provider_error)),
            }
        }
        Err(RunFailure::new(
            LibsyError::MissingFinalResponse,
            &provider_error,
        ))
    }

    async fn drive_stream_run(
        &self,
        request: LibsyRequest,
        headers: Map<String, Json>,
        next: LlmStreamExecutionNextFn,
        tracker: StreamCloseTracker,
        attempt: u32,
    ) -> Result<LibsyResponse, RunFailure> {
        let values = context_values(request.metadata.as_ref(), attempt);
        let mark_metadata = libsy_identity_metadata(request.metadata.as_ref());
        let mut context = LibsyContext::default();
        context.values = values.into_iter().collect();
        let mut steps = self.algorithm.clone().run_stream(context, request);
        let provider_error = Arc::new(Mutex::new(None));
        while let Some(step) = steps.next().await {
            match step {
                Ok(Step::Decision(decision)) => {
                    self.emit_decision(decision.as_ref(), attempt, mark_metadata.clone());
                }
                Ok(Step::CallLlm(call)) => {
                    self.emit_call(call.get_decision(), attempt, mark_metadata.clone());
                    self.serve_stream_call(
                        *call,
                        headers.clone(),
                        next.clone(),
                        tracker.clone(),
                        Arc::clone(&provider_error),
                    )
                    .await
                    .map_err(|error| RunFailure::new(error, &provider_error))?;
                }
                Ok(Step::ReturnToAgent(response)) => return Ok(*response),
                Err(error) => return Err(RunFailure::new(error, &provider_error)),
            }
        }
        Err(RunFailure::new(
            LibsyError::MissingFinalResponse,
            &provider_error,
        ))
    }

    async fn serve_buffered_call(
        &self,
        call: CallLlmRequest,
        headers: Map<String, Json>,
        next: LlmExecutionNextFn,
        provider_error: Arc<Mutex<Option<FlowError>>>,
    ) -> switchyard_libsy::Result<()> {
        let routed = call.get_routed().clone();
        let target_name = routed.decision.selected_model().to_string();
        let is_routed_call = routed.decision.is_routed_call();
        let result = async {
            let metadata = routed.request.metadata.clone();
            let (target_protocol, request) =
                self.apply_target(&target_name, routed.request, headers, false)?;
            let response = match next(request).await {
                Ok(response) => response,
                Err(error) => {
                    if is_routed_call {
                        remember_provider_error(&provider_error, &error);
                    }
                    return Err(flow_to_client_error(error, &target_name));
                }
            };
            let response = decode_response(&self.translation, target_protocol, &response)
                .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
            Ok(LibsyResponse {
                llm_response: LlmResponse::Agg(response),
                metadata: Some(response_metadata(metadata.as_ref(), target_protocol)),
            })
        }
        .await
        .map_err(|source| LibsyError::client_call(target_name, source));
        call.respond(result)
    }

    async fn serve_stream_call(
        &self,
        call: CallLlmRequest,
        headers: Map<String, Json>,
        next: LlmStreamExecutionNextFn,
        tracker: StreamCloseTracker,
        provider_error: Arc<Mutex<Option<FlowError>>>,
    ) -> switchyard_libsy::Result<()> {
        let routed = call.get_routed().clone();
        let target_name = routed.decision.selected_model().to_string();
        let is_routed_call = routed.decision.is_routed_call();
        let result = async {
            let metadata = routed.request.metadata.clone();
            let (target_protocol, request) =
                self.apply_target(&target_name, routed.request, headers, true)?;
            let mut upstream = match next(request).await {
                Ok(upstream) => upstream,
                Err(error) => {
                    if is_routed_call {
                        remember_provider_error(&provider_error, &error);
                    }
                    return Err(flow_to_client_error(error, &target_name));
                }
            };
            let first = match upstream.next().await {
                Some(Ok(first)) => first,
                Some(Err(error)) => {
                    if is_routed_call {
                        remember_provider_error(&provider_error, &error);
                    }
                    return Err(flow_to_client_error(error, &target_name));
                }
                None => {
                    return Err(LlmClientError::InvalidResponse {
                        source: Box::new(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "provider returned an empty stream",
                        )),
                    });
                }
            };
            let response = provider_response_stream(
                upstream,
                wire_format(target_protocol),
                first,
                tracker,
                target_name.clone(),
            )?;
            Ok(LibsyResponse {
                llm_response: LlmResponse::Stream(response),
                metadata: Some(response_metadata(metadata.as_ref(), target_protocol)),
            })
        }
        .await
        .map_err(|source| LibsyError::client_call(target_name, source));
        call.respond(result)
    }

    fn libsy_request(
        &self,
        inbound: WireProtocol,
        original: &LlmRequest,
        streaming: bool,
    ) -> FlowResult<LibsyRequest> {
        let mut llm_request = decode_request(&self.translation, inbound, original)?;
        llm_request.stream = streaming;
        let mut metadata = relay_metadata(&string_headers(&original.headers));
        metadata.wire_format = Some(wire_format(inbound));
        Ok(LibsyRequest {
            llm_request,
            raw_request: Some(original.content.clone()),
            metadata: Some(metadata),
        })
    }

    fn apply_target(
        &self,
        target_name: &str,
        mut request: LibsyRequest,
        headers: Map<String, Json>,
        streaming: bool,
    ) -> Result<(WireProtocol, LlmRequest), LlmClientError> {
        let target =
            self.config
                .targets
                .get(target_name)
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("libsy selected unknown target {target_name:?}"),
                })?;
        request.llm_request.stream = streaming;
        let mut routed = encode_request(
            &self.translation,
            target.protocol,
            &request.llm_request,
            headers,
        )
        .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?;
        let object = routed.content.as_object_mut().ok_or_else(|| {
            LlmClientError::RequestEncoding("translated provider request is not an object".into())
        })?;
        object.insert("model".into(), Json::String(target.model.clone()));
        object.insert("stream".into(), Json::Bool(streaming));
        if let Some(target_headers) = self.target_headers.get(target_name) {
            routed.headers.extend(target_headers.clone());
        }
        routed.headers.insert(
            INTERNAL_DISPATCH_ROUTE_HEADER.into(),
            Json::String(target.protocol.label().into()),
        );
        routed.headers.insert(
            INTERNAL_DISPATCH_URL_HEADER.into(),
            Json::String(dispatch_url(&target.base_url, &target.endpoint)),
        );
        routed.headers.insert(
            INTERNAL_RETRY_AWARE_HEADER.into(),
            Json::String("true".into()),
        );
        Ok((target.protocol, routed))
    }

    fn finish_buffered(&self, inbound: WireProtocol, response: LibsyResponse) -> FlowResult<Json> {
        match response.llm_response {
            LlmResponse::Agg(response) => encode_response(&self.translation, inbound, &response),
            LlmResponse::Stream(_) => Err(FlowError::Internal(
                "libsy returned a stream for a buffered Relay request".into(),
            )),
        }
    }

    fn finish_stream(
        &self,
        inbound: WireProtocol,
        response: LibsyResponse,
        tracker: StreamCloseTracker,
    ) -> FlowResult<LlmJsonStream> {
        let source = response
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.wire_format)
            .ok_or_else(|| {
                FlowError::Internal("libsy streaming response omitted its source format".into())
            })?;
        match response.llm_response {
            LlmResponse::Stream(response) => Ok(relay_response_stream(
                response,
                source,
                wire_format(inbound),
                tracker,
            )),
            LlmResponse::Agg(_) => Err(FlowError::Internal(
                "libsy returned a buffered aggregate for a streaming Relay request".into(),
            )),
        }
    }

    async fn dispatch_fallback_buffered(
        &self,
        inbound: WireProtocol,
        original: LlmRequest,
        next: LlmExecutionNextFn,
        reason: &str,
    ) -> FlowResult<Json> {
        self.emit_fallback(inbound, reason, &original);
        let request = self.fallback_request(inbound, original, false)?;
        next(request).await
    }

    async fn dispatch_fallback_stream(
        &self,
        inbound: WireProtocol,
        original: LlmRequest,
        next: LlmStreamExecutionNextFn,
        reason: &str,
    ) -> FlowResult<LlmJsonStream> {
        self.emit_fallback(inbound, reason, &original);
        let request = self.fallback_request(inbound, original, true)?;
        next(request).await
    }

    fn fallback_request(
        &self,
        inbound: WireProtocol,
        original: LlmRequest,
        streaming: bool,
    ) -> FlowResult<LlmRequest> {
        let target = self.config.default_targets.target(inbound);
        let request = self.libsy_request(inbound, &original, streaming)?;
        self.apply_target(
            target,
            request,
            dispatch_headers(&original.headers),
            streaming,
        )
        .map(|(_, request)| request)
        .map_err(|error| FlowError::Internal(error.to_string()))
    }

    fn emit_requested(&self, request: &LlmRequest, inbound: WireProtocol, attempt: u32) {
        emit_mark(
            "switchyard.routing.requested",
            json!({
                "algorithm": self.algorithm.name(),
                "routing_attempt": attempt,
                "inbound_profile": inbound.label(),
            }),
            identity_metadata(request),
        );
    }

    fn emit_decision(&self, decision: &dyn Decision, attempt: u32, metadata: Json) {
        emit_mark(
            "switchyard.routing.decision",
            json!({
                "algorithm": self.algorithm.name(),
                "routing_attempt": attempt,
                "semantic_target": decision.selected_model(),
                "routing_tier": decision.routing_tier(),
                "is_routed_call": decision.is_routed_call(),
                "reasoning": decision.reasoning(),
            }),
            metadata,
        );
    }

    fn emit_call(&self, decision: &dyn Decision, attempt: u32, metadata: Json) {
        emit_mark(
            "switchyard.routing.call",
            json!({
                "algorithm": self.algorithm.name(),
                "routing_attempt": attempt,
                "semantic_target": decision.selected_model(),
                "routing_tier": decision.routing_tier(),
                "is_routed_call": decision.is_routed_call(),
                "reasoning": decision.reasoning(),
            }),
            metadata,
        );
    }

    fn emit_retry(&self, attempt: u32, reason: &str, request: &LlmRequest) {
        emit_mark(
            "switchyard.routing.retry",
            json!({
                "algorithm": self.algorithm.name(),
                "routing_attempt": attempt,
                "retry_reason": reason,
            }),
            identity_metadata(request),
        );
    }

    fn emit_error(&self, attempt: u32, class: &str, error: &str, request: &LlmRequest) {
        emit_mark(
            "switchyard.routing.error",
            json!({
                "algorithm": self.algorithm.name(),
                "routing_attempt": attempt,
                "error_class": class,
                "error": error,
            }),
            identity_metadata(request),
        );
    }

    fn emit_fallback(&self, inbound: WireProtocol, reason: &str, request: &LlmRequest) {
        emit_mark(
            "switchyard.routing.fallback",
            json!({
                "algorithm": self.algorithm.name(),
                "fallback_reason": reason,
                "fallback_route": self.config.default_targets.target(inbound),
                "inbound_profile": inbound.label(),
            }),
            identity_metadata(request),
        );
    }
}

struct RunFailure {
    error: LibsyError,
    provider_error: Option<FlowError>,
}

impl RunFailure {
    fn new(error: LibsyError, provider_error: &Arc<Mutex<Option<FlowError>>>) -> Self {
        Self {
            error,
            provider_error: provider_error.lock().ok().and_then(|error| error.clone()),
        }
    }

    fn is_retryable(&self) -> bool {
        self.provider_error
            .as_ref()
            .is_some_and(flow_error_is_retryable)
            || libsy_error_is_retryable(&self.error)
    }
}

fn build_algorithm(config: &SwitchyardConfig) -> Result<Arc<dyn Algorithm>, String> {
    let target = |name: &str| {
        config
            .targets
            .contains_key(name)
            .then(|| LlmTarget {
                semantic_name: name.to_string(),
                llm_client: None,
            })
            .ok_or_else(|| format!("algorithm target {name:?} is not configured"))
    };
    match &config.algorithm {
        AlgorithmConfig::Random { seed } => {
            let targets = config
                .targets
                .keys()
                .map(|name| target(name))
                .collect::<Result<Vec<_>, _>>()?;
            let weights = config
                .targets
                .values()
                .map(|target| target.weight)
                .collect::<Vec<_>>();
            Random::new(LlmTargetSet::new(targets), Some(weights), *seed)
                .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
                .map_err(|error| error.to_string())
        }
        AlgorithmConfig::LlmClassifier {
            classifier_target,
            weak_target,
            strong_target,
            base_threshold,
            min_confidence,
            capability_elevated_floor,
            session_affinity,
            message_hash_fallback,
        } => {
            let classifier = target(classifier_target)?;
            if config.targets[classifier_target].protocol == WireProtocol::AnthropicMessages {
                return Err(format!(
                    "classifier target {classifier_target:?} must use openai_chat or openai_responses because the pinned lossless translation policy cannot encode its structured response format for anthropic_messages"
                ));
            }
            LlmTaskClassifier::new(
                classifier,
                target(weak_target)?,
                target(strong_target)?,
                TaskClassifierConfig {
                    base_threshold: *base_threshold,
                    min_confidence: *min_confidence,
                    capability_elevated_floor: *capability_elevated_floor,
                    session_affinity: *session_affinity,
                    message_hash_fallback: *message_hash_fallback,
                    recent_turn_window: None,
                },
            )
            .map(|algorithm| Arc::new(algorithm) as Arc<dyn Algorithm>)
            .map_err(|error| error.to_string())
        }
    }
}

fn validate_config(config: &SwitchyardConfig) -> Result<(), String> {
    if config.version != 2 {
        return Err(format!(
            "unsupported Switchyard config version {}; version 1 used the removed switchyard-server Decision API, use version = 2",
            config.version
        ));
    }
    if config.max_retries > 10 {
        return Err("max_retries must not exceed 10".into());
    }
    if config.targets.is_empty() {
        return Err("targets must not be empty".into());
    }
    if config.enabled_inbound_profiles.is_empty() {
        return Err("enabled_inbound_profiles must not be empty".into());
    }
    for (name, target) in &config.targets {
        if name.trim().is_empty() || target.model.trim().is_empty() {
            return Err("target names and models must be non-empty".into());
        }
        if target.endpoint != target.protocol.endpoint() {
            return Err(format!(
                "target {name:?} endpoint must be {:?} for {}",
                target.protocol.endpoint(),
                target.protocol.label()
            ));
        }
        validate_http_url(&target.base_url, name)?;
        if !target.weight.is_finite() || target.weight < 0.0 {
            return Err(format!(
                "target {name:?} weight must be finite and nonnegative"
            ));
        }
    }
    for &protocol in &config.enabled_inbound_profiles {
        let fallback = config.default_targets.target(protocol);
        let target = config
            .targets
            .get(fallback)
            .ok_or_else(|| format!("default target {fallback:?} is not configured"))?;
        if target.protocol != protocol {
            return Err(format!(
                "default target {fallback:?} must use protocol {}",
                protocol.label()
            ));
        }
    }
    build_algorithm(config).map(|_| ())
}

fn validate_http_url(url: &str, target: &str) -> Result<(), String> {
    let uri = url
        .parse::<Uri>()
        .map_err(|error| format!("target {target:?} base_url is invalid: {error}"))?;
    if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.authority().is_none() {
        return Err(format!("target {target:?} base_url must use http or https"));
    }
    Ok(())
}

fn resolve_json_headers(
    static_headers: &BTreeMap<String, String>,
    environment_headers: &BTreeMap<String, String>,
) -> Result<Map<String, Json>, String> {
    let mut headers = Map::new();
    for (name, value) in static_headers {
        validate_target_header(name, value)?;
        headers.insert(name.clone(), Json::String(value.clone()));
    }
    for (name, variable) in environment_headers {
        if static_headers
            .keys()
            .any(|configured| configured.eq_ignore_ascii_case(name))
        {
            return Err(format!(
                "target header {name:?} cannot appear in both headers and header_env"
            ));
        }
        let value = std::env::var(variable)
            .map_err(|_| format!("environment variable {variable:?} is not set"))?;
        if value.trim().is_empty() {
            return Err(format!("environment variable {variable:?} is blank"));
        }
        validate_target_header(name, &value)?;
        headers.insert(name.clone(), Json::String(value));
    }
    Ok(headers)
}

fn validate_target_header(name: &str, value: &str) -> Result<(), String> {
    let normalized = name.to_ascii_lowercase();
    if matches!(
        normalized.as_str(),
        INTERNAL_DISPATCH_URL_HEADER | INTERNAL_DISPATCH_ROUTE_HEADER | INTERNAL_RETRY_AWARE_HEADER
    ) {
        return Err(format!(
            "target header {name:?} is reserved for Relay dispatch"
        ));
    }
    HeaderName::from_bytes(name.as_bytes())
        .map_err(|error| format!("invalid target header name {name:?}: {error}"))?;
    HeaderValue::from_str(value)
        .map_err(|error| format!("invalid target header value for {name:?}: {error}"))?;
    Ok(())
}

fn response_metadata(source: Option<&Metadata>, protocol: WireProtocol) -> Metadata {
    let mut metadata = source.cloned().unwrap_or_default();
    metadata.wire_format = Some(wire_format(protocol));
    metadata
}

fn context_values(metadata: Option<&Metadata>, attempt: u32) -> BTreeMap<String, String> {
    let mut values = BTreeMap::from([
        ("relay.routing_attempt".into(), attempt.to_string()),
        ("relay.scope_id".into(), task_scope_top().uuid.to_string()),
    ]);
    if let Some(metadata) = metadata {
        for (key, value) in [
            ("relay.session_id", metadata.session_id.as_deref()),
            ("relay.agent_id", metadata.agent_id.as_deref()),
            ("relay.turn_id", metadata.turn_id.as_deref()),
            ("relay.correlation_id", metadata.correlation_id.as_deref()),
        ] {
            if let Some(value) = value {
                values.insert(key.into(), value.into());
            }
        }
    }
    values
}

fn string_headers(headers: &Map<String, Json>) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| value.as_str().map(|value| (name.clone(), value.into())))
        .collect()
}

fn dispatch_headers(headers: &Map<String, Json>) -> Map<String, Json> {
    headers
        .iter()
        .filter(|(name, _)| {
            let normalized = name.to_ascii_lowercase();
            !matches!(
                normalized.as_str(),
                "authorization" | "proxy-authorization" | "x-api-key" | "api-key"
            ) && !normalized.starts_with("x-nemo-relay-")
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn relay_metadata(headers: &BTreeMap<String, String>) -> Metadata {
    let mut metadata = Metadata::from_headers(headers);
    if metadata.turn_id.is_none() {
        metadata.turn_id = header_value(headers, RELAY_TURN_ID_HEADER).map(ToOwned::to_owned);
    }
    if metadata.correlation_id.is_none() {
        metadata.correlation_id =
            header_value(headers, RELAY_REQUEST_ID_HEADER).map(ToOwned::to_owned);
    }
    metadata
}

fn libsy_identity_metadata(metadata: Option<&Metadata>) -> Json {
    json!({
        "session_id": metadata.and_then(|value| value.session_id.as_deref()),
        "agent_id": metadata.and_then(|value| value.agent_id.as_deref()),
        "turn_id": metadata.and_then(|value| value.turn_id.as_deref()),
        "request_id": metadata.and_then(|value| value.correlation_id.as_deref()),
    })
}

fn header_value<'a>(headers: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
        .filter(|value| !value.trim().is_empty())
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

fn remember_provider_error(slot: &Arc<Mutex<Option<FlowError>>>, error: &FlowError) {
    if let Ok(mut stored) = slot.lock() {
        *stored = Some(error.clone());
    }
}

fn flow_error_is_retryable(error: &FlowError) -> bool {
    matches!(error, FlowError::Upstream(failure) if failure.is_retryable())
}

fn libsy_error_is_retryable(error: &LibsyError) -> bool {
    match error {
        LibsyError::ClientCall { source, .. } => match source {
            LlmClientError::Transport { .. }
            | LlmClientError::Timeout { .. }
            | LlmClientError::ContextWindowExceeded { .. } => true,
            LlmClientError::UpstreamHttp { status, .. } => {
                matches!(*status, 408 | 409 | 425 | 429) || *status >= 500
            }
            _ => false,
        },
        _ => false,
    }
}

fn identity_metadata(request: &LlmRequest) -> Json {
    let headers = string_headers(&request.headers);
    let metadata = relay_metadata(&headers);
    json!({
        "session_id": metadata.session_id,
        "agent_id": metadata.agent_id,
        "turn_id": metadata.turn_id,
        "request_id": metadata.correlation_id,
    })
}

fn emit_mark(name: &str, data: Json, metadata: Json) {
    if let Err(error) = event(
        EmitMarkEventParams::builder()
            .name(name)
            .data(data)
            .data_schema(
                DataSchema::builder()
                    .name(ROUTING_MARK_SCHEMA)
                    .version("2")
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

#[cfg(test)]
#[path = "../tests/unit/component_tests.rs"]
mod tests;
