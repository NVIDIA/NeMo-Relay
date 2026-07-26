// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nemo_relay::api::event::Event;
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    BuiltinLlmCodec, EventSanitizeFn, LlmCodecIdentity, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, ToolSanitizeFn,
};
use nemo_relay::codec::resolve::{
    ProviderSurface, detect_response_surface, request_codec as build_request_codec,
    response_codec as build_response_codec,
};
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::plugin::{
    InferenceProvider, PluginError, PluginRegistrationContext, Result as PluginResult,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use super::component::{
    DEFAULT_LOCAL_MODEL_LATENCY_MS, DEFAULT_LOCAL_MODEL_MIN_SCORE, LocalBackendConfig,
    MAX_LOCAL_MODEL_EXCLUDED_LABELS, MAX_LOCAL_MODEL_LABEL_BYTES, MAX_LOCAL_MODEL_LATENCY_MS,
    MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES, MAX_LOCAL_MODEL_REPLACEMENT_BYTES,
    MAX_LOCAL_MODEL_TARGET_PATH_BYTES, MAX_LOCAL_MODEL_TARGET_PATHS,
    PII_DETECTION_PROVIDER_CONTRACT, PiiRedactionConfig, is_valid_json_pointer,
    is_valid_json_pointer_pattern, profile_registration_prefix,
};
use super::overlay::BuiltinCodecName;

const LOCAL_MODEL_CONTRACT_VERSION: u32 = 1;
const MAX_BATCH_ITEMS: usize = 64;
const MAX_BATCH_BYTES: usize = 64 * 1024;
const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_TEXTS_PER_PAYLOAD: usize = 256;
const MAX_PAYLOAD_TEXT_BYTES: usize = 256 * 1024;
const MAX_DETECTIONS_PER_TEXT: usize = 128;

#[derive(Clone)]
struct CompiledLocalBackend {
    provider_name: Arc<String>,
    provider: InferenceProvider,
    model_id: Option<String>,
    detector_profile: Option<String>,
    target_paths: Arc<HashSet<Vec<String>>>,
    target_path_patterns: Arc<Vec<JsonPointerPattern>>,
    min_score: f64,
    excluded_labels: Arc<HashSet<String>>,
    replacement: Arc<String>,
    timeout: Duration,
    legacy_surface: Option<ProviderSurface>,
}

#[derive(Clone)]
struct JsonPointerPattern {
    segments: Vec<String>,
}

impl JsonPointerPattern {
    fn compile(pattern: String) -> Self {
        Self {
            segments: compile_json_pointer(pattern),
        }
    }

    fn matches(&self, path: &[String]) -> bool {
        self.segments.len() == path.len()
            && self
                .segments
                .iter()
                .zip(path)
                .all(|(pattern, segment)| pattern == "*" || pattern == segment)
    }
}

#[derive(Serialize)]
struct LocalModelRequest<'a> {
    version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detector_profile: Option<&'a str>,
    texts: Vec<LocalModelText<'a>>,
}

#[derive(Serialize)]
struct LocalModelText<'a> {
    id: u32,
    text: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalModelResponse {
    version: u32,
    detections: Vec<LocalModelDetection>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LocalModelDetection {
    text_id: u32,
    start_utf8: usize,
    end_utf8: usize,
    label: String,
    score: f64,
}

struct SelectedText {
    text: String,
    eligible: bool,
}

enum EventField {
    Data,
    CategoryProfile,
    Metadata,
}

impl CompiledLocalBackend {
    fn new(
        config: LocalBackendConfig,
        codec_name: Option<String>,
        ctx: &PluginRegistrationContext,
    ) -> PluginResult<Self> {
        if let Some(violation) = validate_local_backend_config(&config).into_iter().next() {
            return Err(PluginError::InvalidConfig(violation.message));
        }
        let provider_name = config
            .backend
            .as_deref()
            .map(str::trim)
            .expect("validated local backend has a provider name")
            .to_string();
        let min_score = config.min_score.unwrap_or(DEFAULT_LOCAL_MODEL_MIN_SCORE);
        let replacement = config
            .replacement
            .unwrap_or_else(|| "[REDACTED]".to_string());
        let max_latency_ms = config
            .max_latency_ms
            .unwrap_or(DEFAULT_LOCAL_MODEL_LATENCY_MS);
        let surface = match codec_name.as_deref() {
            Some(name) => Some(ProviderSurface::from_codec_name(name).ok_or_else(|| {
                PluginError::InvalidConfig(format!("unsupported codec '{name}'"))
            })?),
            None => None,
        };
        let provider = ctx
            .inference_provider(&provider_name, PII_DETECTION_PROVIDER_CONTRACT)
            .map_err(|error| {
                PluginError::RegistrationFailed(format!(
                    "PII redaction inference provider '{provider_name}' is unavailable: {error}"
                ))
            })?;
        Ok(Self {
            provider_name: Arc::new(provider_name),
            provider,
            model_id: config.model_id.map(|value| value.trim().to_string()),
            detector_profile: config
                .detector_profile
                .map(|value| value.trim().to_string()),
            target_paths: Arc::new(
                config
                    .target_paths
                    .into_iter()
                    .map(compile_json_pointer)
                    .collect(),
            ),
            target_path_patterns: Arc::new(
                config
                    .target_path_patterns
                    .into_iter()
                    .map(JsonPointerPattern::compile)
                    .collect(),
            ),
            min_score,
            excluded_labels: Arc::new(
                config
                    .excluded_labels
                    .into_iter()
                    .map(|label| label.trim().to_string())
                    .collect(),
            ),
            replacement: Arc::new(replacement),
            timeout: Duration::from_millis(max_latency_ms),
            legacy_surface: surface,
        })
    }

    fn sanitize_json(&self, value: Json) -> Json {
        self.sanitize_json_values(vec![value])
            .pop()
            .expect("single-value sanitization returns one value")
    }

    fn sanitize_json_values(&self, values: Vec<Json>) -> Vec<Json> {
        self.sanitize_json_roots(
            values
                .into_iter()
                .map(|value| (Vec::new(), value))
                .collect(),
        )
    }

    fn sanitize_json_roots(&self, mut roots: Vec<(Vec<String>, Json)>) -> Vec<Json> {
        let mut texts = Vec::new();
        let mut total_bytes = 0;
        let mut within_budget = true;
        for (path, value) in &mut roots {
            self.collect_strings(
                value,
                path,
                &mut texts,
                &mut total_bytes,
                &mut within_budget,
            );
        }
        let sanitized = self.sanitize_texts(texts);
        let mut index = 0;
        for (path, value) in &mut roots {
            self.replace_strings(value, path, &sanitized, &mut index);
        }
        roots.into_iter().map(|(_, value)| value).collect()
    }

    fn collect_strings(
        &self,
        value: &Json,
        path: &mut Vec<String>,
        texts: &mut Vec<SelectedText>,
        total_bytes: &mut usize,
        within_budget: &mut bool,
    ) {
        match value {
            Json::String(text) if self.matches_path(path) && *within_budget => {
                if texts.len() >= MAX_TEXTS_PER_PAYLOAD {
                    *within_budget = false;
                    return;
                }
                if text.len() > MAX_TEXT_BYTES {
                    texts.push(SelectedText {
                        text: self.replacement.as_str().to_string(),
                        eligible: false,
                    });
                    return;
                }
                let Some(next_total) = total_bytes.checked_add(text.len()) else {
                    *within_budget = false;
                    return;
                };
                if next_total > MAX_PAYLOAD_TEXT_BYTES {
                    *within_budget = false;
                    return;
                }
                *total_bytes = next_total;
                texts.push(SelectedText {
                    text: text.clone(),
                    eligible: true,
                });
            }
            Json::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    self.collect_strings(item, path, texts, total_bytes, within_budget);
                    path.pop();
                }
            }
            Json::Object(fields) => {
                for (key, value) in fields {
                    path.push(super::builtin::escape_json_pointer_segment(key));
                    self.collect_strings(value, path, texts, total_bytes, within_budget);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn replace_strings(
        &self,
        value: &mut Json,
        path: &mut Vec<String>,
        sanitized: &[String],
        index: &mut usize,
    ) {
        match value {
            Json::String(text) if self.matches_path(path) => {
                if let Some(replacement) = sanitized.get(*index) {
                    *text = replacement.clone();
                } else {
                    *text = self.replacement.as_str().to_string();
                }
                *index += 1;
            }
            Json::Array(items) => {
                for (item_index, item) in items.iter_mut().enumerate() {
                    path.push(item_index.to_string());
                    self.replace_strings(item, path, sanitized, index);
                    path.pop();
                }
            }
            Json::Object(fields) => {
                for (key, value) in fields {
                    path.push(super::builtin::escape_json_pointer_segment(key));
                    self.replace_strings(value, path, sanitized, index);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn matches_path(&self, path: &[String]) -> bool {
        (self.target_paths.is_empty() && self.target_path_patterns.is_empty())
            || self.target_paths.contains(path)
            || self
                .target_path_patterns
                .iter()
                .any(|pattern| pattern.matches(path))
    }

    fn sanitize_texts(&self, mut texts: Vec<SelectedText>) -> Vec<String> {
        let eligible = texts
            .iter()
            .enumerate()
            .filter_map(|(index, text)| text.eligible.then_some(index))
            .collect::<Vec<_>>();

        let mut cursor = 0;
        let deadline = Instant::now() + self.timeout;
        while cursor < eligible.len() {
            let start = cursor;
            let mut batch_bytes = 0;
            while cursor < eligible.len() && cursor - start < MAX_BATCH_ITEMS {
                let next_bytes = texts[eligible[cursor]].text.len();
                if cursor > start && batch_bytes + next_bytes > MAX_BATCH_BYTES {
                    break;
                }
                batch_bytes += next_bytes;
                cursor += 1;
            }
            let batch = &eligible[start..cursor];
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                for index in &eligible[start..] {
                    texts[*index].text = self.replacement.as_str().to_string();
                }
                break;
            }
            match self.sanitize_batch(&texts, batch, remaining) {
                Ok(replacements) => {
                    for (index, replacement) in replacements {
                        texts[index].text = replacement;
                    }
                }
                Err(_) => {
                    log::warn!(
                        target: "nemo_relay.plugin",
                        event = "local_model_provider_failed",
                        plugin_kind = "pii_redaction",
                        provider = self.provider_name.as_str(),
                        batch_size = batch.len(),
                        reason = "provider_or_response";
                        "PII local-model provider failed closed"
                    );
                    for index in batch {
                        texts[*index].text = self.replacement.as_str().to_string();
                    }
                }
            }
        }
        texts.into_iter().map(|selected| selected.text).collect()
    }

    fn sanitize_batch(
        &self,
        texts: &[SelectedText],
        batch: &[usize],
        timeout: Duration,
    ) -> PluginResult<Vec<(usize, String)>> {
        let request = LocalModelRequest {
            version: LOCAL_MODEL_CONTRACT_VERSION,
            model_id: self.model_id.as_deref(),
            detector_profile: self.detector_profile.as_deref(),
            texts: batch
                .iter()
                .map(|index| LocalModelText {
                    id: u32::try_from(*index).expect("bounded text index fits u32"),
                    text: &texts[*index].text,
                })
                .collect(),
        };
        let request = serde_json::to_value(request)?;
        let response = self.provider.invoke(request, timeout)?;
        let response: LocalModelResponse = serde_json::from_value(response).map_err(|error| {
            PluginError::RegistrationFailed(format!(
                "local-model provider returned an invalid detection response: {error}"
            ))
        })?;
        self.apply_response(texts, batch, response)
    }

    fn apply_response(
        &self,
        texts: &[SelectedText],
        batch: &[usize],
        response: LocalModelResponse,
    ) -> PluginResult<Vec<(usize, String)>> {
        if response.version != LOCAL_MODEL_CONTRACT_VERSION {
            return Err(PluginError::RegistrationFailed(format!(
                "unsupported local-model response version {}",
                response.version
            )));
        }
        if response.detections.len() > batch.len() * MAX_DETECTIONS_PER_TEXT {
            return Err(PluginError::RegistrationFailed(
                "local-model response exceeded the detection limit".into(),
            ));
        }
        let allowed_ids = batch
            .iter()
            .map(|index| u32::try_from(*index).expect("bounded text index fits u32"))
            .collect::<HashSet<_>>();
        let mut detections = HashMap::<u32, Vec<LocalModelDetection>>::new();
        for detection in response.detections {
            if !allowed_ids.contains(&detection.text_id) {
                return Err(PluginError::RegistrationFailed(format!(
                    "local-model response referenced unknown text id {}",
                    detection.text_id
                )));
            }
            if detection.label.trim().is_empty() || detection.label.len() > 128 {
                return Err(PluginError::RegistrationFailed(
                    "local-model response contained an invalid detection label".into(),
                ));
            }
            if !detection.score.is_finite() || !(0.0..=1.0).contains(&detection.score) {
                return Err(PluginError::RegistrationFailed(
                    "local-model response contained an invalid detection score".into(),
                ));
            }
            let text_detections = detections.entry(detection.text_id).or_default();
            if text_detections.len() >= MAX_DETECTIONS_PER_TEXT {
                return Err(PluginError::RegistrationFailed(format!(
                    "local-model response exceeded the per-text detection limit of {MAX_DETECTIONS_PER_TEXT}"
                )));
            }
            text_detections.push(detection);
        }
        let mut replacements = Vec::new();
        for index in batch {
            let id = u32::try_from(*index).expect("bounded text index fits u32");
            let Some(mut spans) = detections.remove(&id) else {
                continue;
            };
            spans.sort_by_key(|span| (span.start_utf8, span.end_utf8));
            let text = &texts[*index].text;
            let mut previous_end = 0;
            for span in &spans {
                if span.start_utf8 >= span.end_utf8
                    || span.end_utf8 > text.len()
                    || !text.is_char_boundary(span.start_utf8)
                    || !text.is_char_boundary(span.end_utf8)
                    || span.start_utf8 < previous_end
                {
                    return Err(PluginError::RegistrationFailed(
                        "local-model response contained invalid or overlapping UTF-8 spans".into(),
                    ));
                }
                previous_end = span.end_utf8;
            }
            spans.retain(|detection| {
                detection.score >= self.min_score
                    && !self.excluded_labels.contains(&detection.label)
            });
            if spans.is_empty() {
                continue;
            }
            let mut redacted = text.clone();
            for span in spans.iter().rev() {
                redacted.replace_range(span.start_utf8..span.end_utf8, self.replacement.as_str());
            }
            replacements.push((*index, redacted));
        }
        Ok(replacements)
    }

    fn sanitize_request_with_codec(
        &self,
        codec: &dyn LlmCodec,
        request: &LlmRequest,
    ) -> Option<LlmRequest> {
        let annotated = codec.decode(request).ok()?;
        let annotated = serde_json::to_value(annotated).ok()?;
        let (headers, annotated) =
            self.sanitize_request_parts(request.headers.clone(), annotated)?;
        let annotated = serde_json::from_value(annotated).ok()?;
        let mut encoded = codec.encode(&annotated, request).ok()?;
        encoded.headers = headers;
        Some(encoded)
    }

    fn sanitize_raw_request(&self, mut request: LlmRequest) -> Option<LlmRequest> {
        let headers = std::mem::take(&mut request.headers);
        let content = std::mem::take(&mut request.content);
        let (headers, content) = self.sanitize_request_parts(headers, content)?;
        request.headers = headers;
        request.content = content;
        Some(request)
    }

    fn sanitize_request_parts(
        &self,
        headers: Map<String, Json>,
        content: Json,
    ) -> Option<(Map<String, Json>, Json)> {
        let mut values = self.sanitize_json_roots(vec![
            (vec!["headers".to_string()], Json::Object(headers)),
            (Vec::new(), content),
        ]);
        let content = values.pop()?;
        let headers = values.pop()?.as_object()?.clone();
        Some((headers, content))
    }

    fn sanitize_response_with_codec(
        &self,
        codec: &dyn LlmResponseCodec,
        surface: ProviderSurface,
        payload: Json,
    ) -> Option<Json> {
        let codec_name = BuiltinCodecName::from_provider_surface(surface);
        let annotated = codec.decode_response(&payload).ok()?;
        let sanitized = sanitize_serializable(self, annotated).ok()?;
        Some(codec_name.overlay_response_payload(payload, &sanitized))
    }

    fn selected_surface(&self, codec: &LlmCodecIdentity) -> Option<ProviderSurface> {
        match codec {
            LlmCodecIdentity::None => self.legacy_surface,
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat) => {
                Some(ProviderSurface::OpenAIChat)
            }
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiResponses) => {
                Some(ProviderSurface::OpenAIResponses)
            }
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::AnthropicMessages) => {
                Some(ProviderSurface::AnthropicMessages)
            }
            LlmCodecIdentity::Runtime(_) | LlmCodecIdentity::Opaque => None,
        }
    }

    fn uses_compatible_legacy_response_codec(&self, payload: &Json) -> bool {
        self.legacy_surface
            .is_some_and(|surface| detect_response_surface(payload) == Some(surface))
    }

    fn log_codec_failure(&self, direction: &'static str, codec: &LlmCodecIdentity, reason: &str) {
        let codec_kind = match codec {
            LlmCodecIdentity::None => "none",
            LlmCodecIdentity::BuiltIn(_) => "builtin",
            LlmCodecIdentity::Runtime(_) => "runtime",
            LlmCodecIdentity::Opaque => "opaque",
        };
        log::warn!(
            target: "nemo_relay.plugin",
            event = "local_model_codec_failed",
            plugin_kind = "pii_redaction",
            provider = self.provider_name.as_str(),
            direction,
            codec_kind,
            reason;
            "PII local-model payload omitted after codec failure"
        );
    }
}

fn compile_json_pointer(pointer: String) -> Vec<String> {
    pointer.strip_prefix('/').map_or_else(Vec::new, |path| {
        path.split('/').map(str::to_string).collect()
    })
}

pub(super) fn register_local_backend(
    config: PiiRedactionConfig,
    ctx: &mut PluginRegistrationContext,
    profile_name: Option<&str>,
) -> PluginResult<()> {
    let local = config.local.clone().ok_or_else(|| {
        PluginError::InvalidConfig(
            "local settings are required when mode = 'local_model'".to_string(),
        )
    })?;
    let backend = CompiledLocalBackend::new(local, config.codec.clone(), ctx)?;

    if config.mark {
        ctx.register_mark_sanitize_guardrail(
            &registration_name(profile_name, "mark"),
            config.priority,
            event_sanitize_callback(backend.clone(), None),
        )?;
    }
    if config.tool_input {
        ctx.register_tool_sanitize_request_guardrail(
            &registration_name(profile_name, "tool_input"),
            config.priority,
            tool_sanitize_callback(backend.clone()),
        )?;
    }
    if config.tool_output {
        ctx.register_tool_sanitize_response_guardrail(
            &registration_name(profile_name, "tool_output"),
            config.priority,
            tool_sanitize_callback(backend.clone()),
        )?;
    }
    if config.input {
        ctx.register_llm_sanitize_request_guardrail(
            &registration_name(profile_name, "input"),
            config.priority,
            llm_sanitize_request_callback(backend.clone()),
        )?;
    }
    if config.input || config.tool_input {
        ctx.register_scope_sanitize_start_guardrail(
            &registration_name(
                profile_name,
                if profile_name.is_some() {
                    "scope_start"
                } else {
                    "input"
                },
            ),
            config.priority,
            event_sanitize_callback(backend.clone(), Some((config.input, config.tool_input))),
        )?;
    }
    if config.output {
        ctx.register_llm_sanitize_response_guardrail(
            &registration_name(profile_name, "output"),
            config.priority,
            llm_sanitize_response_callback(backend.clone()),
        )?;
    }
    if config.output || config.tool_output {
        ctx.register_scope_sanitize_end_guardrail(
            &registration_name(
                profile_name,
                if profile_name.is_some() {
                    "scope_end"
                } else {
                    "output"
                },
            ),
            config.priority,
            event_sanitize_callback(backend, Some((config.output, config.tool_output))),
        )?;
    }
    Ok(())
}

fn tool_sanitize_callback(backend: CompiledLocalBackend) -> ToolSanitizeFn {
    Arc::new(move |_name, payload| backend.sanitize_json(payload))
}

fn event_sanitize_callback(
    backend: CompiledLocalBackend,
    scope_categories: Option<(bool, bool)>,
) -> EventSanitizeFn {
    Arc::new(move |event, mut fields| {
        if scope_categories.is_some_and(|(sanitize_llm, sanitize_tool)| {
            matches!(event, Event::Scope(_))
                && event
                    .category()
                    .is_some_and(|category| match category.as_str() {
                        "llm" => !sanitize_llm,
                        "tool" => !sanitize_tool,
                        _ => false,
                    })
        }) {
            return fields;
        }
        let specialized_scope = matches!(event, Event::Scope(_))
            && event
                .category()
                .is_some_and(|category| matches!(category.as_str(), "tool" | "llm"));

        let mut selected = Vec::with_capacity(3);
        if !specialized_scope && let Some(data) = fields.data.take() {
            selected.push((EventField::Data, data));
        }
        if !specialized_scope
            && let Some(profile) = fields.category_profile.take()
            && let Ok(profile) = serde_json::to_value(profile)
        {
            selected.push((EventField::CategoryProfile, profile));
        }
        if let Some(metadata) = fields.metadata.take() {
            selected.push((EventField::Metadata, metadata));
        }

        let values = selected
            .iter_mut()
            .map(|(_, value)| std::mem::take(value))
            .collect();
        for ((field, _), value) in selected
            .into_iter()
            .zip(backend.sanitize_json_values(values))
        {
            match field {
                EventField::Data => fields.data = Some(value),
                EventField::CategoryProfile => {
                    fields.category_profile = serde_json::from_value(value).ok();
                }
                EventField::Metadata => fields.metadata = Some(value),
            }
        }
        fields
    })
}

fn llm_sanitize_request_callback(backend: CompiledLocalBackend) -> LlmSanitizeRequestFn {
    Arc::new(move |mut request, context| {
        if backend.target_paths.is_empty() && backend.target_path_patterns.is_empty() {
            request.content = backend.sanitize_json(request.content);
            return Some(request);
        }
        if matches!(context.codec(), LlmCodecIdentity::None) && backend.legacy_surface.is_none() {
            return backend.sanitize_raw_request(request);
        }
        let resolved = context.resolve_codec();
        let fallback = if resolved.is_none() {
            backend
                .selected_surface(context.codec())
                .map(build_request_codec)
        } else {
            None
        };
        let sanitized = resolved
            .as_deref()
            .or(fallback.as_deref())
            .and_then(|codec| backend.sanitize_request_with_codec(codec, &request));
        if sanitized.is_none() {
            backend.log_codec_failure(
                "request",
                context.codec(),
                "codec decode, sanitize, or encode failure",
            );
        }
        sanitized
    })
}

fn llm_sanitize_response_callback(backend: CompiledLocalBackend) -> LlmSanitizeResponseFn {
    Arc::new(move |payload, context| {
        if backend.target_paths.is_empty() && backend.target_path_patterns.is_empty() {
            return Some(backend.sanitize_json(payload));
        }
        if matches!(context.codec(), LlmCodecIdentity::None) && backend.legacy_surface.is_none() {
            return Some(backend.sanitize_json(payload));
        }
        if matches!(context.codec(), LlmCodecIdentity::None)
            && !backend.uses_compatible_legacy_response_codec(&payload)
        {
            backend.log_codec_failure("response", context.codec(), "no compatible legacy codec");
            return None;
        }
        let surface = backend.selected_surface(context.codec());
        let resolved = context.resolve_codec();
        let fallback = if resolved.is_none() {
            surface.map(build_response_codec)
        } else {
            None
        };
        let sanitized = surface
            .zip(resolved.as_deref().or(fallback.as_deref()))
            .and_then(|(surface, codec)| {
                backend.sanitize_response_with_codec(codec, surface, payload)
            });
        if sanitized.is_none() {
            backend.log_codec_failure(
                "response",
                context.codec(),
                "codec decode, sanitize, or encode failure",
            );
        }
        sanitized
    })
}

fn sanitize_serializable<T>(backend: &CompiledLocalBackend, value: T) -> PluginResult<T>
where
    T: Serialize + DeserializeOwned,
{
    let value = serde_json::to_value(value)?;
    serde_json::from_value(backend.sanitize_json(value)).map_err(PluginError::from)
}

fn registration_name(profile_name: Option<&str>, callback_name: &str) -> String {
    profile_name.map_or_else(
        || callback_name.to_string(),
        |profile_name| {
            format!(
                "{}/{callback_name}",
                profile_registration_prefix(profile_name)
            )
        },
    )
}

pub(super) struct LocalConfigViolation {
    pub(super) field: &'static str,
    pub(super) message: String,
}

pub(super) fn validate_local_backend_config(
    config: &LocalBackendConfig,
) -> Vec<LocalConfigViolation> {
    let mut violations = Vec::new();
    let mut push = |field, message| violations.push(LocalConfigViolation { field, message });

    match config.backend.as_deref().map(str::trim) {
        None | Some("") => push(
            "local.backend",
            "local.backend is required when mode = 'local_model'".into(),
        ),
        Some(backend) if backend.len() > MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES => push(
            "local.backend",
            format!(
                "local.backend must not exceed {MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES} UTF-8 bytes"
            ),
        ),
        Some(_) => {}
    }
    if config.allow_network == Some(true) {
        push(
            "local.allow_network",
            "worker-backed local-model providers must not use network inference".into(),
        );
    }
    match config.max_latency_ms {
        Some(0) => push(
            "local.max_latency_ms",
            "local.max_latency_ms must be greater than zero".into(),
        ),
        Some(latency) if latency > MAX_LOCAL_MODEL_LATENCY_MS => push(
            "local.max_latency_ms",
            format!("local.max_latency_ms must not exceed {MAX_LOCAL_MODEL_LATENCY_MS}"),
        ),
        _ => {}
    }
    for (field, value) in [
        ("local.model_id", config.model_id.as_deref()),
        ("local.detector_profile", config.detector_profile.as_deref()),
    ] {
        if value.is_some_and(|value| value.trim().is_empty()) {
            push(field, format!("{field} must be a non-empty string"));
        } else if value.is_some_and(|value| value.len() > MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES) {
            push(
                field,
                format!(
                    "{field} must not exceed {MAX_LOCAL_MODEL_PROVIDER_VALUE_BYTES} UTF-8 bytes"
                ),
            );
        }
    }
    if config.target_paths.len() + config.target_path_patterns.len() > MAX_LOCAL_MODEL_TARGET_PATHS
    {
        push(
            "local.target_paths",
            format!(
                "local.target_paths and local.target_path_patterns must contain at most {MAX_LOCAL_MODEL_TARGET_PATHS} entries in total"
            ),
        );
    }
    if config
        .target_paths
        .iter()
        .any(|path| path.len() > MAX_LOCAL_MODEL_TARGET_PATH_BYTES)
    {
        push(
            "local.target_paths",
            format!(
                "local.target_paths entries must not exceed {MAX_LOCAL_MODEL_TARGET_PATH_BYTES} UTF-8 bytes"
            ),
        );
    }
    if config
        .target_paths
        .iter()
        .any(|path| !is_valid_json_pointer(path))
    {
        push(
            "local.target_paths",
            "local.target_paths entries must be valid JSON pointers".into(),
        );
    }
    if config
        .target_path_patterns
        .iter()
        .any(|path| path.len() > MAX_LOCAL_MODEL_TARGET_PATH_BYTES)
    {
        push(
            "local.target_path_patterns",
            format!(
                "local.target_path_patterns entries must not exceed {MAX_LOCAL_MODEL_TARGET_PATH_BYTES} UTF-8 bytes"
            ),
        );
    }
    if config
        .target_path_patterns
        .iter()
        .any(|path| !is_valid_json_pointer_pattern(path))
    {
        push(
            "local.target_path_patterns",
            "local.target_path_patterns entries must be valid JSON-pointer patterns".into(),
        );
    }
    if config
        .min_score
        .is_some_and(|score| !score.is_finite() || !(0.0..=1.0).contains(&score))
    {
        push(
            "local.min_score",
            "local.min_score must be a finite number between 0 and 1".into(),
        );
    }
    if config.excluded_labels.len() > MAX_LOCAL_MODEL_EXCLUDED_LABELS {
        push(
            "local.excluded_labels",
            format!(
                "local.excluded_labels must contain at most {MAX_LOCAL_MODEL_EXCLUDED_LABELS} entries"
            ),
        );
    }
    if config
        .excluded_labels
        .iter()
        .any(|label| label.trim().is_empty() || label.len() > MAX_LOCAL_MODEL_LABEL_BYTES)
    {
        push(
            "local.excluded_labels",
            format!(
                "local.excluded_labels entries must be non-empty and at most {MAX_LOCAL_MODEL_LABEL_BYTES} UTF-8 bytes"
            ),
        );
    }
    if config
        .excluded_labels
        .iter()
        .map(|label| label.trim())
        .collect::<HashSet<_>>()
        .len()
        != config.excluded_labels.len()
    {
        push(
            "local.excluded_labels",
            "local.excluded_labels must not contain duplicates".into(),
        );
    }
    if config
        .replacement
        .as_ref()
        .is_some_and(|replacement| replacement.len() > MAX_LOCAL_MODEL_REPLACEMENT_BYTES)
    {
        push(
            "local.replacement",
            format!(
                "local.replacement must not exceed {MAX_LOCAL_MODEL_REPLACEMENT_BYTES} UTF-8 bytes"
            ),
        );
    }

    violations
}

#[cfg(test)]
#[path = "../tests/unit/local_tests.rs"]
mod tests;
