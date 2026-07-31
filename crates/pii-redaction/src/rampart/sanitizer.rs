// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Duration;

use nemo_relay::api::event::{Event, EventSanitizeFields};
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
use nemo_relay::error::Result as FlowResult;
use nemo_relay::plugin::{PluginError, Result as PluginResult};
use rayon::{ThreadPool, ThreadPoolBuilder};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as Json};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::builtin::escape_json_pointer_segment;
use crate::overlay::BuiltinCodecName;

use super::RampartPiiConfig;
use super::model::{Detection, RampartDetector};

const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_TEXTS_PER_PAYLOAD: usize = 256;
const MAX_PAYLOAD_TEXT_BYTES: usize = 256 * 1024;
// Cap CPU workers while respecting smaller hosts and container CPU quotas.
const MAX_CONCURRENT_INFERENCE: usize = 3;
// Bound admitted work and its wait so large payloads cannot build a long queue.
const MAX_ADMITTED_INFERENCE: usize = 16;
const MAX_ADMISSION_WAIT: Duration = Duration::from_millis(500);

fn inference_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(MAX_CONCURRENT_INFERENCE)
}

pub(super) trait DetectionModel: Send + Sync {
    fn detect(&self, texts: &[&str]) -> PluginResult<Vec<Detection>>;
}

impl DetectionModel for RampartDetector {
    fn detect(&self, texts: &[&str]) -> PluginResult<Vec<Detection>> {
        RampartDetector::detect(self, texts)
    }
}

#[derive(Clone)]
pub(super) struct RampartSanitizer {
    detector: Arc<dyn DetectionModel>,
    target_paths: Arc<HashSet<Vec<String>>>,
    target_path_patterns: Arc<Vec<JsonPointerPattern>>,
    min_score: f64,
    excluded_labels: Arc<HashSet<String>>,
    replacement: Arc<str>,
    legacy_surface: Option<ProviderSurface>,
    admission_capacity: Arc<Semaphore>,
    execution_admission: Arc<Semaphore>,
    executor: Arc<SanitizerExecutor>,
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

struct SelectedText {
    text: String,
    eligible: bool,
}

enum EventField {
    Data,
    CategoryProfile,
    Metadata,
}

struct SanitizerPermit {
    _admission: OwnedSemaphorePermit,
    _execution: OwnedSemaphorePermit,
    executor: Arc<SanitizerExecutor>,
}

struct SanitizerExecutor {
    pool: ThreadPool,
}

impl SanitizerExecutor {
    fn new(worker_count: usize) -> PluginResult<Self> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|worker| format!("nemo-relay-rampart-{worker}"))
            .build()
            .map_err(|error| {
                PluginError::Internal(format!(
                    "failed to start Rampart inference workers: {error}"
                ))
            })?;
        Ok(Self { pool })
    }

    fn submit(&self, job: impl FnOnce() + Send + 'static) {
        self.pool.spawn_fifo(job);
    }
}

impl RampartSanitizer {
    pub(super) fn new(
        config: RampartPiiConfig,
        detector: Arc<dyn DetectionModel>,
    ) -> PluginResult<Self> {
        let legacy_surface = match config.codec.as_deref() {
            Some(codec) => Some(ProviderSurface::from_codec_name(codec).ok_or_else(|| {
                PluginError::InvalidConfig(format!("unsupported Rampart PII codec '{codec}'"))
            })?),
            None => None,
        };
        let worker_count = inference_worker_count();
        let executor = Arc::new(SanitizerExecutor::new(worker_count)?);
        Ok(Self {
            detector,
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
            min_score: config.min_score,
            excluded_labels: Arc::new(config.excluded_labels.into_iter().collect()),
            replacement: config.replacement.into(),
            legacy_surface,
            admission_capacity: Arc::new(Semaphore::new(MAX_ADMITTED_INFERENCE)),
            execution_admission: Arc::new(Semaphore::new(worker_count)),
            executor,
        })
    }

    async fn admit(&self, surface: &'static str) -> Option<SanitizerPermit> {
        let admission = match Arc::clone(&self.admission_capacity).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.log_admission_failure(surface, "queue_full");
                return None;
            }
        };
        let execution = match tokio::time::timeout(
            MAX_ADMISSION_WAIT,
            Arc::clone(&self.execution_admission).acquire_owned(),
        )
        .await
        {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                self.log_admission_failure(surface, "closed");
                return None;
            }
            Err(_) => {
                self.log_admission_failure(surface, "timeout");
                return None;
            }
        };
        Some(SanitizerPermit {
            _admission: admission,
            _execution: execution,
            executor: Arc::clone(&self.executor),
        })
    }

    fn log_admission_failure(&self, surface: &'static str, reason: &'static str) {
        log::warn!(
            target: "nemo_relay.plugin",
            event = "rampart_pii_inference_failed",
            plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
            reason,
            surface;
            "Rampart PII sanitization failed closed during bounded admission"
        );
    }

    fn fail_closed_payload(&self) -> Json {
        Json::String(self.replacement.to_string())
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
        for (path, value) in &roots {
            let mut path = path.clone();
            self.collect_strings(
                value,
                &mut path,
                &mut texts,
                &mut total_bytes,
                &mut within_budget,
            );
        }

        let sanitized = self.sanitize_texts(texts);
        let mut index = 0;
        for (path, value) in &mut roots {
            let mut path = path.clone();
            self.replace_strings(value, &mut path, &sanitized, &mut index);
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
            Json::String(text) if self.matches_path(path) => {
                if !*within_budget || texts.len() >= MAX_TEXTS_PER_PAYLOAD {
                    *within_budget = false;
                    return;
                }
                if text.len() > MAX_TEXT_BYTES {
                    texts.push(SelectedText {
                        text: self.replacement.to_string(),
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
                    path.push(escape_json_pointer_segment(key));
                    self.collect_strings(value, path, texts, total_bytes, within_budget);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn has_selected_string(&self, value: &Json) -> bool {
        self.has_selected_string_at(value, &mut Vec::new())
    }

    fn has_selected_string_at(&self, value: &Json, path: &mut Vec<String>) -> bool {
        match value {
            Json::String(_) => self.matches_path(path),
            Json::Array(items) => items.iter().enumerate().any(|(index, item)| {
                path.push(index.to_string());
                let selected = self.has_selected_string_at(item, path);
                path.pop();
                selected
            }),
            Json::Object(fields) => fields.iter().any(|(key, value)| {
                path.push(escape_json_pointer_segment(key));
                let selected = self.has_selected_string_at(value, path);
                path.pop();
                selected
            }),
            _ => false,
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
                *text = sanitized
                    .get(*index)
                    .cloned()
                    .unwrap_or_else(|| self.replacement.to_string());
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
                    path.push(escape_json_pointer_segment(key));
                    self.replace_strings(value, path, sanitized, index);
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn matches_path(&self, path: &[String]) -> bool {
        self.target_paths.contains(path)
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
        if !eligible.is_empty() && self.sanitize_batch(&mut texts, &eligible).is_err() {
            log::warn!(
                target: "nemo_relay.plugin",
                event = "rampart_pii_inference_failed",
                plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
                selected_text_count = eligible.len(),
                reason = "model_or_output";
                "Rampart PII inference failed closed"
            );
            for index in eligible {
                texts[index].text = self.replacement.to_string();
            }
        }
        texts.into_iter().map(|selected| selected.text).collect()
    }

    fn sanitize_batch(&self, texts: &mut [SelectedText], batch: &[usize]) -> PluginResult<()> {
        let selected = batch
            .iter()
            .map(|index| texts[*index].text.as_str())
            .collect::<Vec<_>>();
        let detections = self.detector.detect(&selected)?;
        let mut by_text = vec![Vec::<Detection>::new(); batch.len()];
        for detection in detections {
            if detection.text_index >= batch.len()
                || !detection.score.is_finite()
                || !(0.0..=1.0).contains(&detection.score)
            {
                return Err(PluginError::Internal(
                    "Rampart returned an invalid detection".into(),
                ));
            }
            by_text[detection.text_index].push(detection);
        }

        for (original_index, mut detections) in batch.iter().copied().zip(by_text) {
            detections.retain(|detection| {
                detection.score >= self.min_score
                    && !self.excluded_labels.contains(&detection.label)
            });
            if detections.is_empty() {
                continue;
            }
            detections.sort_by_key(|detection| (detection.start_utf8, detection.end_utf8));
            let text = &texts[original_index].text;
            let mut previous_end = 0;
            for detection in &detections {
                if detection.start_utf8 >= detection.end_utf8
                    || detection.end_utf8 > text.len()
                    || !text.is_char_boundary(detection.start_utf8)
                    || !text.is_char_boundary(detection.end_utf8)
                    || detection.start_utf8 < previous_end
                {
                    return Err(PluginError::Internal(
                        "Rampart returned invalid or overlapping UTF-8 spans".into(),
                    ));
                }
                previous_end = detection.end_utf8;
            }
            let mut redacted = text.clone();
            for detection in detections.iter().rev() {
                redacted.replace_range(
                    detection.start_utf8..detection.end_utf8,
                    self.replacement.as_ref(),
                );
            }
            texts[original_index].text = redacted;
        }
        Ok(())
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
        let Json::Object(headers) = values.pop()? else {
            return None;
        };
        Some((headers, content))
    }

    fn sanitize_response_with_codec(
        &self,
        codec: &dyn LlmResponseCodec,
        surface: ProviderSurface,
        payload: Json,
    ) -> Option<Json> {
        if surface == ProviderSurface::OpenAIChat
            && payload
                .get("choices")
                .and_then(Json::as_array)
                .is_some_and(|choices| choices.len() > 1)
            && self.targets_normalized_openai_chat_choice()
        {
            return None;
        }
        let codec_name = BuiltinCodecName::from_provider_surface(surface);
        let annotated = codec.decode_response(&payload).ok()?;
        let sanitized = sanitize_serializable(self, annotated).ok()?;
        Some(codec_name.overlay_response_payload(payload, &sanitized))
    }

    fn targets_normalized_openai_chat_choice(&self) -> bool {
        const CHOICE_ROOTS: [&str; 4] = ["message", "tool_calls", "finish_reason", "api_specific"];

        self.target_paths
            .iter()
            .filter_map(|path| path.first())
            .chain(
                self.target_path_patterns
                    .iter()
                    .filter_map(|pattern| pattern.segments.first()),
            )
            .any(|root| root == "*" || CHOICE_ROOTS.contains(&root.as_str()))
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
            event = "rampart_pii_codec_failed",
            plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
            direction,
            codec_kind,
            reason;
            "Rampart PII payload omitted after codec failure"
        );
    }
}

pub(super) fn tool_sanitize_callback(backend: RampartSanitizer) -> ToolSanitizeFn {
    Arc::new(move |_name, payload| {
        let backend = backend.clone();
        if !backend.has_selected_string(&payload) {
            return Box::pin(async move { Ok(payload) });
        }
        let fallback = backend.fail_closed_payload();
        Box::pin(async move {
            let Some(permit) = backend.admit("tool").await else {
                return Ok(fallback);
            };
            run_inference("tool payload", permit, fallback, move || {
                backend.sanitize_json(payload)
            })
            .await
        })
    })
}

pub(super) fn event_sanitize_callback(
    backend: RampartSanitizer,
    scope_categories: Option<(bool, bool)>,
) -> EventSanitizeFn {
    Arc::new(move |event, mut fields| {
        let backend = backend.clone();
        if skips_event_sanitization(event.as_ref(), scope_categories) {
            return Box::pin(async move { Ok(fields) });
        }
        if !event_has_candidate_fields(event.as_ref(), &fields) {
            return Box::pin(async move { Ok(fields) });
        }
        if !event_fields_have_selected_strings(&backend, event.as_ref(), &fields) {
            return Box::pin(async move { Ok(fields) });
        }
        let fallback = fail_closed_event_fields(event.as_ref(), fields.clone());
        Box::pin(async move {
            let Some(permit) = backend.admit("event").await else {
                return Ok(fallback);
            };
            run_inference("event fields", permit, fallback, move || {
                let specialized_scope = is_specialized_scope(event.as_ref());

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
            .await
        })
    })
}

pub(super) fn llm_sanitize_request_callback(backend: RampartSanitizer) -> LlmSanitizeRequestFn {
    Arc::new(move |request, context| {
        let backend = backend.clone();
        Box::pin(async move {
            let Some(permit) = backend.admit("llm_request").await else {
                return Ok(None);
            };
            run_inference("LLM request", permit, None, move || {
                if matches!(context.codec(), LlmCodecIdentity::None)
                    && backend.legacy_surface.is_none()
                {
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
            .await
        })
    })
}

pub(super) fn llm_sanitize_response_callback(backend: RampartSanitizer) -> LlmSanitizeResponseFn {
    Arc::new(move |payload, context| {
        let backend = backend.clone();
        Box::pin(async move {
            let Some(permit) = backend.admit("llm_response").await else {
                return Ok(None);
            };
            run_inference("LLM response", permit, None, move || {
                if matches!(context.codec(), LlmCodecIdentity::None)
                    && backend.legacy_surface.is_none()
                {
                    return Some(backend.sanitize_json(payload));
                }
                if matches!(context.codec(), LlmCodecIdentity::None)
                    && !backend.uses_compatible_legacy_response_codec(&payload)
                {
                    backend.log_codec_failure(
                        "response",
                        context.codec(),
                        "no compatible legacy codec",
                    );
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
            .await
        })
    })
}

async fn run_inference<T>(
    target: &'static str,
    permit: SanitizerPermit,
    fallback: T,
    operation: impl FnOnce() -> T + Send + 'static,
) -> FlowResult<T>
where
    T: Send + 'static,
{
    let executor = Arc::clone(&permit.executor);
    let (sender, receiver) = tokio::sync::oneshot::channel();
    executor.submit(move || {
        let _permit = permit;
        let _ = sender.send(catch_unwind(AssertUnwindSafe(operation)));
    });
    match receiver.await {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(_)) => {
            log::error!(
                target: "nemo_relay.plugin",
                event = "rampart_pii_inference_failed",
                plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
                reason = "dedicated_executor_panic",
                target,
                panicked = true;
                "Rampart PII inference worker panicked and failed closed"
            );
            Ok(fallback)
        }
        Err(error) => {
            log::error!(
                target: "nemo_relay.plugin",
                event = "rampart_pii_inference_failed",
                plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
                reason = "dedicated_executor_result",
                target;
                "Rampart PII inference worker lost a result and failed closed: {error}"
            );
            Ok(fallback)
        }
    }
}

fn skips_event_sanitization(event: &Event, scope_categories: Option<(bool, bool)>) -> bool {
    scope_categories.is_some_and(|(sanitize_llm, sanitize_tool)| {
        matches!(event, Event::Scope(_))
            && event
                .category()
                .is_some_and(|category| match category.as_str() {
                    "llm" => !sanitize_llm,
                    "tool" => !sanitize_tool,
                    _ => false,
                })
    })
}

fn fail_closed_event_fields(event: &Event, mut fields: EventSanitizeFields) -> EventSanitizeFields {
    if is_specialized_scope(event) {
        fields.metadata = None;
        fields
    } else {
        EventSanitizeFields::default()
    }
}

fn event_has_candidate_fields(event: &Event, fields: &EventSanitizeFields) -> bool {
    if is_specialized_scope(event) {
        fields.metadata.is_some()
    } else {
        fields.data.is_some() || fields.category_profile.is_some() || fields.metadata.is_some()
    }
}

fn event_fields_have_selected_strings(
    backend: &RampartSanitizer,
    event: &Event,
    fields: &EventSanitizeFields,
) -> bool {
    if is_specialized_scope(event) {
        return fields
            .metadata
            .as_ref()
            .is_some_and(|metadata| backend.has_selected_string(metadata));
    }
    fields
        .data
        .as_ref()
        .is_some_and(|data| backend.has_selected_string(data))
        || fields
            .category_profile
            .as_ref()
            .and_then(|profile| serde_json::to_value(profile).ok())
            .is_some_and(|profile| backend.has_selected_string(&profile))
        || fields
            .metadata
            .as_ref()
            .is_some_and(|metadata| backend.has_selected_string(metadata))
}

fn is_specialized_scope(event: &Event) -> bool {
    matches!(event, Event::Scope(_))
        && event
            .category()
            .is_some_and(|category| matches!(category.as_str(), "tool" | "llm"))
}

fn compile_json_pointer(pointer: String) -> Vec<String> {
    pointer.strip_prefix('/').map_or_else(Vec::new, |path| {
        path.split('/').map(str::to_string).collect()
    })
}

fn sanitize_serializable<T>(backend: &RampartSanitizer, value: T) -> PluginResult<T>
where
    T: Serialize + DeserializeOwned,
{
    let value = serde_json::to_value(value)?;
    serde_json::from_value(backend.sanitize_json(value)).map_err(PluginError::from)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    struct NameDetector;

    impl DetectionModel for NameDetector {
        fn detect(&self, texts: &[&str]) -> PluginResult<Vec<Detection>> {
            Ok(texts
                .iter()
                .enumerate()
                .filter_map(|(text_index, text)| {
                    text.find("José").map(|start| Detection {
                        text_index,
                        start_utf8: start,
                        end_utf8: start + "José".len(),
                        label: "GIVEN_NAME".into(),
                        score: 0.99,
                    })
                })
                .collect())
        }
    }

    struct FailingDetector;

    impl DetectionModel for FailingDetector {
        fn detect(&self, _texts: &[&str]) -> PluginResult<Vec<Detection>> {
            Err(PluginError::Internal("model failure".into()))
        }
    }

    struct PanickingDetector;

    impl DetectionModel for PanickingDetector {
        fn detect(&self, _texts: &[&str]) -> PluginResult<Vec<Detection>> {
            panic!("model panic")
        }
    }

    struct CountingDetector(Arc<AtomicUsize>);

    impl DetectionModel for CountingDetector {
        fn detect(&self, _texts: &[&str]) -> PluginResult<Vec<Detection>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(Vec::new())
        }
    }

    struct BlockingDetector {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
    }

    impl DetectionModel for BlockingDetector {
        fn detect(&self, _texts: &[&str]) -> PluginResult<Vec<Detection>> {
            self.started.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            self.finished.store(true, Ordering::Release);
            Ok(Vec::new())
        }
    }

    struct CountingBlockingDetector {
        started: Arc<AtomicUsize>,
        release: Arc<AtomicBool>,
    }

    impl DetectionModel for CountingBlockingDetector {
        fn detect(&self, _texts: &[&str]) -> PluginResult<Vec<Detection>> {
            self.started.fetch_add(1, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Ok(Vec::new())
        }
    }

    fn sanitizer(detector: Arc<dyn DetectionModel>, patterns: Vec<&str>) -> RampartSanitizer {
        RampartSanitizer::new(
            RampartPiiConfig {
                model_path: "/tmp/rampart".into(),
                target_path_patterns: patterns.into_iter().map(str::to_string).collect(),
                ..RampartPiiConfig::default()
            },
            detector,
        )
        .unwrap()
    }

    #[test]
    fn sanitizes_selected_utf8_spans_without_touching_metadata() {
        let sanitizer = sanitizer(
            Arc::new(NameDetector),
            vec!["/messages/*/content", "/message"],
        );
        let value = serde_json::json!({
            "messages": [{"content": "Hello José Rivera"}],
            "message": "José",
            "model": "model-José"
        });
        assert_eq!(
            sanitizer.sanitize_json(value),
            serde_json::json!({
                "messages": [{"content": "Hello [REDACTED] Rivera"}],
                "message": "[REDACTED]",
                "model": "model-José"
            })
        );
    }

    #[test]
    fn model_errors_fail_closed_only_for_selected_values() {
        let sanitizer = sanitizer(Arc::new(FailingDetector), vec!["/message"]);
        assert_eq!(
            sanitizer.sanitize_json(serde_json::json!({
                "message": "private",
                "metadata": "visible"
            })),
            serde_json::json!({
                "message": "[REDACTED]",
                "metadata": "visible"
            })
        );
    }

    #[test]
    fn selected_values_over_payload_budget_fail_closed() {
        let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/*"]);
        let value = Json::Object(
            (0..=MAX_TEXTS_PER_PAYLOAD)
                .map(|index| (index.to_string(), Json::String("safe".into())))
                .collect(),
        );
        let sanitized = sanitizer.sanitize_json(value);
        assert_eq!(
            sanitized
                .as_object()
                .unwrap()
                .values()
                .filter(|value| **value == "[REDACTED]")
                .count(),
            1
        );
    }

    #[test]
    fn selected_payload_uses_one_detector_call() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sanitizer = sanitizer(Arc::new(CountingDetector(Arc::clone(&calls))), vec!["/*"]);
        let value = Json::Object(
            (0..128)
                .map(|index| (index.to_string(), Json::String("safe".into())))
                .collect(),
        );
        assert_eq!(
            sanitizer.sanitize_json(value).as_object().unwrap().len(),
            128
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn async_callback_does_not_block_the_runtime_thread() {
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let backend = sanitizer(
            Arc::new(BlockingDetector {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
                finished,
            }),
            vec!["/message"],
        );
        let callback = tool_sanitize_callback(backend);
        let task = tokio::spawn(callback(
            "tool".into(),
            serde_json::json!({"message": "private"}),
        ));

        tokio::time::timeout(Duration::from_secs(1), async {
            while !started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking detector should start");

        let heartbeat = Arc::new(AtomicUsize::new(0));
        let heartbeat_task = {
            let heartbeat = Arc::clone(&heartbeat);
            tokio::spawn(async move {
                for _ in 0..4 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                    heartbeat.fetch_add(1, Ordering::Relaxed);
                }
            })
        };
        heartbeat_task.await.unwrap();
        assert_eq!(heartbeat.load(Ordering::Relaxed), 4);

        release.store(true, Ordering::Release);
        assert_eq!(
            task.await.unwrap().unwrap(),
            serde_json::json!({"message": "private"})
        );
    }

    #[test]
    fn bounded_fanout_does_not_block_the_runtime_thread() {
        let worker_count = inference_worker_count();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let started = Arc::new(AtomicUsize::new(0));
            let release = Arc::new(AtomicBool::new(false));
            let backend = sanitizer(
                Arc::new(CountingBlockingDetector {
                    started: Arc::clone(&started),
                    release: Arc::clone(&release),
                }),
                vec!["/message"],
            );
            let callback = tool_sanitize_callback(backend);
            let mut tasks = Vec::new();
            for index in 0..MAX_ADMITTED_INFERENCE {
                tasks.push(tokio::spawn(callback(
                    format!("tool-{index}"),
                    serde_json::json!({"message": "private"}),
                )));
            }
            tokio::time::timeout(Duration::from_secs(1), async {
                while started.load(Ordering::Acquire) != worker_count {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("the bounded model slots should start");

            let heartbeat = tokio::spawn(async {
                for _ in 0..4 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            });
            heartbeat
                .await
                .expect("the runtime should remain responsive during fanout");

            release.store(true, Ordering::Release);
            for task in tasks {
                assert_eq!(
                    task.await.unwrap().unwrap(),
                    serde_json::json!({"message": "private"})
                );
            }
            assert_eq!(started.load(Ordering::Acquire), MAX_ADMITTED_INFERENCE);
        });
    }

    #[test]
    fn dedicated_executor_ignores_saturated_tokio_blocking_pool() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let blocker_started = Arc::new(AtomicBool::new(false));
            let blocker_release = Arc::new(AtomicBool::new(false));
            let started = Arc::clone(&blocker_started);
            let release = Arc::clone(&blocker_release);
            let blocker = tokio::task::spawn_blocking(move || {
                started.store(true, Ordering::Release);
                while !release.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
            });
            tokio::time::timeout(Duration::from_secs(1), async {
                while !blocker_started.load(Ordering::Acquire) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("the Tokio blocking-pool fixture should start");

            let calls = Arc::new(AtomicUsize::new(0));
            let callback = tool_sanitize_callback(sanitizer(
                Arc::new(CountingDetector(Arc::clone(&calls))),
                vec!["/message"],
            ));
            let output = tokio::time::timeout(
                Duration::from_millis(250),
                callback(
                    "dedicated-executor".into(),
                    serde_json::json!({"message": "private"}),
                ),
            )
            .await
            .expect("Rampart must not queue behind Tokio's blocking pool")
            .unwrap();
            assert_eq!(output, serde_json::json!({"message": "private"}));
            assert_eq!(calls.load(Ordering::Acquire), 1);

            blocker_release.store(true, Ordering::Release);
            blocker.await.unwrap();
        });
    }

    #[tokio::test(flavor = "current_thread")]
    async fn inference_worker_panics_fail_closed_for_every_surface() {
        use nemo_relay::api::event::{BaseEvent, MarkEvent};
        use nemo_relay::api::runtime::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

        let backend = sanitizer(Arc::new(PanickingDetector), vec!["/message"]);
        let tool = tool_sanitize_callback(backend.clone());
        assert_eq!(
            tool(
                "tool".into(),
                serde_json::json!({"message": "private", "metadata": "visible"}),
            )
            .await
            .unwrap(),
            Json::String("[REDACTED]".into())
        );

        let event = Arc::new(Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("mark")
                .data(serde_json::json!({"message": "private"}))
                .metadata(serde_json::json!({"message": "private"}))
                .build(),
            None,
            None,
        )));
        let fields = event.sanitize_fields();
        assert_eq!(
            event_sanitize_callback(backend.clone(), None)(event, fields)
                .await
                .unwrap(),
            EventSanitizeFields::default()
        );

        let request = LlmRequest {
            headers: Map::new(),
            content: serde_json::json!({"message": "private"}),
        };
        assert!(
            llm_sanitize_request_callback(backend.clone())(
                request,
                LlmSanitizeRequestContext::default(),
            )
            .await
            .unwrap()
            .is_none()
        );
        assert!(
            llm_sanitize_response_callback(backend)(
                serde_json::json!({"message": "private"}),
                LlmSanitizeResponseContext::default(),
            )
            .await
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn bounded_admission_times_out_before_spawning_more_blocking_work() {
        let worker_count = inference_worker_count();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let started = Arc::new(AtomicBool::new(false));
            let release = Arc::new(AtomicBool::new(false));
            let finished = Arc::new(AtomicBool::new(false));
            let backend = sanitizer(
                Arc::new(BlockingDetector {
                    started: Arc::clone(&started),
                    release: Arc::clone(&release),
                    finished,
                }),
                vec!["/message"],
            );
            let execution = Arc::clone(&backend.execution_admission);
            let callback = tool_sanitize_callback(backend);
            let mut active = Vec::new();
            for index in 0..worker_count {
                active.push(tokio::spawn(callback(
                    format!("active-{index}"),
                    serde_json::json!({"message": "private"}),
                )));
            }
            tokio::time::timeout(Duration::from_secs(1), async {
                while !started.load(Ordering::Acquire) || execution.available_permits() != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("blocking detector should start");

            let contending = tokio::time::timeout(
                MAX_ADMISSION_WAIT + Duration::from_millis(100),
                callback(
                    "contending".into(),
                    serde_json::json!({"message": "private", "metadata": "visible"}),
                ),
            )
            .await
            .expect("contending sanitizer should respect the admission deadline")
            .unwrap();
            assert_eq!(contending, Json::String("[REDACTED]".into()));

            release.store(true, Ordering::Release);
            for task in active {
                assert_eq!(
                    task.await.unwrap().unwrap(),
                    serde_json::json!({"message": "private"})
                );
            }
        });
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_callback_keeps_admission_until_blocking_work_finishes() {
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let backend = sanitizer(
            Arc::new(BlockingDetector {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
                finished: Arc::clone(&finished),
            }),
            vec!["/message"],
        );
        let execution = Arc::clone(&backend.execution_admission);
        let worker_count = execution.available_permits();
        let callback = tool_sanitize_callback(backend);
        let active = tokio::spawn(callback(
            "active".into(),
            serde_json::json!({"message": "private"}),
        ));
        tokio::time::timeout(Duration::from_secs(1), async {
            while !started.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("blocking detector should start");
        assert_eq!(execution.available_permits(), worker_count - 1);

        active.abort();
        assert!(active.await.unwrap_err().is_cancelled());
        assert_eq!(
            execution.available_permits(),
            worker_count - 1,
            "cancelling the async caller must not release an in-flight model slot"
        );

        release.store(true, Ordering::Release);
        tokio::time::timeout(Duration::from_secs(1), async {
            while !finished.load(Ordering::Acquire) || execution.available_permits() != worker_count
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("detached inference work should finish");
        assert_eq!(
            callback(
                "recovered".into(),
                serde_json::json!({"message": "private"}),
            )
            .await
            .unwrap(),
            serde_json::json!({"message": "private"})
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_queued_callback_releases_capacity_without_running() {
        let started = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let backend = sanitizer(
            Arc::new(CountingBlockingDetector {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
            }),
            vec!["/message"],
        );
        let admission = Arc::clone(&backend.admission_capacity);
        let execution = Arc::clone(&backend.execution_admission);
        let worker_count = execution.available_permits();
        let callback = tool_sanitize_callback(backend);
        let mut active = Vec::new();
        for index in 0..worker_count {
            active.push(tokio::spawn(callback(
                format!("active-{index}"),
                serde_json::json!({"message": "private"}),
            )));
        }
        tokio::time::timeout(Duration::from_secs(1), async {
            while started.load(Ordering::Acquire) != worker_count
                || execution.available_permits() != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both model slots should start");

        let queued = tokio::spawn(callback(
            "queued".into(),
            serde_json::json!({"message": "private"}),
        ));
        tokio::time::timeout(Duration::from_millis(50), async {
            while admission.available_permits() != MAX_ADMITTED_INFERENCE - worker_count - 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the queued callback should reserve bounded capacity");
        queued.abort();
        assert!(queued.await.unwrap_err().is_cancelled());
        assert_eq!(
            admission.available_permits(),
            MAX_ADMITTED_INFERENCE - worker_count
        );
        assert_eq!(started.load(Ordering::Acquire), worker_count);

        release.store(true, Ordering::Release);
        for task in active {
            task.await.unwrap().unwrap();
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_admission_queue_fails_closed_without_running() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = sanitizer(
            Arc::new(CountingDetector(Arc::clone(&calls))),
            vec!["/message"],
        );
        let mut permits = Vec::new();
        for _ in 0..MAX_ADMITTED_INFERENCE {
            permits.push(
                Arc::clone(&backend.admission_capacity)
                    .try_acquire_owned()
                    .unwrap(),
            );
        }
        let output = tool_sanitize_callback(backend)(
            "queue-full".into(),
            serde_json::json!({"message": "private"}),
        )
        .await
        .unwrap();
        assert_eq!(output, Json::String("[REDACTED]".into()));
        assert_eq!(calls.load(Ordering::Acquire), 0);
        drop(permits);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unselected_tool_payload_bypasses_full_admission_queue() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = sanitizer(
            Arc::new(CountingDetector(Arc::clone(&calls))),
            vec!["/message"],
        );
        let mut permits = Vec::new();
        for _ in 0..MAX_ADMITTED_INFERENCE {
            permits.push(
                Arc::clone(&backend.admission_capacity)
                    .try_acquire_owned()
                    .unwrap(),
            );
        }
        let payload = serde_json::json!({"trace_id": "visible"});
        let output = tool_sanitize_callback(backend)("unselected".into(), payload.clone())
            .await
            .unwrap();
        assert_eq!(output, payload);
        assert_eq!(calls.load(Ordering::Acquire), 0);
        drop(permits);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unselected_specialized_metadata_bypasses_full_admission_queue() {
        use nemo_relay::api::event::{
            BaseEvent, CategoryProfile, EventCategory, ScopeCategory, ScopeEvent,
        };

        let calls = Arc::new(AtomicUsize::new(0));
        let backend = sanitizer(
            Arc::new(CountingDetector(Arc::clone(&calls))),
            vec!["/message"],
        );
        let mut permits = Vec::new();
        for _ in 0..MAX_ADMITTED_INFERENCE {
            permits.push(
                Arc::clone(&backend.admission_capacity)
                    .try_acquire_owned()
                    .unwrap(),
            );
        }
        let event = Arc::new(Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("tool")
                .metadata(serde_json::json!({"trace_id": "visible"}))
                .build(),
            ScopeCategory::Start,
            Default::default(),
            EventCategory::tool(),
            Some(CategoryProfile::default()),
        )));
        let fields = event.sanitize_fields();
        let output = event_sanitize_callback(backend, Some((false, true)))(
            Arc::clone(&event),
            fields.clone(),
        )
        .await
        .unwrap();
        assert_eq!(output, fields);
        assert_eq!(calls.load(Ordering::Acquire), 0);
        drop(permits);
    }

    #[test]
    fn empty_specialized_scope_does_not_require_admission() {
        use nemo_relay::api::event::{
            BaseEvent, CategoryProfile, EventCategory, ScopeCategory, ScopeEvent,
        };

        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("tool")
                .data(serde_json::json!({"message": "[REDACTED]"}))
                .build(),
            ScopeCategory::Start,
            Default::default(),
            EventCategory::tool(),
            Some(CategoryProfile::default()),
        ));
        let fields = event.sanitize_fields();
        assert!(!event_has_candidate_fields(&event, &fields));
        assert_eq!(
            fail_closed_event_fields(&event, fields.clone()),
            fields,
            "specialized data already handled by the tool sanitizer must be preserved"
        );
    }

    #[test]
    fn openai_chat_request_projection_preserves_provider_fields() {
        let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/messages/*/content"]);
        let request = LlmRequest {
            headers: Map::from_iter([("x-vendor".into(), Json::String("José-header".into()))]),
            content: serde_json::json!({
                "model": "model-José",
                "messages": [{"role": "user", "content": "Hello José"}],
                "vendor_trace": "trace-José"
            }),
        };
        let codec = build_request_codec(ProviderSurface::OpenAIChat);
        let sanitized = sanitizer
            .sanitize_request_with_codec(codec.as_ref(), &request)
            .unwrap();

        assert_eq!(
            sanitized.content["messages"][0]["content"],
            "Hello [REDACTED]"
        );
        assert_eq!(sanitized.content["model"], "model-José");
        assert_eq!(sanitized.content["vendor_trace"], "trace-José");
        assert_eq!(sanitized.headers["x-vendor"], "José-header");
    }

    #[test]
    fn openai_chat_response_projection_preserves_provider_fields() {
        let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/message"]);
        let payload = serde_json::json!({
            "id": "chatcmpl-José",
            "model": "model-José",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello José"},
                "finish_reason": "stop"
            }],
            "vendor_trace": "trace-José"
        });
        let codec = build_response_codec(ProviderSurface::OpenAIChat);
        let sanitized = sanitizer
            .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
            .unwrap();

        assert_eq!(
            sanitized["choices"][0]["message"]["content"],
            "Hello [REDACTED]"
        );
        assert_eq!(sanitized["id"], "chatcmpl-José");
        assert_eq!(sanitized["model"], "model-José");
        assert_eq!(sanitized["vendor_trace"], "trace-José");
    }

    #[test]
    fn openai_chat_response_projection_omits_multiple_choices_for_choice_targets() {
        let exact = RampartSanitizer::new(
            RampartPiiConfig {
                model_path: "/tmp/rampart".into(),
                target_paths: vec!["/message".into()],
                ..RampartPiiConfig::default()
            },
            Arc::new(NameDetector),
        )
        .unwrap();
        let wildcard = sanitizer(Arc::new(NameDetector), vec!["/*"]);
        let payload = serde_json::json!({
            "id": "chatcmpl-multi",
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello José"},
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "Private José"},
                    "finish_reason": "stop"
                }
            ]
        });
        let codec = build_response_codec(ProviderSurface::OpenAIChat);

        assert!(
            exact
                .sanitize_response_with_codec(
                    codec.as_ref(),
                    ProviderSurface::OpenAIChat,
                    payload.clone(),
                )
                .is_none()
        );
        assert!(
            wildcard
                .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
                .is_none()
        );
    }

    #[test]
    fn openai_chat_response_projection_keeps_multiple_choices_for_response_targets() {
        let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/model"]);
        let payload = serde_json::json!({
            "id": "chatcmpl-multi",
            "model": "model-José",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "first"},
                    "finish_reason": "stop"
                },
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "second"},
                    "finish_reason": "stop"
                }
            ]
        });
        let codec = build_response_codec(ProviderSurface::OpenAIChat);
        let sanitized = sanitizer
            .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
            .unwrap();

        assert_eq!(sanitized["model"], "model-[REDACTED]");
        assert_eq!(sanitized["choices"][1]["message"]["content"], "second");
    }
}
