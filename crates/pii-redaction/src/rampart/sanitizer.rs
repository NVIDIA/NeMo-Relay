// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

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
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::plugin::{PluginError, Result as PluginResult};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as Json};

use crate::builtin::escape_json_pointer_segment;
use crate::overlay::BuiltinCodecName;

use super::RampartPiiConfig;
use super::model::{Detection, RampartDetector};

const MAX_TEXT_BYTES: usize = 16 * 1024;
const MAX_TEXTS_PER_PAYLOAD: usize = 256;
const MAX_PAYLOAD_TEXT_BYTES: usize = 256 * 1024;

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
        Box::pin(async move {
            run_blocking("tool payload", move || backend.sanitize_json(payload)).await
        })
    })
}

pub(super) fn event_sanitize_callback(
    backend: RampartSanitizer,
    scope_categories: Option<(bool, bool)>,
) -> EventSanitizeFn {
    Arc::new(move |event, mut fields| {
        let backend = backend.clone();
        Box::pin(async move {
            run_blocking("event fields", move || {
                if scope_categories.is_some_and(|(sanitize_llm, sanitize_tool)| {
                    matches!(event.as_ref(), Event::Scope(_))
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
                let specialized_scope = matches!(event.as_ref(), Event::Scope(_))
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
            .await
        })
    })
}

pub(super) fn llm_sanitize_request_callback(backend: RampartSanitizer) -> LlmSanitizeRequestFn {
    Arc::new(move |request, context| {
        let backend = backend.clone();
        Box::pin(async move {
            run_blocking("LLM request", move || {
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
            run_blocking("LLM response", move || {
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

async fn run_blocking<T>(
    target: &'static str,
    operation: impl FnOnce() -> T + Send + 'static,
) -> FlowResult<T>
where
    T: Send + 'static,
{
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| {
            FlowError::Internal(format!(
                "Rampart {target} sanitization task failed: {error}"
            ))
        })
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
    }

    impl DetectionModel for BlockingDetector {
        fn detect(&self, _texts: &[&str]) -> PluginResult<Vec<Detection>> {
            self.started.store(true, Ordering::Release);
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
        let backend = sanitizer(
            Arc::new(BlockingDetector {
                started: Arc::clone(&started),
                release: Arc::clone(&release),
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
