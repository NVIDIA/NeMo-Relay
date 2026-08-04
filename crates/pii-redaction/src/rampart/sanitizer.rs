// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
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
use sha2::{Digest, Sha256};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::builtin::escape_json_pointer_segment;
use crate::overlay::BuiltinCodecName;
use crate::trajectory::{
    CustomMarkPayloadPolicy, is_known_content_bearing_mark, is_trusted_scope_metadata_value,
    preserve_analytical_string, preserves_tool_or_function_name,
};

use super::RampartPiiConfig;
use super::model::{Detection, DetectionError, RampartDetector};

const MAX_TEXTS_PER_PAYLOAD: usize = 256;
const MAX_PAYLOAD_TEXT_BYTES: usize = 256 * 1024;
const MAX_CACHE_ENTRIES: usize = 4096;
const MAX_CACHE_DECISION_BYTES: usize = 4 * 1024 * 1024;
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
    fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError>;
}

impl DetectionModel for RampartDetector {
    fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
        RampartDetector::detect(self, texts)
    }
}

#[derive(Clone)]
pub(super) struct RampartSanitizer {
    detector: Arc<dyn DetectionModel>,
    target_paths: Arc<HashSet<Vec<String>>>,
    target_path_patterns: Arc<Vec<JsonPointerPattern>>,
    trajectory_policy: Option<CustomMarkPayloadPolicy>,
    min_score: f64,
    excluded_labels: Arc<HashSet<String>>,
    replacement: Arc<str>,
    legacy_surface: Option<ProviderSurface>,
    admission_capacity: Arc<Semaphore>,
    execution_admission: Arc<Semaphore>,
    executor: Arc<SanitizerExecutor>,
    cache: Arc<Mutex<SanitizationCache>>,
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

enum SelectedText {
    Resolved(String),
    Pending {
        key: TextCacheKey,
        text: Option<String>,
    },
}

type TextCacheKey = [u8; 32];

// Cache decisions rather than text so selected observability content is not retained.
#[derive(Clone)]
enum SanitizationDecision {
    Keep,
    Redact(Arc<[(usize, usize)]>),
    FailClosed,
}

impl SanitizationDecision {
    fn apply(&self, text: &str, replacement: &str) -> String {
        match self {
            Self::Keep => text.to_string(),
            Self::FailClosed => replacement.to_string(),
            Self::Redact(ranges) => {
                let mut redacted = text.to_string();
                for &(start, end) in ranges.iter().rev() {
                    if start >= end
                        || end > redacted.len()
                        || !redacted.is_char_boundary(start)
                        || !redacted.is_char_boundary(end)
                    {
                        return replacement.to_string();
                    }
                    redacted.replace_range(start..end, replacement);
                }
                redacted
            }
        }
    }

    fn cache_weight(&self) -> usize {
        match self {
            Self::Keep | Self::FailClosed => 1,
            Self::Redact(ranges) => ranges.len() * std::mem::size_of::<(usize, usize)>(),
        }
    }
}

struct CacheEntry {
    decision: SanitizationDecision,
    referenced: bool,
    weight: usize,
}

#[derive(Default)]
struct SanitizationCache {
    entries: HashMap<TextCacheKey, CacheEntry>,
    order: VecDeque<TextCacheKey>,
    decision_bytes: usize,
}

impl SanitizationCache {
    fn get(&mut self, key: &TextCacheKey) -> Option<SanitizationDecision> {
        let entry = self.entries.get_mut(key)?;
        entry.referenced = true;
        Some(entry.decision.clone())
    }

    fn insert(&mut self, key: TextCacheKey, decision: SanitizationDecision) {
        let weight = decision.cache_weight();
        if weight > MAX_CACHE_DECISION_BYTES {
            return;
        }
        if let Some(previous) = self.entries.remove(&key) {
            self.decision_bytes = self.decision_bytes.saturating_sub(previous.weight);
            self.order.retain(|existing| existing != &key);
        }
        self.decision_bytes += weight;
        self.entries.insert(
            key,
            CacheEntry {
                decision,
                referenced: false,
                weight,
            },
        );
        self.order.push_back(key);
        self.evict();
    }

    fn evict(&mut self) {
        while self.entries.len() > MAX_CACHE_ENTRIES
            || self.decision_bytes > MAX_CACHE_DECISION_BYTES
        {
            let Some(key) = self.order.pop_front() else {
                break;
            };
            let Some(entry) = self.entries.get_mut(&key) else {
                continue;
            };
            if entry.referenced {
                entry.referenced = false;
                self.order.push_back(key);
                continue;
            }
            let entry = self.entries.remove(&key).expect("cache entry should exist");
            self.decision_bytes = self.decision_bytes.saturating_sub(entry.weight);
        }
    }
}

fn text_cache_key(text: &str) -> TextCacheKey {
    Sha256::digest(text.as_bytes()).into()
}

#[derive(Clone, Copy)]
enum StringSelection {
    Configured,
    All,
    Semantic,
    ScopeMetadata,
}

#[derive(Debug, PartialEq, Eq)]
enum SanitizeError {
    Codec,
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
        let trajectory_policy = config
            .preset
            .as_deref()
            .map(|_| {
                CustomMarkPayloadPolicy::parse(&config.custom_mark_payload_policy).ok_or_else(
                    || {
                        PluginError::InvalidConfig(format!(
                            "unsupported custom-mark payload policy '{}'",
                            config.custom_mark_payload_policy
                        ))
                    },
                )
            })
            .transpose()?;
        Ok(Self {
            detector,
            target_paths: Arc::new(
                config
                    .target_paths
                    .into_iter()
                    .map(compile_json_pointer)
                    .collect(),
            ),
            trajectory_policy,
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
            cache: Arc::new(Mutex::new(SanitizationCache::default())),
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

    fn sanitize_json(&self, value: Json) -> Result<Json, SanitizeError> {
        self.sanitize_json_with_selection(value, StringSelection::Configured)
    }

    fn sanitize_json_with_selection(
        &self,
        value: Json,
        selection: StringSelection,
    ) -> Result<Json, SanitizeError> {
        Ok(self
            .sanitize_json_roots(vec![(Vec::new(), value, selection)])?
            .pop()
            .expect("single-value sanitization returns one value"))
    }

    fn sanitize_json_roots(
        &self,
        mut roots: Vec<(Vec<String>, Json, StringSelection)>,
    ) -> Result<Vec<Json>, SanitizeError> {
        let mut texts = Vec::new();
        let mut total_bytes = 0;
        let mut pending_keys = HashSet::new();
        let mut rejected_fields = 0;
        for (path, value, selection) in &roots {
            let mut path = path.clone();
            self.collect_strings(
                value,
                &mut path,
                *selection,
                None,
                false,
                true,
                &mut texts,
                &mut total_bytes,
                &mut pending_keys,
                &mut rejected_fields,
            )?;
        }

        if rejected_fields > 0 {
            log::warn!(
                target: "nemo_relay.plugin",
                event = "rampart_pii_inference_failed",
                plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
                selected_text_count = pending_keys.len(),
                failed_closed_field_count = rejected_fields,
                reason = "selection_budget";
                "Rampart PII selected-text budget was exceeded and affected fields failed closed"
            );
        }

        let sanitized = self.sanitize_texts(texts);
        let mut index = 0;
        for (path, value, selection) in &mut roots {
            let mut path = path.clone();
            self.replace_strings(
                value, &mut path, *selection, None, false, true, &sanitized, &mut index,
            );
        }
        Ok(roots.into_iter().map(|(_, value, _)| value).collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_strings(
        &self,
        value: &Json,
        path: &mut Vec<String>,
        selection: StringSelection,
        field: Option<&str>,
        preserve: bool,
        selection_root: bool,
        texts: &mut Vec<SelectedText>,
        total_bytes: &mut usize,
        pending_keys: &mut HashSet<TextCacheKey>,
        rejected_fields: &mut usize,
    ) -> Result<(), SanitizeError> {
        match value {
            Json::String(text) if self.selects_string(selection, path, field, preserve) => {
                let key = text_cache_key(text);
                if let Some(decision) = self.cached_decision(&key) {
                    texts.push(SelectedText::Resolved(
                        decision.apply(text, self.replacement.as_ref()),
                    ));
                } else if pending_keys.contains(&key) {
                    texts.push(SelectedText::Pending { key, text: None });
                } else if pending_keys.len() < MAX_TEXTS_PER_PAYLOAD
                    && total_bytes
                        .checked_add(text.len())
                        .is_some_and(|next_total| next_total <= MAX_PAYLOAD_TEXT_BYTES)
                {
                    *total_bytes += text.len();
                    pending_keys.insert(key);
                    texts.push(SelectedText::Pending {
                        key,
                        text: Some(text.clone()),
                    });
                } else {
                    *rejected_fields += 1;
                    texts.push(SelectedText::Resolved(self.replacement.to_string()));
                }
            }
            Json::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path.push(index.to_string());
                    let result = self.collect_strings(
                        item,
                        path,
                        selection,
                        field,
                        false,
                        false,
                        texts,
                        total_bytes,
                        pending_keys,
                        rejected_fields,
                    );
                    path.pop();
                    result?;
                }
            }
            Json::Object(fields) => {
                let preserve_name = preserves_tool_or_function_name(field, fields);
                for (key, value) in fields {
                    path.push(escape_json_pointer_segment(key));
                    let preserve = (key == "name" && preserve_name && value.is_string())
                        || (matches!(selection, StringSelection::ScopeMetadata)
                            && selection_root
                            && is_trusted_scope_metadata_value(key, value));
                    let result = self.collect_strings(
                        value,
                        path,
                        selection,
                        Some(key),
                        preserve,
                        false,
                        texts,
                        total_bytes,
                        pending_keys,
                        rejected_fields,
                    );
                    path.pop();
                    result?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn has_selected_string_with_selection(&self, value: &Json, selection: StringSelection) -> bool {
        self.has_selected_string_at(value, &mut Vec::new(), selection, None, false, true)
    }

    fn has_selected_string_at(
        &self,
        value: &Json,
        path: &mut Vec<String>,
        selection: StringSelection,
        field: Option<&str>,
        preserve: bool,
        selection_root: bool,
    ) -> bool {
        match value {
            Json::String(_) => self.selects_string(selection, path, field, preserve),
            Json::Array(items) => items.iter().enumerate().any(|(index, item)| {
                path.push(index.to_string());
                let selected =
                    self.has_selected_string_at(item, path, selection, field, false, false);
                path.pop();
                selected
            }),
            Json::Object(fields) => {
                let preserve_name = preserves_tool_or_function_name(field, fields);
                fields.iter().any(|(key, value)| {
                    path.push(escape_json_pointer_segment(key));
                    let preserve = (key == "name" && preserve_name && value.is_string())
                        || (matches!(selection, StringSelection::ScopeMetadata)
                            && selection_root
                            && is_trusted_scope_metadata_value(key, value));
                    let selected = self.has_selected_string_at(
                        value,
                        path,
                        selection,
                        Some(key),
                        preserve,
                        false,
                    );
                    path.pop();
                    selected
                })
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_strings(
        &self,
        value: &mut Json,
        path: &mut Vec<String>,
        selection: StringSelection,
        field: Option<&str>,
        preserve: bool,
        selection_root: bool,
        sanitized: &[String],
        index: &mut usize,
    ) {
        match value {
            Json::String(text) if self.selects_string(selection, path, field, preserve) => {
                *text = sanitized
                    .get(*index)
                    .cloned()
                    .unwrap_or_else(|| self.replacement.to_string());
                *index += 1;
            }
            Json::Array(items) => {
                for (item_index, item) in items.iter_mut().enumerate() {
                    path.push(item_index.to_string());
                    self.replace_strings(
                        item, path, selection, field, false, false, sanitized, index,
                    );
                    path.pop();
                }
            }
            Json::Object(fields) => {
                let preserve_name = preserves_tool_or_function_name(field, fields);
                for (key, value) in fields {
                    path.push(escape_json_pointer_segment(key));
                    let preserve = (key == "name" && preserve_name && value.is_string())
                        || (matches!(selection, StringSelection::ScopeMetadata)
                            && selection_root
                            && is_trusted_scope_metadata_value(key, value));
                    self.replace_strings(
                        value,
                        path,
                        selection,
                        Some(key),
                        preserve,
                        false,
                        sanitized,
                        index,
                    );
                    path.pop();
                }
            }
            _ => {}
        }
    }

    fn selects_string(
        &self,
        selection: StringSelection,
        path: &[String],
        field: Option<&str>,
        preserve: bool,
    ) -> bool {
        match selection {
            StringSelection::Configured => self.matches_path(path),
            StringSelection::All => true,
            StringSelection::Semantic | StringSelection::ScopeMetadata => {
                !preserve && !field.is_some_and(preserve_analytical_string)
            }
        }
    }

    fn matches_path(&self, path: &[String]) -> bool {
        self.target_paths.contains(path)
            || self
                .target_path_patterns
                .iter()
                .any(|pattern| pattern.matches(path))
    }

    fn cached_decision(&self, key: &TextCacheKey) -> Option<SanitizationDecision> {
        self.cache.lock().ok()?.get(key)
    }

    fn cache_decision(&self, key: TextCacheKey, decision: SanitizationDecision) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, decision);
        }
    }

    fn sanitize_texts(&self, texts: Vec<SelectedText>) -> Vec<String> {
        let pending = texts
            .iter()
            .filter_map(|selected| match selected {
                SelectedText::Pending {
                    key,
                    text: Some(text),
                } => Some((*key, text.clone())),
                SelectedText::Resolved(_) | SelectedText::Pending { text: None, .. } => None,
            })
            .collect::<Vec<_>>();
        let decisions = self.sanitize_pending_texts(&pending);
        let rendered = pending
            .iter()
            .map(|(key, text)| {
                let decision = decisions
                    .get(key)
                    .cloned()
                    .unwrap_or(SanitizationDecision::FailClosed);
                (*key, decision.apply(text, self.replacement.as_ref()))
            })
            .collect::<HashMap<_, _>>();

        texts
            .into_iter()
            .map(|selected| match selected {
                SelectedText::Resolved(text) => text,
                SelectedText::Pending { key, .. } => rendered
                    .get(&key)
                    .cloned()
                    .unwrap_or_else(|| self.replacement.to_string()),
            })
            .collect()
    }

    fn sanitize_pending_texts(
        &self,
        pending: &[(TextCacheKey, String)],
    ) -> HashMap<TextCacheKey, SanitizationDecision> {
        let mut decisions = HashMap::new();
        let mut groups = vec![(0..pending.len()).collect::<Vec<_>>()];
        while let Some(mut group) = groups.pop() {
            if group.is_empty() {
                continue;
            }
            let texts = group
                .iter()
                .map(|index| pending[*index].1.as_str())
                .collect::<Vec<_>>();
            match self.detect_decisions(&texts) {
                Ok(group_decisions) => {
                    for (index, decision) in group.into_iter().zip(group_decisions) {
                        let key = pending[index].0;
                        self.cache_decision(key, decision.clone());
                        decisions.insert(key, decision);
                    }
                }
                Err(DetectionError::PayloadLimit) if group.len() > 1 => {
                    let right = group.split_off(group.len() / 2);
                    groups.push(right);
                    groups.push(group);
                }
                Err(DetectionError::PayloadLimit) => {
                    let index = group[0];
                    let key = pending[index].0;
                    let decision = SanitizationDecision::FailClosed;
                    self.cache_decision(key, decision.clone());
                    decisions.insert(key, decision);
                    log::warn!(
                        target: "nemo_relay.plugin",
                        event = "rampart_pii_inference_failed",
                        plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
                        selected_text_bytes = pending[index].1.len(),
                        reason = "field_payload_limit";
                        "Rampart PII field exceeded its model budget and failed closed"
                    );
                }
                Err(DetectionError::Model(_)) => {
                    log::warn!(
                        target: "nemo_relay.plugin",
                        event = "rampart_pii_inference_failed",
                        plugin_kind = super::RAMPART_PII_PLUGIN_KIND,
                        selected_text_count = group.len(),
                        reason = "model_or_output";
                        "Rampart PII inference failed closed"
                    );
                    for index in group {
                        decisions.insert(pending[index].0, SanitizationDecision::FailClosed);
                    }
                }
            }
        }
        decisions
    }

    fn detect_decisions(
        &self,
        texts: &[&str],
    ) -> Result<Vec<SanitizationDecision>, DetectionError> {
        let detections = self.detector.detect(texts)?;
        let mut by_text = vec![Vec::<Detection>::new(); texts.len()];
        for detection in detections {
            if detection.text_index >= texts.len()
                || !detection.score.is_finite()
                || !(0.0..=1.0).contains(&detection.score)
            {
                return Err(
                    PluginError::Internal("Rampart returned an invalid detection".into()).into(),
                );
            }
            by_text[detection.text_index].push(detection);
        }

        texts
            .iter()
            .zip(by_text)
            .map(|(text, mut detections)| {
                detections.retain(|detection| {
                    detection.score >= self.min_score
                        && !self.excluded_labels.contains(&detection.label)
                });
                if detections.is_empty() {
                    return Ok(SanitizationDecision::Keep);
                }
                detections.sort_by_key(|detection| (detection.start_utf8, detection.end_utf8));
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
                        )
                        .into());
                    }
                    previous_end = detection.end_utf8;
                }
                Ok(SanitizationDecision::Redact(
                    detections
                        .into_iter()
                        .map(|detection| (detection.start_utf8, detection.end_utf8))
                        .collect::<Vec<_>>()
                        .into(),
                ))
            })
            .collect()
    }

    fn sanitize_request_with_codec(
        &self,
        codec: &dyn LlmCodec,
        request: &LlmRequest,
    ) -> Result<LlmRequest, SanitizeError> {
        let annotated = codec.decode(request).map_err(|_| SanitizeError::Codec)?;
        let annotated = serde_json::to_value(annotated).map_err(|_| SanitizeError::Codec)?;
        let (headers, annotated) = self.sanitize_request_parts(
            request.headers.clone(),
            annotated,
            StringSelection::Configured,
            StringSelection::Configured,
        )?;
        let annotated = serde_json::from_value(annotated).map_err(|_| SanitizeError::Codec)?;
        let mut encoded = codec
            .encode(&annotated, request)
            .map_err(|_| SanitizeError::Codec)?;
        encoded.headers = headers;
        Ok(encoded)
    }

    fn sanitize_raw_request(&self, mut request: LlmRequest) -> Result<LlmRequest, SanitizeError> {
        let headers = std::mem::take(&mut request.headers);
        let content = std::mem::take(&mut request.content);
        let (header_selection, content_selection) = if self.trajectory_policy.is_some() {
            (StringSelection::All, StringSelection::Semantic)
        } else {
            (StringSelection::Configured, StringSelection::Configured)
        };
        let (headers, content) =
            self.sanitize_request_parts(headers, content, header_selection, content_selection)?;
        request.headers = headers;
        request.content = content;
        Ok(request)
    }

    fn sanitize_request_parts(
        &self,
        headers: Map<String, Json>,
        content: Json,
        header_selection: StringSelection,
        content_selection: StringSelection,
    ) -> Result<(Map<String, Json>, Json), SanitizeError> {
        let mut values = self.sanitize_json_roots(vec![
            (
                vec!["headers".to_string()],
                Json::Object(headers),
                header_selection,
            ),
            (Vec::new(), content, content_selection),
        ])?;
        let content = values.pop().ok_or(SanitizeError::Codec)?;
        let headers = values.pop().ok_or(SanitizeError::Codec)?;
        let Json::Object(headers) = headers else {
            return Err(SanitizeError::Codec);
        };
        Ok((headers, content))
    }

    fn sanitize_response_with_codec(
        &self,
        codec: &dyn LlmResponseCodec,
        surface: ProviderSurface,
        payload: Json,
    ) -> Result<Json, SanitizeError> {
        if surface == ProviderSurface::OpenAIChat
            && payload
                .get("choices")
                .and_then(Json::as_array)
                .is_some_and(|choices| choices.len() > 1)
            && self.targets_normalized_openai_chat_choice()
        {
            return Err(SanitizeError::Codec);
        }
        let codec_name = BuiltinCodecName::from_provider_surface(surface);
        let annotated = codec
            .decode_response(&payload)
            .map_err(|_| SanitizeError::Codec)?;
        let sanitized = sanitize_serializable(self, annotated)?;
        Ok(codec_name.overlay_response_payload(payload, &sanitized))
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
        let selection = if backend.trajectory_policy.is_some() {
            StringSelection::All
        } else {
            StringSelection::Configured
        };
        if !backend.has_selected_string_with_selection(&payload, selection) {
            return Box::pin(async move { Ok(payload) });
        }
        let fallback = backend.fail_closed_payload();
        Box::pin(async move {
            let Some(permit) = backend.admit("tool").await else {
                return Ok(fallback);
            };
            run_inference("tool payload", permit, fallback, move || {
                backend.sanitize_json_with_selection(payload, selection)
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
                let mut selected = Vec::with_capacity(3);
                if let Some(selection) =
                    event_field_selection(&backend, event.as_ref(), EventField::Data)
                    && let Some(data) = fields.data.take()
                {
                    selected.push((EventField::Data, data, selection));
                }
                if let Some(selection) =
                    event_field_selection(&backend, event.as_ref(), EventField::CategoryProfile)
                    && let Some(profile) = fields.category_profile.take()
                    && let Ok(profile) = serde_json::to_value(profile)
                {
                    selected.push((EventField::CategoryProfile, profile, selection));
                }
                if let Some(selection) =
                    event_field_selection(&backend, event.as_ref(), EventField::Metadata)
                    && let Some(metadata) = fields.metadata.take()
                {
                    selected.push((EventField::Metadata, metadata, selection));
                }

                let roots = selected
                    .iter_mut()
                    .map(|(_, value, selection)| (Vec::new(), std::mem::take(value), *selection))
                    .collect();
                let sanitized_values = backend.sanitize_json_roots(roots)?;
                for ((field, _, _), value) in selected.into_iter().zip(sanitized_values) {
                    match field {
                        EventField::Data => fields.data = Some(value),
                        EventField::CategoryProfile => {
                            fields.category_profile = serde_json::from_value(value).ok();
                        }
                        EventField::Metadata => fields.metadata = Some(value),
                    }
                }
                Ok(fields)
            })
            .await
        })
    })
}

pub(super) fn llm_sanitize_request_callback(backend: RampartSanitizer) -> LlmSanitizeRequestFn {
    Arc::new(move |request, context| {
        let backend = backend.clone();
        if backend.trajectory_policy.is_some()
            && !request.headers.values().any(|value| {
                backend.has_selected_string_with_selection(value, StringSelection::All)
            })
            && !backend
                .has_selected_string_with_selection(&request.content, StringSelection::Semantic)
        {
            return Box::pin(async move { Ok(Some(request)) });
        }
        Box::pin(async move {
            let Some(permit) = backend.admit("llm_request").await else {
                return Ok(None);
            };
            run_inference("LLM request", permit, None, move || {
                let sanitized = if backend.trajectory_policy.is_some()
                    || (matches!(context.codec(), LlmCodecIdentity::None)
                        && backend.legacy_surface.is_none())
                {
                    backend.sanitize_raw_request(request).map(Some)
                } else {
                    let resolved = context.resolve_codec();
                    let fallback = if resolved.is_none() {
                        backend
                            .selected_surface(context.codec())
                            .map(build_request_codec)
                    } else {
                        None
                    };
                    resolved
                        .as_deref()
                        .or(fallback.as_deref())
                        .ok_or(SanitizeError::Codec)
                        .and_then(|codec| {
                            backend
                                .sanitize_request_with_codec(codec, &request)
                                .map(Some)
                        })
                };
                if matches!(sanitized, Err(SanitizeError::Codec)) {
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
        if backend.trajectory_policy.is_some()
            && !backend.has_selected_string_with_selection(&payload, StringSelection::Semantic)
        {
            return Box::pin(async move { Ok(Some(payload)) });
        }
        Box::pin(async move {
            let Some(permit) = backend.admit("llm_response").await else {
                return Ok(None);
            };
            run_inference("LLM response", permit, None, move || {
                if backend.trajectory_policy.is_some() {
                    return backend
                        .sanitize_json_with_selection(payload, StringSelection::Semantic)
                        .map(Some);
                }
                if matches!(context.codec(), LlmCodecIdentity::None)
                    && backend.legacy_surface.is_none()
                {
                    return backend.sanitize_json(payload).map(Some);
                }
                if matches!(context.codec(), LlmCodecIdentity::None)
                    && !backend.uses_compatible_legacy_response_codec(&payload)
                {
                    backend.log_codec_failure(
                        "response",
                        context.codec(),
                        "no compatible legacy codec",
                    );
                    return Err(SanitizeError::Codec);
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
                    .ok_or(SanitizeError::Codec)
                    .and_then(|(surface, codec)| {
                        backend
                            .sanitize_response_with_codec(codec, surface, payload)
                            .map(Some)
                    });
                if matches!(sanitized, Err(SanitizeError::Codec)) {
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
    operation: impl FnOnce() -> Result<T, SanitizeError> + Send + 'static,
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
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(SanitizeError::Codec))) => Ok(fallback),
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
    let has_selected = |field, value: &Json| {
        event_field_selection(backend, event, field)
            .is_some_and(|selection| backend.has_selected_string_with_selection(value, selection))
    };

    fields
        .data
        .as_ref()
        .is_some_and(|data| has_selected(EventField::Data, data))
        || fields.category_profile.as_ref().is_some_and(|profile| {
            serde_json::to_value(profile)
                .ok()
                .is_some_and(|profile| has_selected(EventField::CategoryProfile, &profile))
        })
        || fields
            .metadata
            .as_ref()
            .is_some_and(|metadata| has_selected(EventField::Metadata, metadata))
}

fn event_field_selection(
    backend: &RampartSanitizer,
    event: &Event,
    field: EventField,
) -> Option<StringSelection> {
    if backend.trajectory_policy.is_none() {
        return (!is_specialized_scope(event) || field == EventField::Metadata)
            .then_some(StringSelection::Configured);
    }

    let unknown_custom_mark = matches!(event, Event::Mark(_))
        && event
            .category()
            .is_some_and(|category| category.as_str() == "custom")
        && !is_known_content_bearing_mark(event.name());
    if unknown_custom_mark {
        return (backend.trajectory_policy == Some(CustomMarkPayloadPolicy::RedactAllLeaves))
            .then_some(StringSelection::All);
    }

    if is_specialized_scope(event) {
        return (field == EventField::Metadata).then_some(StringSelection::ScopeMetadata);
    }

    Some(
        if field == EventField::Metadata && matches!(event, Event::Scope(_)) {
            StringSelection::ScopeMetadata
        } else {
            StringSelection::Semantic
        },
    )
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

fn sanitize_serializable<T>(backend: &RampartSanitizer, value: T) -> Result<T, SanitizeError>
where
    T: Serialize + DeserializeOwned,
{
    let value = serde_json::to_value(value).map_err(|_| SanitizeError::Codec)?;
    serde_json::from_value(backend.sanitize_json(value)?).map_err(|_| SanitizeError::Codec)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    struct NameDetector;

    impl DetectionModel for NameDetector {
        fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
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
        fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
            Err(PluginError::Internal("model failure".into()).into())
        }
    }

    struct PayloadLimitedDetector;

    impl DetectionModel for PayloadLimitedDetector {
        fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
            Err(DetectionError::PayloadLimit)
        }
    }

    struct PanickingDetector;

    impl DetectionModel for PanickingDetector {
        fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
            panic!("model panic")
        }
    }

    struct CountingDetector(Arc<AtomicUsize>);

    impl DetectionModel for CountingDetector {
        fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(Vec::new())
        }
    }

    struct CountingNameDetector(Arc<AtomicUsize>);

    impl DetectionModel for CountingNameDetector {
        fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            NameDetector.detect(texts)
        }
    }

    struct BatchLimitedNameDetector(Arc<AtomicUsize>);

    impl DetectionModel for BatchLimitedNameDetector {
        fn detect(&self, texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            if texts.len() > 1 {
                Err(DetectionError::PayloadLimit)
            } else {
                NameDetector.detect(texts)
            }
        }
    }

    struct BlockingDetector {
        started: Arc<AtomicBool>,
        release: Arc<AtomicBool>,
        finished: Arc<AtomicBool>,
    }

    impl DetectionModel for BlockingDetector {
        fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
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
        fn detect(&self, _texts: &[&str]) -> Result<Vec<Detection>, DetectionError> {
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

    fn trajectory_sanitizer(
        detector: Arc<dyn DetectionModel>,
        custom_mark_payload_policy: &str,
    ) -> RampartSanitizer {
        RampartSanitizer::new(
            RampartPiiConfig {
                model_path: "/tmp/rampart".into(),
                preset: Some("trajectory_context".into()),
                custom_mark_payload_policy: custom_mark_payload_policy.into(),
                ..RampartPiiConfig::default()
            },
            detector,
        )
        .unwrap()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trajectory_preset_sanitizes_multi_message_anthropic_request_without_projection() {
        use nemo_relay::api::runtime::LlmSanitizeRequestContext;

        let backend = trajectory_sanitizer(Arc::new(NameDetector), "preserve");
        let request = LlmRequest {
            headers: Map::from_iter([(
                "x-user-context".into(),
                Json::String("José header".into()),
            )]),
            content: serde_json::json!({
                "model": "claude-José",
                "system": "Help José safely",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Initial prompt from José"}]
                    },
                    {
                        "role": "assistant",
                        "content": [{
                            "type": "tool_use",
                            "id": "toolu_José",
                            "name": "read_file",
                            "input": {"path": "/tmp/José.txt"}
                        }]
                    },
                    {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": "toolu_José",
                            "content": "The file belongs to José"
                        }]
                    }
                ],
                "tools": [{
                    "name": "read_file",
                    "description": "Read files for José",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "José's file path"}
                        }
                    }
                }]
            }),
        };

        let sanitized = llm_sanitize_request_callback(backend)(
            request,
            LlmSanitizeRequestContext::with_identity(LlmCodecIdentity::BuiltIn(
                BuiltinLlmCodec::AnthropicMessages,
            )),
        )
        .await
        .unwrap()
        .expect("trajectory content should remain observable after sanitization");

        assert_eq!(sanitized.headers["x-user-context"], "[REDACTED] header");
        assert_eq!(sanitized.content["model"], "claude-José");
        assert_eq!(sanitized.content["system"], "Help [REDACTED] safely");
        assert_eq!(
            sanitized.content["messages"][0]["content"][0]["text"],
            "Initial prompt from [REDACTED]"
        );
        assert_eq!(
            sanitized.content["messages"][1]["content"][0]["name"],
            "read_file"
        );
        assert_eq!(
            sanitized.content["messages"][1]["content"][0]["id"],
            "toolu_José"
        );
        assert_eq!(
            sanitized.content["messages"][1]["content"][0]["input"]["path"],
            "/tmp/[REDACTED].txt"
        );
        assert_eq!(
            sanitized.content["messages"][2]["content"][0]["tool_use_id"],
            "toolu_José"
        );
        assert_eq!(
            sanitized.content["messages"][2]["content"][0]["content"],
            "The file belongs to [REDACTED]"
        );
        assert_eq!(sanitized.content["tools"][0]["name"], "read_file");
        assert_eq!(
            sanitized.content["tools"][0]["description"],
            "Read files for [REDACTED]"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trajectory_preset_sanitizes_root_scalar_tool_results() {
        let output = tool_sanitize_callback(trajectory_sanitizer(
            Arc::new(NameDetector),
            "preserve",
        ))("read_file".into(), Json::String("Owned by José".into()))
        .await
        .unwrap();

        assert_eq!(output, "Owned by [REDACTED]");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trajectory_preset_preserves_unknown_custom_marks_by_default() {
        use nemo_relay::api::event::{BaseEvent, EventCategory, MarkEvent};

        let calls = Arc::new(AtomicUsize::new(0));
        let event = Arc::new(Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("application.checkpoint")
                .data(serde_json::json!({"message": "José"}))
                .metadata(serde_json::json!({"owner": "José"}))
                .build(),
            Some(EventCategory::custom()),
            None,
        )));
        let fields = event.sanitize_fields();
        let output = event_sanitize_callback(
            trajectory_sanitizer(Arc::new(CountingDetector(Arc::clone(&calls))), "preserve"),
            None,
        )(Arc::clone(&event), fields.clone())
        .await
        .unwrap();

        assert_eq!(output, fields);
        assert_eq!(calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn trajectory_preset_can_inspect_all_unknown_custom_mark_strings() {
        use nemo_relay::api::event::{BaseEvent, EventCategory, MarkEvent};

        let event = Arc::new(Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("application.checkpoint")
                .data(serde_json::json!({"id": "José"}))
                .metadata(serde_json::json!({"owner": "José"}))
                .build(),
            Some(EventCategory::custom()),
            None,
        )));
        let output = event_sanitize_callback(
            trajectory_sanitizer(Arc::new(NameDetector), "redact_all_leaves"),
            None,
        )(Arc::clone(&event), event.sanitize_fields())
        .await
        .unwrap();

        assert_eq!(output.data.unwrap()["id"], "[REDACTED]");
        assert_eq!(output.metadata.unwrap()["owner"], "[REDACTED]");
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
            sanitizer.sanitize_json(value).unwrap(),
            serde_json::json!({
                "messages": [{"content": "Hello [REDACTED] Rivera"}],
                "message": "[REDACTED]",
                "model": "model-José"
            })
        );
    }

    #[test]
    fn content_cache_deduplicates_within_and_across_payloads() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sanitizer = sanitizer(
            Arc::new(CountingNameDetector(Arc::clone(&calls))),
            vec!["/*"],
        );
        let payload = serde_json::json!({
            "first": "Hello José",
            "second": "Hello José"
        });

        let first = sanitizer.sanitize_json(payload.clone()).unwrap();
        let second = sanitizer.sanitize_json(payload).unwrap();

        assert_eq!(first["first"], "Hello [REDACTED]");
        assert_eq!(first["second"], "Hello [REDACTED]");
        assert_eq!(second, first);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn content_cache_evicts_to_its_entry_bound() {
        let mut cache = SanitizationCache::default();
        for index in 0..=MAX_CACHE_ENTRIES {
            cache.insert(
                text_cache_key(&index.to_string()),
                SanitizationDecision::Keep,
            );
        }

        assert_eq!(cache.entries.len(), MAX_CACHE_ENTRIES);
        assert_eq!(cache.order.len(), MAX_CACHE_ENTRIES);
        assert_eq!(cache.decision_bytes, MAX_CACHE_ENTRIES);
    }

    #[test]
    fn payload_limited_batch_splits_without_dropping_the_envelope() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sanitizer = sanitizer(
            Arc::new(BatchLimitedNameDetector(Arc::clone(&calls))),
            vec!["/*"],
        );

        let sanitized = sanitizer
            .sanitize_json(serde_json::json!({
                "first": "José one",
                "second": "José two",
                "metadata": 7
            }))
            .unwrap();

        assert_eq!(sanitized["first"], "[REDACTED] one");
        assert_eq!(sanitized["second"], "[REDACTED] two");
        assert_eq!(sanitized["metadata"], 7);
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn exact_selectors_match_escaped_json_pointer_segments() {
        let sanitizer = RampartSanitizer::new(
            RampartPiiConfig {
                model_path: "/tmp/rampart".into(),
                target_paths: vec!["/a~1b/c~0d".into()],
                ..RampartPiiConfig::default()
            },
            Arc::new(NameDetector),
        )
        .unwrap();
        let value = serde_json::json!({
            "a/b": {"c~d": "Hello José"},
            "a": {"b": {"c~d": "José"}}
        });

        assert_eq!(
            sanitizer.sanitize_json(value).unwrap(),
            serde_json::json!({
                "a/b": {"c~d": "Hello [REDACTED]"},
                "a": {"b": {"c~d": "José"}}
            })
        );
    }

    #[test]
    fn model_errors_fail_closed_only_for_selected_values() {
        let sanitizer = sanitizer(Arc::new(FailingDetector), vec!["/message"]);
        assert_eq!(
            sanitizer
                .sanitize_json(serde_json::json!({
                    "message": "private",
                    "metadata": "visible"
                }))
                .unwrap(),
            serde_json::json!({
                "message": "[REDACTED]",
                "metadata": "visible"
            })
        );
    }

    #[test]
    fn sparse_selected_field_above_16_kib_is_sanitized() {
        let sanitizer = sanitizer(Arc::new(NameDetector), vec!["/message"]);
        let message = format!("{}José", " ".repeat(16 * 1024));
        let sanitized = sanitizer
            .sanitize_json(serde_json::json!({
                "message": message,
                "metadata": "visible"
            }))
            .unwrap();
        let sanitized_message = sanitized["message"].as_str().unwrap();

        assert_eq!(sanitized_message.len(), 16 * 1024 + "[REDACTED]".len());
        assert!(sanitized_message.ends_with("[REDACTED]"));
        assert_eq!(sanitized["metadata"], "visible");
    }

    #[test]
    fn selected_text_count_limit_redacts_only_excess_unique_fields() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sanitizer = sanitizer(Arc::new(CountingDetector(Arc::clone(&calls))), vec!["/*"]);
        let value = Json::Object(
            (0..=MAX_TEXTS_PER_PAYLOAD)
                .map(|index| {
                    (
                        index.to_string(),
                        Json::String(format!("safe-value-{index}")),
                    )
                })
                .collect(),
        );
        let sanitized = sanitizer.sanitize_json(value).unwrap();
        let redacted = sanitized
            .as_object()
            .unwrap()
            .values()
            .filter(|value| value.as_str() == Some("[REDACTED]"))
            .count();

        assert_eq!(redacted, 1);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn selected_payload_byte_limit_has_an_exact_boundary() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sanitizer = sanitizer(
            Arc::new(CountingDetector(Arc::clone(&calls))),
            vec!["/message"],
        );
        let exact = " ".repeat(MAX_PAYLOAD_TEXT_BYTES);
        let sanitized = sanitizer
            .sanitize_json(serde_json::json!({"message": exact}))
            .unwrap();
        assert_eq!(
            sanitized["message"].as_str().unwrap().len(),
            MAX_PAYLOAD_TEXT_BYTES
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        assert_eq!(
            sanitizer
                .sanitize_json(serde_json::json!({
                    "message": " ".repeat(MAX_PAYLOAD_TEXT_BYTES + 1)
                }))
                .unwrap()["message"],
            "[REDACTED]"
        );
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn aggregate_limit_redacts_only_the_field_beyond_the_budget() {
        let calls = Arc::new(AtomicUsize::new(0));
        let backend = sanitizer(
            Arc::new(CountingDetector(Arc::clone(&calls))),
            vec!["/messages/*/content"],
        );
        let output = tool_sanitize_callback(backend)(
            "tool".into(),
            serde_json::json!({
                "messages": [
                    {"content": "first-private-value"},
                    {"content": " ".repeat(MAX_PAYLOAD_TEXT_BYTES)}
                ],
                "metadata": "visible"
            }),
        )
        .await
        .unwrap();

        assert_eq!(output["messages"][0]["content"], "first-private-value");
        assert_eq!(output["messages"][1]["content"], "[REDACTED]");
        assert_eq!(output["metadata"], "visible");
        assert_eq!(calls.load(Ordering::Relaxed), 1);
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
            sanitizer
                .sanitize_json(value)
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
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
            assert_eq!(started.load(Ordering::Acquire), worker_count);
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

    #[tokio::test(flavor = "current_thread")]
    async fn model_window_limit_fails_closed_only_for_affected_fields() {
        use nemo_relay::api::event::{BaseEvent, MarkEvent};
        use nemo_relay::api::runtime::{LlmSanitizeRequestContext, LlmSanitizeResponseContext};

        let backend = sanitizer(Arc::new(PayloadLimitedDetector), vec!["/message"]);
        let private = "must-not-pass-through";
        let tool = tool_sanitize_callback(backend.clone())(
            "tool".into(),
            serde_json::json!({"message": private, "metadata": "visible"}),
        )
        .await
        .unwrap();
        assert_eq!(tool["message"], "[REDACTED]");
        assert_eq!(tool["metadata"], "visible");

        let event = Arc::new(Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("mark")
                .data(serde_json::json!({"message": private}))
                .metadata(serde_json::json!({"message": private}))
                .build(),
            None,
            None,
        )));
        let event_fields = event_sanitize_callback(backend.clone(), None)(
            Arc::clone(&event),
            event.sanitize_fields(),
        )
        .await
        .unwrap();
        assert_eq!(event_fields.data.unwrap()["message"], "[REDACTED]");
        assert_eq!(event_fields.metadata.unwrap()["message"], "[REDACTED]");

        let request = LlmRequest {
            headers: Map::new(),
            content: serde_json::json!({"message": private}),
        };
        let request = llm_sanitize_request_callback(backend.clone())(
            request,
            LlmSanitizeRequestContext::default(),
        )
        .await
        .unwrap()
        .expect("field-level failure should preserve the request envelope");
        assert_eq!(request.content["message"], "[REDACTED]");
        let response = llm_sanitize_response_callback(backend.clone())(
            serde_json::json!({"message": private}),
            LlmSanitizeResponseContext::default(),
        )
        .await
        .unwrap()
        .expect("field-level failure should preserve the response envelope");
        assert_eq!(response["message"], "[REDACTED]");

        let codec = build_response_codec(ProviderSurface::OpenAIChat);
        let payload = serde_json::json!({
            "id": "chatcmpl-payload-limit",
            "model": "model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": private},
                "finish_reason": "stop"
            }]
        });
        let sanitized = backend
            .sanitize_response_with_codec(codec.as_ref(), ProviderSurface::OpenAIChat, payload)
            .unwrap();
        assert_eq!(sanitized["choices"][0]["message"]["content"], "[REDACTED]");
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

        assert_eq!(
            exact.sanitize_response_with_codec(
                codec.as_ref(),
                ProviderSurface::OpenAIChat,
                payload.clone(),
            ),
            Err(SanitizeError::Codec)
        );
        assert_eq!(
            wildcard.sanitize_response_with_codec(
                codec.as_ref(),
                ProviderSurface::OpenAIChat,
                payload,
            ),
            Err(SanitizeError::Codec)
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
