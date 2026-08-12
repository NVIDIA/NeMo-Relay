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
    FailClosed(FailClosedReason),
}

#[derive(Clone, Copy)]
enum FailClosedReason {
    ModelWindowLimit,
    PayloadLimit,
    SanitizerFailure,
}

impl FailClosedReason {
    fn placeholder(self) -> &'static str {
        match self {
            Self::ModelWindowLimit => "[CONTENT OMITTED: sanitizer model window limit exceeded]",
            Self::PayloadLimit => "[CONTENT OMITTED: sanitizer payload limit exceeded]",
            Self::SanitizerFailure => "[CONTENT OMITTED: sanitizer failed closed]",
        }
    }
}

impl SanitizationDecision {
    fn apply(&self, text: &str, replacement: &str) -> String {
        match self {
            Self::Keep => text.to_string(),
            Self::FailClosed(reason) => reason.placeholder().to_string(),
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
            Self::Keep | Self::FailClosed(_) => 1,
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
        Json::String(FailClosedReason::SanitizerFailure.placeholder().to_string())
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
                } else if pending_keys.len() >= MAX_TEXTS_PER_PAYLOAD
                    || total_bytes
                        .checked_add(text.len())
                        .is_none_or(|next_total| next_total > MAX_PAYLOAD_TEXT_BYTES)
                {
                    *rejected_fields += 1;
                    texts.push(SelectedText::Resolved(
                        FailClosedReason::PayloadLimit.placeholder().to_string(),
                    ));
                } else {
                    *total_bytes += text.len();
                    pending_keys.insert(key);
                    texts.push(SelectedText::Pending {
                        key,
                        text: Some(text.clone()),
                    });
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
                *text = sanitized.get(*index).cloned().unwrap_or_else(|| {
                    FailClosedReason::SanitizerFailure.placeholder().to_string()
                });
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
                let decision =
                    decisions
                        .get(key)
                        .cloned()
                        .unwrap_or(SanitizationDecision::FailClosed(
                            FailClosedReason::SanitizerFailure,
                        ));
                (*key, decision.apply(text, self.replacement.as_ref()))
            })
            .collect::<HashMap<_, _>>();

        texts
            .into_iter()
            .map(|selected| match selected {
                SelectedText::Resolved(text) => text,
                SelectedText::Pending { key, .. } => {
                    rendered.get(&key).cloned().unwrap_or_else(|| {
                        FailClosedReason::SanitizerFailure.placeholder().to_string()
                    })
                }
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
                    let decision =
                        SanitizationDecision::FailClosed(FailClosedReason::ModelWindowLimit);
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
                        decisions.insert(
                            pending[index].0,
                            SanitizationDecision::FailClosed(FailClosedReason::SanitizerFailure),
                        );
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
        // OCI's normalized codec flattens or reconstructs provider-native
        // multipart/tool shapes. Until Relay exposes a lossless delta overlay,
        // the copied observability value must be omitted instead of projected.
        if matches!(
            codec.codec_identity(),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI)
        ) {
            return Err(SanitizeError::Codec);
        }
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
        // See the matching request guard above. The trajectory preset takes a
        // separate raw-JSON path and therefore continues to support OCI.
        if surface == ProviderSurface::OCIGenAI {
            return Err(SanitizeError::Codec);
        }
        if surface == ProviderSurface::OpenAIChat
            && payload
                .get("choices")
                .and_then(Json::as_array)
                .is_some_and(|choices| choices.len() > 1)
            && self.targets_normalized_single_projected_response()
        {
            return Err(SanitizeError::Codec);
        }
        if surface == ProviderSurface::GeminiGenerateContent
            && payload
                .get("candidates")
                .and_then(Json::as_array)
                .is_some_and(|candidates| candidates.len() > 1)
            && self.targets_normalized_single_projected_response()
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

    fn targets_normalized_single_projected_response(&self) -> bool {
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
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OCIGenAI) => Some(ProviderSurface::OCIGenAI),
            LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::GeminiGenerateContent) => {
                Some(ProviderSurface::GeminiGenerateContent)
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
#[path = "../../tests/unit/rampart/sanitizer_tests.rs"]
mod tests;
