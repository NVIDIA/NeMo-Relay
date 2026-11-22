// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::Arc;

use regex::Regex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};

use nemo_relay_codecs::resolve::{
    ProviderSurface, detect_response_surface, request_codec as build_request_codec,
};
use nemo_relay_diagnostics::{
    ComponentRecorder, DiagnosticDetailRef, DiagnosticOutcome, ExtensionHandle, Observation,
};
use nemo_relay_plugin_host::{PluginError, Result as PluginResult};
use nemo_relay_runtime::api::event::{CategoryProfile, Event, MetricEnvelope, ScopeCategory};
use nemo_relay_runtime::api::llm::LlmRequest;
use nemo_relay_runtime::api::runtime::{
    BuiltinLlmCodec, EventSanitizeFn, LlmCodecIdentity, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, ToolSanitizeFn,
};
use nemo_relay_runtime::codec::response_codec as build_response_codec;
use nemo_relay_types::codec::redaction::NormalizedEdits;
use nemo_relay_types::codec::traits::{LlmCodec, LlmResponseCodec};

use super::component::{
    BuiltinBackendConfig, DEFAULT_CUSTOM_MARK_PAYLOAD_POLICY,
    validate_metric_string_attribute_allowlist,
};
use super::detectors::BuiltinDetector;
use super::trajectory::{CustomMarkPayloadPolicy, TrajectorySanitizer, is_relay_metric_mark};

#[derive(Clone)]
pub(super) struct CompiledBuiltinBackend {
    action: BuiltinAction,
    target_path_matcher: Arc<TargetPathMatcher>,
    legacy_surface: Option<ProviderSurface>,
    trajectory: Option<TrajectorySanitizer>,
    match_diagnostics: Option<(ComponentRecorder, ExtensionHandle)>,
}

#[derive(Default)]
struct MatchCounts {
    matched_paths: u64,
    eligible_string_leaves: u64,
    affected_values: u64,
}

impl std::fmt::Display for MatchCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "matched_paths={} eligible_string_leaves={} affected_values={}",
            self.matched_paths, self.eligible_string_leaves, self.affected_values
        )
    }
}

/// Compiled exact-pointer, single-segment, and recursive glob selectors.
#[derive(Clone)]
struct TargetPathMatcher {
    selectors: Vec<TargetPathSelector>,
    may_hide_descendant: bool,
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
enum TargetPathSelector {
    Exact(Vec<String>),
    Glob(Vec<TargetPathSegment>),
    RecursiveGlob(Vec<TargetPathSegment>),
}

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
enum TargetPathSegment {
    Exact(String),
    Any,
    AnyDepth,
}

impl TargetPathSelector {
    fn len(&self) -> usize {
        match self {
            Self::Exact(segments) => segments.len(),
            Self::Glob(segments) | Self::RecursiveGlob(segments) => segments.len(),
        }
    }

    fn segment_at(&self, index: usize) -> Option<&str> {
        match self {
            Self::Exact(segments) => Some(&segments[index]),
            Self::Glob(segments) | Self::RecursiveGlob(segments) => match &segments[index] {
                TargetPathSegment::Exact(value) => Some(value),
                TargetPathSegment::Any | TargetPathSegment::AnyDepth => None,
            },
        }
    }

    fn may_hide(&self, other: &Self) -> bool {
        // A recursive selector can match an ancestor and its descendants, even
        // when it is the only selector. Conservatively count before removal.
        if matches!(self, Self::RecursiveGlob(_)) || matches!(other, Self::RecursiveGlob(_)) {
            return true;
        }
        self.len() < other.len()
            && (0..self.len()).all(|index| {
                match (self.segment_at(index), other.segment_at(index)) {
                    (Some(left), Some(right)) => left == right,
                    _ => true,
                }
            })
    }
}

impl TargetPathMatcher {
    fn new(target_paths: &[String], target_path_globs: &[String]) -> Option<Self> {
        let exact = target_paths
            .iter()
            .map(|path| json_pointer_segments(path).map(TargetPathSelector::Exact))
            .collect::<Option<Vec<_>>>()?;
        let globs = target_path_globs
            .iter()
            .map(|path| {
                json_pointer_segments(path).map(|segments| {
                    if segments
                        .iter()
                        .all(|segment| segment != "*" && segment != "**")
                    {
                        TargetPathSelector::Exact(segments)
                    } else {
                        let recursive = segments.iter().any(|segment| segment == "**");
                        let segments = segments
                            .into_iter()
                            .map(|segment| match segment.as_str() {
                                "*" => TargetPathSegment::Any,
                                "**" => TargetPathSegment::AnyDepth,
                                _ => TargetPathSegment::Exact(segment),
                            })
                            .collect();
                        if recursive {
                            TargetPathSelector::RecursiveGlob(segments)
                        } else {
                            TargetPathSelector::Glob(segments)
                        }
                    }
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut seen = BTreeSet::new();
        let selectors: Vec<_> = exact
            .into_iter()
            .chain(globs)
            .filter(|selector| seen.insert(selector.clone()))
            .collect();
        let may_hide_descendant = selectors
            .iter()
            .any(|shorter| selectors.iter().any(|longer| shorter.may_hide(longer)));
        Some(Self {
            selectors,
            may_hide_descendant,
        })
    }

    fn is_empty(&self) -> bool {
        self.selectors.is_empty()
    }

    fn matches(&self, path_segments: &[String]) -> bool {
        self.selectors.iter().any(|selector| match selector {
            TargetPathSelector::Exact(segments) => segments == path_segments,
            TargetPathSelector::RecursiveGlob(segments) => {
                recursive_glob_matches(segments, path_segments, false)
            }
            TargetPathSelector::Glob(segments) => {
                segments.len() == path_segments.len()
                    && segments.iter().zip(path_segments).all(|(selector, segment)| {
                        matches!(selector, TargetPathSegment::Any)
                            || matches!(selector, TargetPathSegment::Exact(value) if value == segment)
                    })
            }
        })
    }

    fn may_match_descendant(&self, path_segments: &[String]) -> bool {
        self.selectors.iter().any(|selector| match selector {
            TargetPathSelector::RecursiveGlob(segments) => {
                recursive_glob_matches(segments, path_segments, true)
            }
            TargetPathSelector::Exact(segments) => {
                segments.len() > path_segments.len() && segments.starts_with(path_segments)
            }
            TargetPathSelector::Glob(segments) => {
                segments.len() > path_segments.len()
                    && segments.iter().zip(path_segments).all(|(selector, segment)| {
                        matches!(selector, TargetPathSegment::Any)
                            || matches!(selector, TargetPathSegment::Exact(value) if value == segment)
                    })
            }
        })
    }

    fn may_select_single_projected_response(&self) -> bool {
        self.selectors.iter().any(|selector| {
            let first = match selector {
                TargetPathSelector::Exact(segments) => segments
                    .first()
                    .map(|segment| TargetPathSegment::Exact(segment.clone())),
                TargetPathSelector::Glob(segments)
                | TargetPathSelector::RecursiveGlob(segments) => segments.first().cloned(),
            };
            match first {
                Some(TargetPathSegment::Any | TargetPathSegment::AnyDepth) => true,
                Some(TargetPathSegment::Exact(segment)) => matches!(
                    segment.as_str(),
                    "message" | "tool_calls" | "finish_reason" | "api_specific"
                ),
                None => false,
            }
        })
    }
}

/// Match whole path segments, retrying only the most recent `**`. This uses
/// constant scratch space and at most O(pattern segments * path segments)
/// segment comparisons; literal comparisons also depend on key byte length.
/// There is no recursive or exponential backtracking. Prefix mode asks whether
/// some strict descendant can match, for diagnostic traversal pruning.
fn recursive_glob_matches(
    pattern: &[TargetPathSegment],
    path: &[String],
    descendant: bool,
) -> bool {
    let (mut pattern_index, mut path_index) = (0, 0);
    let mut recursive = None;
    while path_index < path.len() {
        match pattern.get(pattern_index) {
            Some(TargetPathSegment::AnyDepth) => {
                recursive = Some((pattern_index, path_index));
                pattern_index += 1;
            }
            Some(TargetPathSegment::Any) => {
                pattern_index += 1;
                path_index += 1;
            }
            Some(TargetPathSegment::Exact(value)) if value == &path[path_index] => {
                pattern_index += 1;
                path_index += 1;
            }
            _ => {
                let Some((recursive_index, consumed)) = recursive else {
                    return false;
                };
                let consumed = consumed + 1;
                recursive = Some((recursive_index, consumed));
                pattern_index = recursive_index + 1;
                path_index = consumed;
            }
        }
    }
    if descendant {
        // An unconsumed suffix can be supplied by a descendant. A consumed
        // `**` can also absorb more segments before matching the suffix again.
        return pattern_index < pattern.len() || recursive.is_some();
    }
    pattern[pattern_index..]
        .iter()
        .all(|segment| matches!(segment, TargetPathSegment::AnyDepth))
}

#[derive(Clone)]
enum BuiltinAction {
    Remove,
    Hash {
        matcher: Option<Arc<BuiltinMatcher>>,
    },
    Mask {
        matcher: Option<Arc<BuiltinMatcher>>,
        strategy: BuiltinMaskStrategy,
    },
    Redact {
        matcher: Arc<BuiltinMatcher>,
        replacement: Arc<String>,
    },
    RegexReplace {
        pattern: Arc<BuiltinMatcher>,
        replacement: Arc<String>,
    },
}

#[derive(Clone)]
struct BuiltinMatcher {
    regex: Arc<Regex>,
}

impl BuiltinMatcher {
    fn replace_all(&self, text: &str, mut replacement: impl FnMut(&str) -> String) -> String {
        let mut output = String::with_capacity(text.len());
        let mut cursor = 0;
        for matched in self.regex.find_iter(text) {
            let matched_text = matched.as_str();
            output.push_str(&text[cursor..matched.start()]);
            output.push_str(&replacement(matched_text));
            cursor = matched.end();
        }
        output.push_str(&text[cursor..]);
        output
    }

    fn replace_all_expanded(&self, text: &str, replacement: &str) -> String {
        let mut output = String::with_capacity(text.len());
        let mut cursor = 0;
        for captures in self.regex.captures_iter(text) {
            let matched = captures
                .get(0)
                .expect("regular-expression captures always include the full match");
            output.push_str(&text[cursor..matched.start()]);
            captures.expand(replacement, &mut output);
            cursor = matched.end();
        }
        output.push_str(&text[cursor..]);
        output
    }
}

#[derive(Clone)]
enum BuiltinMaskStrategy {
    Generic {
        mask_char: Arc<String>,
        unmasked_prefix: usize,
        unmasked_suffix: usize,
    },
    DetectorDefault {
        detector: BuiltinDetector,
        mask_char: Arc<String>,
    },
}

impl CompiledBuiltinBackend {
    pub(super) fn new(
        config: BuiltinBackendConfig,
        codec_name: Option<String>,
    ) -> PluginResult<Self> {
        for (index, target_path) in config.target_paths.iter().enumerate() {
            if !is_valid_json_pointer(target_path) {
                return Err(PluginError::InvalidConfig(format!(
                    "builtin.target_paths[{index}] must be a valid RFC 6901 JSON pointer"
                )));
            }
        }
        for (index, target_path_glob) in config.target_path_globs.iter().enumerate() {
            if !is_valid_json_pointer(target_path_glob) {
                return Err(PluginError::InvalidConfig(format!(
                    "builtin.target_path_globs[{index}] must be a valid RFC 6901 JSON pointer"
                )));
            }
        }
        let trajectory = match config.preset.as_deref() {
            Some("trajectory_context") => {
                validate_metric_string_attribute_allowlist(
                    &config.metric_string_attribute_allowlist,
                )
                .map_err(PluginError::InvalidConfig)?;
                if config.detector.is_some()
                    || config.pattern.is_some()
                    || !config.target_paths.is_empty()
                    || !config.target_path_globs.is_empty()
                    || config.mask_char.is_some()
                    || config.unmasked_prefix.is_some()
                    || config.unmasked_suffix.is_some()
                {
                    return Err(PluginError::InvalidConfig(
                        "builtin.preset cannot be combined with matcher, target-path, or mask fields"
                            .to_string(),
                    ));
                }
                let policy = CustomMarkPayloadPolicy::parse(&config.custom_mark_payload_policy)
                    .ok_or_else(|| {
                        PluginError::InvalidConfig(format!(
                            "unsupported custom-mark payload policy '{}'",
                            config.custom_mark_payload_policy
                        ))
                    })?;
                Some(TrajectorySanitizer::new(
                    config
                        .replacement
                        .clone()
                        .unwrap_or_else(|| "[REDACTED]".to_string()),
                    policy,
                    config.metric_string_attribute_allowlist.clone(),
                ))
            }
            Some(other) => {
                return Err(PluginError::InvalidConfig(format!(
                    "unsupported builtin preset '{other}'"
                )));
            }
            None if !config.metric_string_attribute_allowlist.is_empty() => {
                return Err(PluginError::InvalidConfig(
                    "builtin.metric_string_attribute_allowlist requires builtin.preset = 'trajectory_context'"
                        .to_string(),
                ));
            }
            None => None,
        };
        if trajectory.is_none()
            && config.custom_mark_payload_policy != DEFAULT_CUSTOM_MARK_PAYLOAD_POLICY
        {
            return Err(PluginError::InvalidConfig(
                "builtin.custom_mark_payload_policy requires builtin.preset = 'trajectory_context'"
                    .to_string(),
            ));
        }
        let detector = config
            .detector
            .as_deref()
            .map(BuiltinDetector::parse)
            .transpose()?;
        let matcher = compile_builtin_matcher(config.pattern.clone(), detector)?;
        let action = match config.action.as_str() {
            "remove" => BuiltinAction::Remove,
            "hash" => BuiltinAction::Hash { matcher },
            "mask" => BuiltinAction::Mask {
                matcher,
                strategy: build_mask_strategy(&config, detector),
            },
            "redact" | "regex_replace" => {
                let pattern = matcher.ok_or_else(|| {
                    PluginError::InvalidConfig(
                        "builtin.pattern or builtin.detector is required when builtin.action = 'regex_replace' or 'redact'".to_string(),
                    )
                })?;
                let replacement = Arc::new(
                    config
                        .replacement
                        .unwrap_or_else(|| "[REDACTED]".to_string()),
                );
                if config.action == "redact" {
                    BuiltinAction::Redact {
                        matcher: pattern,
                        replacement,
                    }
                } else {
                    BuiltinAction::RegexReplace {
                        pattern,
                        replacement,
                    }
                }
            }
            other => {
                return Err(PluginError::InvalidConfig(format!(
                    "unsupported builtin.action '{other}'"
                )));
            }
        };

        let surface = match codec_name.as_deref() {
            Some(name) => Some(ProviderSurface::from_codec_name(name).ok_or_else(|| {
                PluginError::InvalidConfig(format!("unsupported codec '{name}'"))
            })?),
            None => None,
        };
        let target_paths = TargetPathMatcher::new(&config.target_paths, &config.target_path_globs)
            .ok_or_else(|| {
                PluginError::InvalidConfig(
                    "builtin target paths must be valid RFC 6901 JSON pointers".to_string(),
                )
            })?;
        if trajectory.is_none()
            && matches!(action, BuiltinAction::Remove)
            && target_paths.is_empty()
        {
            return Err(PluginError::InvalidConfig(
                "builtin.action = 'remove' requires at least one builtin.target_paths or builtin.target_path_globs selector"
                    .to_string(),
            ));
        }

        Ok(Self {
            action,
            target_path_matcher: Arc::new(target_paths),
            legacy_surface: surface,
            trajectory,
            match_diagnostics: None,
        })
    }

    pub(super) fn with_match_diagnostics(
        mut self,
        recorder: ComponentRecorder,
        code: ExtensionHandle,
    ) -> Self {
        self.match_diagnostics = Some((recorder, code));
        self
    }

    fn sanitize_json_preorder_dfs(&self, value: Json) -> Json {
        let selected = if self.match_diagnostics.is_some()
            && matches!(self.action, BuiltinAction::Remove)
            && self.target_path_matcher.may_hide_descendant
        {
            let mut selected = MatchCounts::default();
            self.count_selections_at_path(&value, &mut Vec::new(), &mut selected);
            Some(selected)
        } else {
            None
        };
        let mut counts = MatchCounts::default();
        let result = self
            .sanitize_json_preorder_dfs_at_path(value, &mut Vec::new(), &mut counts)
            .unwrap_or(Json::Null);
        if let Some(selected) = selected {
            counts.matched_paths = selected.matched_paths;
            counts.eligible_string_leaves = selected.eligible_string_leaves;
        }
        if let Some((recorder, code)) = &self.match_diagnostics {
            let _ = recorder.record_lazy(|| {
                Observation::extension(
                    *code,
                    None,
                    DiagnosticOutcome::Success,
                    DiagnosticDetailRef::Display(&counts),
                    NonZeroU64::MIN,
                )
            });
        }
        result
    }

    fn count_selections_at_path(
        &self,
        value: &Json,
        path_segments: &mut Vec<String>,
        counts: &mut MatchCounts,
    ) {
        if self.target_path_matcher.matches(path_segments) {
            counts.matched_paths = counts.matched_paths.saturating_add(1);
            if value.is_string() {
                counts.eligible_string_leaves = counts.eligible_string_leaves.saturating_add(1);
            }
        }
        if !self.target_path_matcher.may_match_descendant(path_segments) {
            return;
        }
        match value {
            Json::Array(items) => {
                for (index, item) in items.iter().enumerate() {
                    path_segments.push(index.to_string());
                    self.count_selections_at_path(item, path_segments, counts);
                    path_segments.pop();
                }
            }
            Json::Object(map) => {
                for (key, item) in map {
                    path_segments.push(key.clone());
                    self.count_selections_at_path(item, path_segments, counts);
                    path_segments.pop();
                }
            }
            _ => {}
        }
    }

    /// Sanitize optional metric text without modifying required export fields.
    fn sanitize_metric_envelope(&self, data: Json) -> Option<Json> {
        let mut envelope = match serde_json::from_value::<MetricEnvelope>(data) {
            Ok(envelope) => envelope,
            Err(_) => {
                return log_metric_envelope_omitted("metric envelope deserialization failure");
            }
        };
        if envelope.validate().is_err() {
            return log_metric_envelope_omitted(
                "metric envelope validation failure before redaction",
            );
        }
        for (index, measurement) in envelope.measurements.iter_mut().enumerate() {
            measurement.description = measurement.description.take().and_then(|description| {
                self.sanitize_metric_string_at_path(
                    description,
                    &[
                        "measurements".to_string(),
                        index.to_string(),
                        "description".to_string(),
                    ],
                )
            });
            measurement.attributes = measurement
                .attributes
                .take()
                .map(|attributes| self.sanitize_metric_attributes(attributes, index));
        }
        if envelope.validate().is_err() {
            return log_metric_envelope_omitted(
                "metric envelope validation failure after redaction",
            );
        }
        match serde_json::to_value(envelope) {
            Ok(data) => Some(data),
            Err(_) => log_metric_envelope_omitted("metric envelope serialization failure"),
        }
    }

    /// Sanitize metric text using its payload-relative JSON Pointer.
    fn sanitize_metric_string_at_path(
        &self,
        value: String,
        path_segments: &[String],
    ) -> Option<String> {
        let mut path_segments = path_segments.to_vec();
        self.sanitize_json_preorder_dfs_at_path(
            Json::String(value),
            &mut path_segments,
            &mut MatchCounts::default(),
        )
        .and_then(|value| value.as_str().map(str::to_string))
    }

    /// Sanitize string-valued attributes while retaining analytical values.
    fn sanitize_metric_attributes(&self, attributes: Json, measurement_index: usize) -> Json {
        let Json::Object(attributes) = attributes else {
            return attributes;
        };
        Json::Object(
            attributes
                .into_iter()
                .filter_map(|(key, value)| {
                    self.sanitize_metric_attribute(value, measurement_index, &key)
                        .map(|value| (key, value))
                })
                .collect(),
        )
    }

    /// Sanitize one metric attribute without changing its scalar type.
    fn sanitize_metric_attribute(
        &self,
        value: Json,
        measurement_index: usize,
        attribute_key: &str,
    ) -> Option<Json> {
        let base_path = [
            "measurements".to_string(),
            measurement_index.to_string(),
            "attributes".to_string(),
            attribute_key.to_string(),
        ];
        match value {
            Json::String(value) => self
                .sanitize_metric_string_at_path(value, &base_path)
                .map(Json::String),
            Json::Array(values) if values.iter().all(Json::is_string) => {
                if matches!(self.action, BuiltinAction::Remove)
                    && self.matches_current_preorder_path(&base_path)
                {
                    return None;
                }
                Some(Json::Array(
                    values
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, value)| match value {
                            Json::String(value) => self
                                .sanitize_metric_string_at_path(
                                    value,
                                    &[
                                        base_path[0].clone(),
                                        base_path[1].clone(),
                                        base_path[2].clone(),
                                        base_path[3].clone(),
                                        index.to_string(),
                                    ],
                                )
                                .map(Json::String),
                            _ => unreachable!("checked all metric attribute values are strings"),
                        })
                        .collect(),
                ))
            }
            value => Some(value),
        }
    }

    fn sanitize_tool_result_annotation(&self, profile: &mut CategoryProfile) {
        let Some(annotation) = profile.tool_result_annotation.take() else {
            return;
        };
        let sanitized = self.sanitize_json_preorder_dfs(annotation);
        if !sanitized.is_null() {
            profile.tool_result_annotation = Some(sanitized);
        }
    }

    fn sanitize_json_preorder_dfs_at_path(
        &self,
        value: Json,
        path_segments: &mut Vec<String>,
        counts: &mut MatchCounts,
    ) -> Option<Json> {
        let matched = self.matches_current_preorder_path(path_segments);
        if matched {
            counts.matched_paths = counts.matched_paths.saturating_add(1);
        }
        if !self.target_path_matcher.is_empty()
            && matched
            && matches!(self.action, BuiltinAction::Remove)
        {
            if matches!(&value, Json::String(_)) {
                counts.eligible_string_leaves = counts.eligible_string_leaves.saturating_add(1);
            }
            counts.affected_values = counts.affected_values.saturating_add(1);
            return None;
        }

        match value {
            Json::String(text) => {
                if matched {
                    counts.eligible_string_leaves = counts.eligible_string_leaves.saturating_add(1);
                    let original = self.match_diagnostics.as_ref().map(|_| text.clone());
                    let sanitized = self.sanitize_string_value(text);
                    if original
                        .is_some_and(|original| sanitized.as_ref() != Some(&Json::String(original)))
                    {
                        counts.affected_values = counts.affected_values.saturating_add(1);
                    }
                    sanitized
                } else {
                    Some(Json::String(text))
                }
            }
            Json::Array(items) => Some(Json::Array(
                items
                    .into_iter()
                    .enumerate()
                    .map(|(index, item)| {
                        path_segments.push(index.to_string());
                        let sanitized = self
                            .sanitize_json_preorder_dfs_at_path(item, path_segments, counts)
                            .unwrap_or(Json::Null);
                        path_segments.pop();
                        sanitized
                    })
                    .collect(),
            )),
            Json::Object(map) => Some(Json::Object(
                map.into_iter()
                    .filter_map(|(key, value)| {
                        path_segments.push(key.clone());
                        let sanitized =
                            self.sanitize_json_preorder_dfs_at_path(value, path_segments, counts);
                        path_segments.pop();
                        sanitized.map(|sanitized| (key, sanitized))
                    })
                    .collect(),
            )),
            other => Some(other),
        }
    }

    fn matches_current_preorder_path(&self, path_segments: &[String]) -> bool {
        if self.target_path_matcher.is_empty() {
            return true;
        }
        self.target_path_matcher.matches(path_segments)
    }

    fn sanitize_string_value(&self, text: String) -> Option<Json> {
        match &self.action {
            BuiltinAction::Remove => None,
            BuiltinAction::Hash { matcher } => Some(Json::String(match matcher {
                Some(matcher) => matcher.replace_all(&text, hex_sha256),
                None => hex_sha256(&text),
            })),
            BuiltinAction::Mask { matcher, strategy } => Some(Json::String(match matcher {
                Some(matcher) => {
                    matcher.replace_all(&text, |matched| mask_with_strategy(matched, strategy))
                }
                None => mask_with_strategy(&text, strategy),
            })),
            BuiltinAction::Redact {
                matcher,
                replacement,
            } => Some(Json::String(
                matcher.replace_all(&text, |_| replacement.as_str().to_string()),
            )),
            BuiltinAction::RegexReplace {
                pattern,
                replacement,
            } => Some(Json::String(
                pattern.replace_all_expanded(&text, replacement),
            )),
        }
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

    pub(super) fn is_trajectory(&self) -> bool {
        self.trajectory.is_some()
    }

    fn uses_compatible_legacy_response_codec(&self, payload: &Json) -> bool {
        self.legacy_surface
            .is_some_and(|surface| detect_response_surface(payload) == Some(surface))
    }

    fn sanitize_request_with_codec(
        &self,
        codec: &dyn LlmCodec,
        request: &LlmRequest,
    ) -> Option<LlmRequest> {
        let capability = codec.redaction()?;
        let baseline = serde_json::to_value(codec.decode(request).ok()?).ok()?;
        let sanitized = self.sanitize_json_preorder_dfs(baseline.clone());
        capability
            .edit_request(
                request,
                NormalizedEdits {
                    baseline: &baseline,
                    sanitized: &sanitized,
                    require_complete_output: self
                        .target_path_matcher
                        .may_select_single_projected_response(),
                },
            )
            .ok()?
    }

    fn sanitize_request_headers(&self, headers: Map<String, Json>) -> Map<String, Json> {
        let sanitized = self.sanitize_json_preorder_dfs(Json::Object(
            [("headers".to_string(), Json::Object(headers))]
                .into_iter()
                .collect(),
        ));
        sanitized
            .get("headers")
            .and_then(Json::as_object)
            .cloned()
            .unwrap_or_default()
    }

    fn sanitize_response_annotation(&self, identity: LlmCodecIdentity, annotated: Json) -> Json {
        // OpenAI responses with refusals formerly exposed ordinary text as a scalar.
        // Keep policies that selected that scalar effective when the codec emits parts.
        if matches!(
            identity,
            LlmCodecIdentity::BuiltIn(
                BuiltinLlmCodec::OpenAiChat | BuiltinLlmCodec::OpenAiResponses
            )
        ) && annotated.get("message").is_some_and(Json::is_array)
            && !matches!(self.action, BuiltinAction::Remove)
            && self.target_path_matcher.matches(&["message".to_owned()])
        {
            let mut backend = self.clone();
            let matcher = Arc::make_mut(&mut backend.target_path_matcher);
            for field in ["text", "refusal"] {
                matcher.selectors.push(TargetPathSelector::Glob(vec![
                    TargetPathSegment::Exact("message".to_owned()),
                    TargetPathSegment::Any,
                    TargetPathSegment::Exact(field.to_owned()),
                ]));
            }
            return backend.sanitize_json_preorder_dfs(annotated);
        }
        self.sanitize_json_preorder_dfs(annotated)
    }

    fn sanitize_response_with_codec(
        &self,
        codec: &dyn LlmResponseCodec,
        payload: Json,
    ) -> Option<Json> {
        let capability = codec.redaction()?;
        let baseline = serde_json::to_value(codec.decode_response(&payload).ok()?).ok()?;
        let sanitized = self.sanitize_response_annotation(codec.codec_identity(), baseline.clone());
        // Resolve wire removals while all indexes still refer to the source.
        // A codec may compact the arrays when it applies normalized removals.
        let removing = matches!(self.action, BuiltinAction::Remove);
        let (payload, baseline, sanitized) = if removing {
            let wire = self.sanitize_json_preorder_dfs(payload);
            let decoded = serde_json::to_value(codec.decode_response(&wire).ok()?).ok()?;
            let merged = merge_annotation_edits(&baseline, &sanitized, &decoded)?;
            (wire, decoded, merged)
        } else {
            (payload, baseline, sanitized)
        };
        let payload = capability
            .edit_response(
                &payload,
                NormalizedEdits {
                    baseline: &baseline,
                    sanitized: &sanitized,
                    require_complete_output: self
                        .target_path_matcher
                        .may_select_single_projected_response(),
                },
            )
            .ok()??;
        Some(if removing {
            payload
        } else {
            self.sanitize_json_preorder_dfs(payload)
        })
    }
}

/// Reconcile independent wire and normalized edits against their common source.
/// If both changed a slot differently, its original meaning must remain provable.
fn merge_annotation_edits(source: &Json, normalized: &Json, wire: &Json) -> Option<Json> {
    if normalized == source || normalized == wire {
        return Some(wire.clone());
    }
    if wire == source {
        return Some(normalized.clone());
    }
    match (source, normalized, wire) {
        (Json::Object(before), Json::Object(after), Json::Object(raw)) => {
            let mut result = raw.clone();
            for (key, original) in before {
                match (after.get(key), raw.get(key)) {
                    (Some(edited), Some(current)) if edited != original => {
                        result.insert(
                            key.clone(),
                            merge_annotation_edits(original, edited, current)?,
                        );
                    }
                    (None, _) => {
                        result.remove(key);
                    }
                    (Some(edited), None) if edited != original => return None,
                    _ => {}
                }
            }
            // A policy can remove fields or transform values, never introduce a new key.
            after
                .keys()
                .all(|key| before.contains_key(key))
                .then_some(Json::Object(result))
        }
        (Json::Array(before), Json::Array(after), Json::Array(raw))
            if before.len() == after.len() && before.len() == raw.len() =>
        {
            before
                .iter()
                .zip(after)
                .zip(raw)
                .map(|((original, edited), current)| {
                    merge_annotation_edits(original, edited, current)
                })
                .collect::<Option<Vec<_>>>()
                .map(Json::Array)
        }
        _ => None,
    }
}

pub(super) fn tool_sanitize_callback(backend: CompiledBuiltinBackend) -> ToolSanitizeFn {
    let backend = Arc::new(backend);
    Arc::new(move |_name: String, payload: Json| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
            Ok(match backend.trajectory.as_ref() {
                Some(trajectory) => trajectory.sanitize_tool_payload(payload),
                None => backend.sanitize_json_preorder_dfs(payload),
            })
        })
    })
}

pub(super) fn event_sanitize_callback(backend: CompiledBuiltinBackend) -> EventSanitizeFn {
    event_sanitize_callback_with_scope_categories(backend, None)
}

pub(super) fn scope_event_sanitize_callback(
    backend: CompiledBuiltinBackend,
    sanitize_llm: bool,
    sanitize_tool: bool,
) -> EventSanitizeFn {
    event_sanitize_callback_with_scope_categories(backend, Some((sanitize_llm, sanitize_tool)))
}

fn event_sanitize_callback_with_scope_categories(
    backend: CompiledBuiltinBackend,
    scope_categories: Option<(bool, bool)>,
) -> EventSanitizeFn {
    let backend = Arc::new(backend);
    Arc::new(move |event, mut fields| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
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
                return Ok(fields);
            }

            if let Some(trajectory) = backend.trajectory.as_ref() {
                return Ok(trajectory.sanitize_event_fields(&event, fields));
            }
            if is_relay_metric_mark(&event) {
                fields.data = fields
                    .data
                    .and_then(|data| backend.sanitize_metric_envelope(data));
                fields.metadata = fields
                    .metadata
                    .map(|metadata| backend.sanitize_json_preorder_dfs(metadata));
                fields.category_profile = fields
                    .category_profile
                    .and_then(|profile| sanitize_category_profile_with_backend(&backend, profile));
                return Ok(fields);
            }
            let specialized_scope = matches!(event.as_ref(), Event::Scope(_))
                && event
                    .category()
                    .is_some_and(|category| matches!(category.as_str(), "tool" | "llm"));

            if specialized_scope {
                let sanitize_tool_annotation = scope_categories
                    .is_some_and(|(_, sanitize_tool)| sanitize_tool)
                    && event.scope_category() == Some(ScopeCategory::End)
                    && event
                        .category()
                        .is_some_and(|category| category.as_str() == "tool");
                if sanitize_tool_annotation {
                    fields.category_profile = fields.category_profile.map(|mut profile| {
                        backend.sanitize_tool_result_annotation(&mut profile);
                        profile
                    });
                }
            } else {
                fields.data = fields
                    .data
                    .map(|data| backend.sanitize_json_preorder_dfs(data));
                fields.category_profile = fields
                    .category_profile
                    .and_then(|profile| sanitize_category_profile_with_backend(&backend, profile));
            }

            fields.metadata = fields
                .metadata
                .map(|metadata| backend.sanitize_json_preorder_dfs(metadata));
            Ok(fields)
        })
    })
}

pub(super) fn llm_sanitize_request_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeRequestFn {
    let backend = Arc::new(backend);
    Arc::new(move |mut request: LlmRequest, context| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
            request.headers = backend.sanitize_request_headers(request.headers);
            if backend.target_path_matcher.is_empty() {
                request.content = backend.sanitize_json_preorder_dfs(request.content);
                return Ok(Some(request));
            }
            let resolved = context.resolve_codec();
            let fallback = if resolved.is_none() {
                backend
                    .selected_surface(context.codec())
                    .map(build_request_codec)
            } else {
                None
            };
            let Some(codec) = resolved.as_deref().or(fallback.as_deref()) else {
                log_llm_payload_omitted("request", context.codec(), "no usable request codec");
                return Ok(None);
            };
            let sanitized = backend.sanitize_request_with_codec(codec, &request);
            if sanitized.is_none() {
                log_llm_payload_omitted(
                    "request",
                    context.codec(),
                    "codec decode, sanitize, or encode failure",
                );
            }
            Ok(sanitized)
        })
    })
}

pub(super) fn trajectory_llm_request_projection_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeRequestFn {
    let backend = Arc::new(backend);
    Arc::new(move |request: LlmRequest, context| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
            if matches!(context.codec(), LlmCodecIdentity::None) {
                return Ok(backend.legacy_surface.is_some().then_some(request));
            }
            let projected = (|| {
                let resolved = context.resolve_codec();
                let fallback = if resolved.is_none() {
                    backend
                        .selected_surface(context.codec())
                        .map(build_request_codec)
                } else {
                    None
                };
                let codec = resolved.as_deref().or(fallback.as_deref())?;
                let annotated = codec.decode(&request).ok()?;
                let sanitized = backend
                    .trajectory
                    .as_ref()?
                    .sanitize_annotated_request(annotated)?;
                codec.redaction()?.project_request(&sanitized).ok()?
            })();
            Ok(projected)
        })
    })
}

pub(super) fn llm_sanitize_response_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeResponseFn {
    let backend = Arc::new(backend);
    Arc::new(move |payload: Json, context| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
            if backend.target_path_matcher.is_empty() {
                return Ok(Some(backend.sanitize_json_preorder_dfs(payload)));
            }
            if matches!(context.codec(), LlmCodecIdentity::None)
                && !backend.uses_compatible_legacy_response_codec(&payload)
            {
                log_llm_payload_omitted(
                    "response",
                    context.codec(),
                    "no active response codec or compatible legacy codec",
                );
                return Ok(None);
            }
            let resolved = context.resolve_codec();
            let fallback = if resolved.is_none() {
                backend
                    .selected_surface(context.codec())
                    .map(build_response_codec)
            } else {
                None
            };
            let Some(codec) = resolved.as_deref().or(fallback.as_deref()) else {
                log_llm_payload_omitted("response", context.codec(), "no usable response codec");
                return Ok(None);
            };
            let sanitized = backend.sanitize_response_with_codec(codec, payload);
            if sanitized.is_none() {
                log_llm_payload_omitted(
                    "response",
                    context.codec(),
                    "codec decode, sanitize, or encode failure",
                );
            }
            Ok(sanitized)
        })
    })
}

pub(super) fn trajectory_llm_response_projection_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeResponseFn {
    let backend = Arc::new(backend);
    Arc::new(move |payload: Json, context| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
            if matches!(context.codec(), LlmCodecIdentity::None) {
                return Ok(backend.legacy_surface.is_some().then_some(payload));
            }
            let projected = (|| {
                let resolved = context.resolve_codec();
                let fallback = if resolved.is_none() {
                    backend
                        .selected_surface(context.codec())
                        .map(build_response_codec)
                } else {
                    None
                };
                let codec = resolved.as_deref().or(fallback.as_deref())?;
                let annotated = codec.decode_response(&payload).ok()?;
                let sanitized = backend
                    .trajectory
                    .as_ref()?
                    .sanitize_annotated_response(annotated)?;
                codec.redaction()?.project_response(&sanitized).ok()?
            })();
            Ok(projected)
        })
    })
}

pub(super) fn is_valid_json_pointer(pointer: &str) -> bool {
    if pointer.is_empty() {
        return true;
    }
    pointer.strip_prefix('/').is_some_and(|path| {
        path.split('/').all(|segment| {
            let mut characters = segment.chars();
            while let Some(character) = characters.next() {
                if character == '~' && !matches!(characters.next(), Some('0' | '1')) {
                    return false;
                }
            }
            true
        })
    })
}

fn log_llm_payload_omitted(direction: &str, codec: &LlmCodecIdentity, reason: &str) {
    let codec_kind = match codec {
        LlmCodecIdentity::None => "none",
        LlmCodecIdentity::BuiltIn(_) => "builtin",
        LlmCodecIdentity::Runtime(_) => "runtime",
        LlmCodecIdentity::Opaque => "opaque",
    };
    log::warn!(
        target: "nemo_relay.plugin",
        event = "pii_llm_payload_omitted",
        codec_kind,
        reason;
        "PII redaction omitted an LLM {direction} payload"
    );
}

fn log_metric_envelope_omitted(reason: &str) -> Option<Json> {
    log::warn!(
        target: "nemo_relay.plugin",
        event = "pii_metric_envelope_omitted",
        reason;
        "PII redaction omitted a metric envelope"
    );
    None
}

fn sanitize_category_profile_with_backend(
    backend: &CompiledBuiltinBackend,
    profile: CategoryProfile,
) -> Option<CategoryProfile> {
    match sanitize_serializable_with_backend(backend, profile) {
        Ok(profile) => Some(profile),
        Err(_) => {
            log::warn!(
                target: "nemo_relay.plugin",
                event = "pii_category_profile_omitted",
                reason = "category profile redaction round-trip failure";
                "PII redaction omitted a category profile"
            );
            None
        }
    }
}

fn json_pointer_segments(pointer: &str) -> Option<Vec<String>> {
    if pointer.is_empty() {
        return Some(Vec::new());
    }
    pointer
        .strip_prefix('/')
        .map(|path| path.split('/').map(unescape_json_pointer_segment).collect())
}

fn unescape_json_pointer_segment(segment: &str) -> String {
    segment.replace("~1", "/").replace("~0", "~")
}

pub(crate) fn hex_sha256(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

pub(crate) fn mask_text(
    text: &str,
    mask_char: &str,
    unmasked_prefix: usize,
    unmasked_suffix: usize,
) -> String {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len <= unmasked_prefix.saturating_add(unmasked_suffix) {
        return text.to_string();
    }

    let mut output = String::new();
    for ch in chars.iter().take(unmasked_prefix) {
        output.push(*ch);
    }
    for _ in 0..(len - unmasked_prefix - unmasked_suffix) {
        output.push_str(mask_char);
    }
    for ch in chars.iter().skip(len - unmasked_suffix) {
        output.push(*ch);
    }
    output
}

fn build_mask_strategy(
    config: &BuiltinBackendConfig,
    detector: Option<BuiltinDetector>,
) -> BuiltinMaskStrategy {
    let mask_char = Arc::new(config.mask_char.clone().unwrap_or_else(|| "*".to_string()));
    match detector {
        Some(detector) if config.unmasked_prefix.is_none() && config.unmasked_suffix.is_none() => {
            BuiltinMaskStrategy::DetectorDefault {
                detector,
                mask_char,
            }
        }
        _ => BuiltinMaskStrategy::Generic {
            mask_char,
            unmasked_prefix: config.unmasked_prefix.unwrap_or(0),
            unmasked_suffix: config.unmasked_suffix.unwrap_or(0),
        },
    }
}

fn mask_with_strategy(text: &str, strategy: &BuiltinMaskStrategy) -> String {
    match strategy {
        BuiltinMaskStrategy::Generic {
            mask_char,
            unmasked_prefix,
            unmasked_suffix,
        } => mask_text(text, mask_char.as_str(), *unmasked_prefix, *unmasked_suffix),
        BuiltinMaskStrategy::DetectorDefault {
            detector,
            mask_char,
        } => detector.default_mask(text, mask_char.as_str()),
    }
}

fn compile_builtin_matcher(
    pattern: Option<String>,
    detector: Option<BuiltinDetector>,
) -> PluginResult<Option<Arc<BuiltinMatcher>>> {
    let pattern_text = match (pattern, detector) {
        (Some(pattern), None) => Some(pattern),
        (None, Some(detector)) => Some(detector.regex_pattern().to_string()),
        (None, None) => None,
        (Some(_), Some(_)) => {
            return Err(PluginError::InvalidConfig(
                "builtin.pattern and builtin.detector cannot both be set".to_string(),
            ));
        }
    };

    let Some(pattern_text) = pattern_text else {
        return Ok(None);
    };

    let pattern = Regex::new(&pattern_text).map_err(|err| {
        PluginError::InvalidConfig(format!(
            "invalid builtin matcher regex '{pattern_text}': {err}"
        ))
    })?;
    Ok(Some(Arc::new(BuiltinMatcher {
        regex: Arc::new(pattern),
    })))
}

fn sanitize_serializable_with_backend<T>(
    backend: &CompiledBuiltinBackend,
    value: T,
) -> PluginResult<T>
where
    T: Serialize + DeserializeOwned,
{
    let value = serde_json::to_value(value).map_err(|err| {
        PluginError::Internal(format!(
            "failed to serialize value for PII redaction: {err}"
        ))
    })?;
    serde_json::from_value(backend.sanitize_json_preorder_dfs(value)).map_err(|err| {
        PluginError::Internal(format!(
            "failed to deserialize sanitized value for PII redaction: {err}"
        ))
    })
}

#[cfg(test)]
#[path = "../tests/unit/builtin_tests.rs"]
mod tests;
