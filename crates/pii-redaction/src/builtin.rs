// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use regex::Regex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value as Json;
use sha2::{Digest, Sha256};

use nemo_relay::api::event::{CategoryProfile, Event};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    ContextualLlmSanitizeRequestFn, ContextualLlmSanitizeResponseFn, EventSanitizeFn,
    LlmSanitizeContext, ToolSanitizeFn,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::resolve::{
    ProviderSurface, detect_request_surface, detect_response_surface,
    request_codec as build_request_codec, response_codec as build_response_codec,
};
use nemo_relay::codec::traits::LlmCodec;
use nemo_relay::plugin::{PluginError, Result as PluginResult};

use super::component::BuiltinBackendConfig;
use super::detectors::BuiltinDetector;
use super::overlay::BuiltinCodecName;
use super::trajectory::{CustomMarkPayloadPolicy, TrajectorySanitizer};

#[derive(Clone)]
pub(super) struct CompiledBuiltinBackend {
    action: BuiltinAction,
    target_paths: Arc<Vec<String>>,
    legacy_surface: Option<ProviderSurface>,
    trajectory: Option<TrajectorySanitizer>,
}

#[derive(Clone)]
enum BuiltinAction {
    Remove,
    Hash {
        matcher: Option<Arc<Regex>>,
    },
    Mask {
        matcher: Option<Arc<Regex>>,
        strategy: BuiltinMaskStrategy,
    },
    Redact {
        matcher: Arc<Regex>,
        replacement: Arc<String>,
    },
    RegexReplace {
        pattern: Arc<Regex>,
        replacement: Arc<String>,
    },
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
        let trajectory = match config.preset.as_deref() {
            Some("trajectory_context") => {
                if config.detector.is_some()
                    || config.pattern.is_some()
                    || !config.target_paths.is_empty()
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
                ))
            }
            Some(other) => {
                return Err(PluginError::InvalidConfig(format!(
                    "unsupported builtin preset '{other}'"
                )));
            }
            None => None,
        };
        if trajectory.is_none() && config.custom_mark_payload_policy != "preserve" {
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

        Ok(Self {
            action,
            target_paths: Arc::new(config.target_paths),
            legacy_surface: surface,
            trajectory,
        })
    }

    fn sanitize_json_preorder_dfs(&self, value: Json) -> Json {
        self.sanitize_json_preorder_dfs_at_path(value, &mut Vec::new())
            .unwrap_or(Json::Null)
    }

    fn sanitize_json_preorder_dfs_at_path(
        &self,
        value: Json,
        path_segments: &mut Vec<String>,
    ) -> Option<Json> {
        if !self.target_paths.is_empty()
            && self.matches_current_preorder_path(path_segments)
            && matches!(self.action, BuiltinAction::Remove)
        {
            return None;
        }

        match value {
            Json::String(text) => {
                if self.matches_current_preorder_path(path_segments) {
                    self.sanitize_string_value(text)
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
                            .sanitize_json_preorder_dfs_at_path(item, path_segments)
                            .unwrap_or(Json::Null);
                        path_segments.pop();
                        sanitized
                    })
                    .collect(),
            )),
            Json::Object(map) => Some(Json::Object(
                map.into_iter()
                    .filter_map(|(key, value)| {
                        path_segments.push(escape_json_pointer_segment(&key));
                        let sanitized =
                            self.sanitize_json_preorder_dfs_at_path(value, path_segments);
                        path_segments.pop();
                        sanitized.map(|sanitized| (key, sanitized))
                    })
                    .collect(),
            )),
            other => Some(other),
        }
    }

    fn matches_current_preorder_path(&self, path_segments: &[String]) -> bool {
        if self.target_paths.is_empty() {
            return true;
        }
        let current_path = render_json_pointer_path(path_segments);
        self.target_paths.iter().any(|path| path == &current_path)
    }

    fn sanitize_string_value(&self, text: String) -> Option<Json> {
        match &self.action {
            BuiltinAction::Remove => None,
            BuiltinAction::Hash { matcher } => Some(Json::String(match matcher {
                Some(matcher) => matcher
                    .replace_all(&text, |captures: &regex::Captures<'_>| {
                        hex_sha256(
                            captures
                                .get(0)
                                .map(|capture| capture.as_str())
                                .unwrap_or(""),
                        )
                    })
                    .into_owned(),
                None => hex_sha256(&text),
            })),
            BuiltinAction::Mask { matcher, strategy } => Some(Json::String(match matcher {
                Some(matcher) => matcher
                    .replace_all(&text, |captures: &regex::Captures<'_>| {
                        mask_with_strategy(
                            captures
                                .get(0)
                                .map(|capture| capture.as_str())
                                .unwrap_or(""),
                            strategy,
                        )
                    })
                    .into_owned(),
                None => mask_with_strategy(&text, strategy),
            })),
            BuiltinAction::Redact {
                matcher,
                replacement,
            } => Some(Json::String(
                matcher
                    .replace_all(&text, replacement.as_str())
                    .into_owned(),
            )),
            BuiltinAction::RegexReplace {
                pattern,
                replacement,
            } => Some(Json::String(
                pattern
                    .replace_all(&text, replacement.as_str())
                    .into_owned(),
            )),
        }
    }

    fn selected_surface(&self, context: LlmSanitizeContext) -> Option<ProviderSurface> {
        if context.has_active_codec {
            return context
                .codec_name
                .and_then(ProviderSurface::from_codec_name);
        }
        self.legacy_surface
    }

    fn uses_compatible_legacy_request_codec(&self, request: &LlmRequest) -> bool {
        self.legacy_surface
            .is_some_and(|surface| detect_request_surface(&request.content) == Some(surface))
    }

    fn uses_compatible_legacy_response_codec(&self, payload: &Json) -> bool {
        self.legacy_surface
            .is_some_and(|surface| detect_response_surface(payload) == Some(surface))
    }

    fn sanitize_request_with_codec(
        &self,
        context: LlmSanitizeContext,
        request: &LlmRequest,
    ) -> Option<LlmRequest> {
        let codec = build_request_codec(self.selected_surface(context)?);
        let annotated = codec.decode(request).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        codec
            .encode(&sanitized_annotated, request)
            .ok()
            .or_else(|| {
                self.sanitize_request_target_paths_incrementally(
                    codec.as_ref(),
                    request,
                    sanitized_annotated,
                )
            })
    }

    fn sanitize_request_target_paths_incrementally(
        &self,
        codec: &dyn LlmCodec,
        request: &LlmRequest,
        sanitized_annotated: AnnotatedLlmRequest,
    ) -> Option<LlmRequest> {
        let sanitized = serde_json::to_value(sanitized_annotated).ok()?;
        let mut sanitized_request = request.clone();

        for target_path in self.target_paths.iter() {
            let target_segments = json_pointer_segments(target_path)?;
            let Some(target_value) = sanitized_json_pointer_value(&sanitized, &target_segments)
            else {
                continue;
            };
            let target_value = target_value.clone();
            let current_annotated = codec.decode(&sanitized_request).ok()?;
            let mut current = serde_json::to_value(&current_annotated).ok()?;
            let current_value = sanitized_json_pointer_value(&current, &target_segments)?;
            if current_value == &target_value {
                continue;
            }
            replace_sanitized_json_pointer_value(&mut current, &target_segments, target_value)?;
            let updated = serde_json::from_value(current).ok()?;
            sanitized_request = codec.encode(&updated, &sanitized_request).ok()?;
        }

        Some(sanitized_request)
    }

    fn sanitize_response_with_codec(
        &self,
        context: LlmSanitizeContext,
        payload: Json,
    ) -> Option<Json> {
        let surface = self.selected_surface(context)?;
        let codec = build_response_codec(surface);
        let codec_name = BuiltinCodecName::from_provider_surface(surface);
        let annotated = codec.decode_response(&payload).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        Some(codec_name.overlay_response_payload(payload, &sanitized_annotated))
    }
}

pub(super) fn tool_sanitize_callback(backend: CompiledBuiltinBackend) -> ToolSanitizeFn {
    Arc::new(
        move |_name: &str, payload: Json| match backend.trajectory.as_ref() {
            Some(trajectory) => trajectory.sanitize_tool_payload(payload),
            None => backend.sanitize_json_preorder_dfs(payload),
        },
    )
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

        if let Some(trajectory) = backend.trajectory.as_ref() {
            return trajectory.sanitize_event_fields(event, fields);
        }
        let specialized_scope = matches!(event, Event::Scope(_))
            && event
                .category()
                .is_some_and(|category| matches!(category.as_str(), "tool" | "llm"));

        if !specialized_scope {
            fields.data = fields
                .data
                .map(|data| backend.sanitize_json_preorder_dfs(data));
            fields.category_profile = fields.category_profile.and_then(|profile| {
                sanitize_serializable_with_backend::<CategoryProfile>(&backend, profile).ok()
            });
        }

        fields.metadata = fields
            .metadata
            .map(|metadata| backend.sanitize_json_preorder_dfs(metadata));
        fields
    })
}

pub(super) fn llm_sanitize_request_callback(
    backend: CompiledBuiltinBackend,
) -> ContextualLlmSanitizeRequestFn {
    Arc::new(move |mut request: LlmRequest, context| {
        if let Some(trajectory) = backend.trajectory.as_ref() {
            request.content = trajectory.sanitize_provider_payload(request.content);
            return Some(request);
        }
        if backend.target_paths.is_empty() {
            request.content = backend.sanitize_json_preorder_dfs(request.content);
            return Some(request);
        }
        if !context.has_active_codec && !backend.uses_compatible_legacy_request_codec(&request) {
            log_llm_payload_omitted("request", context, "no compatible legacy codec");
            return None;
        }
        let sanitized = backend.sanitize_request_with_codec(context, &request);
        if sanitized.is_none() {
            log_llm_payload_omitted(
                "request",
                context,
                "codec decode, sanitize, or encode failure",
            );
        }
        sanitized
    })
}

pub(super) fn llm_sanitize_response_callback(
    backend: CompiledBuiltinBackend,
) -> ContextualLlmSanitizeResponseFn {
    Arc::new(move |payload: Json, context| {
        if let Some(trajectory) = backend.trajectory.as_ref() {
            return Some(trajectory.sanitize_provider_payload(payload));
        }
        if backend.target_paths.is_empty() {
            return Some(backend.sanitize_json_preorder_dfs(payload));
        }
        if !context.has_active_codec && !backend.uses_compatible_legacy_response_codec(&payload) {
            log_llm_payload_omitted("response", context, "no compatible legacy codec");
            return None;
        }
        let sanitized = backend
            .sanitize_response_with_codec(context, payload)
            .map(|payload| backend.sanitize_json_preorder_dfs(payload));
        if sanitized.is_none() {
            log_llm_payload_omitted(
                "response",
                context,
                "codec decode, sanitize, or encode failure",
            );
        }
        sanitized
    })
}

fn log_llm_payload_omitted(direction: &str, context: LlmSanitizeContext, reason: &str) {
    log::warn!(
        target: "nemo_relay.plugin",
        event = "pii_llm_payload_omitted",
        codec_name = context.codec_name.unwrap_or("unknown"),
        has_active_codec = context.has_active_codec,
        reason;
        "PII redaction omitted an LLM {direction} payload"
    );
}

fn json_pointer_segments(pointer: &str) -> Option<Vec<String>> {
    pointer
        .strip_prefix('/')
        .map(|path| path.split('/').map(unescape_json_pointer_segment).collect())
}

fn unescape_json_pointer_segment(segment: &str) -> String {
    segment.replace("~1", "/").replace("~0", "~")
}

fn sanitized_json_pointer_value<'a>(value: &'a Json, segments: &[String]) -> Option<&'a Json> {
    segments
        .iter()
        .try_fold(value, |value, segment| match value {
            Json::Object(values) => values.get(segment),
            Json::Array(values) => segment
                .parse::<usize>()
                .ok()
                .and_then(|index| values.get(index)),
            _ => None,
        })
}

fn replace_sanitized_json_pointer_value(
    value: &mut Json,
    segments: &[String],
    replacement: Json,
) -> Option<()> {
    let (last, parents) = segments.split_last()?;
    let parent = parents
        .iter()
        .try_fold(value, |value, segment| match value {
            Json::Object(values) => values.get_mut(segment),
            Json::Array(values) => segment
                .parse::<usize>()
                .ok()
                .and_then(|index| values.get_mut(index)),
            _ => None,
        })?;
    match parent {
        Json::Object(values) => {
            values.insert(last.clone(), replacement);
            Some(())
        }
        Json::Array(values) => {
            let index = last.parse::<usize>().ok()?;
            let value = values.get_mut(index)?;
            *value = replacement;
            Some(())
        }
        _ => None,
    }
}

fn render_json_pointer_path(path_segments: &[String]) -> String {
    if path_segments.is_empty() {
        return String::new();
    }
    let mut rendered = String::new();
    for segment in path_segments {
        rendered.push('/');
        rendered.push_str(segment);
    }
    rendered
}

fn escape_json_pointer_segment(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
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
) -> PluginResult<Option<Arc<Regex>>> {
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
    Ok(Some(Arc::new(pattern)))
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
