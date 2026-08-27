// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use regex::Regex;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value as Json};
use sha2::{Digest, Sha256};

use nemo_relay::api::event::{CategoryProfile, Event, MetricEnvelope, ScopeCategory};
use nemo_relay::api::llm::LlmRequest;
use nemo_relay::api::runtime::{
    BuiltinLlmCodec, EventSanitizeFn, LlmCodecIdentity, LlmSanitizeRequestFn,
    LlmSanitizeResponseFn, ToolSanitizeFn,
};
use nemo_relay::codec::request::AnnotatedLlmRequest;
use nemo_relay::codec::resolve::{
    ProviderSurface, detect_response_surface, request_codec as build_request_codec,
    response_codec as build_response_codec,
};
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::plugin::{PluginError, Result as PluginResult};

use super::component::{BuiltinBackendConfig, validate_metric_string_attribute_allowlist};
use super::detectors::BuiltinDetector;
use super::overlay::BuiltinCodecName;
use super::trajectory::{CustomMarkPayloadPolicy, TrajectorySanitizer, is_relay_metric_mark};

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
        for (index, target_path) in config.target_paths.iter().enumerate() {
            if !is_valid_json_pointer(target_path) {
                return Err(PluginError::InvalidConfig(format!(
                    "builtin.target_paths[{index}] must be a valid RFC 6901 JSON pointer"
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

    /// Sanitize optional metric text without modifying required export fields.
    fn sanitize_metric_envelope(&self, data: Json) -> Option<Json> {
        let mut envelope = serde_json::from_value::<MetricEnvelope>(data).ok()?;
        envelope.validate().ok()?;
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
        envelope.validate().ok()?;
        serde_json::to_value(envelope).ok()
    }

    /// Sanitize metric text using its payload-relative JSON Pointer.
    fn sanitize_metric_string_at_path(
        &self,
        value: String,
        path_segments: &[String],
    ) -> Option<String> {
        let mut path_segments = path_segments.to_vec();
        self.sanitize_json_preorder_dfs_at_path(Json::String(value), &mut path_segments)
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
            escape_json_pointer_segment(attribute_key),
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

    fn sanitize_request_with_codec(
        &self,
        codec: &dyn LlmCodec,
        request: &LlmRequest,
    ) -> Option<LlmRequest> {
        let annotated = codec.decode(request).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        codec
            .encode(&sanitized_annotated, request)
            .ok()
            .or_else(|| {
                self.sanitize_request_target_paths_incrementally(
                    codec,
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
            let current_annotated = codec.decode(&sanitized_request).ok()?;
            let mut current = serde_json::to_value(&current_annotated).ok()?;
            match (
                sanitized_json_pointer_value(&sanitized, &target_segments),
                sanitized_json_pointer_value(&current, &target_segments),
            ) {
                (None, None) => continue,
                (Some(target_value), Some(current_value)) if current_value == target_value => {
                    continue;
                }
                (Some(target_value), Some(_)) => {
                    replace_sanitized_json_pointer_value(
                        &mut current,
                        &target_segments,
                        target_value.clone(),
                    )?;
                }
                (None, Some(_)) if matches!(self.action, BuiltinAction::Remove) => {
                    remove_sanitized_json_pointer_value(&mut current, &target_segments)?;
                }
                _ => return None,
            }
            let updated = serde_json::from_value(current).ok()?;
            sanitized_request = codec.encode(&updated, &sanitized_request).ok()?;
        }

        Some(sanitized_request)
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
            && self.targets_normalized_single_projected_response()
        {
            return None;
        }
        // Gemini responses with multiple candidates: the normalized layer only projects
        // candidate[0], so candidate[1+] would survive in the raw payload unredacted.
        // Fail closed identically to the OpenAI Chat multi-choice guard.
        if surface == ProviderSurface::GeminiGenerateContent
            && payload
                .get("candidates")
                .and_then(Json::as_array)
                .is_some_and(|candidates| candidates.len() > 1)
            && self.targets_normalized_single_projected_response()
        {
            return None;
        }
        let codec_name = BuiltinCodecName::from_provider_surface(surface);
        let annotated = codec.decode_response(&payload).ok()?;
        let annotated_json = serde_json::to_value(&annotated).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        let sanitized_json = serde_json::to_value(&sanitized_annotated).ok()?;
        let payload = codec_name.overlay_response_payload(payload, &sanitized_annotated);
        let payload = self.sanitize_json_preorder_dfs(payload);
        let target_segments = self
            .target_paths
            .iter()
            .map(|target_path| json_pointer_segments(target_path))
            .collect::<Option<Vec<_>>>()?;
        let has_normalized_target = target_segments.iter().any(|target_segments| {
            sanitized_json_pointer_value(&annotated_json, target_segments).is_some()
                || sanitized_json_pointer_value(&sanitized_json, target_segments).is_some()
        });
        if !has_normalized_target {
            return Some(payload);
        }
        let projected = codec.decode_response(&payload).ok()?;
        let projected = serde_json::to_value(projected).ok()?;
        Self::normalized_response_targets_match(
            &target_segments,
            &annotated_json,
            &sanitized_json,
            &projected,
        )
        .then_some(payload)
    }

    fn normalized_response_targets_match(
        target_paths: &[Vec<String>],
        annotated: &Json,
        sanitized: &Json,
        projected: &Json,
    ) -> bool {
        target_paths.iter().all(|target_segments| {
            let original = sanitized_json_pointer_value(annotated, target_segments);
            let expected = sanitized_json_pointer_value(sanitized, target_segments);
            if original.is_none() && expected.is_none() {
                return true;
            }
            sanitized_json_pointer_value(projected, target_segments) == expected
        })
    }

    fn targets_normalized_single_projected_response(&self) -> bool {
        self.target_paths.iter().any(|path| {
            json_pointer_segments(path)
                .and_then(|segments| segments.into_iter().next())
                .is_some_and(|segment| {
                    matches!(
                        segment.as_str(),
                        "message" | "tool_calls" | "finish_reason" | "api_specific"
                    )
                })
        })
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
                fields.category_profile = fields.category_profile.and_then(|profile| {
                    sanitize_serializable_with_backend::<CategoryProfile>(&backend, profile).ok()
                });
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
                fields.category_profile = fields.category_profile.and_then(|profile| {
                    sanitize_serializable_with_backend::<CategoryProfile>(&backend, profile).ok()
                });
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
            if let Some(trajectory) = backend.trajectory.as_ref() {
                request.headers = trajectory
                    .sanitize_tool_payload(Json::Object(request.headers))
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                request.content = trajectory.sanitize_provider_payload(request.content);
                return Ok(Some(request));
            }
            request.headers = backend.sanitize_request_headers(request.headers);
            if backend.target_paths.is_empty() {
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

pub(super) fn llm_sanitize_response_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeResponseFn {
    let backend = Arc::new(backend);
    Arc::new(move |payload: Json, context| {
        let backend = Arc::clone(&backend);
        Box::pin(async move {
            if let Some(trajectory) = backend.trajectory.as_ref() {
                return Ok(Some(trajectory.sanitize_provider_payload(payload)));
            }
            if backend.target_paths.is_empty() {
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
            let Some(surface) = backend.selected_surface(context.codec()) else {
                log_llm_payload_omitted(
                    "response",
                    context.codec(),
                    "no recognized response codec surface",
                );
                return Ok(None);
            };
            let resolved = context.resolve_codec();
            let fallback = if resolved.is_none() {
                Some(build_response_codec(surface))
            } else {
                None
            };
            let Some(codec) = resolved.as_deref().or(fallback.as_deref()) else {
                log_llm_payload_omitted("response", context.codec(), "no usable response codec");
                return Ok(None);
            };
            let sanitized = backend.sanitize_response_with_codec(codec, surface, payload);
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

fn remove_sanitized_json_pointer_value(value: &mut Json, segments: &[String]) -> Option<()> {
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
        Json::Object(values) => values.remove(last).map(|_| ()),
        Json::Array(_) | Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => None,
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
