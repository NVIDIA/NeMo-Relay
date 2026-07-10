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
    EventSanitizeFn, LlmSanitizeRequestFn, LlmSanitizeResponseFn, ToolSanitizeFn,
};
use nemo_relay::codec::resolve::{
    ProviderSurface, request_codec as build_request_codec, response_codec as build_response_codec,
};
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::plugin::{PluginError, Result as PluginResult};

use super::component::BuiltinBackendConfig;
use super::detectors::BuiltinDetector;
use super::overlay::BuiltinCodecName;

#[derive(Clone)]
pub(super) struct CompiledBuiltinBackend {
    action: BuiltinAction,
    target_paths: Arc<Vec<String>>,
    request_codec: Option<Arc<dyn LlmCodec>>,
    response_codec: Option<Arc<dyn LlmResponseCodec>>,
    codec_name: Option<BuiltinCodecName>,
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
            request_codec: surface.map(build_request_codec).map(Arc::from),
            response_codec: surface.map(build_response_codec).map(Arc::from),
            codec_name: surface.map(BuiltinCodecName::from_provider_surface),
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

    fn sanitize_request_with_codec(&self, request: &LlmRequest) -> Option<LlmRequest> {
        let codec = self.request_codec.as_ref()?;
        let annotated = codec.decode(request).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        codec.encode(&sanitized_annotated, request).ok()
    }

    fn sanitize_response_with_codec(&self, payload: Json) -> Option<Json> {
        let codec = self.response_codec.as_ref()?;
        let codec_name = self.codec_name?;
        let annotated = codec.decode_response(&payload).ok()?;
        let sanitized_annotated = sanitize_serializable_with_backend(self, annotated).ok()?;
        Some(codec_name.overlay_response_payload(payload, &sanitized_annotated))
    }
}

pub(super) fn tool_sanitize_callback(backend: CompiledBuiltinBackend) -> ToolSanitizeFn {
    Arc::new(move |_name: &str, payload: Json| backend.sanitize_json_preorder_dfs(payload))
}

pub(super) fn event_sanitize_callback(backend: CompiledBuiltinBackend) -> EventSanitizeFn {
    Arc::new(move |event, mut fields| {
        if matches!(event, Event::Scope(_))
            && event
                .category()
                .is_some_and(|category| matches!(category.as_str(), "tool" | "llm"))
        {
            return fields;
        }

        fields.data = fields
            .data
            .map(|data| backend.sanitize_json_preorder_dfs(data));
        fields.metadata = fields
            .metadata
            .map(|metadata| backend.sanitize_json_preorder_dfs(metadata));
        fields.category_profile = fields.category_profile.and_then(|profile| {
            sanitize_serializable_with_backend::<CategoryProfile>(&backend, profile).ok()
        });
        fields
    })
}

pub(super) fn llm_sanitize_request_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeRequestFn {
    Arc::new(move |mut request: LlmRequest| {
        if let Some(encoded) = backend.sanitize_request_with_codec(&request) {
            return encoded;
        }
        request.content = backend.sanitize_json_preorder_dfs(request.content);
        request
    })
}

pub(super) fn llm_sanitize_response_callback(
    backend: CompiledBuiltinBackend,
) -> LlmSanitizeResponseFn {
    Arc::new(move |payload: Json| {
        if backend.target_paths.is_empty() {
            return backend.sanitize_json_preorder_dfs(payload);
        }

        let payload = backend
            .sanitize_response_with_codec(payload.clone())
            .unwrap_or(payload);
        backend.sanitize_json_preorder_dfs(payload)
    })
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
