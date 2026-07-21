// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Derives stable Adaptive Cache Governor (ACG) profile keys from structured
//! LLM requests.

use crate::acg::canonicalize::{canonicalize_value, sha256_hex};
use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ContentPart, Message, MessageContent, ToolDefinition,
};

const HASH_PREFIX_LEN: usize = 16;

struct AcgKeyParts<'a> {
    model: &'a str,
    system_hash: String,
    tool_hash: String,
    response_format_hash: Option<String>,
    has_stable_scaffold: bool,
}

/// Derive the stable ACG learning key used to bucket observations and hot-cache state.
///
/// The learning key intentionally excludes the full role sequence because normal
/// multi-turn conversations grow every request. Instead it uses a coarse
/// conversation class plus the stable template fingerprints that should remain
/// distinct across prompt families.
pub(crate) fn derive_acg_learning_key(
    agent_id: &str,
    annotated_request: &AnnotatedLlmRequest,
) -> String {
    let parts = derive_key_parts(annotated_request);
    let seed_fingerprint =
        (!parts.has_stable_scaffold).then(|| learning_seed_fingerprint(annotated_request));
    let seed_hash = seed_fingerprint
        .as_deref()
        .map(short_hash)
        .unwrap_or("stable-scaffold");
    let key = format!(
        "{agent_id}::model={}::seed={seed_hash}::system={}::tools={}",
        parts.model, parts.system_hash, parts.tool_hash
    );
    parts
        .response_format_hash
        .map(|hash| format!("{key}::response_format={hash}"))
        .unwrap_or(key)
}

/// Derive the exact ACG profile key used for diagnostics and debug output.
///
/// This preserves the full message role signature so logs can still explain why
/// a concrete live request shape differs from previous observations.
pub(crate) fn derive_acg_profile_key(
    agent_id: &str,
    annotated_request: &AnnotatedLlmRequest,
) -> String {
    let parts = derive_key_parts(annotated_request);
    let anchor_fingerprint = layered_anchor_fingerprint(annotated_request);
    let anchor_hash = anchor_fingerprint
        .as_deref()
        .map(short_hash)
        .unwrap_or("no-anchor");
    let role_signature = annotated_request
        .messages
        .iter()
        .map(message_role_tag)
        .collect::<Vec<_>>()
        .join(".");
    format!(
        "{agent_id}::model={}::roles={role_signature}::system={}::anchor={}::tools={}",
        parts.model,
        short_hash(&parts.system_hash),
        anchor_hash,
        short_hash(&parts.tool_hash)
    )
}

fn derive_key_parts(annotated_request: &AnnotatedLlmRequest) -> AcgKeyParts<'_> {
    let system_fingerprint = system_prompt_fingerprint(annotated_request);
    let tool_fingerprint = tool_schema_fingerprint(annotated_request.tools.as_deref());
    let response_format_fingerprint = response_format_fingerprint(annotated_request);
    let has_stable_scaffold = system_fingerprint != "no-system"
        || annotated_request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty())
        || response_format_fingerprint.is_some();

    AcgKeyParts {
        model: annotated_request.model.as_deref().unwrap_or("unknown"),
        system_hash: if has_stable_scaffold {
            system_fingerprint
        } else {
            short_hash(&system_fingerprint).to_string()
        },
        tool_hash: if has_stable_scaffold {
            tool_fingerprint
        } else {
            short_hash(&tool_fingerprint).to_string()
        },
        response_format_hash: response_format_fingerprint,
        has_stable_scaffold,
    }
}

fn message_role_tag(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
        Message::Tool { .. } => "tool",
    }
}

fn system_prompt_fingerprint(annotated_request: &AnnotatedLlmRequest) -> String {
    let system_content = annotated_request
        .messages
        .iter()
        .filter_map(|message| match message {
            Message::System { content, .. } => Some(extract_text(content)),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    if system_content.is_empty() {
        "no-system".to_string()
    } else {
        sha256_hex(&system_content)
    }
}

fn layered_anchor_fingerprint(annotated_request: &AnnotatedLlmRequest) -> Option<String> {
    let messages = &annotated_request.messages;
    if messages.len() < 4 {
        return None;
    }

    let first_user = messages
        .iter()
        .position(|message| matches!(message, Message::User { .. }))?;
    let next_assistant = first_user + 1;
    let next_user = first_user + 2;
    if next_user >= messages.len() {
        return None;
    }

    let Message::User {
        content: first_user_content,
        ..
    } = &messages[first_user]
    else {
        return None;
    };
    let Message::Assistant {
        content: assistant_content,
        ..
    } = &messages[next_assistant]
    else {
        return None;
    };
    let assistant_content = assistant_content.as_ref()?;
    if !matches!(messages[next_user], Message::User { .. }) {
        return None;
    }

    let anchor = [
        "user",
        &extract_text(first_user_content),
        "assistant",
        &extract_text(assistant_content),
    ]
    .join("\n");
    Some(sha256_hex(&anchor))
}

fn learning_seed_fingerprint(annotated_request: &AnnotatedLlmRequest) -> String {
    annotated_request
        .messages
        .iter()
        .find_map(|message| match message {
            Message::System { .. } => None,
            Message::User { content, .. } => {
                Some(format!("user:{}", sha256_hex(&extract_text(content))))
            }
            Message::Assistant {
                content: Some(content),
                ..
            } => Some(format!("assistant:{}", sha256_hex(&extract_text(content)))),
            Message::Assistant { content: None, .. } => Some("assistant:no-content".to_string()),
            Message::Tool { content, .. } => {
                Some(format!("tool:{}", sha256_hex(&extract_text(content))))
            }
        })
        .unwrap_or_else(|| "no-seed".to_string())
}

fn tool_schema_fingerprint(tools: Option<&[ToolDefinition]>) -> String {
    let Some(tools) = tools else {
        return "no-tools".to_string();
    };

    let canonical_tools = tools
        .iter()
        .filter_map(|tool| serde_json::to_value(tool).ok())
        .filter_map(|tool| canonicalize_value(&tool).ok())
        .collect::<Vec<_>>()
        .join("|");

    if canonical_tools.is_empty() {
        "tools-unavailable".to_string()
    } else {
        sha256_hex(&canonical_tools)
    }
}

fn response_format_fingerprint(annotated_request: &AnnotatedLlmRequest) -> Option<String> {
    annotated_request
        .extra
        .get("response_format")
        .filter(|value| !value.is_null())
        .and_then(|value| canonicalize_value(value).ok())
        .map(|value| sha256_hex(&value))
}

fn extract_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => text.clone(),
                ContentPart::ImageUrl { image_url } => format!(
                    "[image:{}:{}]",
                    image_url.detail.as_deref().unwrap_or("none"),
                    sha256_hex(&image_url.url)
                ),
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn short_hash(value: &str) -> &str {
    value.get(..HASH_PREFIX_LEN).unwrap_or(value)
}

#[cfg(test)]
#[path = "../tests/unit/acg_profile_tests.rs"]
mod tests;
