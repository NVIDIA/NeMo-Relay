// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AnnotatedLlmRequest to PromptIR construction pipeline.

use chrono::Utc;
use uuid::Uuid;

use nemo_relay_types::codec::content::ContentKind;
use std::borrow::Cow;

use nemo_relay_types::codec::request::{
    AnnotatedLlmRequest, Message, MessageContent, ToolCall, ToolDefinition,
};

use crate::acg::canonicalize::{
    canonicalize_serializable, canonicalize_value, normalize_whitespace, sha256_hex,
};
use crate::acg::error::Result;
use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel, SpanId,
    ToolSchemaHash,
};

/// Build a normalized [`PromptIR`] from an annotated LLM request.
///
/// The builder preserves prompt order, inserts tool-schema blocks before the
/// first non-system message when tools are present, and computes the request
/// hashes needed by downstream Adaptive Cache Governor (ACG) analysis.
///
/// # Parameters
/// - `request`: Annotated LLM request to normalize.
///
/// # Returns
/// A [`Result`] containing the constructed [`PromptIR`].
///
/// # Errors
/// Returns an error when tool definitions or request components cannot be
/// serialized into the canonical form required by the IR.
pub fn build_prompt_ir(request: &AnnotatedLlmRequest) -> Result<PromptIR> {
    let mut blocks: Vec<PromptBlock> = Vec::new();
    let mut sequence_index: u32 = 0;
    let mut inserted_scaffold_blocks = false;
    let structured_output = request
        .extra
        .get("response_format")
        .filter(|value| !value.is_null())
        .map(canonicalize_value)
        .transpose()?;

    if let Some(instructions) = &request.instructions {
        blocks.push(build_text_block(
            &mut sequence_index,
            instructions,
            PromptRole::System,
            ProvenanceLabel::System,
            Some("instructions"),
        )?);
    }

    for message in &request.messages {
        if !inserted_scaffold_blocks
            && !matches!(message, Message::System { .. } | Message::Developer { .. })
        {
            append_tool_schema_blocks(&mut blocks, &mut sequence_index, request.tools.as_deref())?;
            append_structured_output_block(
                &mut blocks,
                &mut sequence_index,
                structured_output.as_deref(),
            );
            inserted_scaffold_blocks = true;
        }

        append_message_blocks(&mut blocks, &mut sequence_index, message)?;
    }

    if !inserted_scaffold_blocks {
        append_tool_schema_blocks(&mut blocks, &mut sequence_index, request.tools.as_deref())?;
        append_structured_output_block(
            &mut blocks,
            &mut sequence_index,
            structured_output.as_deref(),
        );
    }

    let tool_schema_hashes = match &request.tools {
        Some(tools) => Some(build_tool_schema_hashes(tools)?),
        None => None,
    };
    let source_request_hash = Some(compute_request_hash(request)?);

    Ok(PromptIR {
        ir_id: Uuid::new_v4(),
        blocks,
        tool_schema_hashes,
        structured_output_schema_id: structured_output.as_deref().map(sha256_hex),
        source_request_hash,
        created_at: Utc::now(),
    })
}

fn append_message_blocks(
    blocks: &mut Vec<PromptBlock>,
    sequence_index: &mut u32,
    message: &Message,
) -> Result<()> {
    match message {
        Message::System { content, .. } => blocks.push(build_text_block(
            sequence_index,
            content,
            PromptRole::System,
            ProvenanceLabel::System,
            None,
        )?),
        Message::Developer { content, .. } => blocks.push(build_text_block(
            sequence_index,
            content,
            PromptRole::System,
            ProvenanceLabel::Developer,
            Some("developer"),
        )?),
        Message::User { content, .. } => blocks.push(build_text_block(
            sequence_index,
            content,
            PromptRole::User,
            ProvenanceLabel::User,
            None,
        )?),
        Message::Assistant {
            content,
            tool_calls,
            ..
        } => append_assistant_blocks(
            blocks,
            sequence_index,
            content.as_ref(),
            tool_calls.as_deref(),
        )?,
        Message::Tool {
            content,
            tool_call_id,
        } => blocks.push(build_tool_result_block(
            sequence_index,
            content,
            tool_call_id,
        )?),
        Message::Function { content, name } => {
            let content = MessageContent::Text(content.clone().unwrap_or_default());
            blocks.push(build_tool_result_block(sequence_index, &content, name)?);
        }
        Message::ToolCallItem { name, .. } => blocks.push(build_serialized_message_block(
            sequence_index,
            message,
            PromptRole::Assistant,
            ProvenanceLabel::Developer,
            Some(name),
        )?),
        Message::ToolResultItem { call_id, .. } => blocks.push(build_serialized_message_block(
            sequence_index,
            message,
            PromptRole::Tool,
            ProvenanceLabel::Tool,
            Some(call_id),
        )?),
        Message::ProviderNative { kind, .. } => blocks.push(build_serialized_message_block(
            sequence_index,
            message,
            PromptRole::User,
            ProvenanceLabel::Developer,
            Some(kind),
        )?),
    }

    Ok(())
}

fn append_assistant_blocks(
    blocks: &mut Vec<PromptBlock>,
    sequence_index: &mut u32,
    content: Option<&MessageContent>,
    tool_calls: Option<&[ToolCall]>,
) -> Result<()> {
    if let Some(content) = content {
        blocks.push(build_text_block(
            sequence_index,
            content,
            PromptRole::Assistant,
            ProvenanceLabel::Developer,
            None,
        )?);
    }

    if let Some(tool_calls) = tool_calls {
        for call in tool_calls {
            blocks.push(build_tool_call_block(sequence_index, call)?);
        }
    }

    Ok(())
}

/// Preserve part boundaries for structured input. Only explicit prose is
/// normalized; payload values keep their exact strings and opaque data is hashed.
fn analysis_content(
    content: &MessageContent,
    normalize_prose: bool,
) -> Result<(String, BlockContentType)> {
    let view = content.content_view();
    if view
        .parts()
        .all(|entry| matches!(entry.kind, ContentKind::Text(_)))
    {
        let text = view.text().unwrap_or(Cow::Borrowed(""));
        let rendered = if normalize_prose {
            normalize_whitespace(&text)
        } else {
            text.into_owned()
        };
        return Ok((rendered, BlockContentType::Text));
    }
    let MessageContent::Parts(parts) = content else {
        unreachable!("non-text entries require multipart content")
    };
    let mut opaque = false;
    let mut rendered = Vec::with_capacity(parts.len());
    for entry in view.parts() {
        let (kind, field, text) = match entry.kind {
            ContentKind::Text(text) => ("text", "text", Some(text)),
            ContentKind::Refusal(text) => ("refusal", "refusal", Some(text)),
            _ => ("", "", None),
        };
        if let Some(text) = text {
            let text = if normalize_prose {
                normalize_whitespace(text)
            } else {
                text.to_owned()
            };
            rendered.push(serde_json::Value::Object(serde_json::Map::from_iter([
                ("type".into(), serde_json::Value::String(kind.into())),
                (field.into(), serde_json::Value::String(text)),
            ])));
            continue;
        }
        let index = entry
            .location
            .index
            .expect("multipart entries have an index");
        let opaque_kind = match entry.kind {
            ContentKind::ImageUrl(_) => Some("image_url"),
            ContentKind::Image(_) => Some("image"),
            ContentKind::Audio(_) => Some("audio"),
            ContentKind::File(_) => Some("file"),
            ContentKind::ProviderNative { .. } => Some("provider_native"),
            _ => None,
        };
        if let Some(kind) = opaque_kind {
            opaque = true;
            let fingerprint = sha256_hex(&canonicalize_serializable(&parts[index])?);
            rendered.push(serde_json::json!({"type": kind, "fingerprint": fingerprint}));
        } else {
            rendered.push(serde_json::to_value(&parts[index])?);
        }
    }
    Ok((
        canonicalize_value(&serde_json::Value::Array(rendered))?,
        if opaque {
            BlockContentType::Opaque
        } else {
            BlockContentType::StructuredInput
        },
    ))
}

fn build_serialized_message_block(
    seq: &mut u32,
    message: &Message,
    role: PromptRole,
    provenance: ProvenanceLabel,
    suffix: Option<&str>,
) -> Result<PromptBlock> {
    let canonical = canonicalize_serializable(message)?;
    let (content, content_type) = match message {
        Message::ProviderNative { .. } => (
            canonicalize_value(&serde_json::json!({
                "type": "provider_native", "fingerprint": sha256_hex(&canonical),
            }))?,
            BlockContentType::Opaque,
        ),
        Message::ToolResultItem { .. } => (canonical, BlockContentType::ToolResult),
        _ => (canonical, BlockContentType::StructuredInput),
    };
    let index = *seq;
    *seq += 1;
    Ok(PromptBlock {
        span_id: generate_span_id(role, index, suffix),
        sequence_index: index,
        role,
        content,
        content_type,
        provenance,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    })
}

fn generate_span_id(role: PromptRole, index: u32, suffix: Option<&str>) -> SpanId {
    let role_str = match role {
        PromptRole::System => "system",
        PromptRole::User => "user",
        PromptRole::Assistant => "assistant",
        PromptRole::Tool => "tool",
    };

    match suffix {
        Some(suffix) => SpanId(format!("{role_str}-{index}-{suffix}")),
        None => SpanId(format!("{role_str}-{index}")),
    }
}

fn build_text_block(
    seq: &mut u32,
    content: &MessageContent,
    role: PromptRole,
    provenance: ProvenanceLabel,
    suffix: Option<&str>,
) -> Result<PromptBlock> {
    let (text, content_type) = analysis_content(content, true)?;
    let index = *seq;
    let span_id = generate_span_id(role, index, suffix);
    *seq += 1;

    Ok(PromptBlock {
        span_id,
        sequence_index: index,
        role,
        content: text,
        content_type,
        provenance,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    })
}

fn build_tool_call_block(seq: &mut u32, call: &ToolCall) -> Result<PromptBlock> {
    let content = canonicalize_serializable(call)?;
    let index = *seq;
    let span_id = generate_span_id(PromptRole::Assistant, index, Some(&call.function.name));
    *seq += 1;

    Ok(PromptBlock {
        span_id,
        sequence_index: index,
        role: PromptRole::Assistant,
        content,
        content_type: BlockContentType::StructuredInput,
        provenance: ProvenanceLabel::Developer,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    })
}

fn build_tool_result_block(
    seq: &mut u32,
    content: &MessageContent,
    tool_call_id: &str,
) -> Result<PromptBlock> {
    let (text, kind) = analysis_content(content, false)?;
    let index = *seq;
    let span_id = generate_span_id(PromptRole::Tool, index, Some(tool_call_id));
    *seq += 1;

    Ok(PromptBlock {
        span_id,
        sequence_index: index,
        role: PromptRole::Tool,
        content: text,
        content_type: if kind == BlockContentType::Opaque {
            kind
        } else {
            BlockContentType::ToolResult
        },
        provenance: ProvenanceLabel::Tool,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    })
}

fn append_tool_schema_blocks(
    blocks: &mut Vec<PromptBlock>,
    seq: &mut u32,
    tools: Option<&[ToolDefinition]>,
) -> Result<()> {
    let Some(tools) = tools else {
        return Ok(());
    };

    for tool in tools {
        blocks.push(build_tool_schema_block(seq, tool)?);
    }

    Ok(())
}

/// Append the canonicalized structured-output contract as a scaffold block.
///
/// The block sits with the tool schemas ahead of the first non-system message so
/// an output contract that never changes stays inside the stable prefix.
///
/// # Parameters
/// - `blocks`: Block list being built.
/// - `seq`: Running sequence index shared by every block builder.
/// - `response_format`: Canonicalized `response_format` value, if the request set one.
fn append_structured_output_block(
    blocks: &mut Vec<PromptBlock>,
    seq: &mut u32,
    response_format: Option<&str>,
) {
    let Some(content) = response_format else {
        return;
    };

    let index = *seq;
    *seq += 1;
    blocks.push(PromptBlock {
        span_id: generate_span_id(PromptRole::System, index, Some("structured-output")),
        sequence_index: index,
        role: PromptRole::System,
        content: content.to_string(),
        content_type: BlockContentType::StructuredOutput,
        provenance: ProvenanceLabel::System,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    });
}

fn build_tool_schema_block(seq: &mut u32, tool: &ToolDefinition) -> Result<PromptBlock> {
    let content = canonicalize_serializable(tool)?;
    let index = *seq;
    let span_id = generate_span_id(PromptRole::System, index, Some(tool_definition_name(tool)));
    *seq += 1;

    Ok(PromptBlock {
        span_id,
        sequence_index: index,
        role: PromptRole::System,
        content,
        content_type: BlockContentType::ToolSchema,
        provenance: ProvenanceLabel::System,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    })
}

fn build_tool_schema_hashes(tools: &[ToolDefinition]) -> Result<Vec<ToolSchemaHash>> {
    tools
        .iter()
        .map(|tool_definition| {
            let value = serde_json::to_value(tool_definition)?;
            let canonical = canonicalize_value(&value)?;
            Ok(ToolSchemaHash {
                tool_name: tool_definition_name(tool_definition).to_string(),
                schema_hash: sha256_hex(&canonical),
            })
        })
        .collect()
}

fn tool_definition_name(tool: &ToolDefinition) -> &str {
    match tool {
        ToolDefinition::Function { function, .. } => &function.name,
        ToolDefinition::ProviderNative { kind, .. } => kind,
    }
}

fn compute_request_hash(request: &AnnotatedLlmRequest) -> Result<String> {
    let value = serde_json::to_value(request)?;
    let canonical = canonicalize_value(&value)?;
    Ok(sha256_hex(&canonical))
}
