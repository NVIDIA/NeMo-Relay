// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AnnotatedLlmRequest to PromptIR construction pipeline.

use chrono::Utc;
use uuid::Uuid;

use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ContentPart, Message, MessageContent, ToolCall, ToolDefinition,
};

use crate::acg::canonicalize::{canonicalize_value, normalize_whitespace, sha256_hex};
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
    let mut inserted_tool_blocks = false;

    if let Some(instructions) = &request.instructions {
        blocks.push(build_text_block(
            &mut sequence_index,
            instructions,
            PromptRole::System,
            ProvenanceLabel::System,
            Some("instructions"),
        ));
    }

    for message in &request.messages {
        if should_insert_tool_blocks_before_message(inserted_tool_blocks, request, message) {
            append_tool_schema_blocks(&mut blocks, &mut sequence_index, request.tools.as_deref())?;
            inserted_tool_blocks = true;
        }

        append_message_blocks(&mut blocks, &mut sequence_index, message)?;
    }

    if !inserted_tool_blocks {
        append_tool_schema_blocks(&mut blocks, &mut sequence_index, request.tools.as_deref())?;
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
        structured_output_schema_id: None,
        source_request_hash,
        created_at: Utc::now(),
    })
}

fn should_insert_tool_blocks_before_message(
    inserted_tool_blocks: bool,
    request: &AnnotatedLlmRequest,
    message: &Message,
) -> bool {
    !inserted_tool_blocks && !matches!(message, Message::System { .. }) && request.tools.is_some()
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
        )),
        Message::Developer { content, .. } => blocks.push(build_text_block(
            sequence_index,
            content,
            PromptRole::System,
            ProvenanceLabel::Developer,
            Some("developer"),
        )),
        Message::User { content, .. } => blocks.push(build_text_block(
            sequence_index,
            content,
            PromptRole::User,
            ProvenanceLabel::User,
            None,
        )),
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
        )),
        Message::Function { content, name } => {
            let content = MessageContent::Text(content.clone().unwrap_or_default());
            blocks.push(build_text_block(
                sequence_index,
                &content,
                PromptRole::Tool,
                ProvenanceLabel::Tool,
                Some(name),
            ));
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
        ));
    }

    if let Some(tool_calls) = tool_calls {
        for call in tool_calls {
            blocks.push(build_tool_call_block(sequence_index, call)?);
        }
    }

    Ok(())
}

fn extract_text(content: &MessageContent) -> String {
    match content {
        MessageContent::Text(text) => text.clone(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text, .. } => text.clone(),
                ContentPart::Refusal { refusal, .. } => refusal.clone(),
                ContentPart::ImageUrl { .. } | ContentPart::Image { .. } => String::new(),
                ContentPart::Audio { .. }
                | ContentPart::File { .. }
                | ContentPart::ToolUse { .. }
                | ContentPart::ToolResult { .. }
                | ContentPart::ProviderNative { .. } => {
                    serde_json::to_string(part).unwrap_or_default()
                }
            })
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn build_serialized_message_block(
    seq: &mut u32,
    message: &Message,
    role: PromptRole,
    provenance: ProvenanceLabel,
    suffix: Option<&str>,
) -> Result<PromptBlock> {
    let value = serde_json::to_value(message)?;
    let content = MessageContent::Text(canonicalize_value(&value)?);
    Ok(build_text_block(seq, &content, role, provenance, suffix))
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
) -> PromptBlock {
    let text = normalize_whitespace(&extract_text(content));
    let index = *seq;
    let span_id = generate_span_id(role, index, suffix);
    *seq += 1;

    PromptBlock {
        span_id,
        sequence_index: index,
        role,
        content: text,
        content_type: BlockContentType::Text,
        provenance,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    }
}

fn build_tool_call_block(seq: &mut u32, call: &ToolCall) -> Result<PromptBlock> {
    let call_value = serde_json::to_value(call)?;
    let canonical = canonicalize_value(&call_value)?;
    let content = normalize_whitespace(&canonical);
    let index = *seq;
    let span_id = generate_span_id(PromptRole::Assistant, index, Some(&call.function.name));
    *seq += 1;

    Ok(PromptBlock {
        span_id,
        sequence_index: index,
        role: PromptRole::Assistant,
        content,
        content_type: BlockContentType::Text,
        provenance: ProvenanceLabel::Developer,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    })
}

fn build_tool_result_block(
    seq: &mut u32,
    content: &MessageContent,
    tool_call_id: &str,
) -> PromptBlock {
    let text = normalize_whitespace(&extract_text(content));
    let index = *seq;
    let span_id = generate_span_id(PromptRole::Tool, index, Some(tool_call_id));
    *seq += 1;

    PromptBlock {
        span_id,
        sequence_index: index,
        role: PromptRole::Tool,
        content: text,
        content_type: BlockContentType::ToolResult,
        provenance: ProvenanceLabel::Tool,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    }
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

fn build_tool_schema_block(seq: &mut u32, tool: &ToolDefinition) -> Result<PromptBlock> {
    let tool_value = serde_json::to_value(tool)?;
    let canonical = canonicalize_value(&tool_value)?;
    let content = normalize_whitespace(&canonical);
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
