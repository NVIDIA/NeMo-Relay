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

const RESPONSE_FORMAT_EXTRA_KEY: &str = "response_format";

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
    let mut inserted_static_contract_blocks = false;
    let structured_output_contract = build_structured_output_contract(request)?;
    let has_static_contract_blocks =
        request.tools.is_some() || structured_output_contract.is_some();

    for message in &request.messages {
        if should_insert_static_contract_blocks_before_message(
            inserted_static_contract_blocks,
            has_static_contract_blocks,
            message,
        ) {
            append_static_contract_blocks(
                &mut blocks,
                &mut sequence_index,
                request.tools.as_deref(),
                structured_output_contract.as_ref(),
            )?;
            inserted_static_contract_blocks = true;
        }

        append_message_blocks(&mut blocks, &mut sequence_index, message)?;
    }

    if !inserted_static_contract_blocks {
        append_static_contract_blocks(
            &mut blocks,
            &mut sequence_index,
            request.tools.as_deref(),
            structured_output_contract.as_ref(),
        )?;
    }

    let tool_schema_hashes = match &request.tools {
        Some(tools) => Some(build_tool_schema_hashes(tools)?),
        None => None,
    };
    let structured_output_schema_id = structured_output_contract
        .as_ref()
        .map(|contract| contract.schema_id.clone());
    let source_request_hash = Some(compute_request_hash(request)?);

    Ok(PromptIR {
        ir_id: Uuid::new_v4(),
        blocks,
        tool_schema_hashes,
        structured_output_schema_id,
        source_request_hash,
        created_at: Utc::now(),
    })
}

struct StructuredOutputContract {
    content: String,
    schema_id: String,
}

fn should_insert_static_contract_blocks_before_message(
    inserted_static_contract_blocks: bool,
    has_static_contract_blocks: bool,
    message: &Message,
) -> bool {
    !inserted_static_contract_blocks
        && has_static_contract_blocks
        && !matches!(message, Message::System { .. })
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
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                ContentPart::ImageUrl { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
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

fn append_static_contract_blocks(
    blocks: &mut Vec<PromptBlock>,
    seq: &mut u32,
    tools: Option<&[ToolDefinition]>,
    structured_output_contract: Option<&StructuredOutputContract>,
) -> Result<()> {
    append_tool_schema_blocks(blocks, seq, tools)?;

    if let Some(contract) = structured_output_contract {
        blocks.push(build_structured_output_block(seq, contract));
    }

    Ok(())
}

fn build_tool_schema_block(seq: &mut u32, tool: &ToolDefinition) -> Result<PromptBlock> {
    let tool_value = serde_json::to_value(tool)?;
    let canonical = canonicalize_value(&tool_value)?;
    let content = normalize_whitespace(&canonical);
    let index = *seq;
    let span_id = generate_span_id(PromptRole::System, index, Some(&tool.function.name));
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

fn build_structured_output_contract(
    request: &AnnotatedLlmRequest,
) -> Result<Option<StructuredOutputContract>> {
    let Some(response_format) = request.extra.get(RESPONSE_FORMAT_EXTRA_KEY) else {
        return Ok(None);
    };

    let canonical = canonicalize_value(response_format)?;
    let content = normalize_whitespace(&canonical);
    Ok(Some(StructuredOutputContract {
        schema_id: sha256_hex(&canonical),
        content,
    }))
}

fn build_structured_output_block(
    seq: &mut u32,
    contract: &StructuredOutputContract,
) -> PromptBlock {
    let index = *seq;
    let span_id = generate_span_id(PromptRole::System, index, Some("structured-output"));
    *seq += 1;

    PromptBlock {
        span_id,
        sequence_index: index,
        role: PromptRole::System,
        content: contract.content.clone(),
        content_type: BlockContentType::StructuredOutput,
        provenance: ProvenanceLabel::System,
        sensitivity: SensitivityLabel::default(),
        token_metadata: None,
    }
}

fn build_tool_schema_hashes(tools: &[ToolDefinition]) -> Result<Vec<ToolSchemaHash>> {
    tools
        .iter()
        .map(|tool_definition| {
            let value = serde_json::to_value(tool_definition)?;
            let canonical = canonicalize_value(&value)?;
            Ok(ToolSchemaHash {
                tool_name: tool_definition.function.name.clone(),
                schema_hash: sha256_hex(&canonical),
            })
        })
        .collect()
}

fn compute_request_hash(request: &AnnotatedLlmRequest) -> Result<String> {
    let value = serde_json::to_value(request)?;
    let canonical = canonicalize_value(&value)?;
    Ok(sha256_hex(&canonical))
}
