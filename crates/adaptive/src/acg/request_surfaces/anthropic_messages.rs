// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages request-surface applier.

use nemo_relay::api::llm::LlmRequest;
use serde_json::{Value, json};

use crate::acg::debug as acg_debug;
use crate::acg::prompt_ir::PromptIR;
use crate::acg::request_surfaces::RequestSurfaceApplier;
use crate::acg::translation::{AnthropicCacheTtl, AnthropicHintDirective, HintDirective, HintPlan};

pub(crate) struct AnthropicMessages;

impl RequestSurfaceApplier for AnthropicMessages {
    fn apply(
        &self,
        request: &LlmRequest,
        prompt_ir: &PromptIR,
        plan: &HintPlan,
    ) -> crate::acg::Result<LlmRequest> {
        let mut translated = request.clone();
        let content = &mut translated.content;

        for directive in &plan.directives {
            match directive {
                HintDirective::Anthropic(AnthropicHintDirective::CanonicalizeToolSchemas) => {
                    let _ = crate::acg::request_surfaces::canonicalize_tools(content);
                }
                HintDirective::Anthropic(AnthropicHintDirective::CacheBreakpoint {
                    target,
                    ..
                }) => {
                    if let Some(target_block_index) =
                        crate::acg::request_surfaces::resolve_target_block_index(prompt_ir, target)
                    {
                        let target_block = &prompt_ir.blocks[target_block_index];
                        let (target_kind, injected) = match target_block.content_type {
                            crate::acg::prompt_ir::BlockContentType::ToolSchema => {
                                let tool_index = crate::acg::request_surfaces::prompt_ir_tool_index(
                                    prompt_ir,
                                    target_block_index,
                                );
                                ("tool", inject_cache_control_tool(content, tool_index))
                            }
                            _ if target_block.role == crate::acg::prompt_ir::PromptRole::System => {
                                ("system", inject_cache_control_system(content))
                            }
                            _ => {
                                let message_index =
                                    crate::acg::request_surfaces::prompt_ir_message_index(
                                        prompt_ir,
                                        target_block_index,
                                        false,
                                    );
                                (
                                    "message",
                                    inject_cache_control_message(content, message_index),
                                )
                            }
                        };
                        acg_debug::emit(
                            "anthropic_surface_breakpoint_apply",
                            json!({
                                "target": format!("{target:?}"),
                                "resolved_index": target_block_index,
                                "target_kind": target_kind,
                                "role": format!("{:?}", target_block.role),
                                "content_type": format!("{:?}", target_block.content_type),
                                "injected": injected,
                            }),
                        );
                    } else {
                        acg_debug::emit(
                            "anthropic_surface_breakpoint_apply",
                            json!({
                                "target": format!("{target:?}"),
                                "resolved_index": serde_json::Value::Null,
                                "injected": false,
                                "reason": "target_block_not_resolved",
                            }),
                        );
                    }
                }
                HintDirective::Anthropic(AnthropicHintDirective::ApplyTtl { ttl }) => match ttl {
                    AnthropicCacheTtl::OneHour => {
                        apply_ttl_to_cache_controls(content, "1h");
                        acg_debug::emit(
                            "anthropic_surface_ttl_apply",
                            json!({
                                "ttl": "1h",
                            }),
                        );
                    }
                },
                HintDirective::OpenAI(_) => {}
            }
        }

        Ok(translated)
    }
}

fn inject_cache_control_system(content: &mut Value) -> bool {
    if let Some(system) = content.get_mut("system") {
        if system.is_string() {
            let text = system.as_str().unwrap_or_default().to_string();
            *system = json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"}
            }]);
            return true;
        }
        if let Some(arr) = system.as_array_mut()
            && let Some(last_block) = arr.last_mut()
            && let Some(obj) = last_block.as_object_mut()
        {
            obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            return true;
        }
    }
    false
}

fn inject_cache_control_message(content: &mut Value, message_index: usize) -> bool {
    if let Some(messages) = content.get_mut("messages").and_then(Value::as_array_mut)
        && let Some(message) = messages.get_mut(message_index)
    {
        if let Some(msg_content) = message.get_mut("content") {
            if msg_content.is_string() {
                let text = msg_content.as_str().unwrap_or_default().to_string();
                *msg_content = json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": {"type": "ephemeral"}
                }]);
                return true;
            }
            if let Some(arr) = msg_content.as_array_mut()
                && let Some(last_block) = arr.last_mut()
                && let Some(obj) = last_block.as_object_mut()
            {
                obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
                return true;
            }
        }
        if let Some(obj) = message.as_object_mut() {
            obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            return true;
        }
    }
    false
}

fn inject_cache_control_tool(content: &mut Value, tool_index: usize) -> bool {
    if let Some(tools) = content.get_mut("tools").and_then(Value::as_array_mut)
        && let Some(tool) = tools.get_mut(tool_index)
        && let Some(obj) = tool.as_object_mut()
    {
        obj.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
        return true;
    }
    false
}

fn apply_ttl_to_cache_controls(content: &mut Value, ttl: &str) {
    if let Some(system) = content.get_mut("system") {
        apply_ttl_to_value(system, ttl);
    }
    if let Some(messages) = content.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages.iter_mut() {
            apply_ttl_to_value(message, ttl);
        }
    }
    if let Some(tools) = content.get_mut("tools").and_then(Value::as_array_mut) {
        for tool in tools.iter_mut() {
            apply_ttl_to_value(tool, ttl);
        }
    }
}

fn apply_ttl_to_value(value: &mut Value, ttl: &str) {
    match value {
        Value::Object(map) => {
            if let Some(cache_control) = map.get_mut("cache_control")
                && let Some(cache_control_obj) = cache_control.as_object_mut()
            {
                cache_control_obj.insert("ttl".to_string(), json!(ttl));
            }
            for child in map.values_mut() {
                apply_ttl_to_value(child, ttl);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                apply_ttl_to_value(item, ttl);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/acg/anthropic_messages_surface_tests.rs"]
mod tests;
