// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for acg module surface in the NeMo Flow adaptive crate.

use nemo_flow_adaptive::acg::prompt_ir::PromptIR;
use nemo_flow_adaptive::acg::{
    AcgError, AgentIdentity, CacheTelemetryEvent, CapabilityRegistry, sha256_hex,
};
use std::sync::Arc;
use uuid::Uuid;

#[test]
fn acg_module_surface_foundational_symbols_compile_from_canonical_namespace() {
    let _: Option<PromptIR> = None;

    let agent_identity = AgentIdentity {
        agent_id: "research-agent".to_string(),
        template_version: "v1".to_string(),
        toolset_hash: "toolset-hash".to_string(),
        model_family: "claude".to_string(),
        tenant_scope: "tenant-a".to_string(),
    };
    assert_eq!(agent_identity.agent_id, "research-agent");

    let error = AcgError::Internal("test".to_string());
    assert_eq!(error.to_string(), "internal error: test");
}

#[test]
fn acg_module_surface_shared_utility_symbols_compile_from_canonical_namespace() {
    let _: Option<CacheTelemetryEvent> = None;

    let registry = CapabilityRegistry::new();
    assert!(registry.list_backend_ids().is_empty());

    let digest = sha256_hex("stable-prefix");
    assert!(digest.starts_with("sha256:"));
}

#[test]
fn acg_module_surface_analysis_symbols_compile_from_canonical_namespace() {
    use nemo_flow_adaptive::acg::profile::{BlockStabilityScore, StabilityClass};
    use nemo_flow_adaptive::acg::prompt_ir::{
        BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel,
        SpanId,
    };
    use nemo_flow_adaptive::acg::retention::RetentionThresholds;
    use nemo_flow_adaptive::acg::stability::{StabilityThresholds, analyze_stability};

    let _: Option<BlockStabilityScore> = None;
    let thresholds = RetentionThresholds::default();
    assert_eq!(thresholds.ephemeral_max_secs, 5.0);

    let observation = PromptIR {
        ir_id: Uuid::nil(),
        blocks: vec![PromptBlock {
            span_id: SpanId("system-prefix".to_string()),
            sequence_index: 0,
            role: PromptRole::System,
            content: "Keep responses concise.".to_string(),
            content_type: BlockContentType::Text,
            provenance: ProvenanceLabel::System,
            sensitivity: SensitivityLabel::Public,
            token_metadata: None,
        }],
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: chrono::Utc::now(),
    };

    let analysis = analyze_stability(&[observation], &StabilityThresholds::default());
    assert_eq!(analysis.stable_prefix_length, 1);
    assert_eq!(analysis.scores[0].classification, StabilityClass::Stable);
}

#[test]
fn acg_module_surface_variable_extractor_keeps_regex_detection_behavior() {
    use nemo_flow_adaptive::acg::prompt_ir::SpanId;
    use nemo_flow_adaptive::acg::variable_extractor::{
        default_variable_patterns, extract_variables,
    };

    let patterns = default_variable_patterns();
    let extraction = extract_variables(
        "trace req_abc12345def at 2026-04-09T14:30:00Z",
        &SpanId("trace-span".to_string()),
        &patterns,
    )
    .expect("expected dynamic content to be detected");

    assert_eq!(
        extraction.template_content,
        "trace {{request_id}} at {{iso8601_timestamp}}"
    );
    assert_eq!(extraction.variables.len(), 2);
    assert_eq!(extraction.variables[0].pattern_name, "request_id");
    assert_eq!(extraction.variables[1].pattern_name, "iso8601_timestamp");
}

#[test]
fn acg_module_surface_policy_and_ir_builder_symbols_compile_from_canonical_namespace() {
    use nemo_flow::codec::request::{AnnotatedLlmRequest, Message, MessageContent};
    use nemo_flow_adaptive::acg::ir_builder::build_prompt_ir;
    use nemo_flow_adaptive::acg::policy::{CachePolicy, PolicyEnvelope};
    use nemo_flow_adaptive::acg::{ModelClass, SharingScope};

    let _: Option<PolicyEnvelope<CachePolicy>> = None;

    let envelope = PolicyEnvelope {
        agent_identity: AgentIdentity {
            agent_id: "research-agent".to_string(),
            template_version: "v2".to_string(),
            toolset_hash: "toolset-hash".to_string(),
            model_family: "claude".to_string(),
            tenant_scope: "tenant-a".to_string(),
        },
        policy_version: "2026-04-13".to_string(),
        created_at: chrono::Utc::now(),
        policy: CachePolicy {
            min_stability_score: 0.95,
            min_evidence_count: 8,
            default_sharing_scope: SharingScope::Session,
            warm_first_enabled: true,
            max_fanout_for_warm_first: Some(4),
        },
    };
    assert!(envelope.policy.warm_first_enabled);

    let request = AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Summarize the findings.".to_string()),
                name: None,
            },
        ],
        model: Some("claude-sonnet".to_string()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };

    let prompt_ir = build_prompt_ir(&request).expect("expected canonical PromptIR construction");
    assert_eq!(prompt_ir.blocks.len(), 2);
    assert_eq!(prompt_ir.blocks[0].span_id.0, "system-0");
    assert_eq!(prompt_ir.blocks[1].span_id.0, "user-1");
    let _: ModelClass = ModelClass::Standard;
}

#[test]
fn acg_module_surface_build_prompt_ir_inserts_tool_schema_before_first_non_system_message() {
    use nemo_flow::codec::request::{
        AnnotatedLlmRequest, FunctionDefinition, Message, MessageContent, ToolDefinition,
    };
    use nemo_flow_adaptive::acg::ir_builder::build_prompt_ir;
    use nemo_flow_adaptive::acg::prompt_ir::{BlockContentType, PromptRole};

    let request = AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("Call the weather tool.".to_string()),
                name: None,
            },
        ],
        model: Some("claude-sonnet".to_string()),
        params: None,
        tools: Some(vec![ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Look up the weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })),
            },
        }]),
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };

    let prompt_ir = build_prompt_ir(&request).expect("expected canonical PromptIR construction");

    assert_eq!(prompt_ir.blocks.len(), 3);
    assert_eq!(prompt_ir.blocks[0].role, PromptRole::System);
    assert_eq!(prompt_ir.blocks[0].content_type, BlockContentType::Text);
    assert_eq!(prompt_ir.blocks[1].role, PromptRole::System);
    assert_eq!(
        prompt_ir.blocks[1].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(prompt_ir.blocks[2].role, PromptRole::User);
    assert_eq!(prompt_ir.blocks[2].content_type, BlockContentType::Text);
}

#[test]
fn acg_module_surface_analyze_stability_limits_stable_prefix_when_later_span_is_missing() {
    use nemo_flow_adaptive::acg::profile::StabilityClass;
    use nemo_flow_adaptive::acg::prompt_ir::{
        BlockContentType, PromptBlock, PromptIR, PromptRole, ProvenanceLabel, SensitivityLabel,
        SpanId,
    };
    use nemo_flow_adaptive::acg::stability::{StabilityThresholds, analyze_stability};

    let make_block = |span: &str, index: u32, role: PromptRole, content: &str| PromptBlock {
        span_id: SpanId(span.to_string()),
        sequence_index: index,
        role,
        content: content.to_string(),
        content_type: BlockContentType::Text,
        provenance: match role {
            PromptRole::System => ProvenanceLabel::System,
            PromptRole::User => ProvenanceLabel::User,
            PromptRole::Assistant => ProvenanceLabel::Developer,
            PromptRole::Tool => ProvenanceLabel::Tool,
        },
        sensitivity: SensitivityLabel::Public,
        token_metadata: None,
    };
    let make_ir = |blocks: Vec<PromptBlock>| PromptIR {
        ir_id: Uuid::new_v4(),
        blocks,
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: chrono::Utc::now(),
    };

    let observations = vec![
        make_ir(vec![
            make_block("system-0", 0, PromptRole::System, "You are helpful."),
            make_block("user-1", 1, PromptRole::User, "Summarize this."),
        ]),
        make_ir(vec![make_block(
            "system-0",
            0,
            PromptRole::System,
            "You are helpful.",
        )]),
    ];

    let analysis = analyze_stability(&observations, &StabilityThresholds::default());

    assert_eq!(analysis.stable_prefix_length, 1);
    assert_eq!(analysis.scores.len(), 2);
    assert_eq!(analysis.scores[0].classification, StabilityClass::Stable);
    assert_eq!(
        analysis.scores[1].classification,
        StabilityClass::SemiStable
    );
}

#[test]
fn acg_module_surface_provider_plugin_symbols_compile_from_canonical_namespace() {
    use nemo_flow_adaptive::acg::anthropic_plugin::AnthropicCachePlugin;
    use nemo_flow_adaptive::acg::openai_plugin::OpenAICachePlugin;
    use nemo_flow_adaptive::acg::passthrough::PassthroughPlugin;
    use nemo_flow_adaptive::acg::plugin::ProviderPlugin;
    use nemo_flow_adaptive::acg::plugin_registry::PluginRegistry;

    let capabilities = CapabilityRegistry::with_defaults();
    let anthropic: Arc<dyn ProviderPlugin> = Arc::new(AnthropicCachePlugin::new(&capabilities));
    let openai: Arc<dyn ProviderPlugin> = Arc::new(OpenAICachePlugin);
    let passthrough: Arc<dyn ProviderPlugin> = Arc::new(PassthroughPlugin);

    let mut registry = PluginRegistry::new();
    registry
        .register(anthropic)
        .expect("anthropic plugin registers");
    registry.register(openai).expect("openai plugin registers");
    registry
        .register(passthrough)
        .expect("passthrough plugin registers");

    assert_eq!(
        registry.list_plugin_ids(),
        vec![
            "anthropic".to_string(),
            "openai".to_string(),
            "passthrough".to_string(),
        ]
    );
}
