// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_plugin::{
    ConfigDiagnostic, DiagnosticLevel, Event, Json, LlmJsonStream, LlmRequest, NativePlugin,
    PluginContext, PluginRuntime, ScopeCategory, ScopeType,
};
use serde_json::{Map, json};

struct ExampleNativePlugin;

#[derive(Clone, Debug)]
struct ExampleConfig {
    tag: String,
    block_tools: bool,
    block_llms: bool,
    emit_isolated_scope: bool,
}

#[derive(Clone, Copy)]
enum ConfigField {
    Tag,
    BlockTools,
    BlockLlms,
    EmitIsolatedScope,
}

impl ConfigField {
    const ALL: [Self; 4] = [
        Self::Tag,
        Self::BlockTools,
        Self::BlockLlms,
        Self::EmitIsolatedScope,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Tag => "tag",
            Self::BlockTools => "block_tools",
            Self::BlockLlms => "block_llms",
            Self::EmitIsolatedScope => "emit_isolated_scope",
        }
    }

    const fn expected_type(self) -> &'static str {
        match self {
            Self::Tag => "string",
            Self::BlockTools | Self::BlockLlms | Self::EmitIsolatedScope => "boolean",
        }
    }

    const fn invalid_code(self) -> &'static str {
        match self {
            Self::Tag => "examples.rust_native_policy.invalid_tag",
            Self::BlockTools | Self::BlockLlms | Self::EmitIsolatedScope => {
                "examples.rust_native_policy.invalid_boolean"
            }
        }
    }

    fn accepts(self, value: &Json) -> bool {
        match self {
            Self::Tag => value.is_string(),
            Self::BlockTools | Self::BlockLlms | Self::EmitIsolatedScope => value.is_boolean(),
        }
    }

    fn parse_into(
        self,
        value: &Json,
        config: &mut ExampleConfig,
    ) -> nemo_relay_plugin::Result<()> {
        if !self.accepts(value) {
            return Err(format!(
                "{} must be a {}",
                self.name(),
                self.expected_type()
            ));
        }
        match self {
            Self::Tag => {
                config.tag = value.as_str().expect("checked config field type").to_owned();
            }
            Self::BlockTools => {
                config.block_tools = value.as_bool().expect("checked config field type");
            }
            Self::BlockLlms => {
                config.block_llms = value.as_bool().expect("checked config field type");
            }
            Self::EmitIsolatedScope => {
                config.emit_isolated_scope = value
                    .as_bool()
                    .expect("checked config field type");
            }
        }
        Ok(())
    }
}

impl Default for ExampleConfig {
    fn default() -> Self {
        Self {
            tag: "rust-native-example".into(),
            block_tools: false,
            block_llms: false,
            emit_isolated_scope: true,
        }
    }
}

impl ExampleConfig {
    fn parse(plugin_config: &Map<String, Json>) -> nemo_relay_plugin::Result<Self> {
        let mut config = Self::default();
        for field in ConfigField::ALL {
            if let Some(value) = plugin_config.get(field.name()) {
                field.parse_into(value, &mut config)?;
            }
        }

        Ok(config)
    }
}

impl NativePlugin for ExampleNativePlugin {
    fn plugin_kind(&self) -> &str {
        "examples.rust_native_policy"
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = Vec::new();

        for key in plugin_config.keys() {
            if !ConfigField::ALL
                .iter()
                .any(|field| field.name() == key.as_str())
            {
                diagnostics.push(diagnostic(
                    DiagnosticLevel::Warning,
                    "examples.rust_native_policy.unknown_field",
                    Some(key),
                    format!("unknown config field '{key}' will be ignored"),
                ));
            }
        }

        for field in ConfigField::ALL {
            if let Some(value) = plugin_config.get(field.name()) {
                if !field.accepts(value) {
                    diagnostics.push(diagnostic(
                        DiagnosticLevel::Error,
                        field.invalid_code(),
                        Some(field.name()),
                        format!("{} must be a {}", field.name(), field.expected_type()),
                    ));
                }
            }
        }

        diagnostics
    }

    fn register(
        &mut self,
        plugin_config: &Map<String, Json>,
        ctx: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        let config = ExampleConfig::parse(plugin_config)?;
        let runtime = ctx.runtime();

        ctx.register_subscriber("example_native_subscriber", {
            let runtime = runtime.clone();
            let tag = config.tag.clone();
            move |event| subscriber_mark(&runtime, &tag, event)
        })?;

        ctx.register_tool_sanitize_request_guardrail("example_tool_sanitize_request", 10, {
            let tag = config.tag.clone();
            move |_name, args| tag_json(args, "native_tool_sanitize_request", &tag)
        })?;
        ctx.register_tool_sanitize_response_guardrail("example_tool_sanitize_response", 10, {
            let tag = config.tag.clone();
            move |_name, result| tag_json(result, "native_tool_sanitize_response", &tag)
        })?;
        ctx.register_tool_conditional_execution_guardrail("example_tool_conditional", 10, {
            let block_tools = config.block_tools;
            move |name, _args| {
                Ok(block_tools.then(|| format!("tool '{name}' blocked by Rust native plugin")))
            }
        })?;
        ctx.register_tool_request_intercept("example_tool_request", 20, false, {
            let runtime = runtime.clone();
            let tag = config.tag.clone();
            let emit_isolated_scope = config.emit_isolated_scope;
            move |name, args| {
                emit_runtime_events(&runtime, &tag, emit_isolated_scope)?;
                let mut scope = runtime.scope(
                    "example.native.tool_request",
                    ScopeType::Tool,
                    Some(&json!({ "tool": name, "tag": tag })),
                    None,
                    Some(&args),
                )?;
                let tagged = tag_json(args, "native_tool_request_intercept", &tag);
                scope.close(Some(&tagged), None)?;
                Ok(tagged)
            }
        })?;
        ctx.register_tool_execution_intercept("example_tool_execution", 30, {
            let tag = config.tag.clone();
            move |_name, args, next| {
                let request = tag_json(args, "native_tool_execution_request", &tag);
                let result = next.call(request)?;
                Ok(tag_json(result, "native_tool_execution_response", &tag))
            }
        })?;

        ctx.register_llm_sanitize_request_guardrail("example_llm_sanitize_request", 10, {
            let tag = config.tag.clone();
            move |request| tag_llm_request(request, "native_llm_sanitize_request", &tag)
        })?;
        ctx.register_llm_sanitize_response_guardrail("example_llm_sanitize_response", 10, {
            let tag = config.tag.clone();
            move |response| tag_json(response, "native_llm_sanitize_response", &tag)
        })?;
        ctx.register_llm_conditional_execution_guardrail("example_llm_conditional", 10, {
            let block_llms = config.block_llms;
            move |_request| {
                Ok(block_llms.then(|| "LLM call blocked by Rust native plugin".to_string()))
            }
        })?;
        ctx.register_llm_request_intercept("example_llm_request", 20, false, {
            let tag = config.tag.clone();
            move |_name, request, annotated| {
                Ok((
                    tag_llm_request(request, "native_llm_request_intercept", &tag),
                    annotated,
                ))
            }
        })?;
        ctx.register_llm_execution_intercept("example_llm_execution", 30, {
            let tag = config.tag.clone();
            move |_name, request, next| {
                let request = tag_llm_request(request, "native_llm_execution_request", &tag);
                let response = next.call(request)?;
                Ok(tag_json(response, "native_llm_execution_response", &tag))
            }
        })?;
        ctx.register_llm_stream_execution_intercept("example_llm_stream_execution", 30, {
            let tag = config.tag;
            move |_name, request, next| {
                let request = tag_llm_request(request, "native_llm_stream_execution_request", &tag);
                let stream = next.call(request)?;
                let tag = tag.clone();
                let stream: LlmJsonStream = Box::new(stream.map(move |chunk| {
                    chunk.map(|chunk| {
                        tag_json(chunk, "native_llm_stream_execution_response", &tag)
                    })
                }));
                Ok(stream)
            }
        })?;

        Ok(())
    }
}

fn diagnostic(
    level: DiagnosticLevel,
    code: &str,
    field: Option<&str>,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level,
        code: code.into(),
        component: Some("examples.rust_native_policy".into()),
        field: field.map(str::to_owned),
        message: message.into(),
    }
}

fn subscriber_mark(runtime: &PluginRuntime, tag: &str, event: &Event) {
    if event.scope_category() == Some(ScopeCategory::Start)
        && !event.name().starts_with("example.native")
    {
        let _ = runtime.emit_mark(
            "example.native.subscriber.seen",
            Some(&json!({ "event": event.name(), "tag": tag })),
            None,
        );
    }
}

fn emit_runtime_events(
    runtime: &PluginRuntime,
    tag: &str,
    emit_isolated_scope: bool,
) -> nemo_relay_plugin::Result<()> {
    runtime.emit_mark(
        "example.native.tool_request.seen",
        Some(&json!({ "tag": tag })),
        None,
    )?;

    if !emit_isolated_scope {
        return Ok(());
    }

    let isolated = runtime.create_scope_stack()?;
    isolated.with_current(|| {
        runtime.emit_mark(
            "example.native.isolated.mark",
            Some(&json!({ "tag": tag })),
            None,
        )?;
        let mut scope = runtime.scope(
            "example.native.isolated.scope",
            ScopeType::Custom,
            None,
            Some(&json!({ "visibility": "isolated" })),
            Some(&json!({ "tag": tag })),
        )?;
        scope.close(Some(&json!({ "done": true })), None)
    })
}

fn tag_llm_request(mut request: LlmRequest, key: &str, tag: &str) -> LlmRequest {
    request.headers.insert(
        "x-nemo-relay-native-plugin".into(),
        Json::String(tag.into()),
    );
    request.content = tag_json(request.content, key, tag);
    request
}

fn tag_json(value: Json, key: &str, tag: &str) -> Json {
    match value {
        Json::Object(mut object) => {
            object.insert(key.into(), Json::Bool(true));
            object.insert("native_plugin_tag".into(), Json::String(tag.into()));
            Json::Object(object)
        }
        other => other,
    }
}

nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || ExampleNativePlugin);
