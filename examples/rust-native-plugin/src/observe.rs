// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use nemo_relay_plugin::{
    Event, EventSanitizeFields, Json, LlmRequest, PluginContext, PluginRuntime, ScopeCategory,
};
use serde_json::{Map, json};

use crate::config::ExampleConfig;

pub(crate) fn register(
    context: &mut PluginContext<'_>,
    config: &ExampleConfig,
    runtime: &PluginRuntime,
) -> nemo_relay_plugin::Result<()> {
    if !config.observe.enabled {
        return Ok(());
    }

    context.register_event_metadata_injector(
        "documentation_event_metadata_injector",
        5,
        |_event| async move {
            Ok(BTreeMap::from([(
                "external.injector.transport".into(),
                Json::String("rust_native_plugin".into()),
            )]))
        },
    )?;

    context.register_subscriber("documentation_subscriber", {
        let runtime = runtime.clone();
        let tag = config.tag.clone();
        move |event| subscriber_mark(&runtime, &tag, event)
    })?;

    let register_event_sanitizer =
        |fields: EventSanitizeFields, tag: String, redact_keys: Vec<String>| async move {
            sanitize_event_fields(fields, &tag, &redact_keys)
        };

    context.register_mark_sanitize_guardrail("documentation_mark_sanitizer", 10, {
        let tag = config.tag.clone();
        let redact_keys = config.observe.redact_keys.clone();
        move |_event, fields| register_event_sanitizer(fields, tag.clone(), redact_keys.clone())
    })?;
    context.register_scope_sanitize_start_guardrail(
        "documentation_scope_start_sanitizer",
        10,
        {
            let tag = config.tag.clone();
            let redact_keys = config.observe.redact_keys.clone();
            move |_event, fields| register_event_sanitizer(fields, tag.clone(), redact_keys.clone())
        },
    )?;
    context.register_scope_sanitize_end_guardrail("documentation_scope_end_sanitizer", 10, {
        let tag = config.tag.clone();
        let redact_keys = config.observe.redact_keys.clone();
        move |_event, fields| register_event_sanitizer(fields, tag.clone(), redact_keys.clone())
    })?;

    context.register_tool_sanitize_request_guardrail(
        "documentation_tool_request_sanitizer",
        10,
        {
            let redact_keys = config.observe.redact_keys.clone();
            move |_name, value| {
                let redact_keys = redact_keys.clone();
                async move { Ok(redact_json(value, &redact_keys)) }
            }
        },
    )?;
    context.register_tool_sanitize_response_guardrail(
        "documentation_tool_response_sanitizer",
        10,
        {
            let redact_keys = config.observe.redact_keys.clone();
            move |_name, value| {
                let redact_keys = redact_keys.clone();
                async move { Ok(redact_json(value, &redact_keys)) }
            }
        },
    )?;

    context.register_llm_sanitize_request_guardrail(
        "documentation_llm_request_sanitizer",
        10,
        {
            let redact_keys = config.observe.redact_keys.clone();
            move |mut request, codec_context| {
                let redact_keys = redact_keys.clone();
                async move {
                    if let Some(codec) = codec_context.resolve_codec() {
                        let annotated = codec.decode(&request)?;
                        let annotated = serde_json::to_value(annotated)
                            .map(|value| redact_json(value, &redact_keys))
                            .and_then(serde_json::from_value)
                            .map_err(|error| error.to_string())?;
                        request = codec.encode(&annotated, &request)?;
                    }
                    request.content = redact_json(request.content, &redact_keys);
                    Ok(Some(request))
                }
            }
        },
    )?;
    context.register_llm_sanitize_response_guardrail(
        "documentation_llm_response_sanitizer",
        10,
        {
            let redact_keys = config.observe.redact_keys.clone();
            move |response, codec_context| {
                let redact_keys = redact_keys.clone();
                async move {
                    if let Some(codec) = codec_context.resolve_codec() {
                        let _annotated = codec.decode(&response)?;
                    }
                    Ok(Some(redact_json(response, &redact_keys)))
                }
            }
        },
    )?;

    Ok(())
}

fn sanitize_event_fields(
    mut fields: EventSanitizeFields,
    tag: &str,
    redact_keys: &[String],
) -> nemo_relay_plugin::Result<EventSanitizeFields> {
    fields.data = fields.data.map(|value| redact_json(value, redact_keys));
    fields.metadata = Some(tagged_metadata(fields.metadata, tag, redact_keys));
    if let Some(profile) = fields.category_profile.take() {
        let value = serde_json::to_value(profile).map_err(|error| error.to_string())?;
        fields.category_profile = Some(
            serde_json::from_value(redact_json(value, redact_keys))
                .map_err(|error| error.to_string())?,
        );
    }
    Ok(fields)
}

fn tagged_metadata(metadata: Option<Json>, tag: &str, redact_keys: &[String]) -> Json {
    let mut metadata = match redact_json(metadata.unwrap_or_else(|| json!({})), redact_keys) {
        Json::Object(object) => object,
        other => Map::from_iter([("original".into(), other)]),
    };
    metadata.insert("plugin_tag".into(), Json::String(tag.into()));
    Json::Object(metadata)
}

pub(crate) fn redact_json(value: Json, redact_keys: &[String]) -> Json {
    match value {
        Json::Object(mut object) => {
            for (key, value) in &mut object {
                if redact_keys.iter().any(|candidate| candidate == key) {
                    *value = Json::String("[REDACTED]".into());
                } else {
                    *value = redact_json(value.take(), redact_keys);
                }
            }
            Json::Object(object)
        }
        Json::Array(values) => Json::Array(
            values
                .into_iter()
                .map(|value| redact_json(value, redact_keys))
                .collect(),
        ),
        other => other,
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

pub(crate) fn add_header(mut request: LlmRequest, name: &str, value: &str) -> LlmRequest {
    request
        .headers
        .insert(name.into(), Json::String(value.into()));
    request
}
