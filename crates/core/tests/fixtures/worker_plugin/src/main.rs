// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use nemo_relay_worker::{
    ConfigDiagnostic, DiagnosticLevel, EmitMarkOptions, EventCategory, EventSanitizeFields, Json,
    LlmRequest, METRIC_DATA_SCHEMA_NAME, MetricKind, MetricMeasurement, MetricValueType,
    PendingMarkSpec,
};
use nemo_relay_worker::{
    JsonStream, LlmNext, LlmStreamNext, PluginContext, RuntimeRegistrationKind, ScopeType,
    ToolExecutionInterceptOutcome, ToolNext, WorkerPlugin, WorkerSdkError, serve_plugin,
};
use serde_json::json;

struct FixtureWorkerPlugin;

impl WorkerPlugin for FixtureWorkerPlugin {
    fn plugin_id(&self) -> &str {
        "fixture_worker"
    }

    fn validate(&self, config: &Json) -> Vec<ConfigDiagnostic> {
        if config
            .get("exit_in_validate")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            std::process::exit(42);
        }
        if config
            .get("reject")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "fixture.rejected".into(),
                component: Some("fixture_worker".into()),
                field: Some("reject".into()),
                message: "fixture rejection requested".into(),
            }];
        }
        Vec::new()
    }

    fn register(&self, ctx: &mut PluginContext, config: &Json) -> nemo_relay_worker::Result<()> {
        if fixture_flag(config, "exit_in_register") {
            std::process::exit(43);
        }
        if fixture_flag(config, "register_error") {
            return Err(WorkerSdkError::Callback(
                "fixture registration error requested".into(),
            ));
        }
        if fixture_flag(config, "empty_registration_name") {
            ctx.register_subscriber("", |_| {});
            return Ok(());
        }

        let event_metadata_injector_error = fixture_flag(config, "event_metadata_injector_error");
        let exit_in_event_metadata_injector =
            fixture_flag(config, "exit_in_event_metadata_injector");
        ctx.register_event_metadata_injector(
            "fixture_event_metadata_injector",
            0,
            move |event| async move {
                if exit_in_event_metadata_injector {
                    std::process::exit(44);
                }
                if event_metadata_injector_error {
                    return Err(WorkerSdkError::Callback(
                        "fixture Event metadata injector error requested".into(),
                    ));
                }
                let mut additions = BTreeMap::new();
                if event
                    .name()
                    .starts_with("external-plugin-event-metadata-injection")
                {
                    additions.insert(
                        "external.injector.transport".into(),
                        json!("rust_grpc_worker"),
                    );
                }
                Ok(additions)
            },
        );
        if fixture_flag(config, "event_metadata_injector_only") {
            return Ok(());
        }
        let runtime = ctx
            .runtime()
            .ok_or_else(|| WorkerSdkError::Callback("runtime handle missing".into()))?;
        ctx.register_mark_sanitize_guardrail("fixture_mark_sanitize", 0, |_, fields| async move {
            Ok(mark_event_fields(fields, "worker_plugin_mark"))
        });
        ctx.register_mark_sanitize_guardrail(
            "fixture_mark_sanitize_data",
            1,
            |event, mut fields| {
                let is_metric = event
                    .data_schema()
                    .is_some_and(|schema| schema.name == METRIC_DATA_SCHEMA_NAME);
                async move {
                    if !is_metric {
                        fields.data = Some(json!({"worker_plugin_mark_data": true}));
                    }
                    Ok(fields)
                }
            },
        );
        let nested_publication_runtime = runtime.clone();
        ctx.register_mark_sanitize_guardrail(
            "fixture_nested_publication_order",
            2,
            move |event, fields| {
                let runtime = nested_publication_runtime.clone();
                let emit_nested = event.name() == "worker-nested-order-outer";
                async move {
                    if emit_nested {
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        runtime
                            .emit_mark("worker-nested-order-inner", None, None)
                            .await?;
                    }
                    Ok(fields)
                }
            },
        );
        ctx.register_scope_sanitize_start_guardrail(
            "fixture_scope_start_sanitize",
            0,
            |_, fields| async move { Ok(mark_event_fields(fields, "worker_plugin_scope_start")) },
        );
        ctx.register_scope_sanitize_end_guardrail(
            "fixture_scope_end_sanitize",
            0,
            |_, fields| async move { Ok(mark_event_fields(fields, "worker_plugin_scope_end")) },
        );
        register_fixture_subscriber(ctx, runtime.clone());
        ctx.register_conditional_middleware_guardrail(
            "fixture_initial_gate",
            BTreeSet::from([RuntimeRegistrationKind::Subscriber]),
            "fixture_worker_gate_target",
            |_, _| async { Ok(Some("fixture initial gate".into())) },
        );
        register_fixture_tool_hooks(
            ctx,
            runtime,
            fixture_flag(config, "block_tool"),
            fixture_flag(config, "tool_request_error"),
            fixture_flag(config, "exit_in_tool_request"),
            fixture_flag(config, "emit_tokenomics_metric"),
        );
        register_fixture_llm_hooks(
            ctx,
            fixture_flag(config, "llm_request_error"),
            fixture_flag(config, "llm_stream_open_error"),
        );
        Ok(())
    }
}

fn fixture_flag(config: &Json, key: &str) -> bool {
    config.get(key).and_then(Json::as_bool).unwrap_or(false)
}

fn mark_event_fields(mut fields: EventSanitizeFields, marker: &str) -> EventSanitizeFields {
    let mut metadata = fields.metadata.unwrap_or_else(|| json!({}));
    metadata[marker] = json!(true);
    fields.metadata = Some(metadata);
    fields
}

fn register_fixture_subscriber(ctx: &mut PluginContext, runtime: nemo_relay_worker::PluginRuntime) {
    ctx.register_subscriber("fixture_subscriber", move |event| {
        if event.name() == "worker-plugin-test-outer" {
            let runtime = runtime.clone();
            let _ = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    runtime
                        .emit_mark(
                            "fixture.worker.subscriber.mark",
                            Some(json!("subscriber")),
                            None,
                        )
                        .await
                })
            });
        }
    });
}

fn register_fixture_tool_hooks(
    ctx: &mut PluginContext,
    runtime: nemo_relay_worker::PluginRuntime,
    block_tool: bool,
    tool_request_error: bool,
    exit_in_tool_request: bool,
    emit_tokenomics_metric: bool,
) {
    ctx.register_tool_sanitize_request_guardrail(
        "fixture_tool_sanitize_request",
        0,
        |_name, args| async move { Ok(mark_json(args, "worker_plugin_tool_sanitize_request")) },
    );
    ctx.register_tool_sanitize_response_guardrail(
        "fixture_tool_sanitize_response",
        0,
        |_name, result| async move {
            Ok(mark_json(result, "worker_plugin_tool_sanitize_response"))
        },
    );
    ctx.register_tool_conditional_execution_guardrail(
        "fixture_tool_conditional",
        0,
        move |_name, _args| async move {
            if block_tool {
                Ok(Some("fixture tool blocked".into()))
            } else {
                Ok(None)
            }
        },
    );
    ctx.register_tool_request_intercept("fixture_rewrite_args", 0, false, move |_name, args| {
        let runtime = runtime.clone();
        async move {
            if exit_in_tool_request {
                std::process::exit(44);
            }
            if tool_request_error {
                return Err(WorkerSdkError::Callback(
                    "fixture tool request error requested".into(),
                ));
            }
            if emit_tokenomics_metric {
                runtime
                    .emit_metric(
                        "tokenomics.measurements",
                        vec![MetricMeasurement {
                            name: "example.tokens.saved".into(),
                            kind: MetricKind::Counter,
                            value_type: MetricValueType::U64,
                            value: json!(42),
                            unit: Some("{token}".into()),
                            description: Some("Tokens avoided".into()),
                            attributes: Some(json!({"model": "example-model"})),
                            boundaries: None,
                        }],
                        None,
                    )
                    .await?;
            } else {
                emit_runtime_events(runtime).await?;
            }
            Ok(mark_json(args, "worker_plugin"))
        }
    });
    ctx.register_tool_execution_intercept(
        "fixture_tool_execution",
        0,
        |_name, args, next: ToolNext| async move {
            let result = next
                .call(mark_json(args, "worker_plugin_tool_execution_request"))
                .await?;
            let mut result = result;
            result.result = mark_json(result.result, "worker_plugin_tool_execution");
            Ok(
                ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                    PendingMarkSpec::builder()
                        .name("fixture.worker.tool_execution.mark")
                        .build(),
                ),
            )
        },
    );
}

fn register_fixture_llm_hooks(
    ctx: &mut PluginContext,
    llm_request_error: bool,
    llm_stream_open_error: bool,
) {
    ctx.register_llm_sanitize_request_guardrail(
        "fixture_llm_sanitize_request",
        0,
        |request, _context| async move {
            Ok(Some(mark_llm_request(
                request,
                "worker_plugin_llm_sanitize_request",
            )))
        },
    );
    ctx.register_llm_sanitize_response_guardrail(
        "fixture_llm_sanitize_response",
        0,
        |response, _context| async move {
            Ok(Some(mark_json(
                response,
                "worker_plugin_llm_sanitize_response",
            )))
        },
    );
    ctx.register_llm_conditional_execution_guardrail(
        "fixture_llm_conditional",
        0,
        |_request| async { Ok(None) },
    );
    ctx.register_llm_request_intercept(
        "fixture_llm_request_intercept",
        0,
        false,
        move |_name, request, annotated| async move {
            if llm_request_error {
                return Err(WorkerSdkError::Callback(
                    "fixture LLM request error requested".into(),
                ));
            }
            let (request, annotated) = match annotated {
                Some(mut annotated) => {
                    annotated
                        .extra
                        .insert("worker_plugin_annotated_request".into(), json!(true));
                    (request, Some(annotated))
                }
                None => (
                    mark_llm_request(request, "worker_plugin_llm_request_intercept"),
                    None,
                ),
            };
            Ok(
                nemo_relay_worker::LlmRequestInterceptOutcome::new(request, annotated)
                    .with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("fixture.worker.llm_request.mark")
                            .data(json!({ "source": "worker_request_intercept" }))
                            .metadata(json!({ "fixture": true }))
                            .build(),
                    ),
            )
        },
    );
    ctx.register_llm_execution_intercept(
        "fixture_llm_execution",
        0,
        |_name, request, next: LlmNext| async move {
            let response = next
                .call(mark_llm_request(
                    request,
                    "worker_plugin_llm_execution_request",
                ))
                .await?;
            Ok(mark_json(response, "worker_plugin_llm_execution"))
        },
    );
    ctx.register_llm_stream_execution_intercept(
        "fixture_llm_stream_execution",
        0,
        move |_name, request, next: LlmStreamNext| async move {
            if llm_stream_open_error {
                return Err(WorkerSdkError::Callback(
                    "fixture LLM stream open error requested".into(),
                ));
            }
            let stream = next
                .call(mark_llm_request(
                    request,
                    "worker_plugin_llm_stream_execution_request",
                ))
                .await?;
            let mapped: JsonStream = Box::pin(tokio_stream::StreamExt::map(stream, |chunk| {
                chunk.map(|value| mark_json(value, "worker_plugin_llm_stream_execution"))
            }));
            Ok(mapped)
        },
    );
}

async fn emit_runtime_events(
    runtime: nemo_relay_worker::PluginRuntime,
) -> nemo_relay_worker::Result<()> {
    runtime
        .emit_mark_with_options_and_category(
            "fixture.worker.mark",
            Some(json!("current")),
            None,
            EmitMarkOptions::default(),
            EventCategory::custom(),
        )
        .await?;
    let scope = runtime
        .push_scope(
            None,
            "fixture.worker.scope",
            ScopeType::Custom,
            None,
            None,
            Some(json!("current-scope-input")),
        )
        .await?;
    runtime
        .pop_scope(&scope, Some(json!("current-scope-output")), None)
        .await?;

    let isolated = runtime.create_scope_stack().await?;
    let isolated_scope = runtime
        .push_scope(
            Some(&isolated),
            "fixture.worker.isolated.scope",
            ScopeType::Custom,
            None,
            None,
            Some(json!("isolated-input")),
        )
        .await?;
    let isolated_runtime = runtime.clone();
    runtime
        .with_scope_stack(&isolated, || async move {
            isolated_runtime
                .emit_mark(
                    "fixture.worker.isolated.mark",
                    Some(json!("isolated")),
                    None,
                )
                .await
        })
        .await?;
    runtime
        .pop_scope(&isolated_scope, Some(json!("isolated-output")), None)
        .await?;
    runtime.drop_scope_stack(&isolated).await
}

fn mark_llm_request(mut request: LlmRequest, key: &str) -> LlmRequest {
    request.content = mark_json(request.content, key);
    request
}

fn mark_json(mut value: Json, key: &str) -> Json {
    if let Json::Object(object) = &mut value {
        object.insert(key.into(), json!(true));
    }
    value
}

#[tokio::main]
async fn main() {
    if let Err(error) = serve_plugin(FixtureWorkerPlugin).await {
        eprintln!("fixture worker failed: {error}");
        std::process::exit(1);
    }
}
