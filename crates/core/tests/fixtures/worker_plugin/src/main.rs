// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay_worker::{
    JsonStream, LlmNext, LlmStreamNext, PluginContext, ScopeType, ToolNext, WorkerPlugin,
    WorkerSdkError, serve_plugin,
};
use nemo_relay_worker::{ConfigDiagnostic, DiagnosticLevel, Json, LlmRequest};
use serde_json::json;

struct FixtureWorkerPlugin;

impl WorkerPlugin for FixtureWorkerPlugin {
    fn plugin_id(&self) -> &str {
        if std::env::var("FIXTURE_WORKER_PLUGIN_ID").as_deref() == Ok("other_worker") {
            return "other_worker";
        }
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
        let register_error = config
            .get("register_error")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let exit_in_register = config
            .get("exit_in_register")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        if exit_in_register {
            std::process::exit(43);
        }
        if register_error {
            return Err(WorkerSdkError::Callback(
                "fixture registration error requested".into(),
            ));
        }

        let empty_registration_name = config
            .get("empty_registration_name")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        if empty_registration_name {
            ctx.register_subscriber("", |_| {});
            return Ok(());
        }

        let block_tool = config
            .get("block_tool")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let tool_request_error = config
            .get("tool_request_error")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let llm_request_error = config
            .get("llm_request_error")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        let llm_stream_open_error = config
            .get("llm_stream_open_error")
            .and_then(Json::as_bool)
            .unwrap_or(false);

        let runtime = ctx
            .runtime()
            .ok_or_else(|| WorkerSdkError::Callback("runtime handle missing".into()))?;

        ctx.register_subscriber("fixture_subscriber", {
            let runtime = runtime.clone();
            move |event| {
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
            }
        });

        ctx.register_tool_sanitize_request_guardrail(
            "fixture_tool_sanitize_request",
            0,
            |_name, args| mark_json(args, "worker_plugin_tool_sanitize_request"),
        );
        ctx.register_tool_sanitize_response_guardrail(
            "fixture_tool_sanitize_response",
            0,
            |_name, result| mark_json(result, "worker_plugin_tool_sanitize_response"),
        );
        ctx.register_tool_conditional_execution_guardrail(
            "fixture_tool_conditional",
            0,
            move |_name, _args| {
                if block_tool {
                    Ok(Some("fixture tool blocked".into()))
                } else {
                    Ok(None)
                }
            },
        );
        ctx.register_tool_request_intercept("fixture_rewrite_args", 0, false, {
            let runtime = runtime.clone();
            move |_name, args| {
                if tool_request_error {
                    return Err(WorkerSdkError::Callback(
                        "fixture tool request error requested".into(),
                    ));
                }
                let runtime = runtime.clone();
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(emit_runtime_events(runtime))
                })?;
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
                Ok(mark_json(result, "worker_plugin_tool_execution"))
            },
        );

        ctx.register_llm_sanitize_request_guardrail(
            "fixture_llm_sanitize_request",
            0,
            |request| mark_llm_request(request, "worker_plugin_llm_sanitize_request"),
        );
        ctx.register_llm_sanitize_response_guardrail(
            "fixture_llm_sanitize_response",
            0,
            |response| mark_json(response, "worker_plugin_llm_sanitize_response"),
        );
        ctx.register_llm_conditional_execution_guardrail(
            "fixture_llm_conditional",
            0,
            |_request| Ok(None),
        );
        ctx.register_llm_request_intercept(
            "fixture_llm_request_intercept",
            0,
            false,
            move |_name, request, annotated| {
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
                Ok(nemo_relay_worker::LlmRequestInterceptOutcome::new(
                    request,
                    annotated,
                ))
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

        Ok(())
    }
}

async fn emit_runtime_events(runtime: nemo_relay_worker::PluginRuntime) -> nemo_relay_worker::Result<()> {
    runtime
        .emit_mark("fixture.worker.mark", Some(json!("current")), None)
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
                .emit_mark("fixture.worker.isolated.mark", Some(json!("isolated")), None)
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
