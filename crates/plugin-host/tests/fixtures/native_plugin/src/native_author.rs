// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) struct FixtureNativePlugin;

struct DropScope {
    runtime: PluginRuntime,
    name: &'static str,
}

impl Drop for DropScope {
    fn drop(&mut self) {
        if let Ok(mut scope) = self
            .runtime
            .scope(self.name, ScopeType::Custom, None, None, None)
        {
            let _ = scope.close(None, None);
        }
    }
}

// Make the distinction between terminal delivery and producer cleanup observable.
struct StreamCleanup {
    marker_path: String,
    _scope: DropScope,
}

impl Drop for StreamCleanup {
    fn drop(&mut self) {
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&self.marker_path, "dropped").expect("write stream cleanup marker");
    }
}

impl NativePlugin for FixtureNativePlugin {
    fn plugin_kind(&self) -> &str {
        "fixture_native"
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        if plugin_config
            .get("reject")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            return vec![ConfigDiagnostic {
                level: DiagnosticLevel::Error,
                code: "fixture.rejected".into(),
                component: Some("fixture_native".into()),
                field: Some("reject".into()),
                message: "fixture rejection requested".into(),
            }];
        }
        vec![]
    }

    fn executor_config(&self) -> NativeExecutorConfig {
        NativeExecutorConfig { worker_threads: 3 }
    }

    fn register<'a>(
        &'a self,
        plugin_config: &'a Map<String, Json>,
        ctx: &'a mut PluginContext<'_>,
    ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
        Box::pin(async move {
            if plugin_config
                .get("registration_mark")
                .and_then(Json::as_bool)
                == Some(true)
            {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                ctx.runtime()
                    .emit_mark("fixture-registration-mark", None, None)?;
            }
            let result: nemo_relay_plugin::Result<()> = async move {
        if plugin_config
            .get("diagnostics_fixture")
            .and_then(Json::as_bool)
            == Some(true)
        {
            return register_diagnostics_fixture(ctx, plugin_config);
        }
        let event_metadata_injector_error = plugin_config
            .get("event_metadata_injector_error")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        register_fixture_event_metadata_injector(ctx, event_metadata_injector_error)?;
        if plugin_config
            .get("event_metadata_injector_only")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            return Ok(());
        }
        let runtime = ctx.runtime();
        let subscriber_kinds = BTreeSet::from([RuntimeRegistrationKind::Subscriber]);
        if plugin_config
            .get("discover_observability")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            let observability_subscribers = runtime
                .list_runtime_registrations(Some(&subscriber_kinds))?
                .into_iter()
                .filter(|registration| {
                    registration.local_name == "opentelemetry"
                        && registration.owner.plugin_kind.as_deref() == Some("observability")
                })
                .count();
            if observability_subscribers != 1 {
                return Err(format!(
                    "expected one observability subscriber during registration; found {observability_subscribers}"
                ));
            }
        }
        ctx.register_subscriber("fixture_subscriber", {
            let runtime = runtime.clone();
            move |event| subscriber_mark(&runtime, event)
        })?;
        if !runtime
            .list_runtime_registrations(Some(&subscriber_kinds))?
            .iter()
            .any(|registration| registration.local_name == "fixture_subscriber")
        {
            return Err("native runtime did not discover its subscriber".into());
        }
        ctx.register_conditional_middleware_guardrail(
            "fixture_initial_gate",
            &subscriber_kinds,
            "missing-initial-target",
            |_, _| Some("fixture initial gate".into()),
        )?;
        if plugin_config
            .get("conditional_gate_test")
            .and_then(Json::as_bool)
            .unwrap_or(false)
        {
            ctx.register_conditional_middleware_guardrail(
                "fixture_blocking_callback_gate",
                &subscriber_kinds,
                "fixture-native-blocked-subscriber",
                |_, _| Some("fixture callback blocked target".into()),
            )?;
            ctx.register_conditional_middleware_guardrail(
                "fixture_allowing_callback_gate",
                &subscriber_kinds,
                "fixture-native-allowed-subscriber",
                |_, _| None,
            )?;
            ctx.register_conditional_middleware_guardrail(
                "fixture_failing_callback_gate",
                &subscriber_kinds,
                "fixture-native-failed-open-subscriber",
                |_, _| panic!("fixture callback failure"),
            )?;
        }
        let dynamic_gate = runtime.register_conditional_middleware_guardrail(
            "fixture_dynamic_gate",
            &subscriber_kinds,
            "missing-dynamic-target",
            |_, _| Some("fixture dynamic gate".into()),
        )?;
        if !runtime.deregister_conditional_middleware_guardrail(&dynamic_gate)? {
            return Err("native runtime did not remove its dynamic gate".into());
        }
        ctx.register_mark_sanitize_guardrail("fixture_mark_sanitize", 0, |_, fields| async move {
            Ok(mark_event_fields(fields, "native_plugin_mark"))
        })?;
        ctx.register_scope_sanitize_start_guardrail(
            "fixture_scope_start_sanitize",
            0,
            |_, fields| async move { Ok(mark_event_fields(fields, "native_plugin_scope_start")) },
        )?;
        ctx.register_scope_sanitize_end_guardrail(
            "fixture_scope_end_sanitize",
            0,
            |_, fields| async move { Ok(mark_event_fields(fields, "native_plugin_scope_end")) },
        )?;

        ctx.register_tool_sanitize_request_guardrail(
            "fixture_tool_sanitize_request",
            0,
            |_name, args| async move { Ok(mark_json(args, "native_plugin_tool_sanitize_request")) },
        )?;
        ctx.register_tool_sanitize_response_guardrail(
            "fixture_tool_sanitize_response",
            0,
            |_name, result| async move {
                Ok(mark_json(result, "native_plugin_tool_sanitize_response"))
            },
        )?;
        ctx.register_tool_conditional_execution_guardrail(
            "fixture_tool_conditional",
            0,
            |_name, _args| async move { Ok(None) },
        )?;
        ctx.register_tool_request_intercept("fixture_rewrite_args", 0, false, {
            let runtime = runtime.clone();
            move |_name, args| {
                let runtime = runtime.clone();
                async move {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    let (mut writer, mut reader) = tokio::io::duplex(8);
                    writer
                        .write_all(b"async")
                        .await
                        .map_err(|error| error.to_string())?;
                    let mut readiness = [0_u8; 5];
                    reader
                        .read_exact(&mut readiness)
                        .await
                        .map_err(|error| error.to_string())?;
                    let before = runtime.current_scope_stack_snapshot()?;
                    let temporary_scope = runtime.push_scope(
                        "fixture.snapshot",
                        ScopeType::Function,
                        None,
                        None,
                        None,
                    )?;
                    let inside = runtime.current_scope_stack_snapshot()?;
                    runtime.pop_scope(&temporary_scope, None, None)?;
                    let after = runtime.current_scope_stack_snapshot()?;
                    emit_runtime_events(&runtime)?;
                    let mut result = mark_json(args, "native_plugin");
                    result["scope_stack_snapshot"] =
                        serde_json::to_value(&before).map_err(|error| error.to_string())?;
                    result["scope_stack_snapshot_inside"] =
                        serde_json::to_value(&inside).map_err(|error| error.to_string())?;
                    result["scope_stack_snapshot_after"] =
                        serde_json::to_value(&after).map_err(|error| error.to_string())?;
                    Ok(result)
                }
            }
        })?;
        register_fixture_tool_execution(ctx, &runtime)?;

        register_fixture_llm_guardrails(ctx, &runtime)?;
        register_fixture_llm_stream_execution(ctx, &runtime)?;
        ctx.register_llm_execution_intercept("fixture_llm_execution_cancellation", -1, {
            let runtime = runtime.clone();
            move |name, request, _context, next| {
                let runtime = runtime.clone();
                async move {
                    if name != "native-fixture-cancelled-unary" {
                        return next.call(request).await;
                    }
                    let _drop_scope = DropScope {
                        runtime,
                        name: "fixture.native.unary.drop",
                    };
                    crate::ASYNC_PENDING_ENTERED.store(true, Ordering::Release);
                    futures::future::pending::<nemo_relay_plugin::Result<Json>>().await
                }
            }
        })?;
        ctx.register_llm_stream_execution_intercept(
            "fixture_llm_stream_execution_cancellation",
            -1,
            {
                let runtime = runtime.clone();
                move |name, request, _context, next| {
                    let runtime = runtime.clone();
                    async move {
                        if name == "native-fixture-cleanup-stream" {
                            let cleanup = StreamCleanup {
                                marker_path: request.content["marker_path"]
                                    .as_str()
                                    .ok_or("missing marker path")?
                                    .to_owned(),
                                _scope: DropScope {
                                    runtime,
                                    name: "fixture.native.stream.cleanup",
                                },
                            };
                            let mode = request.content["mode"]
                                .as_str()
                                .unwrap_or("pending")
                                .to_owned();
                            let mut first = true;
                            let stream: LlmJsonAsyncStream =
                                Box::pin(futures::stream::poll_fn(move |_| {
                                    let _ = &cleanup;
                                    if first {
                                        first = false;
                                        return std::task::Poll::Ready(Some(Ok(
                                            json!({"token": "first"}),
                                        )));
                                    }
                                    match mode.as_str() {
                                        "end" => std::task::Poll::Ready(None),
                                        "error" => std::task::Poll::Ready(Some(Err(
                                            "native cleanup stream failed".into(),
                                        ))),
                                        _ => std::task::Poll::Pending,
                                    }
                                }));
                            return Ok(stream);
                        }
                        if name != "native-fixture-cancelled-stream" {
                            return next.call(request).await;
                        }
                        let drop_scope = DropScope {
                            runtime,
                            name: "fixture.native.stream.drop",
                        };
                        let stream = next.call(request).await?;
                        let stream: LlmJsonAsyncStream = Box::pin(stream.map(move |chunk| {
                            let _ = &drop_scope;
                            chunk
                        }));
                        Ok(stream)
                    }
                }
            },
        )?;

        Ok(())

}.await;
            result.map_err(nemo_relay_plugin::component::PluginError::RegistrationFailed)
        })
    }
}

fn register_fixture_llm_stream_execution(
    ctx: &mut PluginContext<'_>,
    runtime: &PluginRuntime,
) -> nemo_relay_plugin::Result<()> {
    ctx.register_llm_stream_execution_intercept("fixture_llm_stream_execution", 0, {
        let runtime = runtime.clone();
        move |_name, request, _context, next| {
            let runtime = runtime.clone();
            async move {
                if request.content["native_http_error"].as_bool() == Some(true) {
                    return Err(nemo_relay_plugin::http_error_message(
                        403,
                        "provider denied access",
                    )?);
                }
                if let Some(count) = request.content["native_stream_http_error_after_chunks"].as_u64() {
                    let chunks = (0..count)
                        .map(|index| Ok(json!({"before_error":index})))
                        .chain(std::iter::once(Err(nemo_relay_plugin::http_error_message(
                            403,
                            "provider denied access",
                        )?)));
                    return Ok(Box::pin(futures::stream::iter(chunks)) as LlmJsonAsyncStream);
                }
                let inspect_snapshot =
                    request.content["inspect_scope_stack"].as_bool() == Some(true);
                let stream = next
                    .call(mark_llm_request(
                        request,
                        "native_plugin_llm_stream_execution_request",
                    ))
                    .await?;
                let snapshot = if inspect_snapshot {
                    tokio::task::yield_now().await;
                    serde_json::to_value(runtime.current_scope_stack_snapshot()?)
                        .map_err(|error| error.to_string())?
                } else {
                    Json::Null
                };
                let stream: LlmJsonAsyncStream = Box::pin(stream.then(move |chunk| {
                    let runtime = runtime.clone();
                    let snapshot = snapshot.clone();
                    async move {
                        let mut chunk = mark_json(chunk?, "native_plugin_llm_stream_execution");
                        if inspect_snapshot {
                            tokio::task::yield_now().await;
                            chunk["scope_stack_snapshot"] = snapshot;
                            chunk["scope_stack_snapshot_polled"] =
                                serde_json::to_value(runtime.current_scope_stack_snapshot()?)
                                    .map_err(|error| error.to_string())?;
                        }
                        Ok(chunk)
                    }
                }));
                Ok(stream)
            }
        }
    })
}

fn register_fixture_tool_execution(
    ctx: &mut PluginContext<'_>,
    runtime: &PluginRuntime,
) -> nemo_relay_plugin::Result<()> {
    ctx.register_tool_execution_intercept("fixture_tool_execution", 0, {
        let runtime = runtime.clone();
        move |context, next| {
            let runtime = runtime.clone();
            async move {
                let args = context.args;
                let args = mark_json(args, "native_plugin_tool_execution_request");
                let result = if args
                    .get("use_scoped_next")
                    .and_then(Json::as_bool)
                    .unwrap_or(false)
                {
                    let mut scope = runtime.scope(
                        "fixture.native.scoped.next",
                        ScopeType::Custom,
                        None,
                        None,
                        Some(&Json::String("scoped-next-input".into())),
                    )?;
                    let call_result = next.call(args).await;
                    let close_result =
                        scope.close(Some(&Json::String("scoped-next-output".into())), None);
                    close_result?;
                    call_result?
                } else if args
                    .get("use_isolated_next")
                    .and_then(Json::as_bool)
                    .unwrap_or(false)
                {
                    let isolated = runtime.create_scope_stack()?;
                    let previous = runtime.capture_scope_stack_thread()?;
                    isolated.set_thread()?;
                    let mut scope = runtime.scope(
                        "fixture.native.isolated.next",
                        ScopeType::Custom,
                        None,
                        None,
                        Some(&Json::String("isolated-next-input".into())),
                    )?;
                    let call_result = next.call(args).await;
                    let close_result =
                        scope.close(Some(&Json::String("isolated-next-output".into())), None);
                    previous.restore()?;
                    close_result?;
                    call_result?
                } else if args
                    .get("use_concurrent_next")
                    .and_then(Json::as_bool)
                    .unwrap_or(false)
                {
                    let first_next = next.clone();
                    let (first, second) =
                        tokio::join!(first_next.call(args.clone()), next.call(args),);
                    let result = first?;
                    second?;
                    result
                } else {
                    next.call(args).await?
                };
                let mut result = result;
                result.result = mark_json(result.result, "native_plugin_tool_execution");
                Ok(
                    ToolExecutionInterceptOutcome::from(result).with_pending_mark(
                        PendingMarkSpec::builder()
                            .name("fixture.native.tool_execution.mark")
                            .category(EventCategory::custom())
                            .category_profile(CategoryProfile {
                                subtype: Some("fixture.native.tool_execution".into()),
                                ..CategoryProfile::default()
                            })
                            .data(json!({ "source": "native_tool_execution" }))
                            .metadata(json!({ "fixture": true }))
                            .build(),
                    ),
                )
            }
        }
    })?;
    ctx.register_tool_execution_intercept("fixture_tool_execution_nested", 1, {
        let runtime = runtime.clone();
        move |context, next| {
            let runtime = runtime.clone();
            async move {
                let args = context.args;
                if !args
                    .get("use_scoped_next")
                    .and_then(Json::as_bool)
                    .unwrap_or(false)
                {
                    return next.call(args).await.map(Into::into);
                }
                let mut scope = runtime.scope(
                    "fixture.native.scoped.next.downstream",
                    ScopeType::Custom,
                    None,
                    None,
                    Some(&Json::String("scoped-next-downstream-input".into())),
                )?;
                let call_result = next.call(args).await;
                let close_result = scope.close(
                    Some(&Json::String("scoped-next-downstream-output".into())),
                    None,
                );
                close_result?;
                call_result.map(Into::into)
            }
        }
    })?;
    Ok(())
}

fn register_fixture_event_metadata_injector(
    ctx: &mut PluginContext<'_>,
    return_error: bool,
) -> nemo_relay_plugin::Result<()> {
    ctx.register_event_metadata_injector(
        "fixture_event_metadata_injector",
        0,
        move |event| async move {
            if return_error {
                return Err("fixture Event metadata injector error requested".into());
            }
            let mut additions = BTreeMap::new();
            if event
                .name()
                .starts_with("external-plugin-event-metadata-injection")
            {
                additions.insert(
                    "external.injector.transport".into(),
                    json!("native_dynamic_rust"),
                );
            }
            Ok(additions)
        },
    )
}

fn mark_event_fields(mut fields: EventSanitizeFields, marker: &str) -> EventSanitizeFields {
    let mut metadata = fields.metadata.unwrap_or_else(|| json!({}));
    metadata[marker] = json!(true);
    fields.metadata = Some(metadata);
    fields
}

fn subscriber_mark(runtime: &PluginRuntime, event: &Event) {
    if event.name() == "native-plugin-test-outer"
        && event.scope_category() == Some(ScopeCategory::Start)
    {
        let _ = runtime.emit_mark(
            "fixture.native.subscriber.mark",
            Some(&Json::String("subscriber".into())),
            None,
        );
    }
}

fn emit_runtime_events(runtime: &PluginRuntime) -> nemo_relay_plugin::Result<()> {
    runtime.emit_mark(
        "fixture.native.mark",
        Some(&Json::String("current".into())),
        None,
    )?;
    runtime.emit_metric(
        "fixture.native.metric",
        vec![
            MetricMeasurement::builder()
                .name("fixture.native.count")
                .kind(MetricKind::Counter)
                .value_type(MetricValueType::U64)
                .value(json!(1))
                .build(),
        ],
        None,
    )?;
    let scope = runtime.push_scope(
        "fixture.native.scope",
        ScopeType::Custom,
        None,
        None,
        Some(&Json::String("current-scope-input".into())),
    )?;
    runtime.emit_mark(
        "fixture.native.scope.mark",
        Some(&Json::String("inside-current-scope".into())),
        None,
    )?;
    runtime.pop_scope(
        &scope,
        Some(&Json::String("current-scope-output".into())),
        None,
    )?;

    let thread_stack = runtime.create_scope_stack()?;
    {
        let _thread_guard = runtime.bind_scope_stack_thread(&thread_stack)?;
        runtime.emit_mark(
            "fixture.native.thread_stack.mark",
            Some(&Json::String("thread-stack".into())),
            None,
        )?;
    }

    let isolated = runtime.create_scope_stack()?;
    isolated.with_current(|| {
        runtime.emit_mark(
            "fixture.native.isolated.mark",
            Some(&Json::String("isolated".into())),
            None,
        )?;
        let scope = runtime.push_scope(
            "fixture.native.isolated.scope",
            ScopeType::Custom,
            None,
            None,
            Some(&Json::String("isolated-scope-input".into())),
        )?;
        runtime.pop_scope(
            &scope,
            Some(&Json::String("isolated-scope-output".into())),
            None,
        )
    })
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

fn register_diagnostics_fixture(
    ctx: &mut PluginContext<'_>,
    config: &Map<String, Json>,
) -> nemo_relay_plugin::Result<()> {
    use nemo_relay_plugin::diagnostics::*;
    use std::num::NonZeroU64;
    let diagnostics = ctx.runtime().diagnostics();
    let codes = [DiagnosticCode::Extension {
        name: "lookup_failed",
        description: "Fixture internal lookup",
        outcome: DiagnosticOutcome::Failure,
        allowed_classes: &[FailureClass::Unavailable],
    }];
    let component = diagnostics
        .component_lazy(|| ComponentDeclaration {
            name: "fixture.internal",
            codes: &codes,
        })
        .map_err(|e| e.to_string())?;
    for delta in [2, 3] {
        component
            .try_record_lazy(|| {
                let mut observation = Observation::failure(
                    1,
                    FailureClass::Unavailable,
                    "native-private-sentinel",
                    NonZeroU64::new(delta).unwrap(),
                );
                observation.correlation = DiagnosticCorrelation {
                    request_id: Some("native-request".into()),
                    session_id: Some(format!("native-session-{delta}").into()),
                };
                observation
            })
            .map_err(|e| e.to_string())?;
    }
    if config.get("expect_read").and_then(Json::as_bool) == Some(true) {
        let reader = diagnostics.reader().map_err(|e| e.to_string())?;
        let snapshot = reader
            .snapshot(&DiagnosticQuery::default())
            .map_err(|e| e.to_string())?;
        if !snapshot.events.iter().any(|event| {
            event
                .detail
                .as_ref()
                .is_some_and(|d| d.message == "native-private-sentinel")
        }) {
            return Err("native diagnostics did not read its selected context".into());
        }
        reader
            .inspect_memory(&DiagnosticQuery::default())
            .map_err(|e| e.to_string())?;
    } else if !matches!(diagnostics.reader(), Err(DiagnosticError::PermissionDenied)) {
        return Err("native diagnostics unexpectedly granted context reads".into());
    }
    if config.get("registration_failure").and_then(Json::as_bool) == Some(true) {
        return Err("fixture registration failure".into());
    }
    ctx.register_tool_request_intercept("diagnostic_lookup", 0, false, move |_name, args| {
        let component = component.clone();
        async move {
            // Diagnostic admission is independent of the existing execution result.
            let _ = component.try_record_lazy(|| {
                Observation::failure(
                    1,
                    FailureClass::Unavailable,
                    "native-private-sentinel",
                    NonZeroU64::MIN,
                )
            });
            Ok(args)
        }
    })
}

fn register_fixture_llm_guardrails(
    ctx: &mut PluginContext<'_>,
    runtime: &PluginRuntime,
) -> nemo_relay_plugin::Result<()> {
    ctx.register_llm_sanitize_request_guardrail(
        "fixture_llm_sanitize_request",
        0,
        |request, context| async move {
            let request = if let Some(codec) = context.resolve_codec() {
                let annotated = codec.decode(&request)?;
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                codec.encode(&annotated, &request)?
            } else {
                request
            };
            Ok(Some(mark_llm_request(
                request,
                "native_plugin_llm_sanitize_request",
            )))
        },
    )?;
    ctx.register_llm_sanitize_response_guardrail(
        "fixture_llm_sanitize_response",
        0,
        |response, context| async move {
            if let Some(codec) = context.resolve_codec() {
                codec.decode(&response)?;
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                codec.decode(&response)?;
            }
            Ok(Some(mark_json(
                response,
                "native_plugin_llm_sanitize_response",
            )))
        },
    )?;
    ctx.register_llm_conditional_execution_guardrail(
        "fixture_llm_conditional",
        0,
        |_request| async move { Ok(None) },
    )?;
    ctx.register_llm_request_intercept(
        "fixture_llm_request_intercept",
        0,
        false,
        |_name, request, annotated| async move {
            Ok(LlmRequestInterceptOutcome::new(
                mark_llm_request(request, "native_plugin_llm_request_intercept"),
                annotated,
            )
            .with_pending_mark(
                PendingMarkSpec::builder()
                    .name("fixture.native.llm_request.mark")
                    .category(EventCategory::custom())
                    .category_profile(CategoryProfile {
                        subtype: Some("fixture.native.pending".into()),
                        ..CategoryProfile::default()
                    })
                    .data(json!({ "source": "native_request_intercept" }))
                    .metadata(json!({ "fixture": true }))
                    .build(),
            ))
        },
    )?;
    ctx.register_llm_execution_intercept("fixture_llm_execution", 0, {
        let runtime = runtime.clone();
        move |_name, request, _context, next| {
            let runtime = runtime.clone();
            async move {
                if request.content["native_http_error"].as_bool() == Some(true) {
                    return Err(nemo_relay_plugin::http_error_message(
                        403,
                        "provider denied access",
                    )?);
                }
                let inspect_snapshot =
                    request.content["inspect_scope_stack"].as_bool() == Some(true);
                let response = next
                    .call(mark_llm_request(
                        request,
                        "native_plugin_llm_execution_request",
                    ))
                    .await?;
                let mut response = mark_json(response, "native_plugin_llm_execution");
                if inspect_snapshot {
                    tokio::task::yield_now().await;
                    response["scope_stack_snapshot"] =
                        serde_json::to_value(runtime.current_scope_stack_snapshot()?)
                            .map_err(|error| error.to_string())?;
                }
                Ok(response)
            }
        }
    })?;
    Ok(())
}
