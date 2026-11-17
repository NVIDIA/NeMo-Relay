// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked ABI lifecycle and authoring conformance fixture.

use nemo_relay_plugin::abi::future::{FfiFuture, FutureExt as _};
use nemo_relay_plugin::abi::{Json, v1_0::*, v1_1 as b};
use nemo_relay_plugin::{NativeExecutorConfig, abi_executor::ComponentExecutor};
use std::sync::{Arc, Mutex};

mod author {
    use super::super::*;
    use nemo_relay_plugin::checked::{
        LlmJsonAsyncStream, NativePlugin, PluginContext, PluginRuntime,
    };
    include!("native_author.rs");
}
struct Router;
impl b::Plugin for Router {
    fn allows_multiple_components(&self) -> bool {
        true
    }
    fn validate(&self, config: Json) -> Vec<ConfigDiagnostic> {
        if matches!(config, Json::Null) {
            return Vec::new();
        }
        b::Plugin::validate(
            &nemo_relay_plugin::checked::Adapter(Arc::new(author::FixtureNativePlugin)),
            config,
        )
    }
    fn register(
        &self,
        config: Json,
        runtime: b::RuntimeBox,
    ) -> FfiFuture<Result<RegisteredComponent>> {
        let options = nemo_relay_plugin::abi_value::from_abi(config.clone()).ok();
        let legacy = matches!(config, Json::Null)
            || options.as_ref().is_some_and(|value| {
                [
                    "stream_only_fixture",
                    "raw_stream_only_fixture",
                    "raw_sync_fixture",
                    "legacy_sdk_fixture",
                ]
                .iter()
                .any(|key| value[key].as_bool() == Some(true))
            });
        if !legacy {
            return b::Plugin::register(
                &nemo_relay_plugin::checked::Adapter(Arc::new(author::FixtureNativePlugin)),
                config,
                runtime,
            );
        }
        let runtime = match runtime.baseline() {
            Ok(runtime) => runtime,
            Err(error) => return async move { Err(error) }.into_ffi(),
        };
        if matches!(config, Json::Null) {
            Fixture.register(config, runtime)
        } else if options.as_ref().is_some_and(|value| {
            value["stream_only_fixture"].as_bool() == Some(true)
                || value["raw_stream_only_fixture"].as_bool() == Some(true)
        }) {
            Plugin::register(
                &nemo_relay_plugin::checked::Adapter(Arc::new(StreamOnlyAuthor)),
                config,
                runtime,
            )
        } else if options
            .as_ref()
            .is_some_and(|value| value["raw_sync_fixture"].as_bool() == Some(true))
        {
            Plugin::register(
                &nemo_relay_plugin::checked::Adapter(Arc::new(RawAuthor)),
                config,
                runtime,
            )
        } else {
            Plugin::register(
                &nemo_relay_plugin::checked::Adapter(Arc::new(author::FixtureNativePlugin)),
                config,
                runtime,
            )
        }
    }
}
struct Fixture;
struct Owner {
    executor: ComponentExecutor,
    // Keep the receiver alive so component cancellation, rather than receiver
    // drop, is responsible for stopping the deliberately unfinished task.
    _pending: Mutex<FfiFuture<Result<()>>>,
}
impl Component for Owner {
    fn shutdown(&self) -> FfiFuture<Result<()>> {
        self.executor.shutdown()
    }
}
struct TaskState(Arc<RuntimeBox>);
impl Drop for TaskState {
    fn drop(&mut self) {
        let _ = self.0.call_sync("fixture.task_dropped".into(), Json::Null);
    }
}
impl Plugin for Fixture {
    fn allows_multiple_components(&self) -> bool {
        true
    }
    fn validate(&self, _: Json) -> Vec<ConfigDiagnostic> {
        Vec::new()
    }
    fn register(&self, _: Json, runtime: RuntimeBox) -> FfiFuture<Result<RegisteredComponent>> {
        let executor = ComponentExecutor::new(
            NativeExecutorConfig { worker_threads: 1 },
            "checked-fixture",
        )
        .unwrap();
        let runtime = Arc::new(runtime);
        let task_runtime = runtime.clone();
        let task_state = TaskState(task_runtime.clone());
        let pending = executor.execute(async move {
            let _state = task_state;
            task_runtime
                .call_sync("fixture.task_started".into(), Json::Null)
                .unwrap();
            std::future::pending::<()>().await;
            Result::Ok(())
        });
        let component = Owner {
            executor: executor.clone(),
            _pending: Mutex::new(pending),
        };
        executor.execute(async move {
            // This timer must run on the fixture library's Tokio executor.
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            Result::Ok(RegisteredComponent {
                error: Option::None,
                registrations: Vec::new(),
                component: ComponentBox::new(component),
            })
        })
    }
}
nemo_relay_plugin::abi::export_factory_v1_1!(
    kind = "fixture_native",
    create = || Ok(b::PluginBox::new(Router))
);

struct RawAuthor;
impl nemo_relay_plugin::checked::Plugin for RawAuthor {
    fn plugin_kind(&self) -> &str {
        "fixture_native"
    }
    fn register<'a>(
        &'a self,
        _: &'a serde_json::Map<String, nemo_relay_plugin::Json>,
        ctx: &'a mut nemo_relay_plugin::checked::PluginContext<'_>,
    ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
        Box::pin(async move {
            ctx.register_synchronous_callback(
                "checked_raw_rewrite",
                Surface::ToolRequestIntercept,
                0,
                false,
                |invocation, _| {
                    let mut value = nemo_relay_plugin::abi_value::from_abi(invocation.value)?;
                    value["checked_raw_sync"] = serde_json::json!(true);
                    Ok(nemo_relay_plugin::abi_value::into_abi(value))
                },
            )?;
            let runtime = ctx.runtime();
            ctx.register_synchronous_callback(
                "checked_raw_subscriber",
                Surface::Subscriber,
                0,
                false,
                move |invocation, _| {
                    let value = nemo_relay_plugin::abi_value::from_abi(invocation.value)?;
                    let event: nemo_relay_plugin::Event =
                        serde_json::from_value(value).map_err(|e| e.to_string())?;
                    if event.name() == "checked-raw-sync-trigger" {
                        runtime.emit_mark("checked-raw-sync-observed", None, None)?;
                    }
                    if event.name() == "checked-raw-panic" {
                        panic!("raw fixture callback panic");
                    }
                    Ok(Json::Null)
                },
            )?;
            Ok(())
        })
    }
}

struct StreamOnlyAuthor;
impl nemo_relay_plugin::checked::Plugin for StreamOnlyAuthor {
    fn plugin_kind(&self) -> &str {
        "fixture_native"
    }
    fn register<'a>(
        &'a self,
        config: &'a serde_json::Map<String, nemo_relay_plugin::Json>,
        ctx: &'a mut nemo_relay_plugin::checked::PluginContext<'_>,
    ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
        Box::pin(async move {
            if config
                .get("raw_stream_only_fixture")
                .and_then(|v| v.as_bool())
                == Some(true)
            {
                let peek = config
                    .get("raw_stream_peek_fixture")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                ctx.register_synchronous_stream_callback(
                    "checked_raw_stream",
                    0,
                    move |invocation, next| {
                        if invocation.name.as_str() == "raw-stream-panic" {
                            panic!("raw stream factory panic");
                        }
                        let mut stream =
                            futures::executor::block_on(next.open_stream(invocation.value))
                                .map_err(|e| e.message.to_string())?;
                        if !peek {
                            return Ok(stream);
                        }
                        let first = futures::executor::block_on(stream.pull())
                            .map_err(|e| e.message.to_string())?;
                        Ok(StreamBox::new(PeekedStream { first, stream }))
                    },
                )?;
                if peek {
                    ctx.register_synchronous_stream_callback(
                        "checked_raw_nested",
                        10,
                        |invocation, next| {
                            futures::executor::block_on(next.open_stream(invocation.value))
                                .map_err(|e| e.message.to_string())
                        },
                    )?;
                }
                return Ok(());
            }
            ctx.register_llm_stream_execution_intercept(
                "checked_stream_only",
                0,
                |_, request, _, next| async move { next.call(request).await },
            )?;
            Ok(())
        })
    }
}

struct PeekedStream {
    first: Option<Json>,
    stream: StreamBox,
}
impl Stream for PeekedStream {
    fn pull(&mut self) -> FfiFuture<Result<Option<Json>>> {
        use nemo_relay_plugin::abi::future::FutureExt as _;
        match self.first.take() {
            Option::Some(value) => async move { Result::Ok(Option::Some(value)) }.into_ffi(),
            Option::None => self.stream.pull(),
        }
    }
    fn close(&mut self) -> FfiFuture<Result<()>> {
        self.first = Option::None;
        self.stream.close()
    }
}
