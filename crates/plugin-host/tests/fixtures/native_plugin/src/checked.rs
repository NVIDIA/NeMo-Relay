// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked ABI lifecycle and authoring conformance fixture.

use abi_stable::{
    export_root_module,
    library::LibHeader,
    prefix_type::PrefixTypeTrait,
    sabi_trait::prelude::TD_Opaque,
    sabi_types::RRef,
    std_types::{ROption, RStr, RVec},
};
use nemo_relay_plugin::abi::async_ffi::FfiFuture;
use nemo_relay_plugin::abi::{
    Bootstrap, Bootstrap_Ref, Json, NATIVE_API_VERSION, PluginDescriptor, v1_0::*,
};
use nemo_relay_plugin::{NativeExecutorConfig, abi_executor::ComponentExecutor};
use std::sync::{Arc, Mutex, OnceLock};

mod author {
    use super::super::*;
    use nemo_relay_plugin::checked::{
        LlmJsonAsyncStream, NativePlugin, PluginContext, PluginRuntime,
    };
    include!("native_author.rs");
}
struct Router;
impl Plugin for Router {
    fn allows_multiple_components(&self) -> bool {
        true
    }
    fn validate(&self, config: Json) -> RVec<ConfigDiagnostic> {
        if matches!(config, Json::Null) {
            return RVec::new();
        }
        nemo_relay_plugin::checked::Adapter(Arc::new(author::FixtureNativePlugin)).validate(config)
    }
    fn register(
        &self,
        config: Json,
        runtime: RuntimeBox,
    ) -> FfiFuture<Result<RegisteredComponent>> {
        if matches!(config, Json::Null) {
            Fixture.register(config, runtime)
        } else if nemo_relay_plugin::abi_value::from_abi(config.clone())
            .ok()
            .is_some_and(|v| {
                v["stream_only_fixture"].as_bool() == Some(true)
                    || v["raw_stream_only_fixture"].as_bool() == Some(true)
            })
        {
            nemo_relay_plugin::checked::Adapter(Arc::new(StreamOnlyAuthor))
                .register(config, runtime)
        } else if nemo_relay_plugin::abi_value::from_abi(config.clone())
            .ok()
            .is_some_and(|v| v["raw_sync_fixture"].as_bool() == Some(true))
        {
            nemo_relay_plugin::checked::Adapter(Arc::new(RawAuthor)).register(config, runtime)
        } else {
            nemo_relay_plugin::checked::Adapter(Arc::new(author::FixtureNativePlugin))
                .register(config, runtime)
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
    fn validate(&self, _: Json) -> RVec<ConfigDiagnostic> {
        RVec::new()
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
                .into_result()
                .unwrap();
            std::future::pending::<()>().await;
            Result::ROk(())
        });
        let component = Owner {
            executor: executor.clone(),
            _pending: Mutex::new(pending),
        };
        executor.execute(async move {
            // This timer must run on the fixture library's Tokio executor.
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            Result::ROk(RegisteredComponent {
                error: ROption::RNone,
                registrations: RVec::new(),
                component: Component_TO::from_value(component, TD_Opaque),
            })
        })
    }
}
extern "C" fn create() -> Result<PluginBox> {
    Result::ROk(Plugin_TO::from_value(Router, TD_Opaque))
}
extern "C" fn descriptor() -> PluginDescriptor {
    PluginDescriptor {
        native_api: NATIVE_API_VERSION,
        plugin_kind: "fixture_native".into(),
    }
}
extern "C" fn query(name: RStr<'_>) -> ROption<RRef<'static, LibHeader>> {
    if name.as_str() != FACTORY_MODULE {
        return ROption::RNone;
    }
    static HEADER: OnceLock<LibHeader> = OnceLock::new();
    ROption::RSome(RRef::new(HEADER.get_or_init(|| {
        LibHeader::from_module(Factory { create }.leak_into_prefix())
    })))
}
#[export_root_module]
pub fn bootstrap() -> Bootstrap_Ref {
    Bootstrap {
        descriptor,
        query_module: query,
    }
    .leak_into_prefix()
}

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
                                .into_result()
                                .map_err(|e| e.message.to_string())?;
                        if !peek {
                            return Ok(stream);
                        }
                        let first = futures::executor::block_on(stream.pull())
                            .into_result()
                            .map_err(|e| e.message.to_string())?;
                        Ok(Stream_TO::from_value(
                            PeekedStream { first, stream },
                            TD_Opaque,
                        ))
                    },
                )?;
                if peek {
                    ctx.register_synchronous_stream_callback(
                        "checked_raw_nested",
                        10,
                        |invocation, next| {
                            futures::executor::block_on(next.open_stream(invocation.value))
                                .into_result()
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
    first: ROption<Json>,
    stream: StreamBox,
}
impl Stream for PeekedStream {
    fn pull(&mut self) -> FfiFuture<Result<ROption<Json>>> {
        use nemo_relay_plugin::abi::async_ffi::FutureExt as _;
        match std::mem::replace(&mut self.first, ROption::RNone) {
            ROption::RSome(value) => async move { Result::ROk(ROption::RSome(value)) }.into_ffi(),
            ROption::RNone => self.stream.pull(),
        }
    }
    fn close(&mut self) -> FfiFuture<Result<()>> {
        self.first = ROption::RNone;
        self.stream.close()
    }
}
