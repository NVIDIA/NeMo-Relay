// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Identical middleware behavior behind each SDK's authoring adapter.
mod allocation;
use nemo_relay_plugin::{Json, NativePlugin, PluginContext, ToolExecutionInterceptOutcome};
use serde_json::{Map, json};
use std::{
    sync::atomic::{AtomicU64, Ordering},
    task::Poll,
};
static EXECUTIONS: AtomicU64 = AtomicU64::new(0);

struct Benchmark;
async fn scheduling(value: &Json) {
    match value["mode"].as_str().unwrap_or("ready") {
        "pending" => tokio::task::yield_now().await,
        "wakers" => {
            let mut pending = true;
            std::future::poll_fn(|context| {
                if pending {
                    pending = false;
                    for _ in 0..16 {
                        drop(context.waker().clone());
                    }
                    context.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            })
            .await;
        }
        _ => {}
    }
}
fn register(context: &mut PluginContext<'_>) -> nemo_relay_plugin::Result<()> {
    context.register_tool_request_intercept("request", 0, false, |_, value| async move {
        match value["__bench_control"].as_str() {
            Some("start") => {
                EXECUTIONS.store(0, Ordering::Relaxed);
                allocation::start();
                return Ok(Json::Null);
            }
            Some("stop") => {
                let (calls, bytes) = allocation::stop();
                return Ok(json!({"calls":calls,"bytes":bytes,"executions":EXECUTIONS.load(Ordering::Relaxed)}));
            }
            _ => {}
        }
        scheduling(&value).await;
        Ok(value)
    })?;
    context.register_tool_execution_intercept("execute", 0, |context, next| async move {
        EXECUTIONS.fetch_add(1, Ordering::Relaxed);
        scheduling(&context.args).await;
        let result = if context.args["mode"] == "fanout" {
            let (first, second) =
                futures::try_join!(next.call(context.args.clone()), next.call(context.args))?;
            assert_eq!(first.result, second.result);
            first
        } else {
            next.call(context.args).await?
        };
        Ok(ToolExecutionInterceptOutcome::from(result))
    })
}
impl NativePlugin for Benchmark {
    fn plugin_kind(&self) -> &str {
        "bench.native"
    }
    #[cfg(feature = "checked")]
    fn register<'a>(
        &'a self,
        _: &'a Map<String, Json>,
        context: &'a mut PluginContext<'_>,
    ) -> nemo_relay_plugin::PluginFuture<'a, ()> {
        Box::pin(async move { register(context).map_err(Into::into) })
    }
    #[cfg(not(feature = "checked"))]
    fn register(
        &mut self,
        _: &Map<String, Json>,
        context: &mut PluginContext<'_>,
    ) -> nemo_relay_plugin::Result<()> {
        register(context)
    }
}
#[cfg(feature = "checked")]
nemo_relay_plugin::nemo_relay_plugin!(kind = "bench.native", || Benchmark);
#[cfg(not(feature = "checked"))]
nemo_relay_plugin::nemo_relay_plugin!(nemo_relay_register_plugin, || Benchmark);
