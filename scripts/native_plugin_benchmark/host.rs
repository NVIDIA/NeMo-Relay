// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Measure real production loading, request middleware, and tool execution.
mod allocation;
use nemo_relay_plugin_host::dynamic::{NativePluginLoadSpec, load_native_plugins};
use nemo_relay_plugin_host::{
    HostPluginRegistrationContext, lookup_plugin, rollback_registrations,
};
use nemo_relay_runtime::api::{
    subscriber::flush_subscribers,
    tool::{
        ToolCallExecuteParams, ToolExecutionResult, tool_call_execute, tool_request_intercepts,
    },
};
use serde_json::{Map, Value as Json, json};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};
static PROVIDER_CALLS: AtomicU64 = AtomicU64::new(0);

async fn invoke(path: &str, payload: &Json) {
    let result = if path == "request" {
        tool_request_intercepts("bench", payload.clone())
            .await
            .unwrap()
    } else {
        tool_call_execute(
            ToolCallExecuteParams::builder()
                .name("bench")
                .args(payload.clone())
                .func(Arc::new(|args| {
                    Box::pin(async move {
                        PROVIDER_CALLS.fetch_add(1, Ordering::Relaxed);
                        Ok(ToolExecutionResult::new(args))
                    })
                }))
                .build(),
        )
        .await
        .unwrap()
        .result
    };
    assert_eq!(
        &result, payload,
        "middleware must preserve the complete payload"
    );
}
async fn allocations(path: &str, payload: &Json, iterations: usize) -> [u64; 4] {
    tool_request_intercepts("bench", json!({"__bench_control":"start"}))
        .await
        .unwrap();
    PROVIDER_CALLS.store(0, Ordering::Relaxed);
    allocation::start();
    for _ in 0..iterations {
        invoke(path, payload).await;
    }
    flush_subscribers().unwrap();
    let (calls, bytes) = allocation::stop();
    let plugin = tool_request_intercepts("bench", json!({"__bench_control":"stop"}))
        .await
        .unwrap();
    let executions = if path == "execute" {
        iterations as u64
    } else {
        0
    };
    assert_eq!(plugin["executions"].as_u64().unwrap(), executions);
    let fanout = if payload["mode"] == "fanout" { 2 } else { 1 };
    assert_eq!(PROVIDER_CALLS.load(Ordering::Relaxed), executions * fanout);
    [
        calls,
        bytes,
        plugin["calls"].as_u64().unwrap(),
        plugin["bytes"].as_u64().unwrap(),
    ]
}
fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    let iterations: usize = arguments[2].parse().unwrap();
    let rounds: usize = arguments[3].parse().unwrap();
    assert!(iterations > 0 && rounds > 0);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let activation = load_native_plugins([NativePluginLoadSpec { plugin_id: "bench.native".into(), manifest_ref: arguments[1].clone() }]).unwrap();
        let plugin = lookup_plugin("bench.native").unwrap();
        let mut context = HostPluginRegistrationContext::new();
        plugin.register(&Map::new(), &mut context).await.unwrap();
        let mut registrations = context.into_registrations();
        let mut results = Vec::new();
        for path in ["request", "execute"] {
            for mode in ["ready", "pending", "wakers", "fanout"] {
                for shape in ["tiny", "objects", "large"] {
                    let data = match shape {
                        "tiny" => Json::Null,
                        "objects" => Json::Object((0..128).map(|index| (format!("field_{index}"), json!({"index":index,"text":"Unicode \u{03bb}\0","values":[-1,18446744073709551615_u64,0.5]}))).collect()),
                        _ => json!("x".repeat(256 * 1024)),
                    };
                    let payload = json!({"mode":mode,"data":data});
                    for _ in 0..50 { invoke(path, &payload).await; }
                    flush_subscribers().unwrap();
                    let mut samples = Vec::new();
                    for _ in 0..rounds {
                        let started = Instant::now();
                        for _ in 0..iterations { invoke(path, &payload).await; }
                        flush_subscribers().unwrap();
                        samples.push(started.elapsed().as_nanos() as f64 / iterations as f64);
                    }
                    samples.sort_by(f64::total_cmp);
                    let empty = allocations(path, &payload, 0).await;
                    let counted = allocations(path, &payload, iterations).await;
                    let counts: Vec<f64> = counted.into_iter().zip(empty).map(|(total, overhead)| total.saturating_sub(overhead) as f64 / iterations as f64).collect();
                    results.push(json!({"path":path,"mode":mode,"payload":shape,
                        "payload_bytes":serde_json::to_vec(&payload).unwrap().len(),
                        "median_ns":samples[rounds/2],"min_ns":samples[0],"max_ns":samples[rounds-1],
                        "samples_ns":samples,"host_allocations":counts[0],"host_allocated_bytes":counts[1],
                        "plugin_allocations":counts[2],"plugin_allocated_bytes":counts[3]}));
                }
            }
        }
        let outcome = rollback_registrations(&mut registrations);
        assert!(outcome.callbacks_cleared(), "benchmark teardown must succeed");
        drop(plugin);
        drop(activation);
        println!("NATIVE_BENCH_RESULT {}", json!({"iterations":iterations,"rounds":rounds,"results":results}));
    });
}
