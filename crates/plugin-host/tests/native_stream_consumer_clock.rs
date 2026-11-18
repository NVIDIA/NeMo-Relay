// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native plugins read caller-poll timing for the stream call they intercept.

use nemo_relay_plugin::PluginRuntime;
use nemo_relay_plugin_host::dynamic::native_checked::HostRuntime;
use nemo_relay_runtime::api::runtime::stream_consumer_clock::{
    StreamConsumerClock, with_stream_consumer_clock,
};

#[tokio::test]
async fn sdk_captures_and_reports_the_caller_clock() {
    let owner = HostRuntime::new("consumer-clock".into()).unwrap();
    let runtime = PluginRuntime::from_abi(owner.runtime_box());

    // The bridge allocates clocks only while a recorder holds the guard.
    assert!(StreamConsumerClock::if_enabled().is_none());
    let guard = runtime.enable_stream_consumer_timing().unwrap();
    let clock = StreamConsumerClock::if_enabled().expect("timing is enabled");

    // Outside a stream call there is nothing to capture.
    assert!(runtime.capture_stream_consumer_clock().unwrap().is_none());
    let captured = with_stream_consumer_clock(Some(clock.clone()), async {
        runtime.capture_stream_consumer_clock()
    })
    .await
    .unwrap()
    .expect("the stream call's clock is captured");

    // Chunk 0 was requested before it was available; chunk 1 was not requested yet.
    clock.requested();
    assert!(!captured.was_prefetched(0).unwrap());
    clock.delivered();
    assert!(captured.was_prefetched(1).unwrap());

    let captured = std::sync::Arc::new(captured);
    let waiter = std::sync::Arc::clone(&captured);
    let finished = tokio::spawn(async move { waiter.finished().await });
    clock.finish(true);
    let report = finished.await.unwrap().unwrap();
    assert!(report.complete);
    assert_eq!(report.delivered.len(), 1);
    assert!(report.delivered[0] <= report.ended);
    // The handle stays usable after the report until the plugin drops it.
    assert!(captured.was_prefetched(1).unwrap());
    drop(captured);

    drop(guard);
    assert!(StreamConsumerClock::if_enabled().is_none());
    owner.shutdown().await.unwrap();
}
