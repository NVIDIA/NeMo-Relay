// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ptr;

use crate::api::{
    nemo_relay_resource_metrics_collect_free, nemo_relay_resource_metrics_collect_poll,
    nemo_relay_resource_metrics_collect_start, tokio_runtime,
};
use crate::convert::nemo_relay_string_free;
use crate::error::NemoRelayStatus;
use crate::types::FfiResourceMetricsCollection;

#[test]
fn asynchronous_collection_returns_the_active_plugins_snapshot() {
    let _lock = super::TEST_MUTEX
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let config = std::ffi::CString::new(r#"{"components":[{"kind":"resource_metrics","enabled":true,"config":{"gpu":{"enabled":false}}}]}"#).unwrap();
    let mut report = ptr::null_mut();
    unsafe {
        assert_eq!(
            super::activate_test_plugin_config(config.as_ptr(), &mut report),
            NemoRelayStatus::Ok
        );
        nemo_relay_string_free(report);
        let mut collection = ptr::null_mut();
        assert_eq!(
            nemo_relay_resource_metrics_collect_start(&mut collection),
            NemoRelayStatus::Ok
        );
        assert!(!collection.is_null());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut done = false;
        let mut output = ptr::null_mut();
        while !done {
            assert_eq!(
                nemo_relay_resource_metrics_collect_poll(collection, &mut done, &mut output),
                NemoRelayStatus::Ok
            );
            assert!(
                std::time::Instant::now() < deadline,
                "collection did not complete"
            );
            if !done {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        let snapshot: nemo_relay::api::resource_metrics::ResourceMetricsSnapshot =
            serde_json::from_str(std::ffi::CStr::from_ptr(output).to_str().unwrap()).unwrap();
        assert!(snapshot.cpu.is_some());
        assert!(snapshot.gpu.is_none());
        assert_eq!(
            snapshot.measurement_scope,
            nemo_relay::api::resource_metrics::ResourceMeasurementScope::ProcessTree
        );
        nemo_relay_string_free(output);
        nemo_relay_resource_metrics_collect_free(collection);
    }
    assert_eq!(super::close_test_plugin_host(), NemoRelayStatus::Ok);
}

#[test]
fn closed_collection_channel_reports_a_terminal_error() {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let task = tokio_runtime().spawn(std::future::pending::<()>());
    let handle = Box::into_raw(Box::new(FfiResourceMetricsCollection {
        receiver: std::sync::Mutex::new(Some(receiver)),
        task: task.abort_handle(),
    }));
    drop(sender);
    let mut done = false;
    let mut output = ptr::null_mut();
    unsafe {
        assert_eq!(
            nemo_relay_resource_metrics_collect_poll(handle, &mut done, &mut output),
            NemoRelayStatus::Internal
        );
        assert!(done);
        assert!(output.is_null());
        nemo_relay_resource_metrics_collect_free(handle);
    }
}

#[test]
fn collection_handle_reports_pending_then_a_single_completed_result() {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let task = tokio_runtime().spawn(std::future::pending::<()>());
    let handle = Box::into_raw(Box::new(FfiResourceMetricsCollection {
        receiver: std::sync::Mutex::new(Some(receiver)),
        task: task.abort_handle(),
    }));
    let mut done = true;
    let mut json = ptr::null_mut();
    unsafe {
        assert_eq!(
            nemo_relay_resource_metrics_collect_poll(handle, &mut done, &mut json),
            NemoRelayStatus::Ok
        );
        assert!(!done);
        assert!(json.is_null());
        sender.send(Ok("{}".to_owned())).unwrap();
        assert_eq!(
            nemo_relay_resource_metrics_collect_poll(handle, &mut done, &mut json),
            NemoRelayStatus::Ok
        );
        assert!(done);
        assert_eq!(std::ffi::CStr::from_ptr(json).to_str().unwrap(), "{}");
        nemo_relay_string_free(json);
        assert_eq!(
            nemo_relay_resource_metrics_collect_poll(handle, &mut done, &mut json),
            NemoRelayStatus::InvalidArg
        );
        assert!(json.is_null());
        nemo_relay_resource_metrics_collect_free(handle);
    }
    assert!(tokio_runtime().block_on(task).unwrap_err().is_cancelled());
}

#[test]
fn collection_handle_delivers_errors_and_rejects_null_pointers() {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let task = tokio_runtime().spawn(std::future::pending::<()>());
    let handle = Box::into_raw(Box::new(FfiResourceMetricsCollection {
        receiver: std::sync::Mutex::new(Some(receiver)),
        task: task.abort_handle(),
    }));
    sender
        .send(Err(nemo_relay::error::FlowError::InvalidArgument(
            "inactive resource metrics".into(),
        )))
        .unwrap();
    let mut done = false;
    let mut json = ptr::null_mut();
    unsafe {
        assert_eq!(
            nemo_relay_resource_metrics_collect_start(ptr::null_mut()),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_resource_metrics_collect_poll(ptr::null_mut(), &mut done, &mut json),
            NemoRelayStatus::NullPointer
        );
        assert_eq!(
            nemo_relay_resource_metrics_collect_poll(handle, &mut done, &mut json),
            NemoRelayStatus::InvalidArg
        );
        assert!(done);
        assert!(json.is_null());
        nemo_relay_resource_metrics_collect_free(handle);
        nemo_relay_resource_metrics_collect_free(ptr::null_mut());
    }
}
