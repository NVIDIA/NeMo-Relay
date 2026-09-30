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
