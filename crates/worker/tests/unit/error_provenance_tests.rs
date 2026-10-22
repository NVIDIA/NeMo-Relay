// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use prost::Message;

fn downstream_error(message: &str, id: Option<&str>) -> WorkerError {
    WorkerError {
        code: "host.runtime_error".into(),
        message: message.into(),
        downstream_continuation_id: id.map(str::to_owned),
        ..Default::default()
    }
}

#[test]
fn downstream_provenance_roundtrips_without_changing_legacy_errors() {
    for message in ["provider failed", "HTTP 503: provider unavailable"] {
        let legacy = sdk_error_to_worker(worker_error_to_sdk(downstream_error(message, None)));
        let wire = downstream_error(message, Some("next-owned"));
        // Verify the additive field survives the actual protobuf encoding.
        let wire = WorkerError::decode(wire.encode_to_vec().as_slice()).unwrap();
        for error in [
            json_result_to_sdk(JsonResult {
                error: Some(wire.clone()),
                ..Default::default()
            })
            .unwrap_err(),
            tool_execution_result_to_sdk(ToolExecutionResultResponse {
                error: Some(wire.clone()),
                ..Default::default()
            })
            .unwrap_err(),
            stream_chunk_to_json(StreamChunk {
                item: Some(nemo_relay_worker_proto::v1::stream_chunk::Item::Error(
                    wire.clone(),
                )),
            })
            .unwrap_err(),
        ] {
            let mut forwarded = sdk_error_to_worker(error);
            assert_eq!(
                forwarded.downstream_continuation_id.take().as_deref(),
                Some("next-owned")
            );
            assert_eq!(forwarded, legacy);
        }
    }
}

#[test]
fn provenance_stays_on_each_error_and_does_not_mark_new_failures() {
    let first = worker_error_to_sdk(downstream_error("first", Some("next-first")));
    let second = worker_error_to_sdk(downstream_error("second", Some("next-second")));
    // Another completed/retried call cannot replace either error's origin.
    assert!(
        json_result_to_sdk(JsonResult {
            value: Some(infallible_json_envelope(JSON_SCHEMA, &Json::Null)),
            error: None
        })
        .is_ok()
    );
    let local = WorkerSdkError::Callback(first.to_string());
    assert_eq!(
        sdk_error_to_worker(second)
            .downstream_continuation_id
            .as_deref(),
        Some("next-second")
    );
    assert_eq!(
        sdk_error_to_worker(first)
            .downstream_continuation_id
            .as_deref(),
        Some("next-first")
    );
    assert!(
        sdk_error_to_worker(local)
            .downstream_continuation_id
            .is_none()
    );
    assert!(
        sdk_error_to_worker(worker_error_to_sdk(downstream_error("legacy", Some(""))))
            .downstream_continuation_id
            .is_none()
    );
}

#[test]
fn stream_setup_status_preserves_provenance_and_existing_display() {
    let mut status = Status::internal("provider stream failed");
    let legacy_display = WorkerSdkError::Transport(status.to_string()).to_string();
    status.metadata_mut().insert(
        DOWNSTREAM_CONTINUATION_METADATA,
        "next-owned".parse().unwrap(),
    );
    let error = continuation_status_to_sdk(status);
    assert_eq!(error.to_string(), legacy_display);
    let returned = status_from_sdk(error);
    assert_eq!(returned.code(), tonic::Code::Internal);
    assert_eq!(returned.message(), legacy_display);
    assert_eq!(
        returned
            .metadata()
            .get(DOWNSTREAM_CONTINUATION_METADATA)
            .unwrap(),
        "next-owned"
    );
    // An independent transport error has no downstream marker.
    let local = continuation_status_to_sdk(Status::unavailable("worker transport"));
    assert!(
        sdk_error_to_worker(local)
            .downstream_continuation_id
            .is_none()
    );
}
