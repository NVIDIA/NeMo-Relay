// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn test_rejection_message_prefers_string_then_object_then_unknown() {
    assert_eq!(rejection_message(Ok("boom".into()), None), "boom");
    assert_eq!(
        rejection_message(
            Err(napi::Error::from_reason("x")),
            Some(Ok("from object".into()))
        ),
        "from object"
    );
    assert_eq!(
        rejection_message(
            Err(napi::Error::from_reason("x")),
            Some(Err(napi::Error::from_reason("y")))
        ),
        "unknown error"
    );
    assert_eq!(
        rejection_message(Err(napi::Error::from_reason("x")), None),
        "unknown error"
    );
}

#[test]
fn test_queue_status_and_closed_helpers_cover_error_paths() {
    assert!(queue_status_result(napi::Status::Ok).is_ok());
    assert!(
        queue_status_result(napi::Status::GenericFailure)
            .unwrap_err()
            .to_string()
            .contains("failed to queue threadsafe function call")
    );
    assert!(
        closed_tsfn_error()
            .to_string()
            .contains("PromiseAwareFn threadsafe function closed")
    );
}

#[tokio::test]
async fn test_call_completion_and_closed_promise_aware_fn_paths() {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let completion = CallCompletion::new(tx);
    completion.send(Ok(serde_json::json!({"ok": true})));
    assert_eq!(rx.await.unwrap().unwrap(), serde_json::json!({"ok": true}));

    let (tx, rx) = tokio::sync::oneshot::channel();
    let completion = CallCompletion::new(tx);
    completion.send(Err(FlowError::Internal("boom".into())));
    assert!(rx.await.unwrap().unwrap_err().to_string().contains("boom"));

    let promise_aware = PromiseAwareFn {
        tsfn: std::sync::Mutex::new(None),
    };
    assert!(
        promise_aware
            .call(serde_json::json!({"value": 1}))
            .await
            .unwrap_err()
            .to_string()
            .contains("threadsafe function closed")
    );

    let next_json: JsonNextFn = Arc::new(|value| Box::pin(async move { Ok(value) }));
    assert!(
        promise_aware
            .call_with_json_next(serde_json::json!({"value": 1}), next_json)
            .await
            .unwrap_err()
            .to_string()
            .contains("threadsafe function closed")
    );

    let next_stream: JsonStreamNextFn = Arc::new(|value| Box::pin(async move { Ok(vec![value]) }));
    assert!(
        promise_aware
            .call_with_stream_next(serde_json::json!({"value": 1}), next_stream)
            .await
            .unwrap_err()
            .to_string()
            .contains("threadsafe function closed")
    );

    promise_aware.close();
}
