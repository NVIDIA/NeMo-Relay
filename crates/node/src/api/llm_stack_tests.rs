// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use nemo_relay::api::llm::{
    LlmCallExecuteParams, LlmRequest, LlmRequestInterceptOutcome, LlmStreamCallExecuteParams,
    llm_call_execute, llm_request_intercepts, llm_stream_call_execute,
};
use nemo_relay::api::registry::{deregister_llm_request_intercept, register_llm_request_intercept};
use nemo_relay::api::runtime::LlmJsonStream;
use serde_json::json;
use tokio_stream::StreamExt;

/// Poll the Node binding's core LLM futures on a bounded native worker stack.
#[test]
fn llm_futures_complete_on_a_constrained_native_polling_stack() {
    // JavaScript Worker limits cannot constrain napi-rs's Tokio threads.
    // Bound the actual polling worker to 1 MiB. Spawning the task keeps core
    // future construction and polling off this test's caller thread.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .thread_stack_size(1024 * 1024)
        .enable_all()
        .build()
        .expect("create constrained native runtime");

    runtime.block_on(async {
        tokio::spawn(async {
            register_llm_request_intercept(
                "node-native-stack-request",
                1,
                false,
                Arc::new(|_, mut request, annotated| {
                    Box::pin(async move {
                        tokio::task::yield_now().await;
                        request.headers.insert("x-intercepted".into(), json!(true));
                        Ok(LlmRequestInterceptOutcome::new(request, annotated))
                    })
                }),
            )
            .expect("register native request middleware");
            let request = LlmRequest {
                headers: Default::default(),
                content: json!({"messages": [], "model": "test-model"}),
            };
            let intercepted = Box::pin(llm_request_intercepts("native-stack", request.clone()))
                .await
                .expect("intercept request on native worker");
            assert_eq!(intercepted.request.content, request.content);
            assert_eq!(intercepted.request.headers["x-intercepted"], json!(true));

            let result = Box::pin(llm_call_execute(
                LlmCallExecuteParams::builder()
                    .name("native-stack-execute")
                    .request(request.clone())
                    .func(Arc::new(|request| {
                        assert_eq!(request.headers["x-intercepted"], json!(true));
                        Box::pin(async {
                            tokio::task::yield_now().await;
                            Ok(json!({"ok": true}))
                        })
                    }))
                    .build(),
            ))
            .await
            .expect("execute on native worker");
            assert_eq!(result, json!({"ok": true}));

            let mut stream = Box::pin(llm_stream_call_execute(
                LlmStreamCallExecuteParams::builder()
                    .name("native-stack-stream")
                    .request(request)
                    .func(Arc::new(|request| {
                        assert_eq!(request.headers["x-intercepted"], json!(true));
                        Box::pin(async {
                            tokio::task::yield_now().await;
                            Ok(LlmJsonStream::new(tokio_stream::iter([
                                Ok(json!({"token": "hello"})),
                                Ok(json!({"token": "world"})),
                            ])))
                        })
                    }))
                    .collector(Box::new(|_| Ok(())))
                    .finalizer(Box::new(|| json!({"complete": true})))
                    .build(),
            ))
            .await
            .expect("start stream on native worker");
            assert_eq!(
                stream.next().await.unwrap().unwrap(),
                json!({"token": "hello"})
            );
            assert_eq!(
                stream.next().await.unwrap().unwrap(),
                json!({"token": "world"})
            );
            assert!(stream.next().await.is_none());
            deregister_llm_request_intercept("node-native-stack-request")
                .expect("remove native request middleware");
        })
        .await
        .expect("native polling task completes");
    });
}
