// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::llm::{
    LlmAttributes, LlmRequest, LlmStreamCallExecuteParams, llm_stream_call_execute,
};
use nemo_relay::middleware::LlmJsonStream;
use serde_json::json;
use tokio_stream::StreamExt;

/// Read one stream chunk and close the producer.
pub async fn run_example() -> Result<Option<serde_json::Value>, Box<dyn std::error::Error>> {
    let request = LlmRequest {
        headers: Default::default(),
        content: json!({"messages": [{"role": "user", "content": "hello"}]}),
    };

    let mut stream = llm_stream_call_execute(
        LlmStreamCallExecuteParams::builder()
            .name("demo-provider")
            .request(request)
            .func(std::sync::Arc::new(|_req| {
                Box::pin(async move {
                    Ok(LlmJsonStream::new(tokio_stream::iter(vec![Ok(
                        json!({"delta": "hi"}),
                    )])))
                })
            }))
            .collector(Box::new(|_chunk| Ok(())))
            .finalizer(Box::new(|| json!({"text": "hi"})))
            .attributes(LlmAttributes::STREAMING)
            .model_name("demo-model")
            .build(),
    )
    .await?;

    let first_chunk = stream.next().await;
    // Stop early after deciding no more output is needed.
    stream.close().await?;
    Ok(first_chunk.transpose()?)
}
