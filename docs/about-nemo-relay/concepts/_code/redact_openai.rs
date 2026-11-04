// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nemo_relay::middleware::{BuiltinLlmCodec, LlmCodecIdentity};
use nemo_relay::registry::register_llm_sanitize_request_guardrail;
use std::sync::Arc;

fn register_redaction() -> nemo_relay::Result<()> {
    register_llm_sanitize_request_guardrail(
        "redact-openai-chat",
        10,
        Arc::new(|mut request, context| {
            Box::pin(async move {
                if context.codec() == &LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
                    && let Some(codec) = context.resolve_codec()
                    && let Ok(mut annotated) = codec.decode(&request)
                {
                    annotated.messages.clear();
                    if let Ok(encoded) = codec.encode(&annotated, &request) {
                        request = encoded;
                    }
                }
                Ok(Some(request))
            })
        }),
    )?;
    Ok(())
}
