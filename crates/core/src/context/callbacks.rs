// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;

use crate::codec::request::AnnotatedLLMRequest;
use crate::error::Result;
use crate::json::Json;
use crate::types::event::Event;
use crate::types::llm::LLMRequest;

pub type ToolSanitizeFn = Box<dyn Fn(&str, Json) -> Json + Send + Sync>;
pub type ToolConditionalFn = Box<dyn Fn(&str, &Json) -> Result<Option<String>> + Send + Sync>;
pub type ToolInterceptFn = Box<dyn Fn(&str, Json) -> Result<Json> + Send + Sync>;
pub type ToolExecutionNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
pub type ToolExecutionFn = Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;

pub type LlmSanitizeRequestFn = Box<dyn Fn(LLMRequest) -> LLMRequest + Send + Sync>;
pub type LlmSanitizeResponseFn = Box<dyn Fn(Json) -> Json + Send + Sync>;
pub type LlmConditionalFn = Box<dyn Fn(&LLMRequest) -> Result<Option<String>> + Send + Sync>;
pub type LlmRequestInterceptFn = Box<
    dyn Fn(
            &str,
            LLMRequest,
            Option<AnnotatedLLMRequest>,
        ) -> Result<(LLMRequest, Option<AnnotatedLLMRequest>)>
        + Send
        + Sync,
>;
pub type LlmExecutionNextFn =
    Arc<dyn Fn(LLMRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
pub type LlmExecutionFn = Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;
pub type LlmStreamExecutionNextFn = Arc<
    dyn Fn(
            LLMRequest,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
>;
pub type LlmStreamExecutionFn = Arc<
    dyn Fn(
            &str,
            LLMRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<Pin<Box<dyn Stream<Item = Result<Json>> + Send>>>>
                    + Send,
            >,
        > + Send
        + Sync,
>;

pub type EventSubscriberFn = Arc<dyn Fn(&Event) + Send + Sync>;
