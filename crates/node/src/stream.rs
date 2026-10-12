// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming LLM response support for the Node.js NAPI bindings.
//!
//! Provides the `LlmStream` type, an async iterator that yields response chunks
//! from a streaming LLM call. Chunks are received over a Tokio MPSC channel and
//! exposed to JavaScript via the `next()` method.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use nemo_relay::error::Result as FlowResult;
use serde_json::Value as Json;
use std::sync::Arc;

use crate::api::PersistentJsFunction;

/// An async iterator over chunks from a streaming LLM response.
///
/// Obtained from `llmStreamCallExecute()`. Call `next()` repeatedly to consume
/// response chunks. Returns `null` when the stream is fully consumed.
#[napi]
pub struct LlmStream {
    pub(crate) receiver: tokio::sync::Mutex<tokio::sync::mpsc::Receiver<FlowResult<Json>>>,
    pub(crate) cancel: tokio::sync::watch::Sender<bool>,
    pub(crate) closed: tokio::sync::watch::Receiver<Option<std::result::Result<(), String>>>,
    pub(crate) codec_references: Vec<Arc<PersistentJsFunction>>,
    pub(crate) close_result: tokio::sync::Mutex<Option<std::result::Result<(), String>>>,
}

#[napi]
impl LlmStream {
    /// Retrieve the next chunk from the stream.
    ///
    /// Returns the next JSON chunk, or `null` when the stream is exhausted.
    /// Throws if the underlying stream encountered an error.
    #[napi]
    pub async fn next(&self) -> Result<Option<Json>> {
        let mut guard = self.receiver.lock().await;
        let next_item = guard.recv().await;
        match next_item {
            None => Ok(None),
            Some(Ok(value)) => Ok(Some(value)),
            Some(Err(e)) => Err(napi::Error::from_reason(e.to_string())),
        }
    }

    /// Stop the producer and wait for its cleanup to complete.
    ///
    /// Throws a producer error that `next()` has not yet thrown, then any
    /// cleanup error. Further calls return the same result.
    #[napi]
    pub async fn close(&self) -> Result<()> {
        let mut close_result = self.close_result.lock().await;
        if let Some(result) = close_result.as_ref() {
            return result.clone().map_err(napi::Error::from_reason);
        }
        self.cancel.send_replace(true);
        let result = self
            .closed
            .clone()
            .wait_for(Option::is_some)
            .await
            .map_err(|_| {
                napi::Error::from_reason("stream close task ended before releasing the producer")
            })?
            .clone()
            .expect("close state checked above");
        let mut receiver = self.receiver.lock().await;
        receiver.close();
        let mut undelivered_error = None;
        while let Ok(item) = receiver.try_recv() {
            if let Err(error) = item {
                undelivered_error.get_or_insert(error);
            }
        }
        let result = combine_close_errors(undelivered_error.map(|error| error.to_string()), result);
        *close_result = Some(result.clone());
        result.map_err(napi::Error::from_reason)
    }
}

impl Drop for LlmStream {
    fn drop(&mut self) {
        self.cancel.send_replace(true);
    }
}

fn combine_close_errors(
    stream_error: Option<String>,
    cleanup: std::result::Result<(), String>,
) -> std::result::Result<(), String> {
    match (stream_error, cleanup) {
        (None, cleanup) => cleanup,
        (Some(error), Ok(())) => Err(error),
        (Some(error), Err(cleanup)) => {
            Err(format!("{error}; stream cleanup also failed: {cleanup}"))
        }
    }
}
