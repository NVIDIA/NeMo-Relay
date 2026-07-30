// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay stream adapters for Switchyard provider events.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures_util::Stream;
use nemo_relay::api::runtime::{LlmJsonStream, LlmStreamInner};
use nemo_relay::error::{FlowError, Result as FlowResult, UpstreamFailure, UpstreamFailureClass};
use serde_json::Value as Json;
use switchyard_protocol::{LlmClientError, LlmResponseChunk, LlmResponseStream, WireFormat};
use switchyard_translation::{StreamTranslationState, TranslationEngine};
use tokio::sync::oneshot;

/// Completion handles for provider streams owned by one libsy run.
#[derive(Clone, Default)]
pub(crate) struct StreamCloseTracker {
    receivers: Arc<Mutex<Vec<oneshot::Receiver<FlowResult<()>>>>>,
}

impl StreamCloseTracker {
    fn track(&self, receiver: oneshot::Receiver<FlowResult<()>>) {
        if let Ok(mut receivers) = self.receivers.lock() {
            receivers.push(receiver);
        }
    }

    async fn close_all(&self) -> FlowResult<()> {
        let receivers = self
            .receivers
            .lock()
            .map_err(|error| FlowError::Internal(error.to_string()))?
            .drain(..)
            .collect::<Vec<_>>();
        for receiver in receivers {
            match receiver.await {
                Ok(result) => result?,
                Err(_) => {
                    return Err(FlowError::Internal(
                        "Switchyard provider stream close task was dropped".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Converts one Relay provider stream into libsy's neutral stream contract.
pub(crate) fn provider_response_stream(
    upstream: LlmJsonStream,
    source: WireFormat,
    first: Json,
    tracker: StreamCloseTracker,
    model: String,
) -> std::result::Result<LlmResponseStream, LlmClientError> {
    let translation = TranslationEngine::default();
    let mut state = StreamTranslationState::new(source, source);
    let first = decode_provider_event(&translation, &mut state, source, first)?;
    let (closed_tx, closed_rx) = oneshot::channel();
    tracker.track(closed_rx);
    Ok(Box::pin(RelayProviderStream {
        upstream: Some(upstream),
        first: Some(first),
        source,
        translation,
        state,
        closed_tx: Some(closed_tx),
        model,
    }))
}

struct RelayProviderStream {
    upstream: Option<LlmJsonStream>,
    first: Option<LlmResponseChunk>,
    source: WireFormat,
    translation: TranslationEngine,
    state: StreamTranslationState,
    closed_tx: Option<oneshot::Sender<FlowResult<()>>>,
    model: String,
}

impl RelayProviderStream {
    fn decode(&mut self, raw: Json) -> std::result::Result<LlmResponseChunk, LlmClientError> {
        decode_provider_event(&self.translation, &mut self.state, self.source, raw)
    }
}

fn decode_provider_event(
    translation: &TranslationEngine,
    state: &mut StreamTranslationState,
    source: WireFormat,
    raw: Json,
) -> std::result::Result<LlmResponseChunk, LlmClientError> {
    let chunk = translation
        .decode_stream_event(state, source, &raw)
        .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
    ensure_provider_event_succeeded(&chunk)?;
    Ok(chunk)
}

fn ensure_provider_event_succeeded(
    chunk: &LlmResponseChunk,
) -> std::result::Result<(), LlmClientError> {
    match chunk {
        LlmResponseChunk::ProviderEvent { normalized, .. } => {
            for chunk in normalized {
                ensure_provider_event_succeeded(chunk)?;
            }
            Ok(())
        }
        LlmResponseChunk::DecodeError { message } => {
            Err(LlmClientError::ResponseTranslation(message.clone()))
        }
        LlmResponseChunk::StreamError { message } => Err(LlmClientError::UpstreamHttp {
            status: 502,
            body: message.clone(),
        }),
        _ => Ok(()),
    }
}

impl Stream for RelayProviderStream {
    type Item = std::result::Result<LlmResponseChunk, LlmClientError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        if let Some(first) = this.first.take() {
            return Poll::Ready(Some(Ok(first)));
        }
        let Some(upstream) = this.upstream.as_mut() else {
            return Poll::Ready(None);
        };
        match Pin::new(upstream).poll_next(cx) {
            Poll::Ready(Some(Ok(raw))) => Poll::Ready(Some(this.decode(raw))),
            Poll::Ready(Some(Err(error))) => {
                Poll::Ready(Some(Err(flow_to_client_error(error, &this.model))))
            }
            Poll::Ready(None) => {
                this.upstream.take();
                if let Some(closed_tx) = this.closed_tx.take() {
                    let _ = closed_tx.send(Ok(()));
                }
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for RelayProviderStream {
    fn drop(&mut self) {
        let Some(mut upstream) = self.upstream.take() else {
            return;
        };
        let closed_tx = self.closed_tx.take();
        tokio::spawn(async move {
            let result = upstream.close().await;
            if let Some(closed_tx) = closed_tx {
                let _ = closed_tx.send(result);
            }
        });
    }
}

/// Converts libsy's final response stream back into Relay provider events.
pub(crate) fn relay_response_stream(
    upstream: LlmResponseStream,
    source: WireFormat,
    target: WireFormat,
    tracker: StreamCloseTracker,
) -> LlmJsonStream {
    LlmJsonStream::from_closeable(LibsyResponseStream {
        upstream: Some(upstream),
        source,
        target,
        translation: TranslationEngine::default(),
        state: StreamTranslationState::new(source, target),
        buffered: VecDeque::new(),
        finished: false,
        tracker,
    })
}

struct LibsyResponseStream {
    upstream: Option<LlmResponseStream>,
    source: WireFormat,
    target: WireFormat,
    translation: TranslationEngine,
    state: StreamTranslationState,
    buffered: VecDeque<FlowResult<Json>>,
    finished: bool,
    tracker: StreamCloseTracker,
}

impl Stream for LibsyResponseStream {
    type Item = FlowResult<Json>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        loop {
            if let Some(item) = this.buffered.pop_front() {
                return Poll::Ready(Some(item));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            let Some(upstream) = this.upstream.as_mut() else {
                this.finished = true;
                return Poll::Ready(None);
            };
            match upstream.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(event))) => {
                    match this
                        .translation
                        .encode_stream_event(&mut this.state, this.target, event)
                    {
                        Ok(events) => {
                            this.buffered.extend(events.into_iter().map(Ok));
                        }
                        Err(error) => {
                            this.finished = true;
                            return Poll::Ready(Some(Err(FlowError::InvalidArgument(format!(
                                "Switchyard stream translation failed: {error}"
                            )))));
                        }
                    }
                }
                Poll::Ready(Some(Err(error))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(client_to_flow_error(error))));
                }
                Poll::Ready(None) => {
                    this.upstream.take();
                    this.finished = true;
                    if this.source == this.target {
                        return Poll::Ready(None);
                    }
                    match this.translation.finish_stream(&mut this.state, this.target) {
                        Ok(events) => {
                            this.buffered.extend(events.into_iter().map(Ok));
                        }
                        Err(error) => {
                            return Poll::Ready(Some(Err(FlowError::InvalidArgument(format!(
                                "Switchyard stream finalization failed: {error}"
                            )))));
                        }
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl LlmStreamInner for LibsyResponseStream {
    fn close(self: Pin<&mut Self>) -> Pin<Box<dyn Future<Output = FlowResult<()>> + Send + '_>> {
        let this = self.get_mut();
        this.upstream.take();
        this.buffered.clear();
        this.finished = true;
        Box::pin(async move { this.tracker.close_all().await })
    }
}

pub(crate) fn flow_to_client_error(error: FlowError, model: &str) -> LlmClientError {
    match error {
        FlowError::Upstream(failure) => match failure.class {
            UpstreamFailureClass::Timeout => LlmClientError::Timeout {
                source: Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    failure.body,
                )),
            },
            UpstreamFailureClass::Connection => LlmClientError::Transport {
                source: Box::new(std::io::Error::other(failure.body)),
            },
            UpstreamFailureClass::ContextWindow => LlmClientError::ContextWindowExceeded {
                model: model.into(),
                message: failure.body,
            },
            UpstreamFailureClass::ModelUnavailable => LlmClientError::General(failure.body),
            UpstreamFailureClass::InvalidRequest => LlmClientError::InvalidRequest {
                message: failure.body,
            },
            UpstreamFailureClass::Authentication | UpstreamFailureClass::Other
                if failure.status.is_none() =>
            {
                LlmClientError::General(failure.body)
            }
            _ => LlmClientError::UpstreamHttp {
                status: failure.status.unwrap_or(500),
                body: failure.body,
            },
        },
        FlowError::InvalidArgument(message) => LlmClientError::InvalidRequest { message },
        other => LlmClientError::General(other.to_string()),
    }
}

fn client_to_flow_error(error: LlmClientError) -> FlowError {
    match error {
        LlmClientError::Transport { source } => {
            provider_failure(None, source.to_string(), UpstreamFailureClass::Connection)
        }
        LlmClientError::Timeout { source } => {
            provider_failure(None, source.to_string(), UpstreamFailureClass::Timeout)
        }
        LlmClientError::ContextWindowExceeded { message, .. } => {
            provider_failure(None, message, UpstreamFailureClass::ContextWindow)
        }
        LlmClientError::UpstreamHttp { status, body } => {
            let class = if status == 401 || status == 403 {
                UpstreamFailureClass::Authentication
            } else if matches!(status, 408 | 409 | 425 | 429) || status >= 500 {
                UpstreamFailureClass::RetryableStatus
            } else if (400..500).contains(&status) {
                UpstreamFailureClass::InvalidRequest
            } else {
                UpstreamFailureClass::Other
            };
            provider_failure(Some(status), body, class)
        }
        LlmClientError::InvalidRequest { message }
        | LlmClientError::RequestTranslation(message)
        | LlmClientError::RequestEncoding(message)
        | LlmClientError::ResponseTranslation(message)
        | LlmClientError::Configuration { message } => FlowError::InvalidArgument(message),
        other => FlowError::Internal(format!("Switchyard model call failed: {other}")),
    }
}

fn provider_failure(status: Option<u16>, body: String, class: UpstreamFailureClass) -> FlowError {
    FlowError::Upstream(UpstreamFailure {
        status,
        body,
        headers: Default::default(),
        class,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn preserved_translation_replays_unknown_same_protocol_stream_fields() {
        let raw = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "model": "provider/model",
            "system_fingerprint": "fp_provider_specific",
            "choices": [{
                "index": 0,
                "delta": {"content": "hello"},
                "finish_reason": null
            }]
        });
        let upstream = LlmJsonStream::new(futures_util::stream::iter([Ok(raw.clone())]));
        let tracker = StreamCloseTracker::default();
        let neutral = provider_response_stream(
            upstream,
            WireFormat::OpenAiChat,
            raw.clone(),
            tracker.clone(),
            "target".into(),
        )
        .unwrap();
        let mut replay = relay_response_stream(
            neutral,
            WireFormat::OpenAiChat,
            WireFormat::OpenAiChat,
            tracker,
        );
        let output = replay.next().await.and_then(Result::ok);

        assert_eq!(output, Some(raw));
    }
}
