// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned, bounded control messages. Provider bodies never pass through these queues.
use super::{
    control::*,
    protocol::{BrokerDirective, ComponentRole, SensitiveString},
};
use crate::error::CliError;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_tungstenite::tungstenite::{Message, protocol::WebSocketConfig};

pub(crate) const MCP_SOCKET_PATH: &str = "/_nemo-relay/control/v1/mcp";
pub(crate) const WORKER_SOCKET_PATH: &str = "/_nemo-relay/control/v1/worker";
pub(crate) const QUEUE_CAPACITY: usize = 1024;
pub(crate) const GRACE: Duration = Duration::from_secs(30);
pub(crate) const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub(crate) enum Command {
    Challenge(ChallengeRequest),
    RegisterMcp {
        request: McpRegisterRequest,
        credential: SensitiveString,
    },
    RegisterWorker(WorkerRegisterRequest),
    RecoverWorker(WorkerRecoverRequest),
    Ready(SessionRequest<WorkerReadyPayload>),
    Release(SessionRequest<EmptyPayload>),
    ActivationFailed(SessionRequest<ActivationFailedPayload>),
    CancelActivation(SessionRequest<super::control::CancelActivationPayload>),
    Acknowledge {
        request_id: String,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Request {
    pub(crate) request_id: String,
    pub(crate) command: Command,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum Event {
    Reply {
        request_id: String,
        status: u16,
        payload: Value,
    },
    Directive {
        request_id: String,
        directive: BrokerDirective,
    },
    Drain {
        request_id: String,
        request: WorkerDrainRequest,
    },
}

struct Reply {
    status: u16,
    payload: Value,
}

struct PendingRequest {
    request: Request,
    reply: oneshot::Sender<Reply>,
}

struct Connection {
    send: mpsc::Sender<PendingRequest>,
    receive: Arc<Mutex<mpsc::Receiver<Event>>>,
    task: tokio::task::JoinHandle<()>,
    disconnected: Arc<std::sync::Mutex<Option<tokio::time::Instant>>>,
}
impl Connection {
    fn close(&self) {
        self.disconnected
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get_or_insert_with(tokio::time::Instant::now);
        self.task.abort();
    }
}
impl Drop for Connection {
    fn drop(&mut self) {
        self.close();
    }
}
#[derive(Clone, Default)]
pub(crate) struct Client(Arc<Mutex<Option<Connection>>>);

impl Client {
    pub(crate) async fn connect(&self, origin: &str, role: ComponentRole) -> Result<(), CliError> {
        let mut url = super::address::daemon_url(origin)?;
        let scheme = if url.scheme() == "https" { "wss" } else { "ws" };
        url.set_scheme(scheme)
            .map_err(|()| failure("invalid WebSocket scheme"))?;
        url.set_path(if role == ComponentRole::Mcp {
            MCP_SOCKET_PATH
        } else {
            WORKER_SOCKET_PATH
        });
        let config = WebSocketConfig::default()
            .max_message_size(Some(MAX_CONTROL_BODY_BYTES))
            .max_frame_size(Some(MAX_CONTROL_BODY_BYTES));
        // Multiple rustls backends are enabled in the workspace. Select Relay's provider
        // before the WSS connector builds its TLS configuration, as pooled_client does.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let (socket, _) = tokio::time::timeout(
            ATTEMPT_TIMEOUT,
            tokio_tungstenite::connect_async_with_config(url.as_str(), Some(config), true),
        )
        .await
        .map_err(|_| failure("control connection timed out"))?
        .map_err(|error| failure(format!("daemon WebSocket connection failed: {error}")))?;
        let (send, mut requests) = mpsc::channel::<PendingRequest>(QUEUE_CAPACITY);
        let (events, receive) = mpsc::channel(QUEUE_CAPACITY);
        let disconnected = Arc::new(std::sync::Mutex::new(None));
        let connection_loss = disconnected.clone();
        let task = tokio::spawn(async move {
            let (mut writer, mut reader) = socket.split();
            let mut replies = HashMap::<String, oneshot::Sender<Reply>>::new();
            loop {
                tokio::select! {
                    request = requests.recv() => {
                        let Some(PendingRequest { request, reply }) = request else { break };
                        if reply.is_closed() { continue; }
                        // Cancelled or timed-out callers must not consume reply slots forever.
                        replies.retain(|_, sender| !sender.is_closed());
                        if replies.len() == QUEUE_CAPACITY { break; }
                        let Ok(encoded) = serde_json::to_string(&request) else { break };
                        if encoded.len() > MAX_CONTROL_BODY_BYTES { break; }
                        replies.insert(request.request_id, reply);
                        if !matches!(tokio::time::timeout(ATTEMPT_TIMEOUT, writer.send(Message::Text(encoded.into()))).await, Ok(Ok(()))) { break; }
                    }
                    message = reader.next() => match message {
                        Some(Ok(Message::Text(text))) => {
                            let Ok(event) = serde_json::from_str::<Event>(&text) else { break };
                            match event {
                                Event::Reply { request_id, status, payload } => {
                                    if let Some(reply) = replies.remove(&request_id) {
                                        let _ = reply.send(Reply { status, payload });
                                    }
                                }
                                event => if events.try_send(event).is_err() { break; },
                            }
                        }
                        Some(Ok(Message::Ping(bytes))) => {
                            if !matches!(tokio::time::timeout(ATTEMPT_TIMEOUT, writer.send(Message::Pong(bytes))).await, Ok(Ok(()))) { break; }
                        }
                        Some(Ok(Message::Pong(_))) => {},
                        _ => break,
                    }
                }
            }
            // Record transport loss before closing the event channel. Consumers may still
            // have queued directives to process when they observe its eventual EOF.
            connection_loss
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get_or_insert_with(tokio::time::Instant::now);
        });
        *self.0.lock().await = Some(Connection {
            send,
            receive: Arc::new(Mutex::new(receive)),
            task,
            disconnected,
        });
        Ok(())
    }
    pub(crate) async fn recovery_deadline(&self) -> tokio::time::Instant {
        let guard = self.0.lock().await;
        let disconnected = guard.as_ref().and_then(|connection| {
            *connection
                .disconnected
                .lock()
                .unwrap_or_else(|error| error.into_inner())
        });
        disconnected.unwrap_or_else(tokio::time::Instant::now) + GRACE
    }
    pub(crate) async fn request<R: serde::de::DeserializeOwned>(
        &self,
        command: Command,
    ) -> Result<R, CliError> {
        tokio::time::timeout(ATTEMPT_TIMEOUT, async {
            let response = {
                let guard = self.0.lock().await;
                let connection = guard
                    .as_ref()
                    .ok_or_else(|| failure("control connection is closed"))?;
                let (reply, response) = oneshot::channel();
                if connection
                    .send
                    .try_send(PendingRequest {
                        request: Request {
                            request_id: uuid::Uuid::now_v7().to_string(),
                            command,
                        },
                        reply,
                    })
                    .is_err()
                {
                    connection.close();
                    return Err(failure("control writer unavailable"));
                }
                response
            };
            let Reply { status, payload } = response
                .await
                .map_err(|_| failure("control connection lost"))?;
            if !(200..300).contains(&status) {
                let message = payload
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("control command rejected");
                let code = payload.pointer("/error/code").and_then(Value::as_str);
                return Err(
                    if status == 401 && code == Some(ROUTE_CREDENTIAL_REJECTED_CODE) {
                        CliError::RouteCredentialRejected(message.into())
                    } else if status == 401 {
                        CliError::Unauthorized(message.into())
                    } else {
                        failure(message)
                    },
                );
            }
            serde_json::from_value(payload)
                .map_err(|error| failure(format!("invalid control reply: {error}")))
        })
        .await
        .map_err(|_| failure("control operation timed out"))?
    }
    pub(crate) async fn next(&self) -> Result<Event, CliError> {
        let receive = self
            .0
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| failure("control connection is closed"))?
            .receive
            .clone();
        receive
            .lock()
            .await
            .recv()
            .await
            .ok_or_else(|| failure("control connection lost"))
    }
    /// Registration replies are ordered after any pending drain intent on the same socket.
    pub(crate) async fn pending_event(&self) -> Option<Event> {
        let receive = self.0.lock().await.as_ref()?.receive.clone();
        receive.try_lock().ok()?.try_recv().ok()
    }
    pub(crate) async fn acknowledge(&self, request_id: String) -> Result<(), CliError> {
        self.request(Command::Acknowledge { request_id }).await
    }
}
pub(crate) fn failure(message: impl Into<String>) -> CliError {
    CliError::Launch(message.into())
}

/// Returns whether a failed control attempt may succeed on retry. A definitive credential
/// rejection cannot, so callers fall back immediately instead of spending the reconnect grace.
pub(crate) fn is_retryable(error: &CliError) -> bool {
    !matches!(error, CliError::RouteCredentialRejected(_))
}

pub(crate) async fn retry<T, F, Fut>(operation: F) -> Result<T, CliError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, CliError>>,
{
    retry_until(tokio::time::Instant::now() + GRACE, operation).await
}

pub(crate) async fn retry_until<T, F, Fut>(
    deadline: tokio::time::Instant,
    mut operation: F,
) -> Result<T, CliError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, CliError>>,
{
    let mut delay = Duration::from_millis(250);
    loop {
        if tokio::time::Instant::now() >= deadline {
            return Err(failure("control reconnect grace period expired"));
        }
        let result = tokio::time::timeout_at(
            deadline.min(tokio::time::Instant::now() + ATTEMPT_TIMEOUT),
            operation(),
        )
        .await;
        let error = match result {
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(error)) if !is_retryable(&error) => return Err(error),
            Ok(Err(error)) => error,
            Err(_) => failure("control connection/authentication attempt timed out"),
        };
        if tokio::time::Instant::now() >= deadline {
            return Err(failure(format!(
                "control reconnect grace period expired: {error}"
            )));
        }
        let jitter = u64::from(uuid::Uuid::now_v7().as_bytes()[15]) % 100;
        tokio::time::sleep_until(deadline.min(
            tokio::time::Instant::now() + delay.saturating_sub(Duration::from_millis(jitter)),
        ))
        .await;
        delay = (delay * 2).min(Duration::from_secs(2));
    }
}

#[cfg(test)]
#[path = "../../../tests/coverage/daemon/client_tests.rs"]
mod tests;
