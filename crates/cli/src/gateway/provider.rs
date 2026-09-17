// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-owned provider transport. Credentials never cross the plugin ABI.

use nemo_relay::api::runtime::provider::{
    LlmProviderDispatcher, LlmProviderFormat, LlmProviderRequest,
};
use nemo_relay::codec::streaming::SseEventDecoder;

use super::*;
use crate::configuration::CallerCredentialTarget;

struct ProviderTransport {
    client: reqwest::Client,
    targets: BTreeMap<String, CallerCredentialTarget>,
    source: ProviderRoute,
    headers: HeaderMap,
    credential_present: bool,
    response_limit: usize,
}

pub(super) fn dispatcher(
    state: &AppState,
    prepared: &PreparedGatewayRequest,
) -> LlmProviderDispatcher {
    let transport = Arc::new(ProviderTransport {
        client: state.http_no_redirect.clone(),
        targets: state.config.caller_credential_targets.clone(),
        source: prepared.provider,
        headers: prepared.headers.clone(),
        credential_present: prepared
            .authorization
            .source_credential
            .provider_credential_present(),
        response_limit: state.config.max_passthrough_body_bytes,
    });
    let buffered = transport.clone();
    LlmProviderDispatcher::new(
        Arc::new(move |request| {
            let transport = buffered.clone();
            Box::pin(async move { transport.buffered(request).await })
        }),
        Arc::new(move |request| {
            let transport = transport.clone();
            Box::pin(async move { transport.streaming(request).await })
        }),
    )
}

impl ProviderTransport {
    fn headers_for(&self, target: &CallerCredentialTarget) -> Result<HeaderMap, FlowError> {
        let openai = matches!(
            self.source,
            ProviderRoute::OpenAiChatCompletions | ProviderRoute::OpenAiResponses
        );
        let matching_family = match target.format {
            LlmProviderFormat::OpenaiChat | LlmProviderFormat::OpenaiResponses => openai,
            LlmProviderFormat::AnthropicMessages => {
                matches!(self.source, ProviderRoute::AnthropicMessages)
            }
        };
        if !matching_family {
            return Err(FlowError::InvalidArgument(
                "provider target belongs to a different credential family".into(),
            ));
        }
        let credential_names: &[&str] = if openai {
            &["authorization", "api-key"]
        } else {
            &["authorization", "x-api-key", "anthropic-api-key"]
        };
        let mut headers = HeaderMap::new();
        for name in credential_names {
            if let Some(value) = self.headers.get(*name).filter(|value| !value.is_empty()) {
                let mut value = value.clone();
                value.set_sensitive(true);
                headers.insert(HeaderName::from_static(name), value);
            }
        }
        if !self.credential_present || headers.is_empty() {
            return Err(FlowError::InvalidArgument(
                "private provider dispatch requires a caller provider credential".into(),
            ));
        }
        let companion_names: &[&str] = if openai {
            &["chatgpt-account-id", "x-openai-fedramp"]
        } else {
            &["anthropic-version", "anthropic-beta"]
        };
        for name in companion_names {
            if let Some(value) = self.headers.get(*name) {
                headers.insert(HeaderName::from_static(name), value.clone());
            }
        }
        Ok(headers)
    }

    async fn send(
        &self,
        mut request: LlmProviderRequest,
        streaming: bool,
    ) -> Result<reqwest::Response, FlowError> {
        let target = self.targets.get(&request.target).ok_or_else(|| {
            FlowError::InvalidArgument(
                "provider target is not authorized by caller_credential_targets".into(),
            )
        })?;
        let headers = self.headers_for(target)?;
        let content = request.content.as_object_mut().ok_or_else(|| {
            FlowError::InvalidArgument("provider request content must be an object".into())
        })?;
        content.insert("stream".into(), Value::Bool(streaming));
        let response = self
            .client
            .post(&target.url)
            .headers(headers)
            .json(&request.content)
            .send()
            .await
            .map_err(|error| {
                safe_failure(
                    None,
                    if error.is_timeout() {
                        UpstreamFailureClass::Timeout
                    } else {
                        UpstreamFailureClass::Connection
                    },
                )
            })?;
        if !response.status().is_success() {
            let status = response.status().as_u16();
            // Never return provider error bodies, redirect locations, or transport URLs: they
            // can echo credentials. Status retains enough information for plugin retry policy.
            let class = match status {
                401 | 403 => UpstreamFailureClass::Authentication,
                408 | 429 | 500..=599 => UpstreamFailureClass::RetryableStatus,
                _ => UpstreamFailureClass::InvalidRequest,
            };
            return Err(safe_failure(Some(status), class));
        }
        Ok(response)
    }

    async fn buffered(&self, request: LlmProviderRequest) -> Result<Value, FlowError> {
        let mut response = self.send(request, false).await?;
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| malformed_response())? {
            if chunk.len() > self.response_limit.saturating_sub(bytes.len()) {
                return Err(FlowError::InvalidArgument(
                    "provider response exceeds gateway body limit".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let mut value = serde_json::from_slice(&bytes).map_err(|_| malformed_response())?;
        self.redact(&mut value);
        Ok(value)
    }

    async fn streaming(
        self: Arc<Self>,
        request: LlmProviderRequest,
    ) -> Result<LlmJsonStream, FlowError> {
        let response = self.send(request, true).await?;
        let mut bytes = response.bytes_stream();
        let mut decoder = SseEventDecoder::new();
        Ok(LlmJsonStream::new(stream! {
            while let Some(chunk) = bytes.next().await {
                let Ok(chunk) = chunk else {
                    yield Err(malformed_response());
                    return;
                };
                for result in decoder.push_bytes_results(&chunk) {
                    match result {
                        Ok(mut event) => { self.redact(&mut event.data); yield Ok(event.data); }
                        Err(_) => { yield Err(malformed_response()); return; }
                    }
                }
            }
            match decoder.finish() {
                Ok(Some(mut event)) => { self.redact(&mut event.data); yield Ok(event.data); }
                Ok(None) => {}
                Err(_) => yield Err(malformed_response()),
            }
        }))
    }

    // Defense in depth for providers that echo header values in successful JSON or SSE data.
    // Configured endpoints remain trusted recipients; this is not a sandbox for a malicious peer.
    fn redact(&self, value: &mut Value) {
        match value {
            Value::String(text) => {
                for name in ["authorization", "x-api-key", "api-key", "anthropic-api-key"] {
                    if let Some(secret) = self.headers.get(name).and_then(|v| v.to_str().ok()) {
                        let secret = secret.strip_prefix("Bearer ").unwrap_or(secret);
                        if !secret.is_empty() {
                            *text = text.replace(secret, "[REDACTED]");
                        }
                    }
                }
            }
            Value::Array(values) => values.iter_mut().for_each(|value| self.redact(value)),
            Value::Object(values) => {
                let original = std::mem::take(values);
                for (key, mut value) in original {
                    let mut key = Value::String(key);
                    self.redact(&mut key);
                    self.redact(&mut value);
                    values.insert(key.as_str().expect("string key").to_owned(), value);
                }
            }
            _ => {}
        }
    }
}

fn malformed_response() -> FlowError {
    FlowError::Internal("provider returned an unreadable response".into())
}

fn safe_failure(status: Option<u16>, class: UpstreamFailureClass) -> FlowError {
    FlowError::Upstream(UpstreamFailure {
        status,
        body: "private provider call failed".into(),
        headers: BTreeMap::new(),
        class,
    })
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/private_provider_tests.rs"]
mod tests;
