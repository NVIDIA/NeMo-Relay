// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Relay adapters for Switchyard's buffered translation engine.

use nemo_relay::api::llm::LlmRequest;
use nemo_relay::error::{FlowError, Result};
use serde_json::{Map, Value as Json};
use switchyard_protocol::{AggLlmResponse, LlmRequest as ProtocolRequest};
use switchyard_translation::{
    DeterministicIdPolicy, DiagnosticSeverity, LossyConversionPolicy, PreservationPolicy,
    TargetCapabilities, TranslationDiagnostic, TranslationEngine, TranslationPolicy,
    UnknownFieldPolicy, WireFormat,
};

use crate::component::WireProtocol;

pub(crate) fn translation_engine() -> TranslationEngine {
    TranslationEngine::default()
}

pub(crate) fn decode_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &LlmRequest,
) -> Result<ProtocolRequest> {
    let output = engine
        .decode_request(
            wire_format(protocol),
            &request.content,
            &translation_policy(),
        )
        .map_err(translation_error)?;
    ensure_safe_diagnostics(&output.diagnostics)?;
    Ok(output.request)
}

pub(crate) fn encode_request(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    request: &ProtocolRequest,
    headers: Map<String, Json>,
) -> Result<LlmRequest> {
    let output = engine
        .encode_request(
            wire_format(protocol),
            request,
            &request_translation_policy(protocol),
        )
        .map_err(translation_error)?;
    ensure_safe_diagnostics(&output.diagnostics)?;
    Ok(LlmRequest {
        headers,
        content: output.body,
    })
}

pub(crate) fn decode_response(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    response: &Json,
) -> Result<AggLlmResponse> {
    let output = engine
        .decode_response(wire_format(protocol), response, &translation_policy())
        .map_err(translation_error)?;
    ensure_safe_diagnostics(&output.diagnostics)?;
    Ok(output.response)
}

pub(crate) fn encode_response(
    engine: &TranslationEngine,
    protocol: WireProtocol,
    response: &AggLlmResponse,
) -> Result<Json> {
    let output = engine
        .encode_response(wire_format(protocol), response, &translation_policy())
        .map_err(translation_error)?;
    ensure_safe_diagnostics(&output.diagnostics)?;
    Ok(output.body)
}

pub(crate) const fn wire_format(protocol: WireProtocol) -> WireFormat {
    match protocol {
        WireProtocol::OpenaiChat => WireFormat::OpenAiChat,
        WireProtocol::OpenaiResponses => WireFormat::OpenAiResponses,
        WireProtocol::AnthropicMessages => WireFormat::AnthropicMessages,
    }
}

fn translation_policy() -> TranslationPolicy {
    TranslationPolicy {
        unknown_field_policy: UnknownFieldPolicy::Preserve,
        lossy_conversion_policy: LossyConversionPolicy::Reject,
        deterministic_ids: DeterministicIdPolicy::GenerateStable {
            prefix: "relay".into(),
        },
        preservation: PreservationPolicy::InMemory,
        target_capabilities: TargetCapabilities::default(),
    }
}

fn request_translation_policy(protocol: WireProtocol) -> TranslationPolicy {
    let mut policy = translation_policy();
    if protocol == WireProtocol::AnthropicMessages {
        policy
            .target_capabilities
            .supports_json_schema_response_format = Some(false);
    }
    policy
}

fn translation_error(error: switchyard_translation::TranslationError) -> FlowError {
    FlowError::InvalidArgument(format!("Switchyard translation failed: {error}"))
}

fn ensure_safe_diagnostics(diagnostics: &[TranslationDiagnostic]) -> Result<()> {
    let lossy = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity != DiagnosticSeverity::Info)
        .collect::<Vec<_>>();
    if lossy.is_empty() {
        Ok(())
    } else {
        Err(FlowError::InvalidArgument(format!(
            "Switchyard translation was not lossless: {lossy:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const PROTOCOLS: [WireProtocol; 3] = [
        WireProtocol::OpenaiChat,
        WireProtocol::OpenaiResponses,
        WireProtocol::AnthropicMessages,
    ];

    #[test]
    fn same_protocol_buffered_response_preserves_provider_extensions() {
        let engine = translation_engine();
        let original = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "provider/model",
            "system_fingerprint": "fp_exact",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop"
            }]
        });

        let decoded = decode_response(&engine, WireProtocol::OpenaiChat, &original);
        assert!(decoded.is_ok());
        let encoded = decoded
            .and_then(|response| encode_response(&engine, WireProtocol::OpenaiChat, &response));
        assert_eq!(encoded.ok(), Some(original));
    }

    #[test]
    fn same_protocol_request_preserves_unknown_fields() {
        let engine = translation_engine();
        let original = LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "caller/model",
                "messages": [{"role": "user", "content": "hello"}],
                "provider_extension": {"exact": true}
            }),
        };

        let decoded = decode_request(&engine, WireProtocol::OpenaiChat, &original);
        assert!(decoded.is_ok());
        let encoded = decoded.and_then(|request| {
            encode_request(&engine, WireProtocol::OpenaiChat, &request, Map::new())
        });
        assert_eq!(
            encoded.ok().map(|request| request.content),
            Some(original.content)
        );
    }

    #[test]
    fn all_same_protocol_requests_and_responses_preserve_unknown_fields() {
        let engine = translation_engine();
        for protocol in PROTOCOLS {
            let request = request_fixture(protocol, true);
            let decoded = decode_request(
                &engine,
                protocol,
                &LlmRequest {
                    headers: Map::new(),
                    content: request.clone(),
                },
            )
            .expect("request should decode");
            let encoded = encode_request(&engine, protocol, &decoded, Map::new())
                .expect("request should encode");
            assert_eq!(
                encoded.content, request,
                "request mismatch for {protocol:?}"
            );

            let response = response_fixture(protocol, true);
            let decoded =
                decode_response(&engine, protocol, &response).expect("response should decode");
            let encoded =
                encode_response(&engine, protocol, &decoded).expect("response should encode");
            assert_eq!(encoded, response, "response mismatch for {protocol:?}");
        }
    }

    #[test]
    fn every_cross_protocol_pair_translates_common_text_response_data() {
        let engine = translation_engine();
        for source in PROTOCOLS {
            for target in PROTOCOLS {
                if source == target {
                    continue;
                }

                let request = request_fixture(source, false);
                let decoded = decode_request(
                    &engine,
                    source,
                    &LlmRequest {
                        headers: Map::new(),
                        content: request,
                    },
                )
                .expect("request should decode");
                let encoded = encode_request(&engine, target, &decoded, Map::new())
                    .expect("request should encode");
                assert!(
                    encoded.content.to_string().contains("hello"),
                    "translated request lost text for {source:?} -> {target:?}: {}",
                    encoded.content
                );

                let response = response_fixture(source, false);
                let decoded =
                    decode_response(&engine, source, &response).expect("response should decode");
                let encoded =
                    encode_response(&engine, target, &decoded).expect("response should encode");
                let encoded_text = encoded.to_string();
                assert!(
                    encoded_text.contains("world"),
                    "translated response lost text for {source:?} -> {target:?}: {encoded}"
                );
                assert!(
                    encoded_text.contains('7') && encoded_text.contains('3'),
                    "translated response lost usage for {source:?} -> {target:?}: {encoded}"
                );
            }
        }
    }

    fn request_fixture(protocol: WireProtocol, extension: bool) -> Json {
        let mut body = match protocol {
            WireProtocol::OpenaiChat => json!({
                "model": "caller/model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32
            }),
            WireProtocol::OpenaiResponses => json!({
                "model": "caller/model",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}]
                }],
                "max_output_tokens": 32
            }),
            WireProtocol::AnthropicMessages => json!({
                "model": "caller/model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32
            }),
        };
        if extension {
            body["provider_extension"] = json!({"exact": true});
        }
        body
    }

    fn response_fixture(protocol: WireProtocol, extension: bool) -> Json {
        let mut body = match protocol {
            WireProtocol::OpenaiChat => json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "provider/model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "world"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
            }),
            WireProtocol::OpenaiResponses => json!({
                "id": "resp-test",
                "object": "response",
                "model": "provider/model",
                "status": "completed",
                "output": [{
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "world"}]
                }],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
            }),
            WireProtocol::AnthropicMessages => json!({
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "provider/model",
                "content": [{"type": "text", "text": "world"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 3}
            }),
        };
        if extension {
            body["provider_extension"] = json!({"exact": true});
        }
        body
    }
}
