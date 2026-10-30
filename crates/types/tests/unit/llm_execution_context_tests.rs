// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::api::runtime::{BuiltinLlmCodec, LlmCodecIdentity, LlmResponseContext};

struct LeaseProbeCodec {
    allows_estimated_cost: bool,
}

impl LlmCodec for LeaseProbeCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    }

    fn decode(&self, _request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        Ok(AnnotatedLlmRequest::default())
    }

    fn encode(
        &self,
        _annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        Ok(original.clone())
    }
}

impl LlmResponseCodec for LeaseProbeCodec {
    fn codec_identity(&self) -> LlmCodecIdentity {
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    }

    fn allows_estimated_cost(&self, _response: &Json) -> bool {
        self.allows_estimated_cost
    }

    fn decode_response(&self, _response: &Json) -> Result<AnnotatedLlmResponse> {
        Ok(AnnotatedLlmResponse::default())
    }
}

#[test]
fn retained_codecs_work_while_active_and_expire_with_their_lease() {
    let backing = Arc::new(LeaseProbeCodec {
        allows_estimated_cost: false,
    });
    let backing_probe = Arc::downgrade(&backing);
    let request_codec: Arc<dyn LlmCodec> = backing.clone();
    let response_codec: Arc<dyn LlmResponseCodec> = backing.clone();
    let context = LlmExecutionContext::for_non_streaming(Some(request_codec), Some(response_codec));
    drop(backing);

    let (leased_context, guard) = context.lease();
    let retained_request = leased_context.request_codec().resolve_codec().unwrap();
    let retained_response = leased_context
        .response_codec()
        .and_then(LlmResponseContext::resolve_codec)
        .unwrap();
    drop(leased_context);
    drop(context);

    let request = LlmRequest {
        headers: serde_json::Map::new(),
        content: Json::Null,
    };
    let annotated = retained_request.decode(&request).unwrap();
    assert_eq!(
        retained_request.encode(&annotated, &request).unwrap(),
        request
    );
    assert_eq!(
        retained_response.decode_response(&Json::Null).unwrap(),
        AnnotatedLlmResponse::default()
    );
    assert!(!retained_response.allows_estimated_cost(&Json::Null));
    assert!(backing_probe.upgrade().is_some());

    drop(guard);

    assert!(backing_probe.upgrade().is_none());
    assert_eq!(
        retained_request.codec_identity(),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert_eq!(
        retained_response.codec_identity(),
        LlmCodecIdentity::BuiltIn(BuiltinLlmCodec::OpenAiChat)
    );
    assert!(!retained_response.allows_estimated_cost(&Json::Null));
    assert!(matches!(
        retained_request.decode(&request),
        Err(FlowError::InvalidArgument(_))
    ));
    assert!(matches!(
        retained_response.decode_response(&Json::Null),
        Err(FlowError::InvalidArgument(_))
    ));
}

#[test]
fn estimated_cost_defaults_to_false_after_expiry() {
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(LeaseProbeCodec {
        allows_estimated_cost: true,
    });
    let context = LlmExecutionContext::for_non_streaming(None, Some(response_codec));
    let (leased_context, guard) = context.lease();
    let retained = leased_context
        .response_codec()
        .and_then(LlmResponseContext::resolve_codec)
        .unwrap();

    assert!(retained.allows_estimated_cost(&Json::Null));
    drop(guard);
    assert!(!retained.allows_estimated_cost(&Json::Null));
}
