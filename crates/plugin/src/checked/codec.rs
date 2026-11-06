// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;
use nemo_relay_types::codec::response::AnnotatedLlmResponse;
#[derive(Clone)]
pub struct RequestCodec(Arc<a::CodecBox>);
impl RequestCodec {
    /// Decode a request through the invocation's live request codec capability.
    pub fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        decode(local(self.0.decode(encode(request)?))?)
    }
    /// Encode an annotated request while retaining fields from the original provider request.
    pub fn encode(
        &self,
        annotated: &AnnotatedLlmRequest,
        original: &LlmRequest,
    ) -> Result<LlmRequest> {
        decode(local(self.0.encode(encode(annotated)?, encode(original)?))?)
    }
}
#[derive(Clone)]
pub struct ResponseCodec(Arc<a::CodecBox>);
impl ResponseCodec {
    /// Decode a provider response through the invocation's live response codec capability.
    pub fn decode(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        decode(local(self.0.decode(encode(response)?))?)
    }
}
/// Resolve request codec identity and expose operations only when the host reports an available capability.
fn request(value: ROption<a::CodecBox>) -> Result<(LlmCodecIdentity, Option<RequestCodec>)> {
    match value.into_option() {
        None => Ok((LlmCodecIdentity::None, None)),
        Some(v) => {
            let record: Json = decode(local(v.identity())?)?;
            let identity =
                serde_json::from_value(record["identity"].clone()).map_err(|e| e.to_string())?;
            let available = record["available"].as_bool().unwrap_or(false);
            Ok((identity, available.then(|| RequestCodec(Arc::new(v)))))
        }
    }
}
/// Resolve response codec identity and expose operations only when the host reports an available capability.
fn response(value: ROption<a::CodecBox>) -> Result<(LlmCodecIdentity, Option<ResponseCodec>)> {
    match value.into_option() {
        None => Ok((LlmCodecIdentity::None, None)),
        Some(v) => {
            let record: Json = decode(local(v.identity())?)?;
            let identity =
                serde_json::from_value(record["identity"].clone()).map_err(|e| e.to_string())?;
            let available = record["available"].as_bool().unwrap_or(false);
            Ok((identity, available.then(|| ResponseCodec(Arc::new(v)))))
        }
    }
}
pub struct LlmSanitizeRequestContext<'a> {
    pub codec: LlmCodecIdentity,
    resolved: Option<RequestCodec>,
    _borrow: PhantomData<&'a ()>,
}
impl LlmSanitizeRequestContext<'_> {
    /// Bind request codec identity and capabilities to one sanitizer invocation.
    pub(super) fn new(value: ROption<a::CodecBox>) -> Result<Self> {
        let (codec, resolved) = request(value)?;
        Ok(Self {
            codec,
            resolved,
            _borrow: PhantomData,
        })
    }
    pub fn resolve_codec(&self) -> Option<&RequestCodec> {
        self.resolved.as_ref()
    }
}
pub struct LlmSanitizeResponseContext<'a> {
    pub codec: LlmCodecIdentity,
    resolved: Option<ResponseCodec>,
    _borrow: PhantomData<&'a ()>,
}
impl LlmSanitizeResponseContext<'_> {
    /// Bind response codec identity and capabilities to one sanitizer invocation.
    pub(super) fn new(value: ROption<a::CodecBox>) -> Result<Self> {
        let (codec, resolved) = response(value)?;
        Ok(Self {
            codec,
            resolved,
            _borrow: PhantomData,
        })
    }
    pub fn resolve_codec(&self) -> Option<&ResponseCodec> {
        self.resolved.as_ref()
    }
}
pub struct LlmRequestContext {
    pub codec: LlmCodecIdentity,
    resolved: Option<RequestCodec>,
}
impl LlmRequestContext {
    pub fn resolve_codec(&self) -> Option<&RequestCodec> {
        self.resolved.as_ref()
    }
}
pub struct LlmResponseContext {
    pub codec: LlmCodecIdentity,
    resolved: Option<ResponseCodec>,
}
impl LlmResponseContext {
    pub fn resolve_codec(&self) -> Option<&ResponseCodec> {
        self.resolved.as_ref()
    }
}
pub struct LlmExecutionContext {
    request_codec: LlmRequestContext,
    response_codec: Option<LlmResponseContext>,
}
impl LlmExecutionContext {
    /// Build invocation codec context with an optional response capability for unary execution.
    pub(super) fn new(req: ROption<a::CodecBox>, res: ROption<a::CodecBox>) -> Result<Self> {
        let (codec, resolved) = request(req)?;
        let request_codec = LlmRequestContext { codec, resolved };
        let response_codec = if res.is_some() {
            let (codec, resolved) = response(res)?;
            Some(LlmResponseContext { codec, resolved })
        } else {
            None
        };
        Ok(Self {
            request_codec,
            response_codec,
        })
    }
    pub fn request_codec(&self) -> &LlmRequestContext {
        &self.request_codec
    }
    pub fn response_codec(&self) -> Option<&LlmResponseContext> {
        self.response_codec.as_ref()
    }
}
