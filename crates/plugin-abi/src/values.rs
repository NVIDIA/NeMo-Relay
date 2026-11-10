// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Checked wire records for the native value vocabulary.
use crate::{
    Field, Json,
    transfer::{Transfer, VectorWire, record},
    v1_0::*,
};

/// Frozen tagged payload. Only the field selected by `JsonWire::tag` is live.
#[repr(C)]
#[derive(Clone, Copy)]
pub union JsonPayload {
    scalar: u64,
    text: VectorWire<u8>,
    values: VectorWire<JsonWire>,
    fields: VectorWire<FieldWire>,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct JsonWire {
    tag: u8,
    payload: JsonPayload,
}
unsafe impl Transfer for Json {
    type Wire = JsonWire;
    fn lower(self) -> JsonWire {
        let (tag, payload) = match self {
            Self::Null => (0, JsonPayload { scalar: 0 }),
            Self::Bool(value) => (
                1,
                JsonPayload {
                    scalar: u64::from(value),
                },
            ),
            Self::Signed(value) => (
                2,
                JsonPayload {
                    scalar: value as u64,
                },
            ),
            Self::Unsigned(value) => (3, JsonPayload { scalar: value }),
            Self::Float(value) => (
                4,
                JsonPayload {
                    scalar: value.to_bits(),
                },
            ),
            Self::String(value) => (
                5,
                JsonPayload {
                    text: value.lower(),
                },
            ),
            Self::Array(value) => (
                6,
                JsonPayload {
                    values: value.lower(),
                },
            ),
            Self::Object(value) => (
                7,
                JsonPayload {
                    fields: value.lower(),
                },
            ),
        };
        JsonWire { tag, payload }
    }
    unsafe fn take(wire: &mut JsonWire) -> xabi::Result<Self> {
        Ok(match wire.tag {
            0 => Self::Null,
            1 if unsafe { wire.payload.scalar } <= 1 => {
                Self::Bool(unsafe { wire.payload.scalar } == 1)
            }
            2 => Self::Signed(unsafe { wire.payload.scalar } as i64),
            3 => Self::Unsigned(unsafe { wire.payload.scalar }),
            4 => {
                let value = f64::from_bits(unsafe { wire.payload.scalar });
                if !value.is_finite() {
                    return Err(xabi::Error::Export("non-finite JSON number".into()));
                }
                Self::Float(value)
            }
            5 => Self::String(unsafe { String::take(&mut wire.payload.text) }?),
            6 => Self::Array(unsafe { Vec::<Json>::take(&mut wire.payload.values) }?),
            7 => {
                let fields = unsafe { Vec::<Field>::take(&mut wire.payload.fields) }?;
                let mut keys = std::collections::BTreeSet::new();
                if fields.iter().any(|field| !keys.insert(&field.key)) {
                    return Err(xabi::Error::Export("duplicate JSON object key".into()));
                }
                Self::Object(fields)
            }
            _ => return Err(xabi::Error::AbiMismatch("invalid JSON tag".into())),
        })
    }
    unsafe fn release(wire: &mut JsonWire) {
        match wire.tag {
            5 => unsafe { String::release(&mut wire.payload.text) },
            6 => unsafe { Vec::<Json>::release(&mut wire.payload.values) },
            7 => unsafe { Vec::<Field>::release(&mut wire.payload.fields) },
            _ => {}
        }
        wire.tag = 0;
    }
}
record!(
    FieldWire,
    Field {
        key: String,
        value: Json
    }
);
record!(ErrorWire, Error {code:String,message:String,http_status:Option<u16>});
record!(ToolExecutionResultWire, ToolExecutionResult {result:Json,annotation:Option<Json>});
record!(ToolExecutionOutcomeWire, ToolExecutionOutcome {result:ToolExecutionResult,pending_marks:Vec<Json>});
record!(ConfigDiagnosticWire, ConfigDiagnostic {is_error:bool,code:String,component:Option<String>,field:Option<String>,message:String});
record!(InvocationWire, Invocation {surface:Surface,name:String,value:Json,tool_call_id:Option<String>,context:Context});
record!(ContextWire, Context {scope:ScopeBox,request_codec:Option<CodecBox>,response_codec:Option<CodecBox>});
record!(
    RegistrationWire,
    Registration {
        name: String,
        surface: Surface,
        priority: i32,
        break_chain: bool,
        options: Json,
        callback: CallbackBox
    }
);
record!(RegisteredComponentWire, RegisteredComponent {error:Option<Error>,registrations:Vec<Registration>,component:ComponentBox});
record!(
    DiagnosticReceiptWire,
    crate::diagnostics::DiagnosticReceipt {
        sequence: u64,
        total: u64,
        total_present: u32,
        saturated: u32,
        evidence: u32
    }
);
record!(
    DiagnosticCapabilitiesWire,
    crate::diagnostics::DiagnosticCapabilities {
        contract_version: u32,
        availability: u32,
        can_read_context: u32,
        max_components: u32,
        max_codes: u32,
        max_extensions: u32,
        max_request_bytes: u32,
        max_detail_bytes: u32,
        max_correlation_bytes: u32,
        max_response_bytes: usize,
        max_query_bytes: usize,
        max_sessions: u32,
        max_calls: u32,
        adapter_max_session_bytes: usize,
        adapter_session_bytes: usize,
        adapter_registration_scratch_bytes: usize
    }
);

unsafe impl Transfer for Surface {
    type Wire = u8;
    fn lower(self) -> u8 {
        self as u8
    }
    unsafe fn take(wire: &mut u8) -> xabi::Result<Self> {
        Ok(match *wire {
            0 => Self::Subscriber,
            1 => Self::EventMetadataInjector,
            2 => Self::MarkSanitize,
            3 => Self::ScopeSanitizeStart,
            4 => Self::ScopeSanitizeEnd,
            5 => Self::ToolSanitizeRequest,
            6 => Self::ToolSanitizeResponse,
            7 => Self::ToolConditionalExecution,
            8 => Self::ToolRequestIntercept,
            9 => Self::ToolExecutionIntercept,
            10 => Self::LlmSanitizeRequest,
            11 => Self::LlmSanitizeResponse,
            12 => Self::LlmConditionalExecution,
            13 => Self::LlmRequestIntercept,
            14 => Self::LlmExecutionIntercept,
            15 => Self::LlmStreamExecutionIntercept,
            16 => Self::ConditionalMiddleware,
            _ => return Err(xabi::Error::AbiMismatch("invalid surface".into())),
        })
    }
    unsafe fn release(_: &mut u8) {}
}
