// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use wasm_bindgen_test::*;

use nvidia_nat_nexus_wasm::api::*;

// ===========================================================================
// Deregister nonexistent
// ===========================================================================

#[wasm_bindgen_test]
fn test_deregister_nonexistent_tool_guardrails() {
    assert!(!deregister_tool_sanitize_request_guardrail("nx").unwrap());
    assert!(!deregister_tool_sanitize_response_guardrail("nx").unwrap());
    assert!(!deregister_tool_conditional_execution_guardrail("nx").unwrap());
}

#[wasm_bindgen_test]
fn test_deregister_nonexistent_tool_intercepts() {
    assert!(!deregister_tool_request_intercept("nx").unwrap());
    assert!(!deregister_tool_response_intercept("nx").unwrap());
    assert!(!deregister_tool_execution_intercept("nx").unwrap());
}

#[wasm_bindgen_test]
fn test_deregister_nonexistent_llm_guardrails() {
    assert!(!deregister_llm_sanitize_request_guardrail("nx").unwrap());
    assert!(!deregister_llm_sanitize_response_guardrail("nx").unwrap());
    assert!(!deregister_llm_conditional_execution_guardrail("nx").unwrap());
}

#[wasm_bindgen_test]
fn test_deregister_nonexistent_llm_intercepts() {
    assert!(!deregister_llm_request_intercept("nx").unwrap());
    assert!(!deregister_llm_execution_intercept("nx").unwrap());
    assert!(!deregister_llm_stream_execution_intercept("nx").unwrap());
}
