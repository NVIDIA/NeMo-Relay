use wasm_bindgen::prelude::*;
use wasm_bindgen_test::*;

// Tests run in Node.js via `wasm-pack test --node`

use nvagentrt_wasm::api::*;
use nvagentrt_wasm::types::*;

// ---------------------------------------------------------------------------
// Helper: create JS Functions with different arities
// ---------------------------------------------------------------------------
fn js_fn1(arg: &str, body: &str) -> js_sys::Function {
    js_sys::Function::new_with_args(arg, body)
}

fn js_fn2(args: &str, body: &str) -> js_sys::Function {
    js_sys::Function::new_with_args(args, body)
}

fn empty_obj() -> JsValue {
    js_sys::Object::new().into()
}

fn parse_json(s: &str) -> JsValue {
    js_sys::JSON::parse(s).unwrap()
}

// ===========================================================================
// Type constants
// ===========================================================================

#[wasm_bindgen_test]
fn test_scope_type_constants() {
    assert_eq!(SCOPE_TYPE_AGENT, 0);
    assert_eq!(SCOPE_TYPE_FUNCTION, 1);
    assert_eq!(SCOPE_TYPE_TOOL, 2);
    assert_eq!(SCOPE_TYPE_LLM, 3);
    assert_eq!(SCOPE_TYPE_RETRIEVER, 4);
    assert_eq!(SCOPE_TYPE_EMBEDDER, 5);
    assert_eq!(SCOPE_TYPE_RERANKER, 6);
    assert_eq!(SCOPE_TYPE_GUARDRAIL, 7);
    assert_eq!(SCOPE_TYPE_EVALUATOR, 8);
    assert_eq!(SCOPE_TYPE_CUSTOM, 9);
    assert_eq!(SCOPE_TYPE_UNKNOWN, 10);
}

#[wasm_bindgen_test]
fn test_attribute_constants() {
    assert_eq!(SCOPE_PARALLEL, 0b01);
    assert_eq!(SCOPE_RELOCATABLE, 0b10);
    assert_eq!(TOOL_LOCAL, 0b01);
    assert_eq!(LLM_STATELESS, 0b01);
    assert_eq!(LLM_STREAMING, 0b10);
}

#[wasm_bindgen_test]
fn test_scope_type_roundtrip() {
    assert_eq!(scope_type_to_i32(i32_to_scope_type(0)), 0);
    assert_eq!(scope_type_to_i32(i32_to_scope_type(9)), 9);
    assert_eq!(scope_type_to_i32(i32_to_scope_type(99)), SCOPE_TYPE_UNKNOWN);
}

// ===========================================================================
// WasmLLMRequest
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_request_construction() {
    let headers = empty_obj();
    let body = parse_json(r#"{"model":"gpt-4"}"#);
    let req = WasmLLMRequest::new(
        "POST".into(),
        "https://api.example.com/v1/chat".into(),
        headers,
        body,
    )
    .unwrap();
    assert_eq!(req.method(), "POST");
    assert_eq!(req.url(), "https://api.example.com/v1/chat");
}

#[wasm_bindgen_test]
fn test_llm_request_headers_and_body() {
    let headers = js_sys::Object::new();
    js_sys::Reflect::set(&headers, &"Authorization".into(), &"Bearer tok".into()).unwrap();
    let body = parse_json(r#"{"prompt":"hello"}"#);
    let req = WasmLLMRequest::new(
        "POST".into(),
        "https://api.example.com".into(),
        headers.into(),
        body,
    )
    .unwrap();

    let h = req.headers();
    let auth = js_sys::Reflect::get(&h, &"Authorization".into()).unwrap();
    assert_eq!(auth.as_string().unwrap(), "Bearer tok");

    let b = req.body();
    let prompt = js_sys::Reflect::get(&b, &"prompt".into()).unwrap();
    assert_eq!(prompt.as_string().unwrap(), "hello");
}

// ===========================================================================
// Scope operations
// ===========================================================================

#[wasm_bindgen_test]
fn test_get_handle_returns_root() {
    let handle = nv_agentrt_get_handle().unwrap();
    assert!(!handle.uuid().is_empty());
}

#[wasm_bindgen_test]
fn test_push_pop_scope() {
    let scope = nv_agentrt_push_scope("test_wasm_scope", SCOPE_TYPE_AGENT, None, None).unwrap();
    assert_eq!(scope.name(), "test_wasm_scope");
    assert_eq!(scope.scope_type(), SCOPE_TYPE_AGENT);
    nv_agentrt_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_attributes() {
    let scope = nv_agentrt_push_scope(
        "attr_scope",
        SCOPE_TYPE_FUNCTION,
        None,
        Some(SCOPE_PARALLEL | SCOPE_RELOCATABLE),
    )
    .unwrap();
    assert_eq!(scope.attributes(), SCOPE_PARALLEL | SCOPE_RELOCATABLE);
    nv_agentrt_pop_scope(&scope).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_with_parent() {
    // Push parent, note its uuid, then push child with parent (consumes parent)
    let parent = nv_agentrt_push_scope("parent_scope", SCOPE_TYPE_AGENT, None, None).unwrap();
    let parent_uuid = parent.uuid();
    let child =
        nv_agentrt_push_scope("child_scope", SCOPE_TYPE_FUNCTION, Some(parent), None).unwrap();
    assert_eq!(child.parent_uuid().unwrap(), parent_uuid);
    nv_agentrt_pop_scope(&child).unwrap();
    // Pop parent via get_handle (it's still on the stack)
    let current = nv_agentrt_get_handle().unwrap();
    assert_eq!(current.uuid(), parent_uuid);
    nv_agentrt_pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_scope_nesting() {
    let s1 = nv_agentrt_push_scope("nest_1", SCOPE_TYPE_AGENT, None, None).unwrap();
    let s2 = nv_agentrt_push_scope("nest_2", SCOPE_TYPE_FUNCTION, None, None).unwrap();
    let s3 = nv_agentrt_push_scope("nest_3", SCOPE_TYPE_TOOL, None, None).unwrap();
    nv_agentrt_pop_scope(&s3).unwrap();
    nv_agentrt_pop_scope(&s2).unwrap();
    nv_agentrt_pop_scope(&s1).unwrap();
}

#[wasm_bindgen_test]
fn test_all_scope_types() {
    let types = [
        (SCOPE_TYPE_AGENT, "agent_s"),
        (SCOPE_TYPE_FUNCTION, "function_s"),
        (SCOPE_TYPE_TOOL, "tool_s"),
        (SCOPE_TYPE_LLM, "llm_s"),
        (SCOPE_TYPE_RETRIEVER, "retriever_s"),
        (SCOPE_TYPE_EMBEDDER, "embedder_s"),
        (SCOPE_TYPE_RERANKER, "reranker_s"),
        (SCOPE_TYPE_GUARDRAIL, "guardrail_s"),
        (SCOPE_TYPE_EVALUATOR, "evaluator_s"),
        (SCOPE_TYPE_CUSTOM, "custom_s"),
        (SCOPE_TYPE_UNKNOWN, "unknown_s"),
    ];
    for (st, name) in types {
        let scope = nv_agentrt_push_scope(name, st, None, None).unwrap();
        assert_eq!(scope.scope_type(), st);
        nv_agentrt_pop_scope(&scope).unwrap();
    }
}

// ===========================================================================
// Events
// ===========================================================================

#[wasm_bindgen_test]
fn test_event_basic() {
    nv_agentrt_event("test_event", None, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_data() {
    let data = parse_json(r#"{"key":"value"}"#);
    nv_agentrt_event("data_event", None, data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_event_with_parent() {
    let scope = nv_agentrt_push_scope("event_parent", SCOPE_TYPE_AGENT, None, None).unwrap();
    // event() takes Option<WasmScopeHandle> by value — we need to pop after
    let scope_uuid = scope.uuid();
    nv_agentrt_event("child_event", Some(scope), JsValue::NULL, JsValue::NULL).unwrap();
    // Pop via get_handle
    let current = nv_agentrt_get_handle().unwrap();
    assert_eq!(current.uuid(), scope_uuid);
    nv_agentrt_pop_scope(&current).unwrap();
}

// ===========================================================================
// Subscribers
// ===========================================================================

#[wasm_bindgen_test]
fn test_register_deregister_subscriber() {
    let cb = js_fn1("event", "");
    register_subscriber("wasm_sub_1", cb).unwrap();
    let removed = deregister_subscriber("wasm_sub_1").unwrap();
    assert!(removed);
}

#[wasm_bindgen_test]
fn test_duplicate_subscriber_fails() {
    let cb1 = js_fn1("event", "");
    let cb2 = js_fn1("event", "");
    register_subscriber("wasm_dup_sub", cb1).unwrap();
    let result = register_subscriber("wasm_dup_sub", cb2);
    assert!(result.is_err());
    deregister_subscriber("wasm_dup_sub").unwrap();
}

#[wasm_bindgen_test]
fn test_deregister_nonexistent_subscriber() {
    let removed = deregister_subscriber("nonexistent_sub").unwrap();
    assert!(!removed);
}

// ===========================================================================
// Tool lifecycle
// ===========================================================================

#[wasm_bindgen_test]
fn test_tool_call_and_end() {
    let args = parse_json(r#"{"x": 1}"#);
    let handle =
        nv_agentrt_tool_call("test_tool", args, None, None, JsValue::NULL, JsValue::NULL).unwrap();
    assert_eq!(handle.name(), "test_tool");
    assert!(!handle.uuid().is_empty());

    let result = parse_json(r#"{"result": 42}"#);
    nv_agentrt_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_with_attributes() {
    let args = parse_json(r#"{}"#);
    let handle = nv_agentrt_tool_call(
        "attr_tool",
        args,
        None,
        Some(TOOL_LOCAL),
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(handle.attributes(), TOOL_LOCAL);

    let result = parse_json(r#"{}"#);
    nv_agentrt_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_with_data_metadata() {
    let args = parse_json(r#"{}"#);
    let data = parse_json(r#"{"info":"test"}"#);
    let meta = parse_json(r#"{"version":"1.0"}"#);
    let handle = nv_agentrt_tool_call("data_tool", args, None, None, data, meta).unwrap();

    let result = parse_json(r#"{}"#);
    let end_data = parse_json(r#"{"done":true}"#);
    nv_agentrt_tool_call_end(&handle, result, end_data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_with_parent() {
    let scope = nv_agentrt_push_scope("tool_parent", SCOPE_TYPE_AGENT, None, None).unwrap();
    let scope_uuid = scope.uuid();
    let args = parse_json(r#"{}"#);
    let handle = nv_agentrt_tool_call(
        "parented_tool",
        args,
        Some(scope),
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(handle.parent_uuid().unwrap(), scope_uuid);

    let result = parse_json(r#"{}"#);
    nv_agentrt_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();

    let current = nv_agentrt_get_handle().unwrap();
    nv_agentrt_pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_tool_call_generates_events() {
    js_sys::eval("globalThis.__tool_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__tool_events.push(event)");
    register_subscriber("wasm_tool_evt_sub", cb).unwrap();

    let args = parse_json(r#"{}"#);
    let handle =
        nv_agentrt_tool_call("evt_tool", args, None, None, JsValue::NULL, JsValue::NULL).unwrap();
    let result = parse_json(r#"{}"#);
    nv_agentrt_tool_call_end(&handle, result, JsValue::NULL, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__tool_events").unwrap();
    let arr = js_sys::Array::from(&events);
    // Should have at least start + end events
    assert!(
        arr.length() >= 2,
        "Expected at least 2 events for tool call/end"
    );

    deregister_subscriber("wasm_tool_evt_sub").unwrap();
    js_sys::eval("delete globalThis.__tool_events").unwrap();
}

// ===========================================================================
// Tool execute
// ===========================================================================

#[wasm_bindgen_test]
async fn test_tool_execute_basic() {
    let func = js_fn1("args", "return {result: args.x + 1}");
    let args = parse_json(r#"{"x": 10}"#);
    let result = nv_agentrt_tool_call_execute(
        "exec_tool",
        args,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let r = js_sys::Reflect::get(&result, &"result".into()).unwrap();
    assert_eq!(r.as_f64().unwrap(), 11.0);
}

#[wasm_bindgen_test]
async fn test_tool_execute_with_attributes() {
    let func = js_fn1("args", "return {ok: true}");
    let args = parse_json(r#"{}"#);
    let result = nv_agentrt_tool_call_execute(
        "exec_attr_tool",
        args,
        func,
        None,
        Some(TOOL_LOCAL),
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let ok = js_sys::Reflect::get(&result, &"ok".into()).unwrap();
    assert!(ok.as_bool().unwrap());
}

#[wasm_bindgen_test]
async fn test_tool_execute_promise() {
    let func = js_fn1("args", "return Promise.resolve({async_result: args.v * 2})");
    let args = parse_json(r#"{"v": 5}"#);
    let result = nv_agentrt_tool_call_execute(
        "promise_tool",
        args,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let r = js_sys::Reflect::get(&result, &"async_result".into()).unwrap();
    assert_eq!(r.as_f64().unwrap(), 10.0);
}

// ===========================================================================
// Tool guardrails
// ===========================================================================

#[wasm_bindgen_test]
fn test_tool_sanitize_request_guardrail() {
    let guardrail = js_fn2("name, args", "args.sanitized = true; return args");
    register_tool_sanitize_request_guardrail("wasm_tool_san_req", 10, guardrail).unwrap();
    deregister_tool_sanitize_request_guardrail("wasm_tool_san_req").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_sanitize_response_guardrail() {
    let guardrail = js_fn2("name, result", "result.checked = true; return result");
    register_tool_sanitize_response_guardrail("wasm_tool_san_resp", 10, guardrail).unwrap();
    deregister_tool_sanitize_response_guardrail("wasm_tool_san_resp").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_conditional_guardrail() {
    let guardrail = js_fn2("name, args", "return null");
    register_tool_conditional_execution_guardrail("wasm_tool_cond", 10, guardrail).unwrap();
    deregister_tool_conditional_execution_guardrail("wasm_tool_cond").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_conditional_guardrail_blocks() {
    let guardrail = js_fn2("name, args", "return 'blocked by guardrail'");
    register_tool_conditional_execution_guardrail("wasm_tool_block", 10, guardrail).unwrap();
    deregister_tool_conditional_execution_guardrail("wasm_tool_block").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_tool_guardrail_fails() {
    let g1 = js_fn2("name, args", "return args");
    let g2 = js_fn2("name, args", "return args");
    register_tool_sanitize_request_guardrail("wasm_dup_guard", 10, g1).unwrap();
    let result = register_tool_sanitize_request_guardrail("wasm_dup_guard", 20, g2);
    assert!(result.is_err());
    deregister_tool_sanitize_request_guardrail("wasm_dup_guard").unwrap();
}

// ===========================================================================
// Tool intercepts
// ===========================================================================

#[wasm_bindgen_test]
fn test_tool_request_intercept() {
    let func = js_fn2("name, args", "args.intercepted = true; return args");
    register_tool_request_intercept("wasm_tool_req_int", 10, false, func).unwrap();
    deregister_tool_request_intercept("wasm_tool_req_int").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_response_intercept() {
    let func = js_fn2("name, result", "result.processed = true; return result");
    register_tool_response_intercept("wasm_tool_resp_int", 10, false, func).unwrap();
    deregister_tool_response_intercept("wasm_tool_resp_int").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_execution_intercept() {
    let cond = js_fn2("name, args", "return true");
    let exec = js_fn1("args", "return {intercepted: true}");
    register_tool_execution_intercept("wasm_tool_exec_int", 10, cond, exec).unwrap();
    deregister_tool_execution_intercept("wasm_tool_exec_int").unwrap();
}

#[wasm_bindgen_test]
fn test_tool_request_intercept_break_chain() {
    let func = js_fn2("name, args", "return args");
    register_tool_request_intercept("wasm_tool_break", 10, true, func).unwrap();
    deregister_tool_request_intercept("wasm_tool_break").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_tool_intercept_fails() {
    let f1 = js_fn2("name, args", "return args");
    let f2 = js_fn2("name, args", "return args");
    register_tool_request_intercept("wasm_dup_int", 10, false, f1).unwrap();
    let result = register_tool_request_intercept("wasm_dup_int", 20, false, f2);
    assert!(result.is_err());
    deregister_tool_request_intercept("wasm_dup_int").unwrap();
}

#[wasm_bindgen_test]
async fn test_tool_request_intercept_modifies_args() {
    let func = js_fn2("name, args", "args.added = 'yes'; return args");
    register_tool_request_intercept("wasm_tool_req_mod", 10, false, func).unwrap();

    let exec = js_fn1("args", "return args");
    let args = parse_json(r#"{"original": true}"#);
    let result = nv_agentrt_tool_call_execute(
        "mod_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let added = js_sys::Reflect::get(&result, &"added".into()).unwrap();
    assert_eq!(added.as_string().unwrap(), "yes");

    deregister_tool_request_intercept("wasm_tool_req_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_tool_response_intercept_modifies_result() {
    let func = js_fn2(
        "name, result",
        "result.post_processed = true; return result",
    );
    register_tool_response_intercept("wasm_tool_resp_mod", 10, false, func).unwrap();

    let exec = js_fn1("args", "return {value: 42}");
    let args = parse_json(r#"{}"#);
    let result = nv_agentrt_tool_call_execute(
        "resp_mod_tool",
        args,
        exec,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let pp = js_sys::Reflect::get(&result, &"post_processed".into()).unwrap();
    assert!(pp.as_bool().unwrap());

    deregister_tool_response_intercept("wasm_tool_resp_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_tool_execution_intercept_replaces_func() {
    let cond = js_fn2("name, args", "return true");
    let intercept_exec = js_fn1("args", "return {replaced: true}");
    register_tool_execution_intercept("wasm_tool_exec_repl", 10, cond, intercept_exec).unwrap();

    let original = js_fn1("args", "return {original: true}");
    let args = parse_json(r#"{}"#);
    let result = nv_agentrt_tool_call_execute(
        "replaced_tool",
        args,
        original,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let replaced = js_sys::Reflect::get(&result, &"replaced".into()).unwrap();
    assert!(replaced.as_bool().unwrap());

    deregister_tool_execution_intercept("wasm_tool_exec_repl").unwrap();
}

// ===========================================================================
// LLM lifecycle
// ===========================================================================

fn make_llm_request(method: &str, url: &str) -> WasmLLMRequest {
    let headers = js_sys::Object::new();
    let body = parse_json(r#"{}"#);
    WasmLLMRequest::new(method.into(), url.into(), headers.into(), body).unwrap()
}

#[wasm_bindgen_test]
fn test_llm_call_and_end() {
    let req = make_llm_request("POST", "https://api.test.com");
    let handle =
        nv_agentrt_llm_call("test_llm", &req, None, None, JsValue::NULL, JsValue::NULL).unwrap();
    assert_eq!(handle.name(), "test_llm");
    assert!(!handle.uuid().is_empty());

    let response = parse_json(r#"{"choices":[{"text":"hello"}]}"#);
    nv_agentrt_llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_with_attributes() {
    let req = make_llm_request("POST", "https://api.test.com");
    let handle = nv_agentrt_llm_call(
        "attr_llm",
        &req,
        None,
        Some(LLM_STATELESS | LLM_STREAMING),
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(handle.attributes(), LLM_STATELESS | LLM_STREAMING);

    let response = parse_json(r#"{}"#);
    nv_agentrt_llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_with_parent() {
    let scope = nv_agentrt_push_scope("llm_parent", SCOPE_TYPE_AGENT, None, None).unwrap();
    let scope_uuid = scope.uuid();
    let req = make_llm_request("POST", "https://api.test.com");
    let handle = nv_agentrt_llm_call(
        "parented_llm",
        &req,
        Some(scope),
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .unwrap();
    assert_eq!(handle.parent_uuid().unwrap(), scope_uuid);

    let response = parse_json(r#"{}"#);
    nv_agentrt_llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();

    let current = nv_agentrt_get_handle().unwrap();
    nv_agentrt_pop_scope(&current).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_with_data_metadata() {
    let req = make_llm_request("POST", "https://api.test.com");
    let data = parse_json(r#"{"info":"llm_test"}"#);
    let meta = parse_json(r#"{"version":"2.0"}"#);
    let handle = nv_agentrt_llm_call("data_llm", &req, None, None, data, meta).unwrap();

    let response = parse_json(r#"{}"#);
    let end_data = parse_json(r#"{"tokens":100}"#);
    nv_agentrt_llm_call_end(&handle, response, end_data, JsValue::NULL).unwrap();
}

#[wasm_bindgen_test]
fn test_llm_call_generates_events() {
    js_sys::eval("globalThis.__llm_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__llm_events.push(event)");
    register_subscriber("wasm_llm_evt_sub", cb).unwrap();

    let req = make_llm_request("POST", "https://api.test.com");
    let handle =
        nv_agentrt_llm_call("evt_llm", &req, None, None, JsValue::NULL, JsValue::NULL).unwrap();
    let response = parse_json(r#"{}"#);
    nv_agentrt_llm_call_end(&handle, response, JsValue::NULL, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__llm_events").unwrap();
    let arr = js_sys::Array::from(&events);
    assert!(
        arr.length() >= 2,
        "Expected at least 2 events for llm call/end"
    );

    deregister_subscriber("wasm_llm_evt_sub").unwrap();
    js_sys::eval("delete globalThis.__llm_events").unwrap();
}

// ===========================================================================
// LLM execute
// ===========================================================================

#[wasm_bindgen_test]
async fn test_llm_execute_basic() {
    let func = js_fn1("request", "return {response: 'hello from llm'}");
    let req = make_llm_request("POST", "https://api.test.com");
    let result = nv_agentrt_llm_call_execute(
        "exec_llm",
        &req,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let r = js_sys::Reflect::get(&result, &"response".into()).unwrap();
    assert_eq!(r.as_string().unwrap(), "hello from llm");
}

#[wasm_bindgen_test]
async fn test_llm_execute_promise() {
    let func = js_fn1("request", "return Promise.resolve({async: true})");
    let req = make_llm_request("POST", "https://api.test.com");
    let result = nv_agentrt_llm_call_execute(
        "async_llm",
        &req,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let a = js_sys::Reflect::get(&result, &"async".into()).unwrap();
    assert!(a.as_bool().unwrap());
}

// ===========================================================================
// LLM guardrails
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_sanitize_request_guardrail() {
    let guardrail = js_fn1(
        "request",
        "request.url = 'https://sanitized.com'; return request",
    );
    register_llm_sanitize_request_guardrail("wasm_llm_san_req", 10, guardrail).unwrap();
    deregister_llm_sanitize_request_guardrail("wasm_llm_san_req").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_sanitize_response_guardrail() {
    let guardrail = js_fn1("response", "response.sanitized = true; return response");
    register_llm_sanitize_response_guardrail("wasm_llm_san_resp", 10, guardrail).unwrap();
    deregister_llm_sanitize_response_guardrail("wasm_llm_san_resp").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_conditional_guardrail() {
    let guardrail = js_fn1("request", "return null");
    register_llm_conditional_execution_guardrail("wasm_llm_cond", 10, guardrail).unwrap();
    deregister_llm_conditional_execution_guardrail("wasm_llm_cond").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_conditional_guardrail_blocks() {
    let guardrail = js_fn1("request", "return 'blocked'");
    register_llm_conditional_execution_guardrail("wasm_llm_block", 10, guardrail).unwrap();
    deregister_llm_conditional_execution_guardrail("wasm_llm_block").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_llm_guardrail_fails() {
    let g1 = js_fn1("request", "return request");
    let g2 = js_fn1("request", "return request");
    register_llm_sanitize_request_guardrail("wasm_llm_dup_guard", 10, g1).unwrap();
    let result = register_llm_sanitize_request_guardrail("wasm_llm_dup_guard", 20, g2);
    assert!(result.is_err());
    deregister_llm_sanitize_request_guardrail("wasm_llm_dup_guard").unwrap();
}

// ===========================================================================
// LLM intercepts
// ===========================================================================

#[wasm_bindgen_test]
fn test_llm_request_intercept() {
    let func = js_fn1(
        "request",
        "request.url = 'https://intercepted.com'; return request",
    );
    register_llm_request_intercept("wasm_llm_req_int", 10, false, func).unwrap();
    deregister_llm_request_intercept("wasm_llm_req_int").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_response_intercept() {
    let func = js_fn1("response", "response.intercepted = true; return response");
    register_llm_response_intercept("wasm_llm_resp_int", 10, false, func).unwrap();
    deregister_llm_response_intercept("wasm_llm_resp_int").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_execution_intercept() {
    let cond = js_fn1("request", "return true");
    let exec = js_fn1("request", "return {replaced: true}");
    register_llm_execution_intercept("wasm_llm_exec_int", 10, cond, exec).unwrap();
    deregister_llm_execution_intercept("wasm_llm_exec_int").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_stream_response_intercept() {
    let func = js_fn1("event", "return event");
    register_llm_stream_response_intercept("wasm_llm_sse_int", 10, false, func).unwrap();
    deregister_llm_stream_response_intercept("wasm_llm_sse_int").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_stream_execution_intercept() {
    let cond = js_fn1("request", "return true");
    let exec = js_fn1("request", "return {stream_result: true}");
    register_llm_stream_execution_intercept("wasm_llm_stream_exec", 10, cond, exec).unwrap();
    deregister_llm_stream_execution_intercept("wasm_llm_stream_exec").unwrap();
}

#[wasm_bindgen_test]
fn test_llm_request_intercept_break_chain() {
    let func = js_fn1("request", "return request");
    register_llm_request_intercept("wasm_llm_break", 10, true, func).unwrap();
    deregister_llm_request_intercept("wasm_llm_break").unwrap();
}

#[wasm_bindgen_test]
fn test_duplicate_llm_intercept_fails() {
    let f1 = js_fn1("request", "return request");
    let f2 = js_fn1("request", "return request");
    register_llm_request_intercept("wasm_llm_dup_int", 10, false, f1).unwrap();
    let result = register_llm_request_intercept("wasm_llm_dup_int", 20, false, f2);
    assert!(result.is_err());
    deregister_llm_request_intercept("wasm_llm_dup_int").unwrap();
}

#[wasm_bindgen_test]
async fn test_llm_request_intercept_modifies_request() {
    let intercept = js_fn1(
        "request",
        "request.url = 'https://modified.com'; return request",
    );
    register_llm_request_intercept("wasm_llm_req_mod", 10, false, intercept).unwrap();

    let func = js_fn1("request", "return {url: request.url}");
    let req = make_llm_request("POST", "https://original.com");
    let result = nv_agentrt_llm_call_execute(
        "mod_llm",
        &req,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let url = js_sys::Reflect::get(&result, &"url".into()).unwrap();
    assert_eq!(url.as_string().unwrap(), "https://modified.com");

    deregister_llm_request_intercept("wasm_llm_req_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_llm_response_intercept_modifies_response() {
    let intercept = js_fn1(
        "response",
        "response.post_processed = true; return response",
    );
    register_llm_response_intercept("wasm_llm_resp_mod", 10, false, intercept).unwrap();

    let func = js_fn1("request", "return {value: 'test'}");
    let req = make_llm_request("POST", "https://api.test.com");
    let result = nv_agentrt_llm_call_execute(
        "resp_mod_llm",
        &req,
        func,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let pp = js_sys::Reflect::get(&result, &"post_processed".into()).unwrap();
    assert!(pp.as_bool().unwrap());

    deregister_llm_response_intercept("wasm_llm_resp_mod").unwrap();
}

#[wasm_bindgen_test]
async fn test_llm_execution_intercept_replaces_func() {
    let cond = js_fn1("request", "return true");
    let intercept_exec = js_fn1("request", "return {replaced: true}");
    register_llm_execution_intercept("wasm_llm_exec_repl", 10, cond, intercept_exec).unwrap();

    let original = js_fn1("request", "return {original: true}");
    let req = make_llm_request("POST", "https://api.test.com");
    let result = nv_agentrt_llm_call_execute(
        "repl_llm",
        &req,
        original,
        None,
        None,
        JsValue::NULL,
        JsValue::NULL,
    )
    .await
    .unwrap();

    let replaced = js_sys::Reflect::get(&result, &"replaced".into()).unwrap();
    assert!(replaced.as_bool().unwrap());

    deregister_llm_execution_intercept("wasm_llm_exec_repl").unwrap();
}

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
    assert!(!deregister_llm_response_intercept("nx").unwrap());
    assert!(!deregister_llm_execution_intercept("nx").unwrap());
    assert!(!deregister_llm_stream_response_intercept("nx").unwrap());
    assert!(!deregister_llm_stream_execution_intercept("nx").unwrap());
}

// ===========================================================================
// Subscriber event detail tests
// ===========================================================================

#[wasm_bindgen_test]
fn test_subscriber_receives_events() {
    js_sys::eval("globalThis.__wasm_test_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__wasm_test_events.push(event)");
    register_subscriber("wasm_event_collector", cb).unwrap();

    let scope = nv_agentrt_push_scope("sub_test", SCOPE_TYPE_AGENT, None, None).unwrap();
    nv_agentrt_pop_scope(&scope).unwrap();

    let events = js_sys::eval("globalThis.__wasm_test_events").unwrap();
    let arr = js_sys::Array::from(&events);
    assert!(arr.length() > 0, "Expected at least one event");

    deregister_subscriber("wasm_event_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_test_events").unwrap();
}

#[wasm_bindgen_test]
fn test_subscriber_event_properties() {
    js_sys::eval("globalThis.__wasm_evt_props = null; true").unwrap();
    let cb = js_fn1(
        "event",
        "if (!globalThis.__wasm_evt_props) globalThis.__wasm_evt_props = event",
    );
    register_subscriber("wasm_prop_collector", cb).unwrap();

    let scope = nv_agentrt_push_scope("prop_test", SCOPE_TYPE_FUNCTION, None, None).unwrap();
    nv_agentrt_pop_scope(&scope).unwrap();

    let event = js_sys::eval("globalThis.__wasm_evt_props").unwrap();
    assert!(
        !event.is_null() && !event.is_undefined(),
        "Expected an event"
    );

    let uuid = js_sys::Reflect::get(&event, &"uuid".into()).unwrap();
    assert!(uuid.is_string(), "Event should have uuid string");

    let timestamp = js_sys::Reflect::get(&event, &"timestamp".into()).unwrap();
    assert!(timestamp.is_string(), "Event should have timestamp string");

    let event_type = js_sys::Reflect::get(&event, &"event_type".into()).unwrap();
    assert!(
        event_type.as_f64().is_some(),
        "Event should have event_type number"
    );

    deregister_subscriber("wasm_prop_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_evt_props").unwrap();
}

#[wasm_bindgen_test]
fn test_event_mark() {
    js_sys::eval("globalThis.__wasm_mark_events = []; true").unwrap();
    let cb = js_fn1("event", "globalThis.__wasm_mark_events.push(event)");
    register_subscriber("wasm_mark_collector", cb).unwrap();

    let data = parse_json(r#"{"marker":"test"}"#);
    nv_agentrt_event("mark_event", None, data, JsValue::NULL).unwrap();

    let events = js_sys::eval("globalThis.__wasm_mark_events").unwrap();
    let arr = js_sys::Array::from(&events);
    let found = (0..arr.length()).any(|i| {
        let e = arr.get(i);
        let et = js_sys::Reflect::get(&e, &"event_type".into())
            .unwrap()
            .as_f64()
            .unwrap_or(-1.0);
        et == 2.0 // Mark = 2
    });
    assert!(found, "Expected a Mark event (event_type=2)");

    deregister_subscriber("wasm_mark_collector").unwrap();
    js_sys::eval("delete globalThis.__wasm_mark_events").unwrap();
}
