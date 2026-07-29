// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script that regenerates the committed `nemo_relay.h` header.

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    validate_async_registration_parity(&crate_dir);
    let config = cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml"))
        .expect("Unable to read cbindgen.toml");
    let include_guard = config
        .include_guard
        .clone()
        .expect("cbindgen.toml must configure an include guard");

    let bindings = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate FFI header");
    let header_path = format!("{crate_dir}/nemo_relay.h");
    bindings.write_to_file(&header_path);
    // cbindgen intentionally does not expand declarative macros. Keep the
    // macro-generated async registration functions in the generated C ABI.
    let header = std::fs::read_to_string(&header_path).expect("read generated FFI header");
    let marker = format!("\n#endif  /* {include_guard} */\n");
    assert!(
        header.contains(&marker),
        "generated FFI header is missing its configured closing guard"
    );
    let replacement = format!("\n{}\n#endif  /* {include_guard} */\n", ASYNC_REGISTRATIONS);
    let header = header.replacen(&marker, &replacement, 1);
    std::fs::write(header_path, header).expect("write generated FFI header");
}

#[derive(Debug, PartialEq, Eq)]
struct AsyncPrototype {
    name: String,
    parameters: Vec<String>,
}

/// cbindgen does not expand the declarative registration macros. Keep the
/// handwritten C declarations checked against macro-generated exports, their
/// complete parameter lists, and ordering.
fn validate_async_registration_parity(crate_dir: &str) {
    const REGISTRATION_SOURCES: &[&str] = &[
        "src/api/event_registry.rs",
        "src/api/llm_registry.rs",
        "src/api/scope_registry.rs",
        "src/api/tool_registry.rs",
    ];

    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=src");

    let callable_path = format!("{crate_dir}/src/callable.rs");
    let callable = std::fs::read_to_string(&callable_path)
        .unwrap_or_else(|error| panic!("read {callable_path}: {error}"));
    validate_async_callback_abi(&callable);

    let mut expected = Vec::new();
    for source in REGISTRATION_SOURCES {
        let source_path = format!("{crate_dir}/{source}");
        let contents = std::fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {source_path}: {error}"));
        expected.extend(parse_async_macro_invocations(&contents));
    }
    expected.sort_by(|left, right| left.name.cmp(&right.name));

    let mut declared = ASYNC_REGISTRATIONS
        .lines()
        .filter_map(parse_async_prototype)
        .collect::<Vec<_>>();
    declared.sort_by(|left, right| left.name.cmp(&right.name));
    for duplicates in declared.windows(2) {
        assert_ne!(
            duplicates[0].name, duplicates[1].name,
            "ASYNC_REGISTRATIONS contains duplicate declaration for {}",
            duplicates[0].name
        );
    }
    assert_eq!(
        declared, expected,
        "ASYNC_REGISTRATIONS must exactly match the macro-generated Rust FFI exports"
    );
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn rust_type_alias(source: &str, name: &str) -> String {
    let prefix = format!("pub type {name}");
    let start = source
        .find(&prefix)
        .unwrap_or_else(|| panic!("src/callable.rs is missing {name}"));
    let end = source[start..]
        .find(';')
        .map(|offset| start + offset + 1)
        .unwrap_or_else(|| panic!("src/callable.rs has an unterminated {name} alias"));
    normalize_whitespace(&source[start..end])
}

/// Keep the handwritten C typedef block tied to the Rust callback ABI that
/// cbindgen cannot derive through registration macros.
fn validate_async_callback_abi(callable: &str) {
    let enum_start = callable
        .find("pub enum NemoRelayAsyncCallbackState")
        .expect("src/callable.rs is missing NemoRelayAsyncCallbackState");
    let enum_prefix = &callable[..enum_start];
    assert!(
        enum_prefix
            .rsplit_once("#[repr(u32)]")
            .is_some_and(|(_, suffix)| suffix.len() < 256),
        "NemoRelayAsyncCallbackState must retain its u32 representation"
    );
    let enum_end = callable[enum_start..]
        .find("\n}")
        .map(|offset| enum_start + offset)
        .expect("NemoRelayAsyncCallbackState is unterminated");
    let discriminants = callable[enum_start..enum_end]
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("Complete =") || line.starts_with("Pending ="))
        .collect::<Vec<_>>();
    assert_eq!(
        discriminants,
        ["Complete = 0,", "Pending = 1,"],
        "NemoRelayAsyncCallbackState drifted from the C callback-state constants"
    );

    assert_eq!(
        rust_type_alias(callable, "NemoRelayAsyncJsonCb"),
        normalize_whitespace(
            r#"pub type NemoRelayAsyncJsonCb = unsafe extern "C" fn(
                user_data: *mut libc::c_void,
                invocation_json: *const c_char,
                completion: *const NemoRelayAsyncCompletion,
            ) -> u32;"#
        ),
        "NemoRelayAsyncJsonCb drifted from ASYNC_REGISTRATIONS"
    );
    assert_eq!(
        rust_type_alias(callable, "NemoRelayAsyncInterceptCb"),
        normalize_whitespace(
            r#"pub type NemoRelayAsyncInterceptCb = unsafe extern "C" fn(
                user_data: *mut libc::c_void,
                invocation_json: *const c_char,
                next: *const NemoRelayAsyncNext,
                completion: *const NemoRelayAsyncCompletion,
            ) -> u32;"#
        ),
        "NemoRelayAsyncInterceptCb drifted from ASYNC_REGISTRATIONS"
    );
    assert_eq!(
        rust_type_alias(callable, "NemoRelayAsyncStreamInterceptCb"),
        normalize_whitespace(
            r#"pub type NemoRelayAsyncStreamInterceptCb = unsafe extern "C" fn(
                user_data: *mut libc::c_void,
                invocation_json: *const c_char,
                next: *const NemoRelayAsyncNext,
                stream: *const NemoRelayAsyncStream,
            ) -> u32;"#
        ),
        "NemoRelayAsyncStreamInterceptCb drifted from ASYNC_REGISTRATIONS"
    );
    for declaration in [
        "typedef uint32_t NemoRelayAsyncCallbackState;",
        "NEMO_RELAY_ASYNC_CALLBACK_STATE_COMPLETE = 0,",
        "NEMO_RELAY_ASYNC_CALLBACK_STATE_PENDING = 1,",
        "typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncJsonCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncCompletion *completion);",
        "typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncInterceptCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncNext *next, const struct NemoRelayAsyncCompletion *completion);",
        "typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncStreamInterceptCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncNext *next, const struct NemoRelayAsyncStream *stream);",
    ] {
        assert!(
            ASYNC_REGISTRATIONS.contains(declaration),
            "ASYNC_REGISTRATIONS is missing callback ABI declaration: {declaration}"
        );
    }
}

fn parse_async_prototype(line: &str) -> Option<AsyncPrototype> {
    let line = line.strip_prefix("NemoRelayStatus ")?;
    let (name, parameters) = line.split_once('(')?;
    let parameters = parameters.strip_suffix(");")?;
    Some(AsyncPrototype {
        name: name.to_owned(),
        parameters: parameters.split(", ").map(str::to_owned).collect(),
    })
}

fn parse_async_macro_invocations(source: &str) -> Vec<AsyncPrototype> {
    const MACROS: &[(&str, bool)] = &[
        ("global_async_registration!(", false),
        ("scope_async_registration!(", true),
    ];

    let mut prototypes = Vec::new();
    for (prefix, scope_local) in MACROS {
        let mut remaining = source;
        while let Some(start) = remaining.find(prefix) {
            let invocation = &remaining[start + prefix.len()..];
            let Some(end) = invocation.find(");") else {
                break;
            };
            let arguments = invocation[..end]
                .split(',')
                .map(str::trim)
                .collect::<Vec<_>>();
            remaining = &invocation[end + 2..];

            let name = arguments
                .first()
                .copied()
                .expect("async registration macro invocation is missing its export name");
            assert!(
                name.starts_with("nemo_relay_") && name.ends_with("_async"),
                "async registration macro exported unexpected name {name}; expected nemo_relay_*_async"
            );
            let callback_type = arguments
                .get(1)
                .unwrap_or_else(|| panic!("{name} macro invocation is missing its callback type"));
            let mut parameters = Vec::new();
            if *scope_local {
                parameters.push("const char *scope_uuid".to_owned());
            }
            parameters.extend(["const char *name".to_owned(), "int32_t priority".to_owned()]);
            if arguments.contains(&"break_chain") {
                parameters.push("bool break_chain".to_owned());
            }
            parameters.extend([
                format!("{callback_type} cb"),
                "void *user_data".to_owned(),
                "NemoRelayFreeFn free_fn".to_owned(),
            ]);
            prototypes.push(AsyncPrototype {
                name: name.to_owned(),
                parameters,
            });
        }
    }
    prototypes
}

const ASYNC_REGISTRATIONS: &str = r#"
/*
 * Completion-based async middleware registrations generated from Rust macros.
 *
 * Callbacks can run on Relay runtime or publication threads. invocation_json
 * and result-callback strings are borrowed only for the callback invocation;
 * user_data must remain valid and thread-safe until free_fn runs.
 *
 * A callback returning COMPLETE must settle its completion, or finish/reject
 * its stream, before returning. The runtime then releases the callback-owned
 * handles. A callback returning PENDING owns its completion/stream and next
 * references until it settles and releases each handle exactly once. While a
 * handle reference remains valid, duplicate settlement returns
 * NEMO_RELAY_STATUS_INVALID_ARG. After release, callers must not access the
 * handle; doing so is undefined behavior. Relay introduces no
 * implicit timeout; pending work must settle or observe cancellation through
 * nemo_relay_async_completion_is_cancelled or
 * nemo_relay_async_stream_is_cancelled. Each successful streaming next
 * invocation returns a caller-owned invocation handle; cancel it to stop an
 * idle continuation and release it exactly once after completion or
 * cancellation. Cancellation waits for any active result callback, making its
 * user_data unreachable before returning. Result callbacks must return false
 * instead of cancelling their own invocation.
 *
 * invocation_json/result contracts:
 * - event sanitizers: {"event":Event,"fields":EventSanitizeFields}
 *   -> EventSanitizeFields
 * - tool sanitizers, conditional guardrails, and request intercepts:
 *   {"name":string,"value":JSON} -> JSON, string|null, or JSON respectively
 * - tool execution intercepts: {"name":string,"value":JSON}
 *   -> ToolExecutionInterceptOutcome
 * - LLM request/response sanitizers:
 *   {"request":LlmRequest,"context":LlmCodecIdentity} -> LlmRequest|null, or
 *   {"response":JSON,"context":LlmCodecIdentity} -> JSON|null
 * - LLM conditional guardrails: {"request":LlmRequest} -> string|null
 * - LLM request intercepts:
 *   {"name":string,"request":LlmRequest,"annotated":AnnotatedLlmRequest|null}
 *   -> LlmRequestInterceptOutcome
 * - LLM execution and stream execution intercepts:
 *   {"name":string,"request":LlmRequest} -> JSON or incremental stream chunks
 */
typedef uint32_t NemoRelayAsyncCallbackState;
enum {
  NEMO_RELAY_ASYNC_CALLBACK_STATE_COMPLETE = 0,
  NEMO_RELAY_ASYNC_CALLBACK_STATE_PENDING = 1,
};
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncJsonCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncCompletion *completion);
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncInterceptCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncNext *next, const struct NemoRelayAsyncCompletion *completion);
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncStreamInterceptCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncNext *next, const struct NemoRelayAsyncStream *stream);
NemoRelayStatus nemo_relay_register_mark_sanitize_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_scope_sanitize_start_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_scope_sanitize_end_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_mark_sanitize_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_scope_sanitize_start_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_scope_sanitize_end_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_tool_sanitize_request_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_tool_sanitize_response_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_tool_conditional_execution_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_tool_request_intercept_async(const char *name, int32_t priority, bool break_chain, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_tool_execution_intercept_async(const char *name, int32_t priority, NemoRelayAsyncInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_llm_sanitize_request_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_llm_sanitize_response_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_llm_conditional_execution_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_llm_request_intercept_async(const char *name, int32_t priority, bool break_chain, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_llm_execution_intercept_async(const char *name, int32_t priority, NemoRelayAsyncInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_llm_stream_execution_intercept_async(const char *name, int32_t priority, NemoRelayAsyncStreamInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_tool_sanitize_request_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_tool_sanitize_response_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_tool_conditional_execution_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_tool_request_intercept_async(const char *scope_uuid, const char *name, int32_t priority, bool break_chain, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_llm_sanitize_request_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_llm_sanitize_response_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_llm_conditional_execution_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_llm_request_intercept_async(const char *scope_uuid, const char *name, int32_t priority, bool break_chain, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_tool_execution_intercept_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_llm_execution_intercept_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_llm_stream_execution_intercept_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncStreamInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
"#;
