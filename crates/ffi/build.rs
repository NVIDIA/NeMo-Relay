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
        ("global_async_event_registration!(", false),
        ("scope_async_event_registration!(", true),
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

            let Some(name) = arguments.first().copied() else {
                continue;
            };
            if !name.starts_with("nemo_relay_") || !name.ends_with("_async") {
                continue;
            }
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
/* Completion-based async middleware registrations generated from Rust macros. */
typedef uint32_t NemoRelayAsyncCallbackState;
enum {
  NEMO_RELAY_ASYNC_CALLBACK_STATE_COMPLETE = 0,
  NEMO_RELAY_ASYNC_CALLBACK_STATE_PENDING = 1,
};
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncJsonCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncCompletion *completion);
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncInterceptCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncNext *next, const struct NemoRelayAsyncCompletion *completion);
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
NemoRelayStatus nemo_relay_register_llm_stream_execution_intercept_async(const char *name, int32_t priority, NemoRelayAsyncInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
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
NemoRelayStatus nemo_relay_scope_register_llm_stream_execution_intercept_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncInterceptCb cb, void *user_data, NemoRelayFreeFn free_fn);
"#;
