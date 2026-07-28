// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script that regenerates the committed `nemo_relay.h` header.

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    validate_async_registration_parity(&crate_dir);
    let config = cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml"))
        .expect("Unable to read cbindgen.toml");

    if let Ok(bindings) = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        let header_path = format!("{crate_dir}/nemo_relay.h");
        bindings.write_to_file(&header_path);
        // cbindgen intentionally does not expand declarative macros. Keep the
        // macro-generated async registration functions in the generated C ABI.
        let header = std::fs::read_to_string(&header_path).expect("read generated FFI header");
        let marker = "\n#endif  /* NEMO_RELAY_H */\n";
        assert!(
            header.contains(marker),
            "generated FFI header is missing its NEMO_RELAY_H closing guard"
        );
        let header = header.replacen(
            marker,
            &format!("\n{}\n#endif  /* NEMO_RELAY_H */\n", ASYNC_REGISTRATIONS),
            1,
        );
        std::fs::write(header_path, header).expect("write generated FFI header");
    }
}

#[derive(Debug, PartialEq, Eq)]
struct AsyncPrototype<'a> {
    name: &'a str,
    parameters: Vec<&'a str>,
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

    let mut exported = Vec::new();
    for source in REGISTRATION_SOURCES {
        println!("cargo:rerun-if-changed={source}");
        let source_path = format!("{crate_dir}/{source}");
        let contents = std::fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("read {source_path}: {error}"));
        exported.extend(
            contents
                .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                .filter(|token| token.starts_with("nemo_relay_") && token.ends_with("_async"))
                .map(str::to_owned),
        );
    }
    exported.sort();
    exported.dedup();

    let mut declared = ASYNC_REGISTRATIONS
        .lines()
        .filter_map(parse_async_prototype)
        .collect::<Vec<_>>();
    declared.sort_by(|left, right| left.name.cmp(right.name));
    declared.dedup_by(|left, right| left.name == right.name);
    let declared_names = declared
        .iter()
        .map(|prototype| prototype.name.to_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        declared_names, exported,
        "ASYNC_REGISTRATIONS must declare exactly the async Rust FFI exports"
    );

    for prototype in declared {
        assert!(
            exported
                .binary_search_by(|name| name.as_str().cmp(prototype.name))
                .is_ok(),
            "async declaration for {} is not a Rust FFI export",
            prototype.name
        );
        assert_eq!(
            prototype,
            expected_async_prototype(prototype.name),
            "async declaration for {} has a mismatched C prototype",
            prototype.name
        );
    }
}

fn parse_async_prototype(line: &str) -> Option<AsyncPrototype<'_>> {
    let line = line.strip_prefix("NemoRelayStatus ")?;
    let (name, parameters) = line.split_once('(')?;
    let parameters = parameters.strip_suffix(");")?;
    Some(AsyncPrototype {
        name,
        parameters: parameters.split(", ").collect(),
    })
}

fn expected_async_prototype(name: &str) -> AsyncPrototype<'_> {
    let mut parameters = Vec::new();
    if name.starts_with("nemo_relay_scope_") {
        parameters.push("const char *scope_uuid");
    }
    parameters.extend(["const char *name", "int32_t priority"]);
    if name.contains("request_intercept_async") {
        parameters.push("bool break_chain");
    }
    parameters.push(if name.contains("execution_intercept_async") {
        "NemoRelayAsyncInterceptCb cb"
    } else {
        "NemoRelayAsyncJsonCb cb"
    });
    parameters.extend(["void *user_data", "NemoRelayFreeFn free_fn"]);
    AsyncPrototype { name, parameters }
}

const ASYNC_REGISTRATIONS: &str = r#"
/* Completion-based async middleware registrations generated from Rust macros. */
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncJsonCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncCompletion *completion);
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
