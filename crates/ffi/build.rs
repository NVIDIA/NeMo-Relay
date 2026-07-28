// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script that regenerates the committed `nemo_relay.h` header.

fn main() {
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
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

const ASYNC_REGISTRATIONS: &str = r#"
/* Completion-based async middleware registrations generated from Rust macros. */
typedef NemoRelayAsyncCallbackState (*NemoRelayAsyncJsonCb)(void *user_data, const char *invocation_json, const struct NemoRelayAsyncCompletion *completion);
NemoRelayStatus nemo_relay_register_mark_sanitize_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_scope_sanitize_start_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_scope_sanitize_end_guardrail_async(const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_mark_sanitize_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_scope_sanitize_start_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_scope_register_scope_sanitize_end_guardrail_async(const char *scope_uuid, const char *name, int32_t priority, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
NemoRelayStatus nemo_relay_register_tool_request_intercept_async(const char *name, int32_t priority, bool break_chain, NemoRelayAsyncJsonCb cb, void *user_data, NemoRelayFreeFn free_fn);
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
