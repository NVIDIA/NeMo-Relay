// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run with `cargo run -p nemo-relay-pii-redaction --example codec_memory`.
//! Fixture construction, output validation, and registration are outside the measurement.
mod heap;

use nemo_relay_codecs::resolve::{self, ProviderSurface};
use nemo_relay_pii_redaction::component::register_pii_redaction_component;
use nemo_relay_plugin_host::{
    HostPluginRegistrationContext, lookup_plugin, rollback_registrations,
};
use nemo_relay_runtime::api::runtime::{LlmSanitizeResponseContext, global_context};
use nemo_relay_types::api::llm::LlmRequest;
use nemo_relay_types::codec::redaction::NormalizedEdits;
use nemo_relay_types::codec::request::{ContentPart, MessageContent};
use serde_json::{Value, json};

fn sample<I, O>(name: &str, prepare: impl Fn() -> I, run: impl Fn(I) -> O, check: impl Fn(O)) {
    check(run(prepare()));
    let mut requested = [0; 7];
    let mut peaks = [0; 7];
    for (bytes, peak) in requested.iter_mut().zip(&mut peaks) {
        let input = prepare();
        let (output, allocated, live) = heap::measure(|| run(input));
        *bytes = allocated;
        *peak = live;
        check(output);
    }
    requested.sort();
    peaks.sort();
    println!("{name},{},{}", requested[3], peaks[3]);
}

fn projection() {
    let codec = resolve::request_codec(ProviderSurface::OpenAIChat);
    let messages: Vec<_> = (0..128)
        .map(|_| json!({"role":"user","content":"x".repeat(8192)}))
        .collect();
    let wire = LlmRequest {
        headers: Default::default(),
        content: json!({"messages":messages}),
    };
    let annotation = codec.decode(&wire).unwrap();
    let editor = codec.redaction().unwrap();
    sample(
        "project_history",
        || (),
        |()| editor.project_request(&annotation).unwrap().unwrap(),
        |output| assert_eq!(output, wire),
    );
}

fn gemini() {
    let codec = resolve::response_codec(ProviderSurface::GeminiGenerateContent);
    for mixed in [false, true] {
        let text = "x".repeat(8192);
        let mut parts = vec![json!({"text":text}); 128];
        if mixed {
            parts[127]["thoughtSignature"] = json!("retain");
        }
        let expected = vec![text.as_str(); 128].join("\n");
        let wire = json!({"candidates":[{"content":{"parts":parts}}]});
        sample(
            if mixed {
                "gemini_metadata"
            } else {
                "gemini_text"
            },
            || (),
            |()| codec.decode_response(&wire).unwrap(),
            |output| match output.message.unwrap() {
                MessageContent::Text(value) => {
                    assert!(!mixed);
                    assert_eq!(value, expected);
                }
                MessageContent::Parts(parts) => {
                    assert!(mixed);
                    assert_eq!(parts.len(), 128);
                    assert!(parts.iter().all(
                        |part| matches!(part, ContentPart::Text {text: value, ..} if *value == text)
                    ));
                    assert!(
                        matches!(&parts[127], ContentPart::Text {extra, ..} if extra["thoughtSignature"] == "retain")
                    );
                }
            },
        );
    }
}

fn response() -> Value {
    let args = json!({"rows":vec!["x".repeat(4096);256]}).to_string();
    json!({"model":"probe","output":[{"type":"message","content":[{"type":"output_text","text":"keep"}]},
          {"type":"function_call","call_id":"call","name":"lookup","arguments":args}],"usage":{"input_tokens":1,"output_tokens":1}})
}

fn pricing(estimated: bool) {
    let mut wire = response();
    if !estimated {
        wire["usage"]["cost"] = json!(0.5);
    }
    let codec = nemo_relay_runtime::codec::response_codec(ProviderSurface::OpenAIResponses);
    let baseline = serde_json::to_value(codec.decode_response(&wire).unwrap()).unwrap();
    let mut sanitized = baseline.clone();
    sanitized["message"] = json!("edited");
    let mut expected = wire.clone();
    expected["output"][0]["content"][0]["text"] = json!("edited");
    let editor = codec.redaction().unwrap();
    assert_eq!(
        baseline["usage"]["cost"]["source"],
        if estimated {
            "model_pricing"
        } else {
            "provider_reported"
        }
    );
    sample(
        if estimated {
            "priced_estimated"
        } else {
            "priced_reported"
        },
        || (),
        |()| {
            editor
                .edit_response(
                    &wire,
                    NormalizedEdits {
                        baseline: &baseline,
                        sanitized: &sanitized,
                        require_complete_output: true,
                    },
                )
                .unwrap()
                .unwrap()
        },
        |out| assert_eq!(out, expected),
    );
}

fn targeted_pii(concurrent: bool) {
    register_pii_redaction_component().unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let plugin = lookup_plugin("pii_redaction").unwrap();
    let codec = if concurrent {
        nemo_relay_runtime::codec::response_codec(ProviderSurface::OpenAIResponses)
    } else {
        resolve::response_codec(ProviderSurface::OpenAIResponses)
    };
    let ctx = LlmSanitizeResponseContext::for_response_codec(Some(codec));
    for (name, path) in [("pii_no_match", "/absent"), ("pii_one_field", "/private")] {
        let config = json!({"builtin":{"action":"remove","target_paths":[path]}});
        let mut registration = HostPluginRegistrationContext::new();
        runtime
            .block_on(plugin.register(config.as_object().unwrap(), &mut registration))
            .unwrap();
        let entries = global_context()
            .read()
            .unwrap()
            .llm_sanitize_response_entries(&[]);
        assert_eq!(entries.len(), 1);
        let mut wire = response();
        wire["private"] = json!("secret");
        let mut expected = wire.clone();
        if path == "/private" {
            expected.as_object_mut().unwrap().remove("private");
        }
        let sanitize = |input| {
            runtime
                .block_on((entries[0].payload)(input, ctx.clone()))
                .unwrap()
                .unwrap()
        };
        if concurrent {
            let barrier = std::sync::Barrier::new(8);
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    scope.spawn(|| {
                        for _ in 0..8 {
                            barrier.wait();
                            assert_eq!(sanitize(wire.clone()), expected);
                        }
                    });
                }
            });
        } else {
            sample(
                name,
                || wire.clone(),
                sanitize,
                |out| assert_eq!(out, expected),
            );
        }
        rollback_registrations(&mut registration.into_registrations());
    }
}

fn main() {
    // Pin the resolver so local pricing catalogs cannot change the measured branches.
    nemo_relay_runtime::composition::install_runtime_services(|| {
        nemo_relay_runtime::composition::RuntimeServices {
            attach_estimated_cost: |response, _| {
                if let Some(usage) = &mut response.usage {
                    usage.cost.get_or_insert_with(|| {
                        serde_json::from_value(json!({"source":"model_pricing","total":0.5}))
                            .unwrap()
                    });
                }
            },
            ..Default::default()
        }
    });
    if std::env::args().nth(1).as_deref() == Some("--concurrent") {
        targeted_pii(true);
        return;
    }
    println!("fixture,allocated_bytes,peak_extra_live_bytes");
    projection();
    gemini();
    pricing(false);
    pricing(true);
    targeted_pii(false);
}
