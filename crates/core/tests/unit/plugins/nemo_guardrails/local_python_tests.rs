// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::CString;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::json;

use super::*;
use crate::plugins::nemo_guardrails::component::LocalBackendConfig;

fn local_config(module_name: &str) -> NeMoGuardrailsConfig {
    NeMoGuardrailsConfig {
        mode: "local".to_string(),
        codec: Some("openai_chat".to_string()),
        config_yaml: Some("models: []".to_string()),
        colang_content: Some("define flow noop\n  pass".to_string()),
        local: Some(LocalBackendConfig {
            python_module: Some(module_name.to_string()),
        }),
        ..NeMoGuardrailsConfig::default()
    }
}

fn install_fake_guardrails(py: Python<'_>, module_name: &str, version: &str, llm_rails_init: &str) {
    let code = format!(
        r#"
import sys
import types

MODULE_NAME = {module_name:?}

fake_root = types.ModuleType(MODULE_NAME)
fake_root.__version__ = {version:?}
fake_options = types.ModuleType(MODULE_NAME + ".rails.llm.options")

class Result:
    def __init__(self, status, content=None, rail=None):
        self.status = status
        self.content = content
        self.rail = rail

class RailType:
    INPUT = "input"
    OUTPUT = "output"

class RailStatus:
    BLOCKED = "blocked"
    MODIFIED = "modified"
    PASSED = "passed"

class RailsConfig:
    @staticmethod
    def from_content(*, colang_content=None, yaml_content=None):
        return {{"yaml": yaml_content, "colang": colang_content}}

    @staticmethod
    def from_path(path):
        return {{"path": path}}

class LLMRails:
    instances = []

    def __init__(self, config):
        LLMRails.instances.append(self)
{llm_rails_init}

fake_root.Result = Result
fake_root.RailStatus = RailStatus
fake_root.RailsConfig = RailsConfig
fake_root.LLMRails = LLMRails
fake_options.RailType = RailType
fake_options.RailStatus = RailStatus

sys.modules[MODULE_NAME] = fake_root
sys.modules[MODULE_NAME + ".rails"] = types.ModuleType(MODULE_NAME + ".rails")
sys.modules[MODULE_NAME + ".rails.llm"] = types.ModuleType(MODULE_NAME + ".rails.llm")
sys.modules[MODULE_NAME + ".rails.llm.options"] = fake_options
"#
    );
    let code = CString::new(code).unwrap();
    let file_name = CString::new("fake_guardrails.py").unwrap();
    let module_name = CString::new(format!("{module_name}_installer")).unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap();
}

fn py_to_json(obj: &Bound<'_, PyAny>) -> Json {
    pythonize::depythonize(obj).unwrap()
}

#[test]
fn bridge_loads_inline_guardrails_config() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    Python::attach(|py| {
        let module_name = "fake_guardrails_bridge_config";
        install_fake_guardrails(py, module_name, "0.22.0", "        self.config = config");

        let bridge = LocalGuardrailsBridge::new(&local_config(module_name)).unwrap();
        let config = bridge.rails.bind(py).getattr("config").unwrap();
        assert_eq!(
            py_to_json(&config),
            json!({"yaml": "models: []", "colang": "define flow noop\n  pass"})
        );
    });
}

#[test]
fn bridge_parses_pass_block_and_modify_outcomes() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    Python::attach(|py| {
        let module_name = "fake_guardrails_bridge_outcomes";
        install_fake_guardrails(py, module_name, "0.22.0", "        self.config = config");
        let bridge = LocalGuardrailsBridge::new(&local_config(module_name)).unwrap();
        let root = py.import(module_name).unwrap();
        let result_cls = root.getattr("Result").unwrap();
        let status = root.getattr("RailStatus").unwrap();

        let passed = result_cls
            .call1((status.getattr("PASSED").unwrap(),))
            .unwrap();
        assert!(matches!(
            bridge.parse_check_result(&passed).unwrap(),
            LocalCheckOutcome::Passed
        ));

        let blocked = result_cls
            .call1((status.getattr("BLOCKED").unwrap(), "stop", "policy"))
            .unwrap();
        match bridge.parse_check_result(&blocked).unwrap() {
            LocalCheckOutcome::Blocked { rail } => assert_eq!(rail.as_deref(), Some("policy")),
            _ => panic!("expected blocked outcome"),
        }

        let modified = result_cls
            .call1((status.getattr("MODIFIED").unwrap(), "rewritten"))
            .unwrap();
        match bridge.parse_check_result(&modified).unwrap() {
            LocalCheckOutcome::Modified { content } => assert_eq!(content, "rewritten"),
            _ => panic!("expected modified outcome"),
        }
    });
}

#[test]
fn modified_tool_payload_rejects_malformed_content() {
    let error = modified_tool_payload("not-json", "arguments").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("modified tool arguments content that is not valid JSON")
    );

    let error = modified_tool_payload(r#"{"tool_name":"demo"}"#, "result").unwrap_err();
    assert!(
        error
            .to_string()
            .contains("modified tool result content without a 'result' field")
    );
}

#[test]
fn streaming_support_rejects_stream_first_false() {
    let _guard = crate::plugins::nemo_guardrails::test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    Python::attach(|py| {
        let module_name = "fake_guardrails_bridge_streaming";
        install_fake_guardrails(
            py,
            module_name,
            "0.22.0",
            r#"        self.config = types.SimpleNamespace(
            rails=types.SimpleNamespace(
                output=types.SimpleNamespace(
                    flows=["self check output"],
                    streaming=types.SimpleNamespace(enabled=True, stream_first=False),
                )
            )
        )"#,
        );

        let bridge = LocalGuardrailsBridge::new(&local_config(module_name)).unwrap();
        assert!(bridge.has_streaming_output_rails().unwrap());
        let error = bridge.ensure_streaming_output_supported().unwrap_err();
        assert!(error.to_string().contains("stream_first = true"));
    });
}

#[test]
fn stream_text_extraction_handles_supported_codecs() {
    assert_eq!(
        extract_stream_text(
            LocalGuardrailsCodec::OpenAIChat,
            &json!({"choices": [{"delta": {"content": "hel"}}, {"delta": {"content": "lo"}}]})
        ),
        Some("hello".to_string())
    );
    assert_eq!(
        extract_stream_text(
            LocalGuardrailsCodec::OpenAIResponses,
            &json!({"type": "response.output_text.delta", "delta": "hello"})
        ),
        Some("hello".to_string())
    );
    assert_eq!(
        extract_stream_text(
            LocalGuardrailsCodec::AnthropicMessages,
            &json!({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hello"}})
        ),
        Some("hello".to_string())
    );
}
