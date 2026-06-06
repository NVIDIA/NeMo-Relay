// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for Python-facing local NeMo Guardrails integration.

use std::ffi::CString;
use std::fs;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::process::{self, Command, Stdio};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};

use nemo_relay::api::runtime::{NemoRelayContextState, global_context};
use nemo_relay::plugin::{
    PluginComponentSpec, PluginConfig, clear_plugin_configuration, initialize_plugins,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde_json::json;

static NEXT_FAKE_GUARDRAILS_ID: AtomicUsize = AtomicUsize::new(1);
static SERIAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
    let code = CString::new(code).unwrap();
    let file_name = CString::new("nemo_guardrails_coverage_tests.py").unwrap();
    let module_name = CString::new("nemo_guardrails_coverage_tests").unwrap();
    PyModule::from_code(py, &code, &file_name, &module_name).unwrap()
}

fn python_package_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python")
}

struct FakeGuardrailsPackage {
    root: PathBuf,
    module_name: String,
    python_executable: PathBuf,
}

impl FakeGuardrailsPackage {
    fn new(py: Python<'_>, module_name: &str, version: &str, implementation: &str) -> Self {
        let id = NEXT_FAKE_GUARDRAILS_ID.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "nemo_relay_python_fake_guardrails_{}_{}",
            process::id(),
            id
        ));
        let package = root.join(module_name);
        fs::create_dir_all(package.join("rails/llm")).unwrap();
        fs::write(package.join("rails/__init__.py"), "").unwrap();
        fs::write(package.join("rails/llm/__init__.py"), "").unwrap();
        fs::write(package.join("rails/llm/options.py"), fake_options_module()).unwrap();
        fs::write(
            package.join("__init__.py"),
            fake_root_module(version, implementation),
        )
        .unwrap();

        let python_executable = PathBuf::from(python_executable_for_worker(py));

        Self {
            root,
            module_name: module_name.to_string(),
            python_executable,
        }
    }
}

fn python_executable_for_worker(py: Python<'_>) -> String {
    let sys_executable = py
        .import("sys")
        .and_then(|sys| sys.getattr("executable"))
        .and_then(|executable| executable.extract::<String>())
        .ok();

    for executable in [
        std::env::var("PYO3_PYTHON").ok(),
        std::env::var("UV_PYTHON").ok(),
        sys_executable,
        Some("python3".to_string()),
    ]
    .into_iter()
    .flatten()
    {
        let executable = executable.trim();
        if executable.is_empty() {
            continue;
        }
        if Command::new(executable)
            .arg("-c")
            .arg("import sys")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
        {
            return executable.to_string();
        }
    }

    "python3".to_string()
}

impl Drop for FakeGuardrailsPackage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn fake_guardrails_module_prelude(
    module_name: &str,
    python_dir: &str,
    python_executable: &str,
    python_path: &str,
) -> String {
    format!(
        r#"
import sys

sys.path.insert(0, {python_dir:?})

MODULE_NAME = {module_name:?}
PYTHON_EXECUTABLE = {python_executable:?}
PYTHONPATH = {python_path:?}
"#,
        python_dir = python_dir,
        module_name = module_name,
        python_executable = python_executable,
        python_path = python_path,
    )
}

fn fake_options_module() -> &'static str {
    r#"
class RailType:
    INPUT = "input"
    OUTPUT = "output"

class RailStatus:
    BLOCKED = "blocked"
    MODIFIED = "modified"
    PASSED = "passed"
"#
}

fn fake_root_module(version: &str, implementation: &str) -> String {
    format!(
        r#"
import types
from .rails.llm.options import RailStatus

__version__ = {version:?}

class Result:
    def __init__(self, status, content=None, rail=None):
        self.status = status
        self.content = content
        self.rail = rail

class RailsConfig:
    @staticmethod
    def from_content(*, colang_content=None, yaml_content=None):
        return {{"yaml": yaml_content, "colang": colang_content}}

    @staticmethod
    def from_path(path):
        return {{"path": path}}

{implementation}
"#
    )
}

fn check_sequence_guardrails() -> &'static str {
    r#"
class LLMRails:
    def __init__(self, config):
        self.config = config
        self._check_results = [
            Result(RailStatus.MODIFIED, content="sanitized user"),
            Result(RailStatus.BLOCKED, rail="output-policy"),
            Result(RailStatus.MODIFIED, content='{"arguments": {"city": "Boston"}}'),
            Result(RailStatus.MODIFIED, content='{"result": {"ok": true}}'),
        ]

    async def check_async(self, messages, rail_types):
        return self._check_results.pop(0)
"#
}

fn tool_sequence_guardrails() -> &'static str {
    r#"
class LLMRails:
    def __init__(self, config):
        self.config = config
        self._check_results = [
            Result(RailStatus.MODIFIED, content="sanitized user"),
            Result(RailStatus.PASSED),
            Result(RailStatus.MODIFIED, content='{"arguments": {"city": "Boston"}}'),
            Result(RailStatus.MODIFIED, content='{"result": {"ok": true}}'),
        ]

    async def check_async(self, messages, rail_types):
        return self._check_results.pop(0)
"#
}

fn streaming_guardrails() -> &'static str {
    r#"
class LLMRails:
    def __init__(self, config):
        yaml = str(config.get("yaml", ""))
        stream_first = "stream_first_false" not in yaml
        self.config = types.SimpleNamespace(
            rails=types.SimpleNamespace(
                output=types.SimpleNamespace(
                    flows=["self check output"],
                    streaming=types.SimpleNamespace(enabled=True, stream_first=stream_first),
                )
            )
        )
        self._stream_calls = 0

    async def check_async(self, messages, rail_types):
        return Result(RailStatus.PASSED)

    def stream_async(self, *, messages=None, generator=None, include_metadata=False):
        async def _run():
            self._stream_calls += 1
            call_index = self._stream_calls
            async for chunk in generator:
                if call_index == 1:
                    yield chunk
            if call_index > 1:
                yield '{"error": {"message": "Blocked by output rails: output-policy", "type": "guardrails_violation"}}'
        return _run()
"#
}

fn with_isolated_nemo_relay_modules<T>(
    py: Python<'_>,
    native_module: &Bound<'_, PyModule>,
    f: impl FnOnce() -> T,
) -> T {
    let _serial_guard = SERIAL_TEST_MUTEX.lock().unwrap();
    let sys = py.import("sys").unwrap();
    let modules = sys
        .getattr("modules")
        .unwrap()
        .cast_into::<PyDict>()
        .unwrap();
    let saved_modules = modules
        .iter()
        .filter_map(|(name, module)| {
            let name = name.extract::<String>().ok()?;
            if name == "nemo_relay" || name.starts_with("nemo_relay.") {
                Some((name, module.unbind()))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    clear_nemo_relay_modules(&modules);
    modules
        .set_item("nemo_relay._native", native_module.clone())
        .unwrap();

    let result = catch_unwind(AssertUnwindSafe(f));

    clear_nemo_relay_modules(&modules);
    for (name, module) in saved_modules {
        modules.set_item(name, module).unwrap();
    }
    reset_runtime_state();

    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn clear_nemo_relay_modules(modules: &Bound<'_, PyDict>) {
    let module_names = modules
        .iter()
        .filter_map(|(name, _)| name.extract::<String>().ok())
        .filter(|name| name == "nemo_relay" || name.starts_with("nemo_relay."))
        .collect::<Vec<_>>();

    for name in module_names {
        modules.del_item(name).unwrap();
    }
}

fn with_event_loop<T>(py: Python<'_>, f: impl FnOnce(Bound<'_, PyAny>) -> T) -> T {
    let asyncio = py.import("asyncio").unwrap();
    #[cfg(windows)]
    {
        let policy = asyncio
            .getattr("WindowsSelectorEventLoopPolicy")
            .unwrap()
            .call0()
            .unwrap();
        asyncio
            .call_method1("set_event_loop_policy", (policy,))
            .unwrap();
    }
    let event_loop = asyncio.call_method0("new_event_loop").unwrap();
    asyncio
        .call_method1("set_event_loop", (&event_loop,))
        .unwrap();
    let result = catch_unwind(AssertUnwindSafe(|| f(event_loop.clone().into_any())));
    asyncio
        .call_method1("set_event_loop", (py.None(),))
        .unwrap();
    event_loop.call_method0("close").unwrap();
    #[cfg(windows)]
    asyncio
        .call_method1("set_event_loop_policy", (py.None(),))
        .unwrap();
    match result {
        Ok(value) => value,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

fn reset_runtime_state() {
    let _ = clear_plugin_configuration();
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
}

#[test]
fn test_native_pymodule_entrypoint_registers_bindings_without_local_provider_install() {
    let _python = crate::test_support::init_python_test();
    let _serial_guard = SERIAL_TEST_MUTEX.lock().unwrap();
    reset_runtime_state();
    Python::attach(|py| {
        let module = PyModule::new(py, "_native_guardrails_provider").unwrap();
        crate::_native(&module).unwrap();
    });

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let error = runtime
        .block_on(initialize_plugins(PluginConfig {
            version: 1,
            components: vec![PluginComponentSpec {
                kind: "nemo_guardrails".to_string(),
                enabled: true,
                config: serde_json::from_value(json!({
                    "mode": "local",
                    "codec": "openai_chat",
                    "config_path": "./rails"
                }))
                .unwrap(),
            }],
            policy: Default::default(),
        }))
        .unwrap_err();

    reset_runtime_state();
    match error {
        nemo_relay::plugin::PluginError::RegistrationFailed(message) => {
            assert!(
                message.contains(
                    "NeMo Guardrails is required for the built-in NeMo Guardrails local backend"
                ),
                "unexpected message: {message}"
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn test_guardrails_local_runtime_enforces_llm_input_and_output_checks() {
    let _python = crate::test_support::init_python_test();
    reset_runtime_state();

    Python::attach(|py| {
        let native_module = PyModule::new(py, "_native_guardrails_local_runtime").unwrap();
        crate::_native(&native_module).unwrap();

        with_isolated_nemo_relay_modules(py, &native_module, || {
            let fake = FakeGuardrailsPackage::new(
                py,
                "fake_guardrails_local_runtime",
                "0.22.0",
                check_sequence_guardrails(),
            );
            let python_dir = python_package_dir();
            let prelude = fake_guardrails_module_prelude(
                &fake.module_name,
                &python_dir.display().to_string(),
                &fake.python_executable.display().to_string(),
                &fake.root.display().to_string(),
            );
            let module = load_module(
                py,
                &format!(
                    r#"
{prelude}

import nemo_relay

async def run_case():
    stack = nemo_relay.create_scope_stack()
    nemo_relay.set_thread_scope_stack(stack)
    await nemo_relay.plugin.initialize(
        {{
            "version": 1,
            "components": [
                {{
                    "kind": "nemo_guardrails",
                    "enabled": True,
                    "config": {{
                        "mode": "local",
                        "codec": "openai_chat",
                        "config_yaml": "models: []",
                        "input": True,
                        "output": True,
                        "tool_input": True,
                        "tool_output": True,
                        "local": {{
                            "python_module": MODULE_NAME,
                            "python_executable": PYTHON_EXECUTABLE,
                            "python_path": PYTHONPATH,
                        }},
                    }},
                }}
            ],
        }}
    )

    request = nemo_relay.LLMRequest(
        {{}},
        {{
            "model": "gpt-4o-mini",
            "messages": [{{"role": "user", "content": "unsafe"}}],
        }},
    )
    seen_request_messages = []

    async def next_call(req):
        seen_request_messages.append(req.content["messages"][-1]["content"])
        return {{
            "choices": [{{"message": {{"role": "assistant", "content": "safe reply"}}}}],
            "id": "resp_1",
            "model": "gpt-4o-mini",
        }}

    try:
        await nemo_relay.llm.execute(
            "demo",
            request,
            next_call,
            response_codec=nemo_relay.codecs.OpenAIChatCodec(),
        )
    except RuntimeError as error:
        llm_error = str(error)
    else:
        raise AssertionError("expected output rail block")

    return {{
        "llm_error": llm_error,
        "seen_request_messages": seen_request_messages,
    }}
"#,
                    prelude = prelude,
                ),
            );

            let result_json = with_event_loop(py, |event_loop| {
                let coroutine = module.getattr("run_case").unwrap().call0().unwrap();
                let result = event_loop
                    .call_method1("run_until_complete", (coroutine,))
                    .unwrap();
                crate::convert::py_to_json(&result).unwrap()
            });

            assert_eq!(
                result_json["seen_request_messages"][0],
                json!("sanitized user")
            );
            assert!(
                result_json["llm_error"]
                    .as_str()
                    .unwrap()
                    .contains("output rail blocked the LLM call"),
                "unexpected error: {}",
                result_json["llm_error"]
            );
            assert!(
                result_json["llm_error"]
                    .as_str()
                    .unwrap()
                    .contains("output-policy"),
                "unexpected error: {}",
                result_json["llm_error"]
            );
        });
    });

    reset_runtime_state();
}

#[test]
fn test_guardrails_local_runtime_rejects_unsupported_nemoguardrails_version() {
    let _python = crate::test_support::init_python_test();
    reset_runtime_state();

    Python::attach(|py| {
        let native_module = PyModule::new(py, "_native_guardrails_version").unwrap();
        crate::_native(&native_module).unwrap();

        with_isolated_nemo_relay_modules(py, &native_module, || {
            let fake = FakeGuardrailsPackage::new(
                py,
                "fake_guardrails_bad_version",
                "0.21.0",
                check_sequence_guardrails(),
            );
            let python_dir = python_package_dir();
            let prelude = fake_guardrails_module_prelude(
                &fake.module_name,
                &python_dir.display().to_string(),
                &fake.python_executable.display().to_string(),
                &fake.root.display().to_string(),
            );
            let module = load_module(
                py,
                &format!(
                    r#"
{prelude}

import nemo_relay

async def run_case():
    await nemo_relay.plugin.initialize(
        {{
            "version": 1,
            "components": [
                {{
                    "kind": "nemo_guardrails",
                    "enabled": True,
                    "config": {{
                        "mode": "local",
                        "codec": "openai_chat",
                        "config_yaml": "models: []",
                        "input": True,
                        "local": {{
                            "python_module": MODULE_NAME,
                            "python_executable": PYTHON_EXECUTABLE,
                            "python_path": PYTHONPATH,
                        }},
                    }},
                }}
            ],
        }}
    )
"#,
                    prelude = prelude,
                ),
            );

            let error = with_event_loop(py, |event_loop| {
                let coroutine = module.getattr("run_case").unwrap().call0().unwrap();
                event_loop
                    .call_method1("run_until_complete", (coroutine,))
                    .unwrap_err()
                    .to_string()
            });

            assert!(
                error.contains("requires nemoguardrails==0.22.0"),
                "unexpected error: {error}"
            );
            assert!(error.contains("0.21.0"), "unexpected error: {error}");
        });
    });

    reset_runtime_state();
}

#[test]
fn test_guardrails_local_runtime_enforces_streamed_output_rails() {
    let _python = crate::test_support::init_python_test();
    reset_runtime_state();

    Python::attach(|py| {
        let native_module = PyModule::new(py, "_native_guardrails_streaming").unwrap();
        crate::_native(&native_module).unwrap();

        with_isolated_nemo_relay_modules(py, &native_module, || {
            let fake = FakeGuardrailsPackage::new(
                py,
                "fake_guardrails_streaming",
                "0.22.0",
                streaming_guardrails(),
            );
            let python_dir = python_package_dir();
            let prelude = fake_guardrails_module_prelude(
                &fake.module_name,
                &python_dir.display().to_string(),
                &fake.python_executable.display().to_string(),
                &fake.root.display().to_string(),
            );
            let module = load_module(
                py,
                &format!(
                    r#"
{prelude}

event_log = []

import nemo_relay

def plugin_config(config_yaml="models: []"):
    return {{
        "version": 1,
        "components": [
            {{
                "kind": "nemo_guardrails",
                "enabled": True,
                "config": {{
                    "mode": "local",
                    "codec": "openai_chat",
                    "config_yaml": config_yaml,
                    "input": False,
                    "output": True,
                    "local": {{
                        "python_module": MODULE_NAME,
                        "python_executable": PYTHON_EXECUTABLE,
                        "python_path": PYTHONPATH,
                    }},
                }},
            }}
        ],
    }}

async def run_stream(request):
    collected = []

    def next_call(req):
        async def _stream():
            event_log.append("source:hello")
            yield {{"choices": [{{"delta": {{"content": "hello"}}}}]}}
            await __import__("asyncio").sleep(0.01)
            event_log.append("source:world")
            yield {{"choices": [{{"delta": {{"content": "world"}}}}]}}
        return _stream()

    stream = await nemo_relay.llm.stream_execute(
        "demo",
        request,
        next_call,
        collected.append,
        lambda: {{"chunks": collected}},
        response_codec=nemo_relay.codecs.OpenAIChatCodec(),
    )
    chunks = []
    async for chunk in stream:
        event_log.append(f"yield:{{chunk['choices'][0]['delta']['content']}}")
        chunks.append(chunk)
    return chunks

async def run_case():
    stack = nemo_relay.create_scope_stack()
    nemo_relay.set_thread_scope_stack(stack)
    event_log.clear()
    await nemo_relay.plugin.initialize(plugin_config())

    request = nemo_relay.LLMRequest(
        {{}},
        {{
            "model": "gpt-4o-mini",
            "messages": [{{"role": "user", "content": "hello"}}],
        }},
    )

    allowed_chunks = await run_stream(request)

    try:
        await run_stream(request)
    except RuntimeError as error:
        blocked = str(error)
    else:
        raise AssertionError("expected streamed output block")

    nemo_relay.plugin.clear()
    await nemo_relay.plugin.initialize(plugin_config("stream_first_false"))
    try:
        await run_stream(request)
    except RuntimeError as error:
        modified = str(error)
    else:
        raise AssertionError("expected stream_first=false error")

    return {{
        "allowed_chunks": allowed_chunks,
        "blocked": blocked,
        "event_log": event_log,
        "modified": modified,
    }}
"#,
                    prelude = prelude,
                ),
            );

            let result = with_event_loop(py, |event_loop| {
                let coroutine = module.getattr("run_case").unwrap().call0().unwrap();
                let result = event_loop
                    .call_method1("run_until_complete", (coroutine,))
                    .unwrap();
                crate::convert::py_to_json(&result).unwrap()
            });
            assert_eq!(
                result["allowed_chunks"],
                json!([
                    {"choices": [{"delta": {"content": "hello"}}]},
                    {"choices": [{"delta": {"content": "world"}}]}
                ])
            );
            let event_log = result["event_log"].as_array().unwrap();
            for expected in ["source:hello", "source:world", "yield:hello", "yield:world"] {
                assert!(
                    event_log.iter().any(|event| event == expected),
                    "missing event {expected}: {event_log:?}"
                );
            }
            let source_hello = event_log
                .iter()
                .position(|event| event == "source:hello")
                .unwrap();
            let source_world = event_log
                .iter()
                .position(|event| event == "source:world")
                .unwrap();
            let yield_hello = event_log
                .iter()
                .position(|event| event == "yield:hello")
                .unwrap();
            let yield_world = event_log
                .iter()
                .position(|event| event == "yield:world")
                .unwrap();
            assert!(source_hello < yield_hello);
            assert!(source_world < yield_world);
            assert!(yield_hello < yield_world);
            assert!(
                result["blocked"]
                    .as_str()
                    .unwrap()
                    .contains("output rail blocked the LLM call")
            );
            assert!(
                result["modified"]
                    .as_str()
                    .unwrap()
                    .contains("stream_first = true")
            );
        });
    });

    reset_runtime_state();
}

#[test]
fn test_local_guardrails_provider_initializes_and_enforces_managed_core_calls() {
    let _python = crate::test_support::init_python_test();
    reset_runtime_state();

    Python::attach(|py| {
        let native_module = PyModule::new(py, "_native_guardrails_e2e").unwrap();
        crate::_native(&native_module).unwrap();

        with_isolated_nemo_relay_modules(py, &native_module, || {
            let fake = FakeGuardrailsPackage::new(
                py,
                "fake_guardrails_local_e2e",
                "0.22.0",
                tool_sequence_guardrails(),
            );
            let python_dir = python_package_dir();
            let prelude = fake_guardrails_module_prelude(
                &fake.module_name,
                &python_dir.display().to_string(),
                &fake.python_executable.display().to_string(),
                &fake.root.display().to_string(),
            );
            let module = load_module(
                py,
                &format!(
                    r#"
{prelude}

import nemo_relay

async def run_case():
    stack = nemo_relay.create_scope_stack()
    nemo_relay.set_thread_scope_stack(stack)

    await nemo_relay.plugin.initialize(
        {{
            "version": 1,
            "components": [
                {{
                    "kind": "nemo_guardrails",
                    "enabled": True,
                    "config": {{
                        "mode": "local",
                        "codec": "openai_chat",
                        "config_yaml": "models: []",
                        "input": True,
                        "output": True,
                        "tool_input": True,
                        "tool_output": True,
                        "local": {{
                            "python_module": MODULE_NAME,
                            "python_executable": PYTHON_EXECUTABLE,
                            "python_path": PYTHONPATH,
                        }},
                    }},
                }}
            ],
        }}
    )

    request = nemo_relay.LLMRequest(
        {{}},
        {{
            "model": "gpt-4o-mini",
            "messages": [{{"role": "user", "content": "unsafe"}}],
        }},
    )

    seen_request_messages = []
    async def llm_impl(req):
        seen_request_messages.append(req.content["messages"][-1]["content"])
        return {{
            "choices": [{{"message": {{"role": "assistant", "content": "safe reply"}}}}],
            "id": "resp_1",
            "model": req.content["model"],
        }}

    llm_result = await nemo_relay.llm.execute(
        "demo",
        request,
        llm_impl,
        response_codec=nemo_relay.codecs.OpenAIChatCodec(),
    )

    seen_tool_args = []
    async def tool_impl(args):
        seen_tool_args.append(args)
        return {{"raw": True}}

    tool_result = await nemo_relay.tools.execute("weather_lookup", {{"city": "Phoenix"}}, tool_impl)
    return {{
        "llm_result": llm_result,
        "tool_result": tool_result,
        "seen_request_messages": seen_request_messages,
        "seen_tool_args": seen_tool_args,
    }}
"#,
                    prelude = prelude,
                ),
            );
            let result_json = with_event_loop(py, |event_loop| {
                let coroutine = module.getattr("run_case").unwrap().call0().unwrap();
                let result = event_loop
                    .call_method1("run_until_complete", (coroutine,))
                    .unwrap();
                crate::convert::py_to_json(&result).unwrap()
            });

            assert_eq!(
                result_json["llm_result"]["choices"][0]["message"]["content"],
                json!("safe reply")
            );
            assert_eq!(result_json["tool_result"], json!({ "ok": true }));
            assert_eq!(
                result_json["seen_request_messages"][0],
                json!("sanitized user")
            );
            assert_eq!(
                result_json["seen_tool_args"][0],
                json!({ "city": "Boston" })
            );
        });
    });

    reset_runtime_state();
}
