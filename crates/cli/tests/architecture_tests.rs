// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Architectural dependency and source-layout regression tests.

use std::fs;
use std::path::{Path, PathBuf};

use syn::visit::Visit;

fn source_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_owned()];
    while let Some(path) = pending.pop() {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
                files.push(path);
            }
        }
    }
    files
}

fn syntax_paths(source: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("architecture fixture must parse as Rust");
    let mut visitor = PathVisitor::default();
    visitor.visit_file(&file);
    for item in &file.items {
        if let syn::Item::Use(item) = item {
            expand_use_tree(Vec::new(), &item.tree, &mut visitor.paths);
        }
    }
    visitor.paths
}

#[derive(Default)]
struct PathVisitor {
    paths: Vec<String>,
    command_attributes: Vec<String>,
    test_attributes: Vec<String>,
}

#[derive(Default)]
struct StreamingSourceVisitor {
    file: String,
    function: Option<String>,
    violations: Vec<String>,
}

impl StreamingSourceVisitor {
    fn new(file: &str) -> Self {
        Self {
            file: file.to_owned(),
            ..Self::default()
        }
    }

    fn function_name(&self) -> &str {
        self.function.as_deref().unwrap_or("<module>")
    }

    fn allows_request_body_decode(&self) -> bool {
        self.file == "daemon/worker/managed.rs"
            && matches!(self.function_name(), "handle_hook_inner" | "read")
    }

    fn allows_iterator_collect(&self) -> bool {
        matches!(
            (self.file.as_str(), self.function_name()),
            ("daemon/broker/server.rs", "decode_pem_blocks")
                | ("daemon/broker/server.rs", "load_tls_config")
                | ("daemon/broker/server.rs", "prune_expired_mcp_control_state")
                | ("daemon/broker/server.rs", "spawn_maintenance")
                | ("daemon/broker/server.rs", "strip_public_relay_headers")
                | (
                    "daemon/worker/managed.rs",
                    "incompatible_registration_names"
                )
                | ("daemon/worker/managed.rs", "strip_internal_headers")
        )
    }

    fn allows_sse_observation(&self) -> bool {
        self.file == "daemon/worker/managed.rs" && self.function_name() == "finish_stream"
    }

    fn is_delivery_function(&self) -> bool {
        matches!(
            self.function_name(),
            "public_proxy"
                | "forward_to_provider"
                | "forward_to_worker"
                | "forward"
                | "proxy"
                | "proxy_provider"
                | "proxy_managed"
                | "dispatch_unmanaged"
                | "dispatch_observed"
                | "poll_frame"
        )
    }

    fn reject(&mut self, operation: &str) {
        self.violations.push(format!(
            "{} uses {operation} in {}",
            self.file,
            self.function_name()
        ));
    }

    fn with_function(&mut self, name: String, visit: impl FnOnce(&mut Self)) {
        let previous = self.function.replace(name);
        visit(self);
        self.function = previous;
    }
}

impl<'ast> Visit<'ast> for StreamingSourceVisitor {
    fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
        self.with_function(function.sig.ident.to_string(), |visitor| {
            syn::visit::visit_item_fn(visitor, function);
        });
    }

    fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
        self.with_function(function.sig.ident.to_string(), |visitor| {
            syn::visit::visit_impl_item_fn(visitor, function);
        });
    }

    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        let method = call.method.to_string();
        match method.as_str() {
            "bytes" | "text" | "json" | "bytes_stream" => self.reject(&format!(".{method}()")),
            "collect" if !self.allows_iterator_collect() => self.reject(".collect()"),
            "push_bytes_results" if !self.allows_sse_observation() => {
                self.reject("SSE decoding on the delivery path")
            }
            "extend_from_slice" | "extend" | "push_str" if self.is_delivery_function() => {
                self.reject(&format!("response accumulation via .{method}()"))
            }
            _ => {}
        }
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref() {
            let segments = path
                .path
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>();
            let last = segments.last().map(String::as_str).unwrap_or_default();
            if last == "to_bytes" && !self.allows_request_body_decode() {
                self.reject("to_bytes() response aggregation");
            }
            if last == "from_stream" {
                self.reject("Body::from_stream()");
            }
            if segments.iter().any(|segment| segment.contains("Sse"))
                && !self.allows_sse_observation()
            {
                self.reject("SSE construction or decoding on the delivery path");
            }
            let aggregate_constructor = segments
                .iter()
                .rev()
                .take(2)
                .map(String::as_str)
                .collect::<Vec<_>>();
            if self.is_delivery_function()
                && matches!(
                    aggregate_constructor.as_slice(),
                    [constructor, container]
                        if matches!(*container, "Vec" | "String")
                            && matches!(*constructor, "new" | "with_capacity")
                )
            {
                self.reject("response-wide Vec/String construction");
            }
        }
        syn::visit::visit_expr_call(self, call);
    }

    fn visit_use_rename(&mut self, rename: &'ast syn::UseRename) {
        if matches!(
            rename.ident.to_string().as_str(),
            "to_bytes" | "from_stream"
        ) {
            self.reject("an alias for a forbidden aggregation API");
        }
        syn::visit::visit_use_rename(self, rename);
    }
}

impl<'ast> Visit<'ast> for PathVisitor {
    fn visit_path(&mut self, path: &'ast syn::Path) {
        self.paths.push(
            path.segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
        );
        syn::visit::visit_path(self, path);
    }

    fn visit_attribute(&mut self, attribute: &'ast syn::Attribute) {
        let path = attribute
            .path()
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
        let name = path.rsplit("::").next().unwrap_or_default();
        if matches!(name, "arg" | "command" | "value") {
            self.command_attributes.push(name.to_owned());
        }
        if matches!(path.as_str(), "test" | "tokio::test") {
            self.test_attributes.push(path);
        }
        syn::visit::visit_attribute(self, attribute);
    }
}

fn expand_use_tree(prefix: Vec<String>, tree: &syn::UseTree, output: &mut Vec<String>) {
    match tree {
        syn::UseTree::Path(path) => {
            let mut prefix = prefix;
            prefix.push(path.ident.to_string());
            expand_use_tree(prefix, &path.tree, output);
        }
        syn::UseTree::Name(name) => {
            let mut path = prefix;
            path.push(name.ident.to_string());
            output.push(path.join("::"));
        }
        syn::UseTree::Rename(rename) => {
            let mut path = prefix;
            path.push(rename.ident.to_string());
            output.push(path.join("::"));
        }
        syn::UseTree::Glob(_) => output.push(format!("{}::*", prefix.join("::"))),
        syn::UseTree::Group(group) => {
            for item in &group.items {
                expand_use_tree(prefix.clone(), item, output);
            }
        }
    }
}

#[test]
fn syntax_analysis_expands_grouped_imports_and_ignores_comments() {
    let paths = syntax_paths(
        r#"
        // use crate::commands::ignored;
        use crate::{commands::install, agents::{codex, claude as other}};
        "#,
    );
    assert!(paths.contains(&"crate::commands::install".to_string()));
    assert!(paths.contains(&"crate::agents::codex".to_string()));
    assert!(paths.contains(&"crate::agents::claude".to_string()));
    assert!(!paths.iter().any(|path| path.contains("ignored")));
}

#[test]
fn retired_top_level_agent_modules_do_not_return() {
    let src = source_root();
    for path in [
        "adapters",
        "alignment",
        "plugin_host",
        "plugin_install",
        "hermes.rs",
        "coding_agent.rs",
        "sidecar",
        "sidecar.rs",
    ] {
        assert!(!src.join(path).exists(), "retired module returned: {path}");
    }
}

#[test]
fn shared_services_do_not_depend_on_commands() {
    let src = source_root();
    for path in rust_files(&src) {
        if path.starts_with(src.join("commands")) || path == src.join("main.rs") {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        let paths = syntax_paths(&source);
        assert!(
            !paths.iter().any(|path| path.starts_with("crate::commands")),
            "shared module depends on command layer: {}",
            path.display()
        );
    }
}

#[test]
fn clap_syntax_is_owned_exclusively_by_commands() {
    let src = source_root();
    for path in rust_files(&src) {
        if path.starts_with(src.join("commands")) {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        let file = syn::parse_file(&source).unwrap();
        let mut visitor = PathVisitor::default();
        visitor.visit_file(&file);
        assert!(
            !visitor.paths.iter().any(|path| path.starts_with("clap"))
                && visitor.command_attributes.is_empty(),
            "{} contains command syntax",
            path.display()
        );
    }
}

#[test]
fn tests_are_not_embedded_in_the_source_tree() {
    let src = source_root();
    for path in rust_files(&src) {
        let source = fs::read_to_string(&path).unwrap();
        let file = syn::parse_file(&source).unwrap();
        let mut visitor = PathVisitor::default();
        visitor.visit_file(&file);
        assert!(
            !source.contains("#[cfg(test)]\nmod tests {")
                && !source.contains("#[cfg(test)]\r\nmod tests {")
                && visitor.test_attributes.is_empty(),
            "test body found under src instead of crates/cli/tests: {}",
            path.display()
        );
    }
}

#[test]
fn agent_directories_do_not_import_one_another_or_commands() {
    let agents = source_root().join("agents");
    for (agent, forbidden) in [("codex", ["agents::claude"]), ("claude", ["agents::codex"])] {
        for path in rust_files(&agents.join(agent)) {
            let source = fs::read_to_string(&path).unwrap();
            let paths = syntax_paths(&source);
            assert!(
                !paths.iter().any(|path| path.starts_with("crate::commands")),
                "{} imports commands",
                path.display()
            );
            for module in forbidden {
                assert!(
                    !paths.iter().any(|path| path.contains(module)),
                    "{} imports {module}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn retired_horizontal_and_monolithic_modules_do_not_return() {
    let src = source_root();
    for path in [
        "agents/install",
        "agents/host.rs",
        "agents/adapters.rs",
        "agents/alignment.rs",
        "commands/arguments.rs",
        "configuration/setup.rs",
    ] {
        assert!(!src.join(path).exists(), "retired module returned: {path}");
    }
}

#[test]
fn shared_installation_is_agent_neutral() {
    let installation = source_root().join("installation");
    for path in rust_files(&installation) {
        let source = fs::read_to_string(&path).unwrap();
        for marker in ["crate::agents", "CodingAgent", "IntegrationHost"] {
            assert!(
                !source.contains(marker),
                "{} contains host-selection marker {marker}",
                path.display()
            );
        }
    }
}

#[test]
fn all_target_is_command_only() {
    let src = source_root();
    for path in rust_files(&src) {
        if path.starts_with(src.join("commands")) {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        for marker in ["IntegrationHost", "InstallTarget", "CodingAgent::All"] {
            assert!(
                !source.contains(marker),
                "{} contains command target marker {marker}",
                path.display()
            );
        }
    }
}

const OPERATIONAL_LOG_TARGETS: &[&str] = &[
    "nemo_relay.logging",
    "nemo_relay.runtime",
    "nemo_relay.plugin",
    "nemo_relay.worker",
    "nemo_relay.observability",
    "nemo_relay.cli",
    "nemo_relay.configuration",
    "nemo_relay.server",
    "nemo_relay.gateway",
    "nemo_relay.session",
    "nemo_relay.agent",
    "nemo_relay.bootstrap",
    "nemo_relay.mcp",
    "nemo_relay.hook",
    "nemo_relay.installation",
    "nemo_relay.diagnostics",
    "nemo_relay.daemon",
    "nemo_relay.daemon.mcp",
    "nemo_relay.daemon.worker",
];

#[derive(Default)]
struct LogMacroVisitor {
    failures: Vec<String>,
}

impl<'ast> Visit<'ast> for LogMacroVisitor {
    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        let path = item
            .path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
        let macro_name = path.rsplit("::").next().unwrap_or_default();
        if matches!(macro_name, "info" | "warn" | "error" | "debug" | "trace") {
            let tokens = item.tokens.to_string();
            let target = string_field(&tokens, "target :");
            if target
                .as_deref()
                .is_none_or(|target| !OPERATIONAL_LOG_TARGETS.contains(&target))
            {
                self.failures.push(format!(
                    "{path}! has an invalid or missing target: {tokens}"
                ));
            }
            let event = string_field(&tokens, "event =");
            if event.as_deref().is_none_or(|event| !is_snake_case(event)) {
                self.failures
                    .push(format!("{path}! has an invalid or missing event: {tokens}"));
            }
            for field in [
                "error =",
                "payload =",
                "body =",
                "headers =",
                "credentials =",
                "configuration =",
                "environment =",
                "argv =",
            ] {
                if tokens.contains(field) {
                    self.failures.push(format!(
                        "{path}! uses privacy-sensitive field {field:?}: {tokens}"
                    ));
                }
            }
        }
        syn::visit::visit_macro(self, item);
    }
}

fn string_field(tokens: &str, marker: &str) -> Option<String> {
    let suffix = tokens.split_once(marker)?.1.trim_start();
    let suffix = suffix.strip_prefix('"')?;
    Some(suffix.split_once('"')?.0.to_string())
}

fn is_snake_case(value: &str) -> bool {
    !value.is_empty()
        && !value.starts_with('_')
        && !value.ends_with('_')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

#[test]
fn operational_log_calls_follow_the_stable_contract() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for src in [
        crate_root.join("src"),
        crate_root.join("../core/src"),
        crate_root.join("../adaptive/src"),
        crate_root.join("../pii-redaction/src"),
    ] {
        for path in rust_files(&src) {
            let source = fs::read_to_string(&path).unwrap();
            let file = syn::parse_file(&source).unwrap();
            let mut visitor = LogMacroVisitor::default();
            visitor.visit_file(&file);
            assert!(
                visitor.failures.is_empty(),
                "{}: {}",
                path.display(),
                visitor.failures.join("; ")
            );
        }
    }
}

#[test]
fn operational_direct_stderr_is_limited_to_emergency_and_ui_boundaries() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let allowed_cli = [
        "src/lib.rs",
        "src/commands/mod.rs",
        "src/agents/codex/launch.rs",
        "src/hooks/delivery.rs",
        "src/hooks/response.rs",
        "src/plugins/lifecycle/render.rs",
        "src/daemon/hook/mod.rs",
    ];
    for path in rust_files(&crate_root.join("src")) {
        let source = fs::read_to_string(&path).unwrap();
        if source.contains("eprintln!") {
            let relative = path.strip_prefix(crate_root).unwrap();
            assert!(
                allowed_cli
                    .iter()
                    .any(|allowed| relative == Path::new(allowed)),
                "operational direct stderr found in {}",
                path.display()
            );
        }
    }
    for path in rust_files(&crate_root.join("../core/src")) {
        let source = fs::read_to_string(&path).unwrap();
        assert!(
            !source.contains("eprintln!"),
            "core operational direct stderr found in {}",
            path.display()
        );
    }
    for (root, allowed) in [
        (
            crate_root.join("../adaptive/src"),
            ["src/acg/debug.rs"].as_slice(),
        ),
        (crate_root.join("../pii-redaction/src"), [].as_slice()),
    ] {
        for path in rust_files(&root) {
            let source = fs::read_to_string(&path).unwrap();
            if source.contains("eprintln!") {
                let relative = path.strip_prefix(&root).unwrap();
                assert!(
                    allowed
                        .iter()
                        .any(|allowed| relative == Path::new(allowed.trim_start_matches("src/"))),
                    "operational direct stderr found in {}",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn daemon_streaming_modules_do_not_use_response_aggregation_apis() {
    let src = source_root();
    for relative in [
        "daemon/broker/server.rs",
        "daemon/common/transport.rs",
        "daemon/worker/runtime.rs",
        "daemon/worker/managed.rs",
    ] {
        let path = src.join(relative);
        let source = fs::read_to_string(&path).unwrap();
        let file = syn::parse_file(&source).unwrap();
        let mut visitor = StreamingSourceVisitor::new(relative);
        visitor.visit_file(&file);
        assert!(
            visitor.violations.is_empty(),
            "daemon streaming architecture violations:\n{}",
            visitor.violations.join("\n")
        );
    }
}

#[test]
fn streaming_source_analysis_detects_formatted_aliased_and_manual_aggregation() {
    let fixture = syn::parse_file(
        r#"
        use axum::body::to_bytes as aggregate;
        async fn forward(body: Body) {
            let _ = body.collect()
                .await;
            let mut response = Vec::new();
            response.extend_from_slice(b"data");
            let _ = Body::from_stream(response);
            let _ = SseEventDecoder::new();
        }
        "#,
    )
    .unwrap();
    let mut visitor = StreamingSourceVisitor::new("daemon/broker/server.rs");
    visitor.visit_file(&fixture);
    assert!(
        visitor.violations.len() >= 5,
        "fixture escaped streaming architecture analysis: {:?}",
        visitor.violations
    );
}

#[test]
fn shared_runtime_subsystems_do_not_dispatch_host_variants() {
    let src = source_root();
    for subsystem in [
        "installation",
        "process",
        "configuration",
        "diagnostics",
        "gateway",
        "sessions",
        "hooks",
        "filesystem",
    ] {
        for path in rust_files(&src.join(subsystem)) {
            let source = fs::read_to_string(&path).unwrap();
            for marker in ["CodingAgent::Codex", "CodingAgent::ClaudeCode"] {
                assert!(
                    !source.contains(marker),
                    "{} dispatches host variant {marker}",
                    path.display()
                );
            }
        }
    }
}
