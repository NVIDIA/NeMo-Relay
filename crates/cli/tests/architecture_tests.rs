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
        let name = attribute
            .path()
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_default();
        if matches!(name.as_str(), "arg" | "command" | "value") {
            self.command_attributes.push(name);
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
        use crate::{commands::install, agents::{codex, hermes as other}};
        "#,
    );
    assert!(paths.contains(&"crate::commands::install".to_string()));
    assert!(paths.contains(&"crate::agents::codex".to_string()));
    assert!(paths.contains(&"crate::agents::hermes".to_string()));
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
        assert!(
            !source.contains("#[cfg(test)]\nmod tests {")
                && !source.contains("#[cfg(test)]\r\nmod tests {"),
            "inline test module found under src: {}",
            path.display()
        );
    }
}

#[test]
fn agent_directories_do_not_import_one_another_or_commands() {
    let agents = source_root().join("agents");
    for (agent, forbidden) in [
        ("codex", ["agents::claude", "agents::hermes"]),
        ("claude", ["agents::codex", "agents::hermes"]),
        ("hermes", ["agents::codex", "agents::claude"]),
    ] {
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
            for marker in [
                "CodingAgent::Codex",
                "CodingAgent::ClaudeCode",
                "CodingAgent::Hermes",
            ] {
                assert!(
                    !source.contains(marker),
                    "{} dispatches host variant {marker}",
                    path.display()
                );
            }
        }
    }
}
