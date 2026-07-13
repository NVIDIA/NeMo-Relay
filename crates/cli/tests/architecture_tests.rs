// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Architectural dependency and source-layout regression tests.

use std::fs;
use std::path::{Path, PathBuf};

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
        assert!(
            !source.contains("crate::commands"),
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
        for marker in [
            "use clap::",
            "clap::Parser",
            "#[arg(",
            "#[command(",
            "#[value(",
        ] {
            assert!(
                !source.contains(marker),
                "{} contains command syntax marker {marker}",
                path.display()
            );
        }
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
            assert!(
                !source.contains("crate::commands"),
                "{} imports commands",
                path.display()
            );
            for module in forbidden {
                assert!(
                    !source.contains(module),
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
