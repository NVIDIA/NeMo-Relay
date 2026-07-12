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
