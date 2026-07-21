// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::{SkillLoad, SkillLoadSource, detect, precomputed};

fn expected(names: &[&str], source: SkillLoadSource) -> Vec<SkillLoad> {
    names
        .iter()
        .map(|name| SkillLoad {
            name: (*name).to_string(),
            source,
        })
        .collect()
}

fn assert_detects(tool: &str, args: Value, names: &[&str], source: SkillLoadSource) {
    assert_eq!(
        detect(tool, &args),
        expected(names, source),
        "tool={tool}, args={args}"
    );
}

fn assert_rejected(tool: &str, args: Value) {
    assert!(detect(tool, &args).is_empty(), "tool={tool}, args={args}");
}

#[test]
fn first_class_tools_accept_every_supported_name_field() {
    for (tool, args, name) in [
        ("Skill", json!({"skill": "review"}), "review"),
        ("skill_view", json!({"skill_name": "testing"}), "testing"),
        (
            "skill-view",
            json!({"name": "mlops/axolotl"}),
            "mlops/axolotl",
        ),
        (
            "skill_view",
            json!({"request": {"name": "nested-skill"}}),
            "nested-skill",
        ),
    ] {
        assert_detects(tool, args, &[name], SkillLoadSource::SkillTool);
    }
}

#[test]
fn first_class_tools_reject_missing_empty_and_non_string_names() {
    for (tool, args) in [
        ("Skill", json!({})),
        ("Skill", json!({"skill": "  "})),
        ("skill_view", json!({"name": 7})),
        ("skill_catalog", json!({"skill": "review"})),
    ] {
        assert_rejected(tool, args);
    }
}

#[test]
fn structured_readers_accept_every_supported_tool_and_path_field() {
    for (tool, args) in [
        ("Read", json!({"path": "/skills/review/SKILL.md"})),
        ("read_file", json!({"file_path": "/skills/review/SKILL.md"})),
        (
            "read_text_file",
            json!({"filepath": "/skills/review/SKILL.md"}),
        ),
        ("file_read", json!({"filename": "/skills/review/SKILL.md"})),
        (
            "mcp__filesystem__read_file",
            json!({"file": "/skills/review/SKILL.md"}),
        ),
        (
            "mcp__filesystem__read_multiple_files",
            json!({"paths": ["/skills/review/SKILL.md"]}),
        ),
        (
            "mcp__resources__read_resource",
            json!({"uri": "file:///skills/review/SKILL.md"}),
        ),
        (
            "mcp__resources__read_resource",
            json!({"path": "/skills/review/SKILL.md"}),
        ),
        (
            "mcp__filesystem__get_file_contents",
            json!({"path": "/skills/review/SKILL.md"}),
        ),
        (
            "mcp__filesystem__read_file_content",
            json!({"absolute_path": "/skills/review/SKILL.md"}),
        ),
        (
            "read_file",
            json!({"uri": "skill://catalog/review/SKILL.md"}),
        ),
    ] {
        assert_detects(tool, args, &["review"], SkillLoadSource::StructuredRead);
    }
}

#[test]
fn structured_readers_support_posix_windows_relative_nested_and_duplicate_paths() {
    assert_detects(
        "read_multiple_files",
        json!({
            "request": {
                "paths": [
                    "/workspace/.agents/skills/review/SKILL.md",
                    "C:\\Users\\me\\.codex\\skills\\test-runner\\skill.MD",
                    "relative/skills/authoring/SKILL.md",
                    "/workspace/.agents/skills/review/SKILL.md"
                ]
            }
        }),
        &["review", "test-runner", "authoring"],
        SkillLoadSource::StructuredRead,
    );
}

#[test]
fn structured_readers_allow_zero_offset_but_reject_every_partial_read_control() {
    assert_detects(
        "Read",
        json!({"file_path": "/skills/review/SKILL.md", "offset": 0}),
        &["review"],
        SkillLoadSource::StructuredRead,
    );

    for args in [
        json!({"file_path": "/skills/review/SKILL.md", "offset": 1}),
        json!({"file_path": "/skills/review/SKILL.md", "offset": -1}),
        json!({"file_path": "/skills/review/SKILL.md", "limit": 2000}),
        json!({"file_path": "/skills/review/SKILL.md", "range": "1:20"}),
        json!({"file_path": "/skills/review/SKILL.md", "head": true}),
        json!({"file_path": "/skills/review/SKILL.md", "tail": 20}),
        json!({"file_path": "/skills/review/SKILL.md", "start_line": 1}),
        json!({"file_path": "/skills/review/SKILL.md", "end_line": 20}),
        json!({"file_path": "/skills/review/SKILL.md", "line_start": 1}),
        json!({"file_path": "/skills/review/SKILL.md", "line_end": 20}),
        json!({
            "file_path": "/skills/review/SKILL.md",
            "options": {"limit": 20}
        }),
    ] {
        assert_rejected("Read", args);
    }
}

#[test]
fn structured_readers_reject_non_skill_paths_missing_parents_and_non_read_tools() {
    for (tool, args) in [
        ("Read", json!({"file_path": "/skills/review/README.md"})),
        ("Read", json!({"file_path": "SKILL.md"})),
        ("Read", json!({"file_path": "/SKILL.md"})),
        ("Read", json!({"file_path": "C:\\SKILL.md"})),
        ("Read", json!({"file_path": "/skills/./SKILL.md"})),
        ("Read", json!({"file_path": "/skills/../SKILL.md"})),
        (
            "write_file",
            json!({"file_path": "/skills/review/SKILL.md"}),
        ),
        ("edit_file", json!({"file_path": "/skills/review/SKILL.md"})),
        ("list_directory", json!({"path": "/skills/review/SKILL.md"})),
        ("thread", json!({"path": "/skills/review/SKILL.md"})),
        ("spread", json!({"path": "/skills/review/SKILL.md"})),
        ("unread", json!({"path": "/skills/review/SKILL.md"})),
        (
            "mcp__resources__list_resource",
            json!({"uri": "file:///skills/review/SKILL.md"}),
        ),
        (
            "mcp__filesystem__download_file_contents",
            json!({"absolute_path": "/skills/review/SKILL.md"}),
        ),
        (
            "mcp__filesystem__read_file_contents",
            json!({"path": "/skills/review/SKILL.md"}),
        ),
    ] {
        assert_rejected(tool, args);
    }
}

#[test]
fn new_structured_reader_schemas_reject_partial_controls_and_malformed_uris() {
    for (tool, args) in [
        (
            "mcp__resources__read_resource",
            json!({"uri": "file:///skills/review/SKILL.md", "limit": 2000}),
        ),
        (
            "read_file",
            json!({"absolute_path": "/skills/review/SKILL.md", "offset": 1}),
        ),
        (
            "mcp__filesystem__get_file_contents",
            json!({"path": "/skills/review/SKILL.md", "range": "1:20"}),
        ),
        (
            "mcp__resources__read_resource",
            json!({"uri": "file:///skills/review/SKILL.md?raw=true"}),
        ),
        (
            "mcp__resources__read_resource",
            json!({"uri": "file:///SKILL.md"}),
        ),
    ] {
        assert_rejected(tool, args);
    }
}

#[test]
fn shell_detection_accepts_complete_cat_bat_and_batcat_commands() {
    for (tool, command, names) in [
        (
            "Bash",
            "cat -n '/workspace/skills/review/SKILL.md'",
            &["review"][..],
        ),
        (
            "exec_command",
            "/usr/bin/bat --plain C:\\skills\\testing\\SKILL.md",
            &["testing"][..],
        ),
        (
            "terminal",
            "\"C:\\Tools\\bat.exe\" --plain C:\\skills\\review\\SKILL.md",
            &["review"][..],
        ),
        (
            "run_shell_command",
            "batcat /skills/review/SKILL.md /skills/testing/SKILL.md /skills/review/SKILL.md",
            &["review", "testing"][..],
        ),
    ] {
        assert_detects(
            tool,
            json!({"command": command}),
            names,
            SkillLoadSource::ShellRead,
        );
    }
}

#[test]
fn shell_detection_accepts_complete_powershell_get_content_forms() {
    for command in [
        "Get-Content -Raw -LiteralPath 'C:\\skills\\review\\SKILL.md'",
        "Get-Content -Encoding utf8 -Path C:\\skills\\review\\SKILL.md",
        "C:\\Windows\\System32\\Get-Content.exe C:\\skills\\review\\SKILL.md",
    ] {
        assert_detects(
            "powershell",
            json!({"cmd": command}),
            &["review"],
            SkillLoadSource::ShellRead,
        );
    }
}

#[test]
fn shell_detection_accepts_exact_complete_reader_wrappers() {
    for (tool, command) in [
        (
            "exec_command",
            "sh -c 'cat /workspace/skills/review/SKILL.md'",
        ),
        (
            "shell",
            "bash -c 'bat --plain /workspace/skills/review/SKILL.md'",
        ),
        ("Bash", "bash -lc 'cat /workspace/skills/review/SKILL.md'"),
        (
            "exec_command",
            "zsh -c 'cat /workspace/skills/review/SKILL.md'",
        ),
        (
            "terminal",
            "zsh -lc 'batcat --plain /workspace/skills/review/SKILL.md'",
        ),
        ("shell", "fish -c 'cat /workspace/skills/review/SKILL.md'"),
        (
            "exec_command",
            "fish --command='bat --plain /workspace/skills/review/SKILL.md'",
        ),
        (
            "terminal",
            "powershell -Command Get-Content -Raw -Path C:\\skills\\review\\SKILL.md",
        ),
        (
            "exec_command",
            r#"pwsh -Command "Get-Content -Raw -LiteralPath 'C:\skills\review\SKILL.md'""#,
        ),
        ("sh", "cat /workspace/skills/review/SKILL.md"),
        ("zsh", "cat /workspace/skills/review/SKILL.md"),
        ("fish", "cat /workspace/skills/review/SKILL.md"),
        (
            "pwsh",
            "Get-Content -Raw -Path C:\\skills\\review\\SKILL.md",
        ),
    ] {
        assert_detects(
            tool,
            json!({"command": command}),
            &["review"],
            SkillLoadSource::ShellRead,
        );
    }
}

#[test]
fn shell_detection_rejects_partial_transformed_and_compound_commands() {
    for command in [
        "sed -n '1,200p' /skills/review/SKILL.md",
        "head /skills/review/SKILL.md",
        "tail /skills/review/SKILL.md",
        "bat -r 1:20 /skills/review/SKILL.md",
        "bat --line-range 1:20 /skills/review/SKILL.md",
        "bat --line-range=1:20 /skills/review/SKILL.md",
        "Get-Content -TotalCount 20 /skills/review/SKILL.md",
        "Get-Content -Tail 20 /skills/review/SKILL.md",
        "Get-Content -Head 20 /skills/review/SKILL.md",
        "Get-Content -First 20 /skills/review/SKILL.md",
        "Get-Content -Last 20 /skills/review/SKILL.md",
        "cat /skills/review/SKILL.md | head",
        "cat /skills/review/SKILL.md > /tmp/copy",
        "cat /skills/review/SKILL.md < /tmp/input",
        "cat /skills/review/SKILL.md && echo done",
        "cat /skills/review/SKILL.md || echo failed",
        "cat /skills/review/SKILL.md; echo done",
        "cat /skills/review/SKILL.md\necho done",
        "cat $(find /skills -name SKILL.md)",
        "cat `find /skills -name SKILL.md`",
    ] {
        assert_rejected("shell", json!({"command": command}));
    }
}

#[test]
fn shell_detection_rejects_unsupported_wrapper_forms_and_aliases() {
    for command in [
        "sh -lc 'cat /skills/review/SKILL.md'",
        "bash -lc 'sh -c \"cat /skills/review/SKILL.md\"'",
        "zsh -c 'bash -c \"cat /skills/review/SKILL.md\"'",
        "fish -c 'zsh -c \"cat /skills/review/SKILL.md\"'",
        "bash -lc 'cat /skills/review/SKILL.md | head'",
        "fish -c 'cat /skills/review/SKILL.md | head'",
        "bash -lc 'cat /skills/review/SKILL.md > /tmp/copy'",
        "bash -lc 'cat $(find /skills -name SKILL.md)'",
        "bash -lc 'cat /skills/review/SKILL.md' ignored",
        "powershell -NoProfile -Command Get-Content C:\\skills\\review\\SKILL.md",
        "powershell -Command \"Get-Content C:\\skills\\review\\SKILL.md; Write-Host done\"",
        "powershell -Command gc C:\\skills\\review\\SKILL.md",
        "pwsh -Command cat C:\\skills\\review\\SKILL.md",
        "cmd /c type C:\\skills\\review\\SKILL.md",
    ] {
        assert_rejected("shell", json!({"command": command}));
    }
}

#[test]
fn shell_detection_rejects_malformed_unknown_and_non_command_inputs() {
    for (tool, args) in [
        ("shell", json!({"command": "cat '/skills/review/SKILL.md"})),
        (
            "shell",
            json!({"command": "cp /skills/review/SKILL.md /tmp"}),
        ),
        ("shell", json!({"command": ""})),
        ("shell", json!({"command": 7})),
        ("shell", json!({"script": "cat /skills/review/SKILL.md"})),
        ("python", json!({"command": "cat /skills/review/SKILL.md"})),
        (
            "powershell",
            json!({"cmd": "cat C:\\skills\\review\\SKILL.md"}),
        ),
        (
            "pwsh",
            json!({"command": "gc C:\\skills\\review\\SKILL.md"}),
        ),
    ] {
        assert_rejected(tool, args);
    }
}

#[test]
fn precomputed_detections_validate_sources_names_and_deduplicate() {
    assert_eq!(
        precomputed(Some(&json!({
            "nemo_relay.skill_loads": [
                {"skill_name": "review", "source": "structured_read"},
                {"skill_name": "testing", "source": "skill_tool"},
                {"skill_name": "authoring", "source": "shell_read"},
                {"skill_name": "review", "source": "structured_read"},
                {"skill_name": "", "source": "shell_read"},
                {"skill_name": "ignored", "source": "unknown"},
                {"source": "shell_read"},
                "malformed"
            ]
        }))),
        Some(vec![
            SkillLoad {
                name: "review".into(),
                source: SkillLoadSource::StructuredRead,
            },
            SkillLoad {
                name: "testing".into(),
                source: SkillLoadSource::SkillTool,
            },
            SkillLoad {
                name: "authoring".into(),
                source: SkillLoadSource::ShellRead,
            },
        ])
    );
}

#[test]
fn precomputed_detections_reject_missing_and_malformed_envelopes() {
    for metadata in [
        None,
        Some(json!(null)),
        Some(json!([])),
        Some(json!({})),
        Some(json!({"nemo_relay.skill_loads": {}})),
    ] {
        assert_eq!(precomputed(metadata.as_ref()), None);
    }
    assert_eq!(
        precomputed(Some(&json!({"nemo_relay.skill_loads": []}))),
        Some(Vec::new())
    );
}
