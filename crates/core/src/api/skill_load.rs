// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private detection helpers for eager skill-load observability marks.

use std::collections::HashSet;

use serde_json::Value;
use strum::{EnumString, IntoStaticStr};

pub(crate) const HANDLED_METADATA_KEY: &str = "nemo_relay.skill_load_handled";
pub(crate) const PRECOMPUTED_METADATA_KEY: &str = "nemo_relay.skill_loads";

const CAT_REJECTED_OPTIONS: &[&str] = &["-h", "--help", "--version"];
const BAT_REJECTED_OPTIONS: &[&str] = &[
    "-h",
    "--help",
    "--version",
    "-r",
    "--line-range",
    "--list-languages",
    "--list-themes",
    "--generate-completion",
    "--diagnostic",
    "--acknowledgements",
];
const POWERSHELL_NON_READ_OPTIONS: &[&str] = &["-?", "-help", "--help"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumString, IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(crate) enum SkillLoadSource {
    SkillTool,
    StructuredRead,
    ShellRead,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SkillLoad {
    pub(crate) name: String,
    pub(crate) source: SkillLoadSource,
}

pub(crate) fn detect(tool_name: &str, args: &Value) -> Vec<SkillLoad> {
    let normalized_tool = normalize_identifier(tool_name);
    let source_and_names = if matches!(normalized_tool.as_str(), "skill" | "skillview") {
        (SkillLoadSource::SkillTool, skill_tool_names(args))
    } else if is_structured_reader(tool_name) && !has_partial_read_controls(args) {
        (
            SkillLoadSource::StructuredRead,
            structured_skill_names(args),
        )
    } else if is_shell_tool(&normalized_tool) {
        (
            SkillLoadSource::ShellRead,
            shell_skill_names(&normalized_tool, args),
        )
    } else {
        return Vec::new();
    };

    deduplicate(source_and_names.1)
        .into_iter()
        .map(|name| SkillLoad {
            name,
            source: source_and_names.0,
        })
        .collect()
}

pub(crate) fn precomputed(metadata: Option<&Value>) -> Option<Vec<SkillLoad>> {
    let entries = metadata?
        .as_object()?
        .get(PRECOMPUTED_METADATA_KEY)?
        .as_array()?;
    let mut seen = HashSet::new();
    Some(
        entries
            .iter()
            .filter_map(|entry| {
                let entry = entry.as_object()?;
                let name = entry.get("skill_name")?.as_str()?.trim();
                let source = entry.get("source")?.as_str()?.parse().ok()?;
                (!name.is_empty() && seen.insert(name.to_string())).then(|| SkillLoad {
                    name: name.to_string(),
                    source,
                })
            })
            .collect(),
    )
}

fn skill_tool_names(args: &Value) -> Vec<String> {
    let mut names = Vec::new();
    visit_named_values(args, |key, value| {
        let Some(value) = value.as_str() else {
            return;
        };
        if matches!(key.as_str(), "skill" | "skillname" | "name") {
            let value = value.trim();
            if !value.is_empty() {
                names.push(value.to_string());
            }
        }
    });
    names
}

fn structured_skill_names(args: &Value) -> Vec<String> {
    let mut names = Vec::new();
    visit_named_values(args, |key, value| {
        if matches!(
            key.as_str(),
            "path" | "filepath" | "filename" | "file" | "paths" | "uri" | "absolutepath"
        ) {
            collect_path_skill_names(value, &mut names);
        }
    });
    names
}

fn shell_skill_names(tool_name: &str, args: &Value) -> Vec<String> {
    let mut commands = Vec::new();
    visit_named_values(args, |key, value| {
        if matches!(key.as_str(), "command" | "cmd")
            && let Some(value) = value.as_str()
        {
            commands.push(value.to_string());
        }
    });
    let allow_direct_cat = !matches!(tool_name, "powershell" | "pwsh");
    commands
        .into_iter()
        .flat_map(|command| complete_reader_paths(&command, allow_direct_cat))
        .filter_map(|path| skill_name_from_path(&path))
        .collect()
}

fn is_structured_reader(tool_name: &str) -> bool {
    const READERS: [&str; 8] = [
        "read",
        "readfile",
        "readtextfile",
        "readmultiplefiles",
        "fileread",
        "readresource",
        "getfilecontents",
        "readfilecontent",
    ];
    let segments = tool_name
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|segment| !segment.is_empty())
        .map(normalize_identifier)
        .collect::<Vec<_>>();
    (1..=segments.len())
        .any(|length| READERS.contains(&segments[segments.len() - length..].concat().as_str()))
}

fn is_shell_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "sh" | "bash"
            | "zsh"
            | "fish"
            | "shell"
            | "shellcommand"
            | "exec"
            | "execcommand"
            | "execute"
            | "terminal"
            | "runcommand"
            | "runshellcommand"
            | "shellexec"
            | "powershell"
            | "pwsh"
    )
}

fn has_partial_read_controls(value: &Value) -> bool {
    let mut partial = false;
    visit_named_values(value, |key, value| {
        if !partial {
            let key = normalize_identifier(&key);
            partial = match key.as_str() {
                "offset" => value.as_i64() != Some(0),
                "limit" | "range" | "head" | "tail" | "startline" | "endline" | "linestart"
                | "lineend" => !value.is_null(),
                _ => false,
            };
        }
    });
    partial
}

fn collect_path_skill_names(value: &Value, names: &mut Vec<String>) {
    match value {
        Value::String(path) => {
            if let Some(name) = skill_name_from_path(path) {
                names.push(name);
            }
        }
        Value::Array(values) => values
            .iter()
            .for_each(|value| collect_path_skill_names(value, names)),
        _ => {}
    }
}

fn visit_named_values(value: &Value, mut visit: impl FnMut(String, &Value)) {
    let mut stack = vec![value];
    while let Some(value) = stack.pop() {
        match value {
            Value::Object(object) => {
                for (key, value) in object {
                    visit(normalize_identifier(key), value);
                    stack.push(value);
                }
            }
            Value::Array(values) => stack.extend(values.iter()),
            _ => {}
        }
    }
}

fn skill_name_from_path(path: &str) -> Option<String> {
    let path = path.trim_matches(['\'', '"']);
    if path.trim() != path || path.ends_with('/') || path.ends_with('\\') {
        return None;
    }
    let components = path
        .split(['/', '\\'])
        .filter(|component| !component.is_empty())
        .collect::<Vec<_>>();
    let [.., parent, file] = components.as_slice() else {
        return None;
    };
    if !file.eq_ignore_ascii_case("SKILL.md")
        || matches!(*parent, "." | "..")
        || parent.ends_with(':')
    {
        return None;
    }
    Some((*parent).to_string())
}

fn complete_reader_paths(command: &str, allow_direct_cat: bool) -> Vec<String> {
    let Some(words) = standalone_command_words(command) else {
        return Vec::new();
    };
    complete_reader_word_paths(&words, allow_direct_cat)
}

fn standalone_command_words(command: &str) -> Option<Vec<String>> {
    if command.contains(['\n', '\r']) {
        return None;
    }
    // Preserve Windows separators: shell-words treats a lone backslash as an escape.
    let escaped_windows_paths = command.replace('\\', "\\\\");
    let words = shell_words::split(&escaped_windows_paths).ok()?;
    if words.is_empty()
        || words.iter().any(|word| {
            word.contains(['|', '&', ';', '<', '>']) || word.contains("$(") || word.contains('`')
        })
    {
        return None;
    }
    Some(words)
}

fn complete_reader_word_paths(words: &[String], allow_direct_cat: bool) -> Vec<String> {
    let Some(executable) = words.first().and_then(|word| executable_name(word)) else {
        return Vec::new();
    };
    match executable.as_str() {
        "cat" if allow_direct_cat => positional_paths(&words[1..], CAT_REJECTED_OPTIONS),
        "bat" | "batcat" => positional_paths(&words[1..], BAT_REJECTED_OPTIONS),
        "get-content" => powershell_content_paths(&words[1..]),
        "sh" | "bash" | "zsh" => posix_shell_wrapper_paths(&executable, &words[1..]),
        "fish" => fish_shell_wrapper_paths(&words[1..]),
        "powershell" | "pwsh" => powershell_wrapper_paths(&words[1..]),
        _ => Vec::new(),
    }
}

fn posix_shell_wrapper_paths(executable: &str, words: &[String]) -> Vec<String> {
    let [flag, command] = words else {
        return Vec::new();
    };
    if !matches!(
        (executable, flag.as_str()),
        ("sh", "-c") | ("bash" | "zsh", "-c" | "-lc")
    ) {
        return Vec::new();
    }
    let Some(command_words) = standalone_command_words(command) else {
        return Vec::new();
    };
    posix_reader_word_paths(&command_words)
}

fn fish_shell_wrapper_paths(words: &[String]) -> Vec<String> {
    let command = match words {
        [flag, command] if flag == "-c" => command.as_str(),
        [argument] => argument.strip_prefix("--command=").unwrap_or_default(),
        _ => return Vec::new(),
    };
    let Some(command_words) = standalone_command_words(command) else {
        return Vec::new();
    };
    posix_reader_word_paths(&command_words)
}

fn powershell_wrapper_paths(words: &[String]) -> Vec<String> {
    let [flag, command @ ..] = words else {
        return Vec::new();
    };
    if !flag.eq_ignore_ascii_case("-command") || command.is_empty() {
        return Vec::new();
    }
    if let [command] = command {
        let Some(command_words) = standalone_command_words(command) else {
            return Vec::new();
        };
        return powershell_reader_word_paths(&command_words);
    }
    powershell_reader_word_paths(command)
}

fn posix_reader_word_paths(words: &[String]) -> Vec<String> {
    let Some(executable) = words.first().and_then(|word| executable_name(word)) else {
        return Vec::new();
    };
    match executable.as_str() {
        "cat" => positional_paths(&words[1..], CAT_REJECTED_OPTIONS),
        "bat" | "batcat" => positional_paths(&words[1..], BAT_REJECTED_OPTIONS),
        _ => Vec::new(),
    }
}

fn powershell_reader_word_paths(words: &[String]) -> Vec<String> {
    let Some(executable) = words.first().and_then(|word| executable_name(word)) else {
        return Vec::new();
    };
    if executable != "get-content" {
        return Vec::new();
    }
    powershell_content_paths(&words[1..])
}

fn positional_paths(words: &[String], rejected_flags: &[&str]) -> Vec<String> {
    if words.iter().any(|word| {
        rejected_flags
            .iter()
            .any(|flag| word.eq_ignore_ascii_case(flag) || word.starts_with(&format!("{flag}=")))
    }) {
        return Vec::new();
    }
    words
        .iter()
        .filter(|word| !word.starts_with('-'))
        .cloned()
        .collect()
}

fn powershell_content_paths(words: &[String]) -> Vec<String> {
    if words.iter().any(|word| {
        let candidate = word
            .split([':', '='])
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        POWERSHELL_NON_READ_OPTIONS
            .iter()
            .any(|option| candidate.eq_ignore_ascii_case(option))
            || (candidate.len() > 1
                && ["-totalcount", "-tail", "-head", "-first", "-last"]
                    .iter()
                    .any(|flag| flag.starts_with(&candidate)))
    }) {
        return Vec::new();
    }

    let mut paths = Vec::new();
    let mut index = 0;
    while index < words.len() {
        let word = &words[index];
        if word.eq_ignore_ascii_case("-path") || word.eq_ignore_ascii_case("-literalpath") {
            if let Some(path) = words.get(index + 1) {
                paths.push(path.clone());
            }
            index += 2;
            continue;
        }
        if !word.starts_with('-') {
            paths.push(word.clone());
        }
        index += 1;
    }
    paths
}

fn executable_name(executable: &str) -> Option<String> {
    executable
        .rsplit(['/', '\\'])
        .next()
        .map(|name| name.to_ascii_lowercase())
        .map(|name| name.strip_suffix(".exe").unwrap_or(&name).to_string())
        .filter(|name| !name.is_empty())
}

fn normalize_identifier(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn deduplicate(names: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    names
        .into_iter()
        .filter(|name| seen.insert(name.clone()))
        .collect()
}

#[cfg(test)]
#[path = "../../tests/unit/skill_load_tests.rs"]
mod tests;
