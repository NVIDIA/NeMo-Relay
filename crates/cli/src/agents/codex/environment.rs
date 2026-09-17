// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Persistent header delivery through Codex's startup .env loader.
//!
//! Append an owned assignment rather than rewriting user dotenv syntax. Expansion preserves
//! the effective project, including values inherited from the launching shell. The gateway
//! unwraps the proof before forwarding that original project to the provider.

use std::fs;
use std::path::{Path, PathBuf};

use crate::configuration::BootstrapChallengeKey;
use crate::filesystem::{atomic_write_private, remove_file_preserving_symlink};
use crate::provider_auth::CODEX_CLIENT_PROOF_PREFIX;

const START: &str = "\n# >>> nemo-relay Codex authentication >>>\n";
const END: &str = "# <<< nemo-relay Codex authentication <<<\n";
const PRESENT: &str = "# original dotenv file: present\n";
const ABSENT: &str = "# original dotenv file: absent\n";

pub(super) fn path(config: &Path) -> PathBuf {
    config.with_file_name(".env")
}

fn read(path: &Path) -> Result<String, String> {
    match fs::read_to_string(path) {
        Ok(raw) => Ok(raw),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(format!("failed to read {}: {error}", path.display())),
    }
}

struct ManagedBlock<'a> {
    range: std::ops::Range<usize>,
    existed: bool,
    token: &'a str,
}

fn block(raw: &str) -> Result<Option<ManagedBlock<'_>>, String> {
    let invalid = || {
        "Codex .env has a modified or incomplete Relay authentication block; restore that block before retrying".to_string()
    };
    let Some(start) = raw.find(START) else {
        if raw.contains(END) || raw.contains(START.trim()) {
            return Err(invalid());
        }
        return Ok(None);
    };
    let tail = &raw[start + START.len()..];
    let end = tail.find(END).ok_or_else(invalid)?;
    let range = start..start + START.len() + end + END.len();
    if raw[range.end..].contains(START) || raw[range.end..].contains(END) {
        return Err(invalid());
    }
    let contents = &tail[..end];
    let (existed, assignment) = if let Some(assignment) = contents.strip_prefix(PRESENT) {
        (true, assignment)
    } else {
        (false, contents.strip_prefix(ABSENT).ok_or_else(invalid)?)
    };
    let prefix = format!("OPENAI_PROJECT=\"{CODEX_CLIENT_PROOF_PREFIX}");
    let token = assignment
        .strip_prefix(&prefix)
        .and_then(|value| value.strip_suffix(";${OPENAI_PROJECT}\"\n"))
        .ok_or_else(invalid)?;
    let hex = token.strip_prefix("hmac-sha256:").ok_or_else(invalid)?;
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid());
    }
    Ok(Some(ManagedBlock {
        range,
        existed,
        token,
    }))
}

fn remove_block(raw: &mut String, range: std::ops::Range<usize>) {
    // Keep a separator if the user appended settings after our block and the original file
    // had no trailing newline. With no later edits, restore the original bytes exactly.
    let separator = range.start > 0
        && range.end < raw.len()
        && !raw[..range.start].ends_with('\n')
        && !raw[range.end..].starts_with('\n');
    raw.replace_range(range, if separator { "\n" } else { "" });
}

pub(super) fn has_proof(config: &Path, key: &BootstrapChallengeKey) -> bool {
    read(&path(config)).ok().is_some_and(|raw| {
        block(&raw)
            .ok()
            .flatten()
            .is_some_and(|block| key.verify_client_token(block.token))
    })
}

pub(super) fn install(config: &Path, token: &str) -> Result<(), String> {
    let path = path(config);
    let mut raw = read(&path)?;
    let existed = if let Some(block) = block(&raw)? {
        let existed = block.existed;
        let range = block.range;
        remove_block(&mut raw, range);
        existed
    } else {
        path.exists()
    };
    // Do not allow malformed user quoting to swallow the appended assignment. Do not put
    // dotenv parser diagnostics in errors: they contain the input line, potentially a secret.
    if dotenvy::from_read_iter(raw.as_bytes()).any(|entry| entry.is_err()) {
        return Err(
            "Codex .env contains invalid dotenv syntax; repair it before installing Relay".into(),
        );
    }
    raw.push_str(START);
    raw.push_str(if existed { PRESENT } else { ABSENT });
    raw.push_str(&format!(
        "OPENAI_PROJECT=\"{CODEX_CLIENT_PROOF_PREFIX}{token};${{OPENAI_PROJECT}}\"\n"
    ));
    raw.push_str(END);
    write(&path, raw.as_bytes())
}

pub(super) fn uninstall(config: &Path) -> Result<(), String> {
    let path = path(config);
    let mut raw = read(&path)?;
    let Some(block) = block(&raw)? else {
        return Ok(());
    };
    let existed = block.existed;
    let range = block.range;
    remove_block(&mut raw, range);
    if raw.is_empty() && !existed {
        remove_file_preserving_symlink(&path)
    } else {
        write(&path, raw.as_bytes())
    }
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let target =
        if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            fs::canonicalize(path)
                .map_err(|error| format!("failed to resolve {}: {error}", path.display()))?
        } else {
            path.to_path_buf()
        };
    atomic_write_private(&target, bytes)
}

#[cfg(test)]
#[path = "../../../tests/coverage/agents/codex_environment_tests.rs"]
mod tests;
