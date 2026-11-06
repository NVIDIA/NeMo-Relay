// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ELF candidate selection and dynamic string tokens from the running loader.

use std::path::Path;

#[derive(Clone, Copy)]
pub(super) struct Target {
    class: u8,
    encoding: u8,
    machine: u16,
}
impl Target {
    /// Read ELF class, byte order, and machine identity from a recognized image header.
    pub(super) fn from_header(bytes: &[u8]) -> Option<Self> {
        if bytes.get(..4)? != b"\x7fELF" || !matches!(*bytes.get(4)?, 1 | 2) {
            return None;
        }
        let machine = bytes.get(18..20)?.try_into().ok()?;
        Some(Self {
            class: bytes[4],
            encoding: bytes[5],
            machine: match bytes[5] {
                1 => u16::from_le_bytes(machine),
                2 => u16::from_be_bytes(machine),
                _ => return None,
            },
        })
    }

    pub(super) fn accepts(self, bytes: &[u8]) -> bool {
        let Some(other) = Self::from_header(bytes) else {
            // Malformed files remain errors when their contents are parsed.
            return true;
        };
        self.class == other.class
            && (self.encoding != other.encoding || self.machine == other.machine)
    }

    /// Check a regular dependency file for the ELF target identity required by its importing image.
    pub(super) fn accepts_file(self, path: &Path) -> Result<bool, String> {
        use std::io::Read;
        #[cfg(unix)]
        let file = {
            use rustix::fs::{Mode, OFlags, openat};
            std::fs::File::from(
                openat(
                    rustix::fs::CWD,
                    path,
                    OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK,
                    Mode::empty(),
                )
                .map_err(|error| error.to_string())?,
            )
        };
        #[cfg(not(unix))]
        let file = std::fs::File::open(path).map_err(|error| error.to_string())?;
        if !file
            .metadata()
            .map_err(|error| error.to_string())?
            .is_file()
        {
            return Err(format!(
                "native dependency '{}' must be a regular file",
                path.display()
            ));
        }
        let mut header = Vec::new();
        file.take(20)
            .read_to_end(&mut header)
            .map_err(|error| error.to_string())?;
        Ok(self.accepts(&header))
    }
}

#[derive(Debug)]
pub(super) struct Tokens {
    pub(super) lib: String,
    pub(super) platform: Option<String>,
}

impl Tokens {
    fn host() -> Result<&'static Self, String> {
        static TOKENS: std::sync::OnceLock<Result<Tokens, String>> = std::sync::OnceLock::new();
        TOKENS
            .get_or_init(host_tokens)
            .as_ref()
            .map_err(Clone::clone)
    }
}

/// Expand each token once, so dollar signs in the origin itself remain literal.
pub(super) fn expand(
    path: &str,
    origin: &str,
    tokens: Option<&Tokens>,
) -> Result<Option<String>, String> {
    let mut result = String::new();
    let mut remaining = path;
    while let Some((before, after)) = remaining.split_once('$') {
        result.push_str(before);
        let token = if let Some(braced) = after.strip_prefix('{') {
            braced.split_once('}')
        } else {
            let end = after
                .bytes()
                .position(|byte| !byte.is_ascii_alphanumeric() && byte != b'_')
                .unwrap_or(after.len());
            Some((&after[..end], &after[end..]))
        };
        match token {
            Some(("ORIGIN", tail)) => {
                result.push_str(origin);
                remaining = tail;
            }
            Some((name @ ("LIB" | "PLATFORM"), tail)) => {
                let values = match tokens {
                    Some(values) => values,
                    None => Tokens::host()?,
                };
                let value = if name == "LIB" {
                    Some(values.lib.as_str())
                } else {
                    values.platform.as_deref()
                };
                let Some(value) = value else { return Ok(None) };
                result.push_str(value);
                remaining = tail;
            }
            _ => {
                result.push('$');
                remaining = after;
            }
        }
    }
    result.push_str(remaining);
    Ok(Some(result))
}

#[cfg(any(target_os = "linux", test))]
const LIB_MARKER: &str = "/__nemo_relay_loader_lib__/";
#[cfg(any(target_os = "linux", test))]
const PLATFORM_MARKER: &str = "/__nemo_relay_loader_platform__/";
#[cfg(any(target_os = "linux", test))]
const END_MARKER: &str = "/__nemo_relay_loader_end__";

#[cfg(any(target_os = "linux", test))]
/// Decode loader token expansions and reject a missing or empty library-directory token.
fn parse_tokens(debug: &str) -> Result<Tokens, String> {
    let value = |marker: &str| {
        debug
            .split_once(marker)
            .and_then(|(_, tail)| tail.split_once(END_MARKER))
            .map(|(value, _)| value.to_owned())
    };
    let lib = value(LIB_MARKER)
        .filter(|value| !value.is_empty())
        .ok_or("cannot determine ELF loader $LIB expansion")?;
    Ok(Tokens {
        lib,
        platform: value(PLATFORM_MARKER),
    })
}

#[cfg(target_os = "linux")]
/// Determine the running ELF loader's `$LIB` and `$PLATFORM` expansions using the host executable interpreter.
fn host_tokens() -> Result<Tokens, String> {
    use object::read::elf::ProgramHeader as _;
    use std::os::unix::ffi::OsStrExt;
    // Read only the host's headers, not the entire executable or any plugin.
    let executable = format!("/proc/{}/exe", std::process::id());
    let data = object::read::ReadCache::new(
        std::fs::File::open(&executable).map_err(|error| error.to_string())?,
    );
    let file = object::File::parse(&data).map_err(|error| error.to_string())?;
    macro_rules! interpreter {
        ($file:expr) => {{
            let file = $file;
            let segment = file
                .elf_program_headers()
                .iter()
                .find(|segment| segment.p_type(file.endian()) == object::elf::PT_INTERP)
                .ok_or("ELF loader tokens require a dynamic executable")?;
            let bytes = segment
                .data(file.endian(), file.data())
                .map_err(|()| "cannot read ELF interpreter segment")?;
            let path =
                std::ffi::CStr::from_bytes_with_nul(bytes).map_err(|error| error.to_string())?;
            std::path::PathBuf::from(std::ffi::OsStr::from_bytes(path.to_bytes()))
        }};
    }
    let interpreter = match file {
        object::File::Elf32(ref file) => interpreter!(file),
        object::File::Elf64(ref file) => interpreter!(file),
        _ => return Err("ELF loader tokens require an ELF host".into()),
    };
    // --list does not execute the host program or library constructors. A missing
    // preload forces a search even when the host RPATH resolves every real import.
    // Debug output exposes exact tokens on glibc versions before --list-diagnostics.
    let output = std::process::Command::new(interpreter)
        .arg("--list")
        .arg(&executable)
        .env("LD_DEBUG", "libs")
        .env(
            "LD_LIBRARY_PATH",
            format!("{LIB_MARKER}$LIB{END_MARKER}:{PLATFORM_MARKER}$PLATFORM{END_MARKER}"),
        )
        .env_remove("LD_DEBUG_OUTPUT")
        .env("LD_PRELOAD", "__nemo_relay_loader_token_probe__.so")
        .env_remove("LD_AUDIT")
        .env_remove("LD_PROFILE")
        .env_remove("LD_PROFILE_OUTPUT")
        .output()
        .map_err(|error| format!("cannot query ELF loader tokens: {error}"))?;
    // A custom executable may need LD_LIBRARY_PATH to finish --list. Its first
    // dependency lookup still exposes the token values before that failure.
    parse_tokens(&String::from_utf8_lossy(&output.stderr))
}

#[cfg(not(target_os = "linux"))]
fn host_tokens() -> Result<Tokens, String> {
    Err("ELF loader tokens are unavailable on this host platform".into())
}

#[cfg(test)]
#[path = "../../tests/unit/native_elf_tests.rs"]
mod tests;
