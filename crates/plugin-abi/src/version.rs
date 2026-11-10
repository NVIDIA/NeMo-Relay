// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fmt, str::FromStr};

/// A native interface generation, independent of the Relay package version.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NativeApiVersion {
    /// Breaking interface generation.
    pub major: u32,
    /// Minimum host interface required by a plugin's build SDK.
    pub minor: u32,
}

impl NativeApiVersion {
    /// Construct an interface version.
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Whether a host implementing `self` supports the plugin's build SDK.
    pub const fn supports(self, plugin: Self) -> bool {
        self.major == plugin.major && self.minor >= plugin.minor
    }
}

impl fmt::Display for NativeApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}

/// Invalid or legacy authored native API identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseNativeApiVersionError {
    /// The pre-stabilization contract requires rebuilding.
    Legacy,
    /// Expected canonical `v<major>.<minor>` notation.
    Invalid,
}

impl fmt::Display for ParseNativeApiVersionError {
    /// Explain why an authored native API identifier cannot be used.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Legacy => "native API 1 is obsolete; rebuild the plugin for v1.0",
            Self::Invalid => "expected native API v<major>.<minor>, such as v1.0",
        })
    }
}
impl std::error::Error for ParseNativeApiVersionError {}

impl FromStr for NativeApiVersion {
    type Err = ParseNativeApiVersionError;
    /// Parse canonical `v<major>.<minor>` notation, rejecting invalid numbers and unsupported notation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        use ParseNativeApiVersionError::{Invalid, Legacy};
        if value == "1" {
            return Err(Legacy);
        }
        let (major, minor) = value
            .strip_prefix('v')
            .and_then(|v| v.split_once('.'))
            .ok_or(Invalid)?;
        let parse = |part: &str| {
            if part.is_empty()
                || !part.bytes().all(|b| b.is_ascii_digit())
                || (part.len() > 1 && part.starts_with('0'))
            {
                return Err(Invalid);
            }
            part.parse::<u32>().map_err(|_| Invalid)
        };
        let major = parse(major)?;
        if major == 0 {
            return Err(Invalid);
        }
        Ok(Self::new(major, parse(minor)?))
    }
}

include!(concat!(env!("OUT_DIR"), "/native_api.rs"));
