// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::io::Read;

fn main() {
    let hook_config = std::env::var_os("NEMO_RELAY_HOOK_CONFIG")
        .expect("NEMO_RELAY_HOOK_CONFIG is required");
    let expected = vec![
        OsString::from("hook-forward"),
        OsString::from("codex"),
        OsString::from("--hook-config"),
        hook_config,
        OsString::from("--fail-closed"),
    ];
    let actual = std::env::args_os().skip(1).collect::<Vec<_>>();
    if actual != expected {
        eprintln!("unexpected hook arguments: {actual:?}");
        std::process::exit(19);
    }

    if let Some(path) = std::env::var_os("NEMO_RELAY_HOOK_INPUT_MARKER") {
        let mut input = Vec::new();
        std::io::stdin().read_to_end(&mut input).unwrap();
        std::fs::write(path, input).unwrap();
    }
    if let Some(path) = std::env::var_os("NEMO_RELAY_HOOK_MARKER") {
        std::fs::write(path, "ok\n").unwrap();
    }
    if std::env::var_os("NEMO_RELAY_HOOK_EMIT_OUTPUT").is_some() {
        println!("hook-stdout");
        eprintln!("hook-stderr");
    }
    if let Ok(code) = std::env::var("NEMO_RELAY_HOOK_EXIT_CODE") {
        std::process::exit(code.parse().unwrap());
    }
}
