// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn loader_tokens_expand_once_and_respect_token_boundaries() {
    let tokens = Tokens {
        lib: "lib/x86_64-linux-gnu".into(),
        platform: Some("haswell".into()),
    };
    for path in ["$ORIGIN/$LIB/$PLATFORM", "${ORIGIN}/${LIB}/${PLATFORM}"] {
        assert_eq!(
            expand(path, "/package/$LIB", Some(&tokens))
                .unwrap()
                .unwrap(),
            "/package/$LIB/lib/x86_64-linux-gnu/haswell"
        );
    }
    assert_eq!(
        expand("$ORIGINATOR/$LIBRARY/${UNKNOWN}", "/package", Some(&tokens))
            .unwrap()
            .unwrap(),
        "$ORIGINATOR/$LIBRARY/${UNKNOWN}"
    );
    let tokens = Tokens {
        platform: None,
        ..tokens
    };
    assert!(
        expand("$ORIGIN/$PLATFORM", "/package", Some(&tokens))
            .unwrap()
            .is_none()
    );
}

#[test]
fn loader_debug_values_preserve_distribution_and_cpu_selections() {
    let debug = "123: search path=/__nemo_relay_loader_lib__/lib/x86_64-linux-gnu/__nemo_relay_loader_end__/glibc-hwcaps/x86-64-v3:/__nemo_relay_loader_platform__/haswell/__nemo_relay_loader_end__/tls (LD_LIBRARY_PATH)";
    let tokens = parse_tokens(debug).unwrap();
    assert_eq!(tokens.lib, "lib/x86_64-linux-gnu");
    assert_eq!(tokens.platform.as_deref(), Some("haswell"));
    assert!(parse_tokens("unrelated loader error").is_err());
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn running_glibc_loader_supplies_token_values() {
    let tokens = Tokens::host().unwrap();
    assert!(!tokens.lib.is_empty());
    assert!(!tokens.lib.contains('$'));
}

/// Candidate selection requires the importing image's class, byte order, and machine.
#[test]
fn elf_candidates_require_matching_class_encoding_and_machine() {
    let header = |class, encoding, machine: u16| {
        let mut bytes = [0; 20];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = class;
        bytes[5] = encoding;
        bytes[18..20].copy_from_slice(&match encoding {
            1 => machine.to_le_bytes(),
            2 => machine.to_be_bytes(),
            _ => unreachable!(),
        });
        bytes
    };
    for class in [1, 2] {
        for encoding in [1, 2] {
            let target = Target::from_header(&header(class, encoding, 62)).unwrap();
            assert!(target.accepts(&header(class, encoding, 62)));
            assert!(!target.accepts(&header(3 - class, encoding, 62)));
            assert!(!target.accepts(&header(class, 3 - encoding, 62)));
            assert!(!target.accepts(&header(class, encoding, 183)));
            assert!(!target.accepts(&header(class, 3 - encoding, 183)));
        }
    }
}
