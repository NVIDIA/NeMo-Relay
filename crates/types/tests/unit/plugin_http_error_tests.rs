// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn http_error_strings_round_trip_every_error_status() {
    for status in 400..=599 {
        for body in ["", "denied: secret redacted", "échec\nmore context"] {
            let message = http_error_message(status, body).unwrap();
            assert_eq!(parse_http_error_message(&message), Some((status, body)));
        }
    }
}

#[test]
fn malformed_or_embedded_http_error_strings_are_not_recognized() {
    for message in [
        "http 403: denied",
        " HTTP 403: denied",
        "wrapped: HTTP 403: denied",
        "HTTP 403:denied",
        "HTTP 0403: denied",
        "HTTP +403: denied",
        "HTTP ４０３: denied",
        "HTTP 399: denied",
        "HTTP 600: denied",
        "HTTP 403",
        "plain failure",
    ] {
        assert_eq!(parse_http_error_message(message), None, "{message}");
    }
    for status in [0, 200, 399, 600, u16::MAX] {
        assert!(http_error_message(status, "invalid").is_err());
    }
}
