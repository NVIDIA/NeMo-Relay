// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use http::HeaderValue;

#[test]
fn accepts_only_one_unambiguous_hyphenated_uuid() {
    const ID: &str = "b09d60fd-1234-4567-abcd-123456789abc";
    for (values, expected) in [
        (vec![], None),
        (vec![ID], Some(ID)),
        (vec!["B09D60FD-1234-4567-ABCD-123456789ABC"], Some(ID)),
        (vec!["not-a-uuid"], None),
        (vec!["b09d60fd12344567abcd123456789abc"], None),
        (vec![" b09d60fd-1234-4567-abcd-123456789abc"], None),
        (vec!["b09d60fd-1234-4567-abcd-123456789abc,other"], None),
        (vec![ID, ID], None),
    ] {
        let mut headers = HeaderMap::new();
        for value in &values {
            headers.append(
                "X-OpenShell-Request-ID",
                HeaderValue::from_str(value).unwrap(),
            );
        }
        assert_eq!(
            request_id(&headers).map(|id| id.to_string()).as_deref(),
            expected,
            "{values:?}",
        );
    }
}
