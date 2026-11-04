// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::merge_usage;
use nemo_relay_types::codec::response::Usage;

#[test]
fn merged_usage_keeps_the_input_split_of_the_source_reporting_it() {
    let split = Usage {
        prompt_tokens: Some(100),
        cache_read_tokens: Some(40),
        uncached_input_tokens: Some(60),
        ..Usage::default()
    };
    let scraped = Usage {
        prompt_tokens: Some(100),
        completion_tokens: Some(20),
        cache_write_tokens: Some(10),
        ..Usage::default()
    };

    for (primary, secondary) in [(&split, &scraped), (&scraped, &split)] {
        let merged = merge_usage(Some(primary), Some(secondary)).unwrap();

        assert_eq!(merged.prompt_tokens, Some(100));
        assert_eq!(merged.completion_tokens, Some(20));
        assert_eq!(merged.cache_read_tokens, Some(40));
        assert_eq!(merged.cache_write_tokens, None);
        assert_eq!(merged.uncached_input_tokens, Some(60));
    }
}
