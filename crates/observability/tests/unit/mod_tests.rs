// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::merge_usage;
use nemo_relay_types::codec::response::CacheWriteTokenCounts;
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
        cache_write_tokens: Some(vec![CacheWriteTokenCounts {
            count: 10,
            duration: None,
        }]),
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

#[test]
fn merged_usage_keeps_a_duration_split_of_the_same_cache_write_total() {
    let write = |count, duration| CacheWriteTokenCounts { count, duration };
    let flat = Usage {
        cache_write_tokens: Some(vec![write(50, None)]),
        ..Usage::default()
    };
    let split = Usage {
        cache_write_tokens: Some(vec![write(30, Some(300)), write(20, Some(3_600))]),
        ..Usage::default()
    };
    let other_total = Usage {
        cache_write_tokens: Some(vec![write(30, Some(300)), write(10, Some(3_600))]),
        ..Usage::default()
    };

    // Same total: the source that knows the durations wins, in either order.
    for (primary, secondary) in [(&flat, &split), (&split, &flat)] {
        let merged = merge_usage(Some(primary), Some(secondary)).unwrap();
        assert_eq!(merged.cache_write_tokens, split.cache_write_tokens);
    }
    // A partly tagged list still loses to one that tags more of the same total.
    let partial = Usage {
        cache_write_tokens: Some(vec![write(30, Some(300)), write(20, None)]),
        ..Usage::default()
    };
    let merged = merge_usage(Some(&partial), Some(&split)).unwrap();
    assert_eq!(merged.cache_write_tokens, split.cache_write_tokens);
    // Different totals describe different writes, so the primary source stays.
    let merged = merge_usage(Some(&flat), Some(&other_total)).unwrap();
    assert_eq!(merged.cache_write_tokens, flat.cache_write_tokens);
}
