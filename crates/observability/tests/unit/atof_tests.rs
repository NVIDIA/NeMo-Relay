// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn default_config_uses_cwd_append_and_timestamped_filename() {
    let config = AtofExporterConfig::default();
    let AtofSinkConfig::File(file) = config.sink else {
        panic!("default ATOF config must use a file sink");
    };

    assert_eq!(file.output_directory, std::env::current_dir().unwrap());
    assert_eq!(file.mode, AtofExporterMode::Append);
    assert_eq!(AtofExporterMode::Append.as_str(), "append");
    assert_eq!(AtofExporterMode::Overwrite.as_str(), "overwrite");
    assert!(file.filename.starts_with("nemo-relay-events-"));
    assert!(file.filename.ends_with(".jsonl"));
    assert_eq!(
        file.filename.len(),
        "nemo-relay-events-YYYY-MM-DD-HH.MM.SS.jsonl".len()
    );
}

#[test]
fn endpoint_and_exporter_config_builders_preserve_values() {
    let dir = temp_dir("atof-config-builders");
    let endpoint =
        AtofEndpointConfig::new("http://127.0.0.1:9/events", AtofEndpointTransport::HttpPost)
            .with_header("x-test", "enabled")
            .with_timeout_millis(42)
            .with_field_name_policy(AtofEndpointFieldNamePolicy::ReplaceDots);
    let config = AtofExporterConfig::new()
        .with_output_directory(&dir)
        .with_mode(AtofExporterMode::Overwrite)
        .with_filename("custom.jsonl")
        .with_stream_sink(endpoint.clone());

    assert_eq!(
        endpoint.headers.get("x-test").map(String::as_str),
        Some("enabled")
    );
    assert_eq!(endpoint.timeout_millis, 42);
    assert_eq!(
        endpoint.field_name_policy,
        AtofEndpointFieldNamePolicy::ReplaceDots
    );
    assert_eq!(AtofEndpointFieldNamePolicy::Preserve.as_str(), "preserve");
    assert_eq!(
        AtofEndpointFieldNamePolicy::parse("replace_dots"),
        Some(AtofEndpointFieldNamePolicy::ReplaceDots)
    );
    assert_eq!(config.path(), None);
    assert_eq!(config.sink, AtofSinkConfig::Stream(endpoint));
}

#[test]
fn atof_config_helpers_cover_file_path_and_replace_dots_policy() {
    let config = AtofExporterConfig::new()
        .with_output_directory("output")
        .with_filename("events.jsonl");
    assert_eq!(config.path(), Some(PathBuf::from("output/events.jsonl")));
    assert_eq!(
        AtofEndpointFieldNamePolicy::ReplaceDots.as_str(),
        "replace_dots"
    );
}

fn temp_dir(prefix: &str) -> PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("nemo-relay-{prefix}-{id}"));
    fs::create_dir_all(&path).unwrap();
    path.canonicalize().unwrap()
}
