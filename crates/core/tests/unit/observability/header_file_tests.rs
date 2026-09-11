// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for file-backed observability headers.

use std::collections::HashMap;
use std::fs;

use tonic::service::Interceptor;

use super::*;

#[test]
fn reads_current_value_and_strips_only_trailing_whitespace() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "Bearer first\n").unwrap();
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);

    assert_eq!(
        resolve_header_files(&files).unwrap()["authorization"],
        "Bearer first"
    );
    fs::write(&path, "Bearer second\n\n").unwrap();
    assert_eq!(
        resolve_header_files(&files).unwrap()["authorization"],
        "Bearer second"
    );
}

#[test]
fn activation_requires_existing_files_and_unique_sources() {
    let headers = HashMap::from([("Authorization".to_string(), "static".to_string())]);
    let files = HashMap::from([("authorization".to_string(), "/missing".to_string())]);
    assert!(
        validate_header_files(&headers, &HashMap::new(), &files)
            .unwrap_err()
            .contains("unique across headers and header_env")
    );

    let files = HashMap::from([("authorization".to_string(), "/missing".to_string())]);
    assert!(
        validate_header_files(&HashMap::new(), &HashMap::new(), &files)
            .unwrap_err()
            .contains("does not exist")
    );

    let files = HashMap::from([("invalid header".to_string(), "/missing".to_string())]);
    assert!(
        validate_header_files(&HashMap::new(), &HashMap::new(), &files)
            .unwrap_err()
            .contains("invalid header name")
    );

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    fs::write(&path, "not valid\nvalue").unwrap();
    let files = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    assert!(validate_header_files(&HashMap::new(), &HashMap::new(), &files).is_ok());
    let header_env = HashMap::from([("Authorization".to_string(), "TOKEN".to_string())]);
    assert!(
        validate_header_files(&HashMap::new(), &header_env, &files)
            .unwrap_err()
            .contains("unique across headers and header_env")
    );
}

#[test]
fn runtime_rejects_blank_or_invalid_values_without_echoing_them() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    fs::write(&path, "\n").unwrap();
    assert!(resolve_header_files(&files).unwrap_err().contains("blank"));
    fs::write(&path, "bad\nvalue").unwrap();
    assert!(
        resolve_header_files(&files)
            .unwrap_err()
            .contains("invalid value")
    );
    fs::write(&path, [0xff]).unwrap();
    assert!(
        resolve_header_files(&files)
            .unwrap_err()
            .contains("could not read")
    );
    fs::remove_file(&path).unwrap();
    assert!(
        resolve_header_files(&files)
            .unwrap_err()
            .contains("could not read")
    );
}

#[test]
fn grpc_interceptor_reads_the_current_value_for_each_request() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("token");
    let files: HeaderFiles = HashMap::from([(
        "authorization".to_string(),
        path.to_string_lossy().into_owned(),
    )]);
    let resolver = HeaderFileResolver::new(files);
    let mut interceptor = HeaderFileInterceptor::new(resolver);

    fs::write(&path, "Bearer first\n").unwrap();
    let request = interceptor.call(tonic::Request::new(())).unwrap();
    assert_eq!(
        request
            .metadata()
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer first"
    );

    fs::write(&path, "Bearer second\n").unwrap();
    let request = interceptor.call(tonic::Request::new(())).unwrap();
    assert_eq!(
        request
            .metadata()
            .get("authorization")
            .unwrap()
            .to_str()
            .unwrap(),
        "Bearer second"
    );
}
