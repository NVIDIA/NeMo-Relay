// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Rampart model loading and verification.

use super::*;
use std::fs;

#[test]
fn batches_bound_padded_token_volume() {
    let windows = (0..16)
        .map(|_| Window {
            text_index: 0,
            input_ids: vec![0; 32],
            token_type_ids: vec![0; 32],
            offsets: vec![None; 32],
        })
        .chain((0..2).map(|_| Window {
            text_index: 0,
            input_ids: vec![0; MODEL_MAX_TOKENS],
            token_type_ids: vec![0; MODEL_MAX_TOKENS],
            offsets: vec![None; MODEL_MAX_TOKENS],
        }))
        .collect::<Vec<_>>();
    let batches = inference_batches(&windows, 16);
    assert_eq!(batches.iter().map(Vec::len).collect::<Vec<_>>(), [16, 1, 1]);
    assert!(batches.iter().all(|batch| {
        batch.len()
            * batch
                .iter()
                .map(|index| windows[*index].input_ids.len())
                .max()
                .unwrap()
            <= MAX_PADDED_TOKENS_PER_BATCH
    }));
}

#[test]
fn compute_budget_charges_padded_tokens_instead_of_string_count() {
    let short_windows = (0..100)
        .map(|text_index| Window {
            text_index,
            input_ids: vec![101, 1, 102],
            token_type_ids: vec![0; 3],
            offsets: vec![None, Some((0, 1)), None],
        })
        .collect::<Vec<_>>();
    assert_eq!(padded_token_volume(&short_windows, 16), 300);
    assert!(padded_token_volume(&short_windows, 16) <= MODEL_MAX_TOKENS);

    let long_windows = [
        Window {
            text_index: 0,
            input_ids: vec![0; MODEL_MAX_TOKENS],
            token_type_ids: vec![0; MODEL_MAX_TOKENS],
            offsets: vec![None; MODEL_MAX_TOKENS],
        },
        Window {
            text_index: 0,
            input_ids: vec![0; 156],
            token_type_ids: vec![0; 156],
            offsets: vec![None; 156],
        },
    ];
    assert_eq!(padded_token_volume(&long_windows, 16), 668);
    assert!(padded_token_volume(&long_windows, 16) > MODEL_MAX_TOKENS);
    assert!(padded_token_volume(&long_windows, 16) <= 2 * MODEL_MAX_TOKENS);
}

#[test]
fn overlap_merge_is_deterministic() {
    let merged = merge_overlapping_spans(vec![
        Span {
            start: 0,
            end: 10,
            label: "GIVEN_NAME".into(),
            score: 0.8,
            source: SpanSource::Model,
        },
        Span {
            start: 5,
            end: 12,
            label: "SURNAME".into(),
            score: 0.9,
            source: SpanSource::Model,
        },
        Span {
            start: 20,
            end: 24,
            label: "PHONE".into(),
            score: 0.7,
            source: SpanSource::Model,
        },
        Span {
            start: 24,
            end: 28,
            label: "PHONE".into(),
            score: 0.8,
            source: SpanSource::Model,
        },
    ]);
    assert_eq!(merged.len(), 2);
    assert_eq!((merged[0].start, merged[0].end), (0, 12));
    assert_eq!(merged[0].label, "SURNAME");
    assert_eq!((merged[1].start, merged[1].end), (20, 28));
}

#[test]
fn deterministic_span_wins_an_equal_model_tie() {
    let merged = merge_overlapping_spans(vec![
        Span {
            start: 0,
            end: 11,
            label: "GOVERNMENT_ID".into(),
            score: 1.0,
            source: SpanSource::Model,
        },
        Span {
            start: 0,
            end: 11,
            label: "SSN".into(),
            score: 1.0,
            source: SpanSource::Deterministic,
        },
    ]);
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].label, "SSN");
}

#[test]
fn confidence_uses_stable_softmax_and_rejects_non_finite_logits() {
    let logits = tract_ndarray::Array3::from_shape_vec((1, 1, 3), vec![0.0, 2.0, 1.0]).unwrap();
    let (label, score) = max_label_and_score(&logits, 0, 0, 3).unwrap();
    let expected = 1.0 / (1.0 + (-1.0_f64).exp() + (-2.0_f64).exp());
    assert_eq!(label, 1);
    assert!((score - expected).abs() < 1e-7);

    let invalid =
        tract_ndarray::Array3::from_shape_vec((1, 1, 3), vec![0.0, f32::NAN, 1.0]).unwrap();
    assert!(max_label_and_score(&invalid, 0, 0, 3).is_err());
}

#[test]
fn model_file_verification_rejects_missing_and_modified_files() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("model.bin");
    let expected = {
        let digest = Sha256::digest(b"trusted model");
        let mut value = String::with_capacity(digest.len() * 2);
        for byte in digest {
            write!(value, "{byte:02x}").unwrap();
        }
        value
    };

    let missing = open_verified_model_file(&path, "model.bin", &expected).unwrap_err();
    assert!(missing.to_string().contains("missing required file"));

    fs::write(&path, b"trusted model").unwrap();
    open_verified_model_file(&path, "model.bin", &expected).unwrap();
    fs::write(&path, b"modified model").unwrap();
    let modified = open_verified_model_file(&path, "model.bin", &expected).unwrap_err();
    assert!(modified.to_string().contains("SHA-256 verification"));
}

#[test]
fn label_map_requires_contiguous_numeric_ids_and_outside_label_zero() {
    let valid = HashMap::from([
        ("2".into(), "I-GIVEN_NAME".into()),
        ("0".into(), "O".into()),
        ("1".into(), "B-GIVEN_NAME".into()),
    ]);
    assert_eq!(
        parse_labels(valid).unwrap(),
        ["O", "B-GIVEN_NAME", "I-GIVEN_NAME"]
    );

    for invalid in [
        HashMap::from([("label".into(), "O".into())]),
        HashMap::from([("0".into(), "B-GIVEN_NAME".into())]),
        HashMap::from([("0".into(), "O".into()), ("2".into(), "B-NAME".into())]),
    ] {
        assert!(parse_labels(invalid).is_err());
    }
}

#[test]
fn model_contract_requires_special_tokens_and_normalizes_bio_labels() {
    let tokenizer = RampartTokenizer::from_vocab_reader("[UNK]\n[CLS]\n".as_bytes()).unwrap();
    assert_eq!(required_token_id(&tokenizer, "[CLS]").unwrap(), 1);
    assert!(required_token_id(&tokenizer, "[SEP]").is_err());

    assert_eq!(split_bio_label("O"), None);
    assert_eq!(split_bio_label("B-GIVEN_NAME"), Some(("B", "GIVEN_NAME")));
    assert_eq!(split_bio_label("I-GIVEN_NAME"), Some(("I", "GIVEN_NAME")));
    assert_eq!(split_bio_label("EMAIL"), Some(("B", "EMAIL")));
}
