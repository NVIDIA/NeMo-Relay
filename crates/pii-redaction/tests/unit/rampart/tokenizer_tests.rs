// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for Rampart tokenization and offset projection.

use super::*;

fn tokenizer() -> RampartTokenizer {
    let tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "hello",
        ",",
        "jose",
        "##ph",
        "野",
        "alice",
        "-",
        "rivera",
        "alicerivera",
    ];
    let vocab = tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| (token.to_string(), index as u32))
        .collect();
    RampartTokenizer {
        vocab,
        unknown_id: 1,
    }
}

#[test]
fn normalizes_bert_text_and_preserves_original_utf8_offsets() {
    let encoded = tokenizer().encode("Héllo, JOSEPH 野").unwrap();
    assert_eq!(encoded.ids, [5, 6, 7, 8, 9]);
    assert_eq!(
        encoded.offsets,
        [(0, 6), (6, 7), (8, 12), (12, 14), (15, 18)]
    );
}

#[test]
fn preserves_hyphen_token_and_original_offsets() {
    let encoded = tokenizer().encode("Alice-Rivera").unwrap();
    assert_eq!(encoded.ids, [10, 11, 12]);
    assert_eq!(encoded.offsets, [(0, 5), (5, 6), (6, 12)]);
}

#[test]
fn unknown_word_covers_the_complete_original_token() {
    let encoded = tokenizer().encode("unlisted").unwrap();
    assert_eq!(encoded.ids, [1]);
    assert_eq!(encoded.offsets, [(0, 8)]);
}

#[test]
fn preserves_exact_special_tokens_before_normalization() {
    let encoded = tokenizer().encode("Héllo [CLS] JOSEPH").unwrap();
    assert_eq!(encoded.ids, [5, 2, 7, 8]);
    assert_eq!(encoded.offsets, [(0, 6), (7, 12), (13, 17), (17, 19)]);
}

#[test]
fn removes_controls_and_combining_marks_without_shifting_original_offsets() {
    let encoded = tokenizer().encode("Alice\0Cafe\u{301} Rivera").unwrap();
    assert_eq!(encoded.ids, [1, 12]);
    assert_eq!(encoded.offsets, [(0, 10), (13, 19)]);
}

#[test]
fn removes_bert_controls_before_whitespace_normalization() {
    for control in ['\u{000b}', '\u{000c}', '\u{0085}'] {
        let text = format!("Alice{control}Rivera");
        let encoded = tokenizer().encode(&text).unwrap();
        assert_eq!(encoded.ids, [13], "control U+{:04X}", control as u32);
        assert_eq!(
            encoded.offsets,
            [(0, text.len())],
            "control U+{:04X}",
            control as u32
        );
    }
}

#[test]
fn keeps_bert_whitespace_as_token_separators() {
    for whitespace in ['\t', '\n', '\r', '\u{2003}', '\u{2028}', '\u{2029}'] {
        let text = format!("Alice{whitespace}Rivera");
        let separator_end = 5 + whitespace.len_utf8();
        let encoded = tokenizer().encode(&text).unwrap();
        assert_eq!(
            encoded.ids,
            [10, 12],
            "whitespace U+{:04X}",
            whitespace as u32
        );
        assert_eq!(
            encoded.offsets,
            [(0, 5), (separator_end, text.len())],
            "whitespace U+{:04X}",
            whitespace as u32
        );
    }
}

#[test]
fn isolates_unicode_punctuation_and_chinese_characters() {
    let encoded = tokenizer().encode("Alice’野").unwrap();
    assert_eq!(encoded.ids, [10, 1, 9]);
    assert_eq!(encoded.offsets, [(0, 5), (5, 8), (8, 11)]);
}

#[test]
fn vocabulary_parser_accepts_crlf_and_rejects_invalid_contracts() {
    let parsed =
        RampartTokenizer::from_vocab_reader("[PAD]\r\n[UNK]\r\nhello\r\n".as_bytes()).unwrap();
    assert_eq!(parsed.token_to_id("[UNK]"), Some(1));
    assert_eq!(parsed.token_to_id("hello"), Some(2));

    let duplicate = RampartTokenizer::from_vocab_reader("[UNK]\nhello\nhello\n".as_bytes())
        .err()
        .expect("duplicate vocabulary should fail");
    assert!(duplicate.to_string().contains("duplicate tokens"));

    let missing_unknown = RampartTokenizer::from_vocab_reader("[PAD]\nhello\n".as_bytes())
        .err()
        .expect("vocabulary without [UNK] should fail");
    assert!(missing_unknown.to_string().contains("missing [UNK]"));
}

#[test]
fn overlong_words_use_one_unknown_token_with_complete_offsets() {
    let text = "a".repeat(MAX_CHARS_PER_WORD + 1);
    let encoded = tokenizer().encode(&text).unwrap();

    assert_eq!(encoded.ids, [1]);
    assert_eq!(encoded.offsets, [(0, text.len())]);
}

#[test]
fn encountered_special_tokens_must_exist_in_the_vocabulary() {
    let tokenizer = RampartTokenizer::from_vocab_reader("[UNK]\nhello\n".as_bytes()).unwrap();
    let error = tokenizer
        .encode("hello [CLS]")
        .err()
        .expect("missing special token should fail");

    assert!(error.to_string().contains("missing a special token"));
}
