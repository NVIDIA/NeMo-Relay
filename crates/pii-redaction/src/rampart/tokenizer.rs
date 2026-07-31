// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::io::BufRead;

use nemo_relay::plugin::{PluginError, Result as PluginResult};
use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

const UNKNOWN_TOKEN: &str = "[UNK]";
const CONTINUATION_PREFIX: &str = "##";
const MAX_CHARS_PER_WORD: usize = 100;
const SPECIAL_TOKENS: &[&str] = &["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

pub(super) struct EncodedText {
    pub(super) ids: Vec<u32>,
    pub(super) offsets: Vec<(usize, usize)>,
}

pub(super) struct RampartTokenizer {
    vocab: HashMap<String, u32>,
    unknown_id: u32,
}

#[derive(Clone, Copy)]
struct MappedChar {
    value: char,
    original_start: usize,
    original_end: usize,
    isolate: bool,
}

struct Piece {
    id: u32,
    start: usize,
    end: usize,
}

impl RampartTokenizer {
    pub(super) fn from_vocab_reader(reader: impl BufRead) -> PluginResult<Self> {
        let mut vocab = HashMap::new();
        for (index, line) in reader.lines().enumerate() {
            let line = line.map_err(|error| {
                invalid_tokenizer(format!("failed to read Rampart vocab.txt: {error}"))
            })?;
            let token = line.strip_suffix('\r').unwrap_or(&line).to_string();
            let id = u32::try_from(index)
                .map_err(|_| invalid_tokenizer("Rampart vocabulary is too large"))?;
            if vocab.insert(token, id).is_some() {
                return Err(invalid_tokenizer(
                    "Rampart vocabulary contains duplicate tokens",
                ));
            }
        }
        let unknown_id = vocab
            .get(UNKNOWN_TOKEN)
            .copied()
            .ok_or_else(|| invalid_tokenizer("Rampart vocabulary is missing [UNK]"))?;
        Ok(Self { vocab, unknown_id })
    }

    pub(super) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }

    pub(super) fn encode(&self, text: &str) -> PluginResult<EncodedText> {
        let mut ids = Vec::new();
        let mut offsets = Vec::new();
        let mut cursor = 0;
        while cursor < text.len() {
            let Some((special_start, special)) = next_special_token(text, cursor) else {
                self.encode_segment(&text[cursor..], cursor, &mut ids, &mut offsets)?;
                break;
            };
            self.encode_segment(&text[cursor..special_start], cursor, &mut ids, &mut offsets)?;
            let special_end = special_start + special.len();
            ids.push(self.token_to_id(special).ok_or_else(|| {
                invalid_tokenizer("Rampart vocabulary is missing a special token")
            })?);
            offsets.push((special_start, special_end));
            cursor = special_end;
        }
        Ok(EncodedText { ids, offsets })
    }

    fn encode_segment(
        &self,
        text: &str,
        base_offset: usize,
        ids: &mut Vec<u32>,
        offsets: &mut Vec<(usize, usize)>,
    ) -> PluginResult<()> {
        for token in basic_tokens(text, base_offset) {
            for piece in self.wordpiece(&token)? {
                ids.push(piece.id);
                offsets.push(original_offsets(&token, piece.start, piece.end)?);
            }
        }
        Ok(())
    }

    fn wordpiece(&self, token: &[MappedChar]) -> PluginResult<Vec<Piece>> {
        let normalized = token.iter().map(|item| item.value).collect::<String>();
        if token.len() > MAX_CHARS_PER_WORD {
            return Ok(vec![Piece {
                id: self.unknown_id,
                start: 0,
                end: normalized.len(),
            }]);
        }

        let mut pieces = Vec::new();
        let mut start = 0;
        while start < normalized.len() {
            let mut end = normalized.len();
            let mut matched = None;
            while start < end {
                let candidate = if start == 0 {
                    &normalized[start..end]
                } else {
                    let mut value = String::with_capacity(CONTINUATION_PREFIX.len() + end - start);
                    value.push_str(CONTINUATION_PREFIX);
                    value.push_str(&normalized[start..end]);
                    if let Some(id) = self.vocab.get(&value).copied() {
                        matched = Some(Piece { id, start, end });
                        break;
                    }
                    end -= normalized[start..end]
                        .chars()
                        .next_back()
                        .expect("non-empty WordPiece candidate")
                        .len_utf8();
                    continue;
                };
                if let Some(id) = self.vocab.get(candidate).copied() {
                    matched = Some(Piece { id, start, end });
                    break;
                }
                end -= candidate
                    .chars()
                    .next_back()
                    .expect("non-empty WordPiece candidate")
                    .len_utf8();
            }
            let Some(piece) = matched else {
                return Ok(vec![Piece {
                    id: self.unknown_id,
                    start: 0,
                    end: normalized.len(),
                }]);
            };
            start = piece.end;
            pieces.push(piece);
        }
        Ok(pieces)
    }
}

fn basic_tokens(text: &str, base_offset: usize) -> Vec<Vec<MappedChar>> {
    let mut tokens = Vec::new();
    let mut current = Vec::new();
    for item in normalize(text, base_offset) {
        if item.value.is_whitespace() {
            push_current(&mut tokens, &mut current);
        } else if item.isolate || item.value.is_ascii_punctuation() || item.value.is_punctuation() {
            push_current(&mut tokens, &mut current);
            tokens.push(vec![item]);
        } else {
            current.push(item);
        }
    }
    push_current(&mut tokens, &mut current);
    tokens
}

fn push_current(tokens: &mut Vec<Vec<MappedChar>>, current: &mut Vec<MappedChar>) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

fn normalize(text: &str, base_offset: usize) -> Vec<MappedChar> {
    let mut normalized = Vec::with_capacity(text.chars().count());
    for (relative_start, original) in text.char_indices() {
        let original_start = base_offset + relative_start;
        let original_end = original_start + original.len_utf8();
        if original == '\0' || original == '\u{fffd}' || is_control(original) {
            continue;
        }
        let cleaned = if is_whitespace(original) {
            ' '
        } else {
            original
        };
        let isolate = is_chinese_char(cleaned);
        for decomposed in std::iter::once(cleaned).nfd() {
            if decomposed.is_mark_nonspacing() {
                continue;
            }
            for value in decomposed.to_lowercase() {
                normalized.push(MappedChar {
                    value,
                    original_start,
                    original_end,
                    isolate,
                });
            }
        }
    }
    normalized
}

fn next_special_token(text: &str, cursor: usize) -> Option<(usize, &'static str)> {
    SPECIAL_TOKENS
        .iter()
        .filter_map(|token| {
            text[cursor..]
                .find(token)
                .map(|relative| (cursor + relative, *token))
        })
        .min_by_key(|(start, token)| (*start, std::cmp::Reverse(token.len())))
}

fn original_offsets(
    token: &[MappedChar],
    normalized_start: usize,
    normalized_end: usize,
) -> PluginResult<(usize, usize)> {
    let mut cursor = 0;
    let mut original_start = None;
    let mut original_end = None;
    for item in token {
        let next = cursor + item.value.len_utf8();
        if next > normalized_start && cursor < normalized_end {
            original_start.get_or_insert(item.original_start);
            original_end = Some(item.original_end);
        }
        cursor = next;
    }
    original_start
        .zip(original_end)
        .ok_or_else(|| PluginError::Internal("Rampart tokenizer returned an invalid offset".into()))
}

fn is_whitespace(value: char) -> bool {
    matches!(value, '\t' | '\n' | '\r') || value.is_whitespace()
}

fn is_control(value: char) -> bool {
    !matches!(value, '\t' | '\n' | '\r') && value.is_other()
}

fn is_chinese_char(value: char) -> bool {
    matches!(
        value as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn invalid_tokenizer(message: impl Into<String>) -> PluginError {
    PluginError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests {
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
}
