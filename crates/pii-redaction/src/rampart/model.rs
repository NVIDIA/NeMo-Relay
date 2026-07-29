// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use nemo_relay::plugin::{PluginError, Result as PluginResult};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tract_onnx::prelude::*;

use super::prefilter::PreparedText;
use super::tokenizer::RampartTokenizer;

const MODEL_MAX_TOKENS: usize = 512;
const SPECIAL_TOKEN_COUNT: usize = 2;
const CONTENT_TOKEN_BUDGET: usize = MODEL_MAX_TOKENS - SPECIAL_TOKEN_COUNT;
const WINDOW_OVERLAP_TOKENS: usize = 64;
const MAX_PADDED_TOKENS_PER_BATCH: usize = MODEL_MAX_TOKENS;

const MODEL_FILES: &[(&str, &str)] = &[
    (
        "config.json",
        "003b84bbcd489f5e782fe5cad8f3249c3653ec880089abb1ccc398a0d895e3e6",
    ),
    (
        "onnx/model_q4.onnx",
        "9f27d24949b0581701071ea5ef522d77ccd3f50c525cc91eac4d265b0fc2afe5",
    ),
    (
        "special_tokens_map.json",
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    ),
    (
        "tokenizer.json",
        "98ade711428b42a1b5343c403a73344535e92de8e19359cdb567ef34da210259",
    ),
    (
        "tokenizer_config.json",
        "0088a6f8bcdd4014184fb068b83ebb12896a9db2bb269a71f73de83fef24bceb",
    ),
    (
        "vocab.txt",
        "0fbe6b50061feabb9be68af471e9aa6df07a4bc428bdca4b0eff1fcd3612dee5",
    ),
];

#[derive(Clone, Debug)]
pub(super) struct Detection {
    pub(super) text_index: usize,
    pub(super) start_utf8: usize,
    pub(super) end_utf8: usize,
    pub(super) label: String,
    pub(super) score: f64,
}

pub(super) struct RampartDetector {
    tokenizer: RampartTokenizer,
    plan: Arc<TypedRunnableModel>,
    labels: Arc<[String]>,
    cls_id: i64,
    sep_id: i64,
    pad_id: i64,
    max_windows_per_payload: usize,
    inference_batch_size: usize,
    // Tokenization and inference share one admission point to bound aggregate
    // request-local tensor memory under concurrent Relay traffic.
    inference_lock: Mutex<()>,
}

#[derive(Deserialize)]
struct ModelConfig {
    id2label: HashMap<String, String>,
}

struct VerifiedModelFiles {
    config: File,
    model: File,
    vocab: File,
}

struct Window {
    text_index: usize,
    input_ids: Vec<i64>,
    token_type_ids: Vec<i64>,
    offsets: Vec<Option<(usize, usize)>>,
}

#[derive(Clone)]
struct Span {
    start: usize,
    end: usize,
    label: String,
    score: f64,
    source: SpanSource,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SpanSource {
    Model,
    Deterministic,
}

struct SpanAccumulator {
    start: usize,
    end: usize,
    label: String,
    score_total: f64,
    token_count: usize,
}

impl SpanAccumulator {
    fn finish(self) -> Span {
        Span {
            start: self.start,
            end: self.end,
            label: self.label,
            score: self.score_total / self.token_count as f64,
            source: SpanSource::Model,
        }
    }
}

impl RampartDetector {
    pub(super) fn load(
        model_root: PathBuf,
        max_windows_per_payload: usize,
        inference_batch_size: usize,
    ) -> PluginResult<Self> {
        let model_root = model_root.canonicalize().map_err(|error| {
            invalid_model(format!(
                "Rampart model directory '{}' is unavailable: {error}",
                model_root.display()
            ))
        })?;
        if !model_root.is_dir() {
            return Err(invalid_model(format!(
                "Rampart model path '{}' is not a directory",
                model_root.display()
            )));
        }
        let files = verify_model_files(&model_root)?;

        let config: ModelConfig = serde_json::from_reader(BufReader::new(files.config))
            .map_err(|error| invalid_model(format!("invalid Rampart config.json: {error}")))?;
        let labels = parse_labels(config.id2label)?;

        let tokenizer = RampartTokenizer::from_vocab_reader(BufReader::new(files.vocab))?;
        let cls_id = required_token_id(&tokenizer, "[CLS]")?;
        let sep_id = required_token_id(&tokenizer, "[SEP]")?;
        let pad_id = required_token_id(&tokenizer, "[PAD]")?;

        let framework = tract_onnx::onnx()
            .with_ignore_value_info(true)
            .with_ignore_output_shapes(true);
        let mut model_reader = BufReader::new(files.model);
        let model = framework
            .model_for_read(&mut model_reader)
            .map_err(|error| {
                invalid_model(format!("failed to load Rampart ONNX model: {error}"))
            })?;
        validate_model_inputs(&model)?;
        let outputs = model
            .output_outlets()
            .map_err(|error| invalid_model(format!("invalid Rampart model outputs: {error}")))?;
        if outputs.len() != 1 {
            return Err(invalid_model(
                "Rampart ONNX model must expose exactly one logits output",
            ));
        }
        let plan = model
            .into_optimized()
            .and_then(|model| model.into_runnable())
            .map_err(|error| {
                invalid_model(format!("failed to optimize Rampart ONNX model: {error}"))
            })?;

        let detector = Self {
            tokenizer,
            plan,
            labels: labels.into(),
            cls_id,
            sep_id,
            pad_id,
            max_windows_per_payload,
            inference_batch_size,
            inference_lock: Mutex::new(()),
        };
        detector.detect(&["warmup"])?;
        Ok(detector)
    }

    pub(super) fn detect(&self, texts: &[&str]) -> PluginResult<Vec<Detection>> {
        let _guard = self.inference_lock.lock().map_err(|error| {
            PluginError::Internal(format!("Rampart inference lock poisoned: {error}"))
        })?;
        let prepared = texts
            .iter()
            .map(|text| PreparedText::new(text))
            .collect::<Vec<_>>();
        let masked_texts = prepared
            .iter()
            .map(PreparedText::masked)
            .collect::<Vec<_>>();
        let windows = self.build_windows(&masked_texts)?;
        if windows.is_empty() {
            return Ok(prepared
                .iter()
                .enumerate()
                .flat_map(|(text_index, text)| {
                    text.spans().iter().map(move |span| Detection {
                        text_index,
                        start_utf8: span.start,
                        end_utf8: span.end,
                        label: span.label.into(),
                        score: 1.0,
                    })
                })
                .collect());
        }

        let mut model_spans_by_text = vec![Vec::new(); texts.len()];
        for batch in inference_batches(&windows, self.inference_batch_size) {
            let logits = self.infer_batch(&windows, &batch)?;
            for (batch_index, window_index) in batch.iter().copied().enumerate() {
                let window = &windows[window_index];
                model_spans_by_text[window.text_index].extend(self.decode_window(
                    window,
                    &logits,
                    batch_index,
                )?);
            }
        }

        let mut detections = Vec::new();
        for (text_index, model_spans) in model_spans_by_text.into_iter().enumerate() {
            let mut spans = prepared[text_index]
                .spans()
                .iter()
                .map(|span| Span {
                    start: span.start,
                    end: span.end,
                    label: span.label.into(),
                    score: 1.0,
                    source: SpanSource::Deterministic,
                })
                .collect::<Vec<_>>();
            for mut span in model_spans {
                let Some((start, end)) = prepared[text_index].project(span.start, span.end) else {
                    return Err(inference_error(
                        "Rampart prefilter returned an invalid UTF-8 span",
                    ));
                };
                span.start = start;
                span.end = end;
                spans.push(span);
            }
            for span in merge_overlapping_spans(spans) {
                let text = texts[text_index];
                if span.start >= span.end
                    || span.end > text.len()
                    || !text.is_char_boundary(span.start)
                    || !text.is_char_boundary(span.end)
                {
                    return Err(inference_error(
                        "Rampart tokenizer returned an invalid UTF-8 span",
                    ));
                }
                detections.push(Detection {
                    text_index,
                    start_utf8: span.start,
                    end_utf8: span.end,
                    label: span.label,
                    score: span.score,
                });
            }
        }
        Ok(detections)
    }

    fn build_windows(&self, texts: &[&str]) -> PluginResult<Vec<Window>> {
        let step = CONTENT_TOKEN_BUDGET - WINDOW_OVERLAP_TOKENS;
        let mut windows = Vec::new();
        for (text_index, text) in texts.iter().copied().enumerate() {
            let encoding = self.tokenizer.encode(text)?;
            let ids = &encoding.ids;
            let offsets = &encoding.offsets;

            for start in (0..ids.len()).step_by(step) {
                let end = (start + CONTENT_TOKEN_BUDGET).min(ids.len());
                let mut input_ids = Vec::with_capacity(end - start + SPECIAL_TOKEN_COUNT);
                input_ids.push(self.cls_id);
                input_ids.extend(ids[start..end].iter().map(|id| i64::from(*id)));
                input_ids.push(self.sep_id);

                let mut window_type_ids = Vec::with_capacity(end - start + SPECIAL_TOKEN_COUNT);
                window_type_ids.push(0);
                window_type_ids.extend(std::iter::repeat_n(0, end - start));
                window_type_ids.push(0);

                let mut window_offsets = Vec::with_capacity(end - start + SPECIAL_TOKEN_COUNT);
                window_offsets.push(None);
                window_offsets.extend(offsets[start..end].iter().copied().map(Some));
                window_offsets.push(None);
                windows.push(Window {
                    text_index,
                    input_ids,
                    token_type_ids: window_type_ids,
                    offsets: window_offsets,
                });
                if windows.len() > self.max_windows_per_payload {
                    return Err(inference_error(format!(
                        "selected content exceeded max_windows_per_payload={}",
                        self.max_windows_per_payload
                    )));
                }
                if end == ids.len() {
                    break;
                }
            }
        }
        Ok(windows)
    }

    fn infer_batch(
        &self,
        windows: &[Window],
        batch: &[usize],
    ) -> PluginResult<tract_ndarray::Array3<f32>> {
        let max_length = batch
            .iter()
            .map(|index| windows[*index].input_ids.len())
            .max()
            .ok_or_else(|| inference_error("Rampart inference batch must not be empty"))?;
        let shape = (batch.len(), max_length);
        let mut input_ids = tract_ndarray::Array2::from_elem(shape, self.pad_id);
        let mut attention_mask = tract_ndarray::Array2::zeros(shape);
        let mut token_type_ids = tract_ndarray::Array2::zeros(shape);
        for (batch_index, window_index) in batch.iter().copied().enumerate() {
            let window = &windows[window_index];
            for (token_index, input_id) in window.input_ids.iter().copied().enumerate() {
                input_ids[[batch_index, token_index]] = input_id;
                attention_mask[[batch_index, token_index]] = 1_i64;
                token_type_ids[[batch_index, token_index]] = window.token_type_ids[token_index];
            }
        }

        let outputs = self
            .plan
            .run(tvec![
                input_ids.into_tensor().into(),
                attention_mask.into_tensor().into(),
                token_type_ids.into_tensor().into(),
            ])
            .map_err(|error| inference_error(format!("Rampart inference failed: {error}")))?;
        let output = outputs
            .first()
            .ok_or_else(|| inference_error("Rampart inference returned no logits"))?;
        let view = output
            .to_plain_array_view::<f32>()
            .map_err(|error| inference_error(format!("invalid Rampart logits: {error}")))?;
        let expected = [batch.len(), max_length, self.labels.len()];
        if view.shape() != expected {
            return Err(inference_error(format!(
                "Rampart logits shape must be {expected:?}, got {:?}",
                view.shape()
            )));
        }
        view.to_owned()
            .into_dimensionality::<tract_ndarray::Ix3>()
            .map_err(|error| inference_error(format!("invalid Rampart logits rank: {error}")))
    }

    fn decode_window(
        &self,
        window: &Window,
        logits: &tract_ndarray::Array3<f32>,
        batch_index: usize,
    ) -> PluginResult<Vec<Span>> {
        let mut spans = Vec::new();
        let mut current: Option<SpanAccumulator> = None;
        for (token_index, offset) in window.offsets.iter().copied().enumerate() {
            let (label_index, score) =
                max_label_and_score(logits, batch_index, token_index, self.labels.len())?;
            let raw_label = &self.labels[label_index];
            let Some((prefix, label)) = split_bio_label(raw_label) else {
                if let Some(span) = current.take() {
                    spans.push(span.finish());
                }
                continue;
            };
            let Some((start, end)) = offset.filter(|(start, end)| start < end) else {
                if let Some(span) = current.take() {
                    spans.push(span.finish());
                }
                continue;
            };

            let continue_span = current
                .as_ref()
                .is_some_and(|span| prefix == "I" && span.label == label);
            if continue_span {
                let span = current.as_mut().expect("checked current span");
                span.end = span.end.max(end);
                span.score_total += score;
                span.token_count += 1;
            } else {
                if let Some(span) = current.take() {
                    spans.push(span.finish());
                }
                current = Some(SpanAccumulator {
                    start,
                    end,
                    label: label.to_ascii_uppercase(),
                    score_total: score,
                    token_count: 1,
                });
            }
        }
        if let Some(span) = current {
            spans.push(span.finish());
        }
        Ok(spans)
    }
}

fn validate_model_inputs(model: &InferenceModel) -> PluginResult<()> {
    let names = model
        .input_outlets()
        .map_err(|error| invalid_model(format!("invalid Rampart model inputs: {error}")))?
        .iter()
        .map(|outlet| model.node(outlet.node).name.as_str())
        .collect::<Vec<_>>();
    if names != ["input_ids", "attention_mask", "token_type_ids"] {
        return Err(invalid_model(format!(
            "Rampart ONNX inputs must be input_ids, attention_mask, and token_type_ids; got {names:?}"
        )));
    }
    Ok(())
}

fn parse_labels(raw_labels: HashMap<String, String>) -> PluginResult<Vec<String>> {
    let mut labels = raw_labels
        .into_iter()
        .map(|(index, label)| {
            index
                .parse::<usize>()
                .map(|index| (index, label))
                .map_err(|_| invalid_model("Rampart label IDs must be unsigned integers"))
        })
        .collect::<PluginResult<Vec<_>>>()?;
    labels.sort_by_key(|(index, _)| *index);
    if labels.is_empty()
        || labels[0].1 != "O"
        || labels
            .iter()
            .enumerate()
            .any(|(expected, (actual, _))| expected != *actual)
    {
        return Err(invalid_model(
            "Rampart label IDs must be contiguous from zero with label 0 set to O",
        ));
    }
    Ok(labels.into_iter().map(|(_, label)| label).collect())
}

fn required_token_id(tokenizer: &RampartTokenizer, token: &str) -> PluginResult<i64> {
    tokenizer.token_to_id(token).map(i64::from).ok_or_else(|| {
        invalid_model(format!(
            "Rampart tokenizer is missing required token {token}"
        ))
    })
}

fn inference_batches(windows: &[Window], max_batch_size: usize) -> Vec<Vec<usize>> {
    let mut ordered = (0..windows.len()).collect::<Vec<_>>();
    ordered.sort_by_key(|index| windows[*index].input_ids.len());
    let mut batches = Vec::new();
    let mut batch = Vec::new();
    let mut max_tokens = 0;
    for index in ordered {
        let next_max_tokens = max_tokens.max(windows[index].input_ids.len());
        let padded_tokens = (batch.len() + 1) * next_max_tokens;
        if !batch.is_empty()
            && (batch.len() >= max_batch_size || padded_tokens > MAX_PADDED_TOKENS_PER_BATCH)
        {
            batches.push(std::mem::take(&mut batch));
            max_tokens = 0;
        }
        max_tokens = max_tokens.max(windows[index].input_ids.len());
        batch.push(index);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    batches
}

fn max_label_and_score(
    logits: &tract_ndarray::Array3<f32>,
    batch_index: usize,
    token_index: usize,
    label_count: usize,
) -> PluginResult<(usize, f64)> {
    let mut label_index = 0;
    let mut maximum = f32::NEG_INFINITY;
    for index in 0..label_count {
        let value = logits[[batch_index, token_index, index]];
        if !value.is_finite() {
            return Err(inference_error(
                "Rampart logits must contain only finite values",
            ));
        }
        if value > maximum {
            maximum = value;
            label_index = index;
        }
    }
    let denominator = (0..label_count)
        .map(|index| (logits[[batch_index, token_index, index]] - maximum).exp() as f64)
        .sum::<f64>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(inference_error(
            "Rampart logits produced an invalid confidence score",
        ));
    }
    Ok((label_index, 1.0 / denominator))
}

fn split_bio_label(label: &str) -> Option<(&str, &str)> {
    if label == "O" {
        return None;
    }
    label
        .strip_prefix("B-")
        .map(|label| ("B", label))
        .or_else(|| label.strip_prefix("I-").map(|label| ("I", label)))
        .or(Some(("B", label)))
}

fn merge_overlapping_spans(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|left, right| {
        (left.start, std::cmp::Reverse(left.end))
            .cmp(&(right.start, std::cmp::Reverse(right.end)))
            .then_with(|| right.score.total_cmp(&left.score))
            .then_with(|| left.label.cmp(&right.label))
    });
    let mut merged: Vec<Span> = Vec::new();
    for span in spans {
        let Some(previous) = merged.last_mut() else {
            merged.push(span);
            continue;
        };
        if span.start > previous.end || (span.start == previous.end && span.label != previous.label)
        {
            merged.push(span);
            continue;
        }
        let span_wins = (
            span.score,
            span.end - span.start,
            span.source,
            span.label.as_str(),
        ) > (
            previous.score,
            previous.end - previous.start,
            previous.source,
            previous.label.as_str(),
        );
        previous.start = previous.start.min(span.start);
        previous.end = previous.end.max(span.end);
        previous.score = previous.score.max(span.score);
        if span_wins {
            previous.label = span.label;
        }
    }
    merged
}

fn verify_model_files(model_root: &Path) -> PluginResult<VerifiedModelFiles> {
    let mut files = HashMap::new();
    for (relative_path, expected) in MODEL_FILES {
        let path = model_root.join(relative_path);
        files.insert(
            *relative_path,
            open_verified_model_file(&path, relative_path, expected)?,
        );
    }
    Ok(VerifiedModelFiles {
        config: required_verified_file(&mut files, "config.json")?,
        model: required_verified_file(&mut files, "onnx/model_q4.onnx")?,
        vocab: required_verified_file(&mut files, "vocab.txt")?,
    })
}

fn open_verified_model_file(path: &Path, display_name: &str, expected: &str) -> PluginResult<File> {
    let mut file = File::open(path).map_err(|error| {
        invalid_model(format!(
            "Rampart model is missing required file '{display_name}': {error}"
        ))
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|error| {
            invalid_model(format!(
                "failed to read Rampart model file '{display_name}': {error}"
            ))
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut actual = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(actual, "{byte:02x}").expect("writing to a string cannot fail");
    }
    if actual != expected {
        return Err(invalid_model(format!(
            "Rampart model file '{display_name}' failed SHA-256 verification"
        )));
    }
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        invalid_model(format!(
            "failed to rewind Rampart model file '{display_name}': {error}"
        ))
    })?;
    Ok(file)
}

fn required_verified_file(
    files: &mut HashMap<&'static str, File>,
    relative_path: &'static str,
) -> PluginResult<File> {
    files.remove(relative_path).ok_or_else(|| {
        invalid_model(format!(
            "Rampart integrity manifest is missing '{relative_path}'"
        ))
    })
}

fn invalid_model(message: impl Into<String>) -> PluginError {
    PluginError::InvalidConfig(message.into())
}

fn inference_error(message: impl Into<String>) -> PluginError {
    PluginError::Internal(message.into())
}

#[cfg(test)]
mod tests {
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
}
