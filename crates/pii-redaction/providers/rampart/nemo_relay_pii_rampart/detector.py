# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded ONNX token-classification adapter for the Rampart PII model."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np  # ty: ignore[unresolved-import]
import onnxruntime as ort  # ty: ignore[unresolved-import]
from huggingface_hub import snapshot_download  # ty: ignore[unresolved-import]
from tokenizers import Tokenizer  # ty: ignore[unresolved-import]

DEFAULT_MODEL_ID = "nationaldesignstudio/rampart"
DEFAULT_MODEL_REVISION = "b1993e4e68b082835b80ffc65acc03325ea2e501"
CONTRACT_VERSION = 1
MAX_TEXTS_PER_REQUEST = 64
MAX_TEXT_BYTES = 16 * 1024
MAX_REQUEST_TEXT_BYTES = 64 * 1024
MODEL_MAX_TOKENS = 512
SPECIAL_TOKEN_COUNT = 2
CONTENT_TOKEN_BUDGET = MODEL_MAX_TOKENS - SPECIAL_TOKEN_COUNT
WINDOW_OVERLAP_TOKENS = 64
MAX_MODEL_REFERENCE_BYTES = 1024
MAX_CACHE_PATH_BYTES = 4096

_MODEL_FILE_SHA256 = {
    "config.json": "003b84bbcd489f5e782fe5cad8f3249c3653ec880089abb1ccc398a0d895e3e6",
    "onnx/model_q4.onnx": "9f27d24949b0581701071ea5ef522d77ccd3f50c525cc91eac4d265b0fc2afe5",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "tokenizer.json": "98ade711428b42a1b5343c403a73344535e92de8e19359cdb567ef34da210259",
    "tokenizer_config.json": "0088a6f8bcdd4014184fb068b83ebb12896a9db2bb269a71f73de83fef24bceb",
}


@dataclass(frozen=True)
class RampartSettings:
    """Activation-time settings for one Rampart worker."""

    model_id: str = DEFAULT_MODEL_ID
    revision: str = DEFAULT_MODEL_REVISION
    cache_dir: str | None = None
    local_files_only: bool = True
    max_windows_per_request: int = 128
    inference_batch_size: int = 16
    max_pending_requests: int = 8
    intra_op_threads: int | None = None

    @classmethod
    def from_config(cls, config: Any) -> RampartSettings:
        """Parse and validate dynamic-plugin configuration."""
        if not isinstance(config, dict):
            raise TypeError("plugin config must be a JSON object")
        allowed = {
            "model_id",
            "revision",
            "cache_dir",
            "local_files_only",
            "max_windows_per_request",
            "inference_batch_size",
            "max_pending_requests",
            "intra_op_threads",
        }
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise ValueError(f"unknown plugin config field(s): {', '.join(unknown)}")

        model_id = _bounded_string(
            config.get("model_id", DEFAULT_MODEL_ID),
            "model_id",
            MAX_MODEL_REFERENCE_BYTES,
        )
        revision = _bounded_string(
            config.get("revision", DEFAULT_MODEL_REVISION),
            "revision",
            MAX_MODEL_REFERENCE_BYTES,
        )
        if not _uses_explicit_model_path(model_id) and model_id != DEFAULT_MODEL_ID:
            raise ValueError(f"model_id must be {DEFAULT_MODEL_ID!r} or an explicit local directory")
        if revision != DEFAULT_MODEL_REVISION:
            raise ValueError(f"revision must be the pinned Rampart revision {DEFAULT_MODEL_REVISION!r}")
        cache_dir = config.get("cache_dir")
        if cache_dir is not None:
            cache_dir = _bounded_string(cache_dir, "cache_dir", MAX_CACHE_PATH_BYTES)
        local_files_only = config.get("local_files_only", True)
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")
        if not local_files_only:
            raise ValueError("local_files_only must remain true; prefetch the pinned model before enabling the plugin")

        return cls(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            max_windows_per_request=_bounded_integer(
                config.get("max_windows_per_request", 128),
                "max_windows_per_request",
                1,
                512,
            ),
            inference_batch_size=_bounded_integer(
                config.get("inference_batch_size", 16),
                "inference_batch_size",
                1,
                64,
            ),
            max_pending_requests=_bounded_integer(
                config.get("max_pending_requests", 8),
                "max_pending_requests",
                1,
                64,
            ),
            intra_op_threads=_optional_bounded_integer(config.get("intra_op_threads"), "intra_op_threads", 1, 64),
        )


@dataclass(frozen=True)
class _InputText:
    text_id: int
    text: str


@dataclass(frozen=True)
class _Window:
    text_id: int
    input_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int] | None, ...]


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    label: str
    score: float


class _Tokenizer(Protocol):
    def encode(self, sequence: str, add_special_tokens: bool = True) -> Any: ...

    def token_to_id(self, token: str) -> int | None: ...


class _Session(Protocol):
    def get_inputs(self) -> list[Any]: ...

    def get_outputs(self) -> list[Any]: ...

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray[Any, Any]]) -> list[Any]: ...


class RampartDetector:
    """Load one Rampart model and perform bounded, serialized inference."""

    def __init__(
        self,
        settings: RampartSettings,
        tokenizer: _Tokenizer,
        session: _Session,
        labels: dict[int, str],
    ) -> None:
        self.settings = settings
        self._tokenizer = tokenizer
        self._session = session
        self._labels = labels
        if set(labels) != set(range(len(labels))):
            raise ValueError("Rampart label IDs must be contiguous from zero")
        self._lock = threading.Lock()
        self._cls_id = _required_token_id(tokenizer, "[CLS]")
        self._sep_id = _required_token_id(tokenizer, "[SEP]")
        self._pad_id = _required_token_id(tokenizer, "[PAD]")
        self._validate_model_contract()

    @classmethod
    def load(cls, settings: RampartSettings) -> RampartDetector:
        """Resolve model files and initialize an optimized CPU session."""
        model_root = resolve_verified_model_root(settings)
        config = json.loads((model_root / "config.json").read_text(encoding="utf-8"))
        raw_labels = config.get("id2label")
        if not isinstance(raw_labels, dict):
            raise ValueError("Rampart config.json must contain an id2label object")
        labels = {int(index): str(label) for index, label in raw_labels.items()}
        if not labels or labels.get(0) != "O":
            raise ValueError("Rampart label map must define label 0 as O")

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.inter_op_num_threads = 1
        if settings.intra_op_threads is not None:
            options.intra_op_num_threads = settings.intra_op_threads
        session = ort.InferenceSession(
            str(model_root / "onnx" / "model_q4.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(str(model_root / "tokenizer.json"))
        detector = cls(settings, tokenizer, session, labels)
        detector._detect_texts([_InputText(0, "warmup")])
        return detector

    def detect_request(self, request: Any) -> dict[str, Any]:
        """Validate one provider request and return versioned UTF-8 spans."""
        texts, requested_model = _parse_request(request)
        if requested_model is not None and requested_model != DEFAULT_MODEL_ID:
            raise ValueError(f"request model_id {requested_model!r} does not match loaded model {DEFAULT_MODEL_ID!r}")
        profile = request.get("detector_profile")
        if profile not in (None, "default"):
            raise ValueError(f"unsupported detector_profile {profile!r}")

        with self._lock:
            detections = self._detect_texts(texts)
        return {
            "version": CONTRACT_VERSION,
            "detections": detections,
        }

    def _validate_model_contract(self) -> None:
        inputs = {entry.name for entry in self._session.get_inputs()}
        expected_inputs = {"input_ids", "attention_mask", "token_type_ids"}
        if inputs != expected_inputs:
            raise ValueError(f"Rampart ONNX inputs must be {sorted(expected_inputs)}, got {sorted(inputs)}")
        outputs = self._session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "logits":
            raise ValueError("Rampart ONNX model must expose one logits output")

    def _detect_texts(self, texts: list[_InputText]) -> list[dict[str, Any]]:
        windows = self._build_windows(texts)
        spans_by_text: dict[int, list[_Span]] = {item.text_id: [] for item in texts}
        for start in range(0, len(windows), self.settings.inference_batch_size):
            batch = windows[start : start + self.settings.inference_batch_size]
            logits = self._infer(batch)
            for window, window_logits in zip(batch, logits, strict=True):
                spans_by_text[window.text_id].extend(self._decode_window(window, window_logits))

        detections = []
        text_by_id = {item.text_id: item.text for item in texts}
        byte_offsets_by_id = {text_id: _utf8_offsets(text) for text_id, text in text_by_id.items()}
        for text_id, spans in spans_by_text.items():
            for span in _merge_overlapping_spans(spans):
                byte_offsets = byte_offsets_by_id[text_id]
                detections.append(
                    {
                        "text_id": text_id,
                        "start_utf8": byte_offsets[span.start],
                        "end_utf8": byte_offsets[span.end],
                        "label": span.label,
                        "score": span.score,
                    }
                )
        return detections

    def _build_windows(self, texts: list[_InputText]) -> list[_Window]:
        windows = []
        step = CONTENT_TOKEN_BUDGET - WINDOW_OVERLAP_TOKENS
        for item in texts:
            inference_text = item.text.replace("-", " ")
            encoding = self._tokenizer.encode(inference_text, add_special_tokens=False)
            ids = list(encoding.ids)
            type_ids = list(encoding.type_ids)
            offsets = list(encoding.offsets)
            if not (len(ids) == len(type_ids) == len(offsets)):
                raise ValueError("tokenizer returned inconsistent token metadata")
            if any(start < 0 or end < start or end > len(inference_text) for start, end in offsets):
                raise ValueError("tokenizer returned invalid character offsets")
            for start in range(0, len(ids), step):
                end = min(start + CONTENT_TOKEN_BUDGET, len(ids))
                windows.append(
                    _Window(
                        text_id=item.text_id,
                        input_ids=(self._cls_id, *ids[start:end], self._sep_id),
                        token_type_ids=(0, *type_ids[start:end], 0),
                        offsets=(None, *offsets[start:end], None),
                    )
                )
                if len(windows) > self.settings.max_windows_per_request:
                    raise ValueError(
                        f"request exceeded max_windows_per_request={self.settings.max_windows_per_request}"
                    )
                if end == len(ids):
                    break
        return windows

    def _infer(self, windows: list[_Window]) -> np.ndarray[Any, np.dtype[np.float32]]:
        max_length = max(len(window.input_ids) for window in windows)
        shape = (len(windows), max_length)
        input_ids = np.full(shape, self._pad_id, dtype=np.int64)
        attention_mask = np.zeros(shape, dtype=np.int64)
        token_type_ids = np.zeros(shape, dtype=np.int64)
        for index, window in enumerate(windows):
            length = len(window.input_ids)
            input_ids[index, :length] = window.input_ids
            attention_mask[index, :length] = 1
            token_type_ids[index, :length] = window.token_type_ids
        result = self._session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        logits = np.asarray(result[0], dtype=np.float32)
        expected = (len(windows), max_length, len(self._labels))
        if logits.shape != expected:
            raise ValueError(f"Rampart logits shape must be {expected}, got {logits.shape}")
        if not np.isfinite(logits).all():
            raise ValueError("Rampart logits must contain only finite values")
        return logits

    def _decode_window(self, window: _Window, logits: np.ndarray[Any, Any]) -> list[_Span]:
        label_ids = np.argmax(logits, axis=-1)
        maxima = np.max(logits, axis=-1)
        scores = 1.0 / np.exp(logits - maxima[:, None]).sum(axis=-1)
        spans = []
        current_label: str | None = None
        current_start = 0
        current_end = 0
        current_score = 0.0
        current_count = 0

        def finish() -> None:
            nonlocal current_label, current_start, current_end, current_score, current_count
            if current_label is not None:
                score = current_score / current_count
                spans.append(_Span(current_start, current_end, current_label, score))
            current_label = None
            current_start = 0
            current_end = 0
            current_score = 0.0
            current_count = 0

        for index, offset in enumerate(window.offsets):
            if offset is None or offset[0] >= offset[1]:
                finish()
                continue
            raw_label = self._labels.get(int(label_ids[index]))
            score = float(scores[index])
            prefix, label = _split_bio_label(raw_label)
            if label is None:
                finish()
                continue
            if current_label is None or prefix == "B" or label != current_label:
                finish()
                current_label = label
                current_start = offset[0]
                current_end = offset[1]
                current_score = score
                current_count = 1
            else:
                current_end = max(current_end, offset[1])
                current_score += score
                current_count += 1
        finish()
        return spans


def _resolve_model_root(settings: RampartSettings) -> Path:
    local_path = _explicit_model_root(settings.model_id)
    if local_path is not None:
        return local_path
    resolved = snapshot_download(
        settings.model_id,
        revision=settings.revision,
        cache_dir=settings.cache_dir,
        allow_patterns=list(_MODEL_FILE_SHA256),
        local_files_only=True,
    )
    return Path(resolved)


def resolve_verified_model_root(settings: RampartSettings) -> Path:
    """Resolve and verify the pinned model assets without loading ONNX Runtime."""
    model_root = _resolve_model_root(settings)
    _verify_model_files(model_root)
    return model_root


def prefetch_verified_model(cache_dir: str | None = None) -> Path:
    """Download and verify the pinned Rampart assets outside plugin activation."""
    if cache_dir is not None:
        cache_dir = _bounded_string(cache_dir, "cache_dir", MAX_CACHE_PATH_BYTES)
    model_root = Path(
        snapshot_download(
            DEFAULT_MODEL_ID,
            revision=DEFAULT_MODEL_REVISION,
            cache_dir=cache_dir,
            allow_patterns=list(_MODEL_FILE_SHA256),
            local_files_only=False,
        )
    )
    _verify_model_files(model_root)
    return model_root


def _verify_model_files(
    model_root: Path,
    expected: Mapping[str, str] = _MODEL_FILE_SHA256,
) -> None:
    for relative_path, expected_sha256 in expected.items():
        path = model_root / relative_path
        if not path.is_file():
            raise ValueError(f"Rampart model is missing required file {relative_path!r}")
        with path.open("rb") as model_file:
            digest = hashlib.file_digest(model_file, "sha256").hexdigest()
        if digest != expected_sha256:
            raise ValueError(f"Rampart model file {relative_path!r} failed SHA-256 verification")


def _explicit_model_root(model_id: str) -> Path | None:
    candidate = Path(model_id).expanduser()
    if not _uses_explicit_model_path(model_id):
        return None
    if not candidate.is_dir():
        raise ValueError(f"local Rampart model directory does not exist: {model_id}")
    return candidate.resolve()


def _uses_explicit_model_path(model_id: str) -> bool:
    return Path(model_id).expanduser().is_absolute() or model_id.startswith(("./", "../", ".\\", "..\\", "~/", "~\\"))


def _parse_request(request: Any) -> tuple[list[_InputText], str | None]:
    if not isinstance(request, dict):
        raise TypeError("local-model request must be a JSON object")
    allowed = {"version", "model_id", "detector_profile", "texts"}
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise ValueError(f"unknown local-model request field(s): {', '.join(unknown)}")
    version = request.get("version")
    if isinstance(version, bool) or version != CONTRACT_VERSION:
        raise ValueError(f"local-model request version must be {CONTRACT_VERSION}")
    model_id = request.get("model_id")
    if model_id is not None:
        model_id = _nonempty_string(model_id, "model_id")
    profile = request.get("detector_profile")
    if profile is not None:
        _nonempty_string(profile, "detector_profile")
    raw_texts = request.get("texts")
    if not isinstance(raw_texts, list) or not raw_texts:
        raise TypeError("texts must be a non-empty array")
    if len(raw_texts) > MAX_TEXTS_PER_REQUEST:
        raise ValueError(f"texts must contain at most {MAX_TEXTS_PER_REQUEST} items")

    texts = []
    seen_ids = set()
    total_bytes = 0
    for item in raw_texts:
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise TypeError("each texts item must contain exactly id and text")
        text_id = item["id"]
        text = item["text"]
        if isinstance(text_id, bool) or not isinstance(text_id, int) or not 0 <= text_id <= 2**32 - 1:
            raise TypeError("text id must be an unsigned 32-bit integer")
        if text_id in seen_ids:
            raise ValueError(f"duplicate text id {text_id}")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        text_bytes = len(text.encode("utf-8"))
        if text_bytes > MAX_TEXT_BYTES:
            raise ValueError(f"text {text_id} exceeds {MAX_TEXT_BYTES} UTF-8 bytes")
        total_bytes += text_bytes
        if total_bytes > MAX_REQUEST_TEXT_BYTES:
            raise ValueError(f"request text exceeds {MAX_REQUEST_TEXT_BYTES} UTF-8 bytes")
        seen_ids.add(text_id)
        texts.append(_InputText(text_id, text))
    return texts, model_id


def _split_bio_label(raw_label: str | None) -> tuple[str | None, str | None]:
    if raw_label is None or raw_label == "O":
        return None, None
    if raw_label.startswith(("B-", "I-")) and len(raw_label) > 2:
        return raw_label[0], raw_label[2:].upper()
    return "B", raw_label.upper()


def _merge_overlapping_spans(spans: list[_Span]) -> list[_Span]:
    merged: list[_Span] = []
    for span in sorted(spans, key=lambda item: (item.start, -item.end, -item.score, item.label)):
        if (
            not merged
            or span.start > merged[-1].end
            or (span.start == merged[-1].end and span.label != merged[-1].label)
        ):
            merged.append(span)
            continue
        previous = merged[-1]
        winner = _preferred_span(previous, span)
        merged[-1] = _Span(
            start=min(previous.start, span.start),
            end=max(previous.end, span.end),
            label=winner.label,
            score=max(previous.score, span.score),
        )
    return merged


def _preferred_span(left: _Span, right: _Span) -> _Span:
    left_key = (left.score, left.end - left.start, left.label)
    right_key = (right.score, right.end - right.start, right.label)
    return left if left_key >= right_key else right


def _utf8_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        offsets.append(total)
    return offsets


def _required_token_id(tokenizer: _Tokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"tokenizer is missing required token {token}")
    return token_id


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _bounded_string(value: Any, name: str, maximum_bytes: int) -> str:
    value = _nonempty_string(value, name)
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} must not exceed {maximum_bytes} UTF-8 bytes")
    return value


def _bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _optional_bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    return _bounded_integer(value, name, minimum, maximum)
