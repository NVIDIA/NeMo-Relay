# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np  # ty: ignore[unresolved-import]
import pytest

import nemo_relay_pii_rampart.detector as detector_module
from nemo_relay_pii_rampart.detector import (
    DEFAULT_MODEL_ID,
    RampartDetector,
    RampartSettings,
    _explicit_model_root,
    _merge_overlapping_spans,
    _parse_request,
    _Span,
    _verify_model_files,
    prefetch_verified_model,
)


class FakeTokenizer:
    _tokens = {"[PAD]": 0, "[CLS]": 2, "[SEP]": 3}

    def token_to_id(self, token: str) -> int | None:
        return self._tokens.get(token)

    def encode(self, sequence: str, add_special_tokens: bool = True) -> SimpleNamespace:
        del add_special_tokens
        words = sequence.split()
        ids = []
        offsets = []
        cursor = 0
        for index, word in enumerate(words):
            start = sequence.index(word, cursor)
            end = start + len(word)
            ids.append(10 + index)
            offsets.append((start, end))
            cursor = end
        return SimpleNamespace(ids=ids, type_ids=[0] * len(ids), offsets=offsets)


class FakeSession:
    def __init__(self, label_ids: list[int], scores: list[float]) -> None:
        self._label_ids = label_ids
        self._scores = scores

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=name) for name in ("input_ids", "attention_mask", "token_type_ids")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="logits")]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names == ["logits"]
        shape = (*input_feed["input_ids"].shape, 5)
        logits = np.full(shape, -10.0, dtype=np.float32)
        logits[:, :, 0] = 10.0
        for token_index, (label_id, score) in enumerate(zip(self._label_ids, self._scores, strict=True), start=1):
            logits[:, token_index, 0] = 0.0
            logits[:, token_index, label_id] = np.log(score / (1.0 - score) * 4.0)
        return [logits]


class NonFiniteSession(FakeSession):
    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        logits = super().run(output_names, input_feed)[0]
        logits[0, 0, 0] = np.nan
        return [logits]


def detector(
    label_ids: list[int],
    scores: list[float],
    *,
    max_windows_per_request: int | None = None,
) -> RampartDetector:
    config = {} if max_windows_per_request is None else {"max_windows_per_request": max_windows_per_request}
    return RampartDetector(
        RampartSettings.from_config(config),
        FakeTokenizer(),
        FakeSession(label_ids, scores),
        {0: "O", 1: "B-GIVEN_NAME", 2: "I-GIVEN_NAME", 3: "B-CITY", 4: "I-CITY"},
    )


def test_settings_validate_unknown_and_bounded_fields() -> None:
    settings = RampartSettings.from_config({})
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.local_files_only is True
    with pytest.raises(ValueError, match="unknown plugin config"):
        RampartSettings.from_config({"surprise": True})
    with pytest.raises(TypeError, match="max_pending_requests"):
        RampartSettings.from_config({"max_pending_requests": True})
    with pytest.raises(ValueError, match="model_id"):
        RampartSettings.from_config({"model_id": "x" * 1025})
    with pytest.raises(ValueError, match="explicit local directory"):
        RampartSettings.from_config({"model_id": "other/model"})
    with pytest.raises(ValueError, match="pinned Rampart revision"):
        RampartSettings.from_config({"revision": "main"})
    with pytest.raises(ValueError, match="prefetch"):
        RampartSettings.from_config({"local_files_only": False})


def test_model_files_are_verified_before_loading(tmp_path: Path) -> None:
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"trusted model")
    expected = {"model.bin": hashlib.sha256(b"trusted model").hexdigest()}

    _verify_model_files(tmp_path, expected)
    model_file.write_bytes(b"modified model")
    with pytest.raises(ValueError, match="SHA-256 verification"):
        _verify_model_files(tmp_path, expected)


def test_model_file_verification_rejects_missing_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required file"):
        _verify_model_files(tmp_path, {"missing.bin": "0" * 64})


def test_prefetch_uses_the_pinned_snapshot_and_verifies_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def snapshot(model_id: str, **kwargs: Any) -> str:
        observed["model_id"] = model_id
        observed.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(detector_module, "snapshot_download", snapshot)
    monkeypatch.setattr(
        detector_module, "_verify_model_files", lambda model_root: observed.setdefault("root", model_root)
    )

    assert prefetch_verified_model("/tmp/rampart-cache") == tmp_path
    assert observed["model_id"] == DEFAULT_MODEL_ID
    assert observed["revision"] == detector_module.DEFAULT_MODEL_REVISION
    assert observed["local_files_only"] is False
    assert observed["cache_dir"] == "/tmp/rampart-cache"
    assert observed["root"] == tmp_path


def test_local_model_directories_require_explicit_path_syntax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    assert _explicit_model_root(str(model_root)) == model_root.resolve()
    settings = RampartSettings.from_config({"model_id": str(model_root)})
    assert settings.model_id == str(model_root)

    shadow = tmp_path / "nationaldesignstudio" / "rampart"
    shadow.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert _explicit_model_root(DEFAULT_MODEL_ID) is None
    with pytest.raises(ValueError, match="does not exist"):
        _explicit_model_root("./missing")


def test_request_validation_rejects_duplicate_ids_and_byte_overflow() -> None:
    with pytest.raises(ValueError, match="duplicate text id"):
        _parse_request(
            {
                "version": 1,
                "texts": [
                    {"id": 0, "text": "one"},
                    {"id": 0, "text": "two"},
                ],
            }
        )
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        _parse_request({"version": 1, "texts": [{"id": 0, "text": "é" * 9000}]})


def test_detector_returns_utf8_byte_offsets_and_model_labels() -> None:
    value = detector([1, 2], [0.99, 0.98]).detect_request({"version": 1, "texts": [{"id": 7, "text": "José Rivera"}]})
    assert value["version"] == 1
    assert len(value["detections"]) == 1
    detection = value["detections"][0]
    assert detection["text_id"] == 7
    assert detection["start_utf8"] == 0
    assert detection["end_utf8"] == len("José Rivera".encode())
    assert detection["label"] == "GIVEN_NAME"
    assert 0.9 <= detection["score"] <= 1.0

    city = detector([3, 4], [0.99, 0.98]).detect_request({"version": 1, "texts": [{"id": 0, "text": "New York"}]})
    assert city["detections"][0]["label"] == "CITY"


def test_detector_rejects_model_and_profile_mismatch() -> None:
    current = detector([], [])
    with pytest.raises(ValueError, match="does not match loaded model"):
        current.detect_request(
            {
                "version": 1,
                "model_id": "other/model",
                "texts": [{"id": 0, "text": ""}],
            }
        )
    with pytest.raises(ValueError, match="unsupported detector_profile"):
        current.detect_request(
            {
                "version": 1,
                "detector_profile": "strict",
                "texts": [{"id": 0, "text": ""}],
            }
        )


def test_local_model_path_keeps_the_logical_model_identity(tmp_path: Path) -> None:
    current = RampartDetector(
        RampartSettings.from_config({"model_id": str(tmp_path)}),
        FakeTokenizer(),
        FakeSession([], []),
        {0: "O", 1: "B-GIVEN_NAME", 2: "I-GIVEN_NAME", 3: "B-CITY", 4: "I-CITY"},
    )
    result = current.detect_request(
        {
            "version": 1,
            "model_id": DEFAULT_MODEL_ID,
            "texts": [{"id": 0, "text": ""}],
        }
    )
    assert result == {"version": 1, "detections": []}


def test_overlapping_window_spans_are_coalesced() -> None:
    spans = _merge_overlapping_spans(
        [
            _Span(0, 10, "GIVEN_NAME", 0.8),
            _Span(5, 12, "SURNAME", 0.9),
            _Span(20, 24, "PHONE", 0.7),
            _Span(24, 28, "PHONE", 0.8),
            _Span(28, 30, "TAX_ID", 0.9),
        ]
    )
    assert spans == [
        _Span(0, 12, "SURNAME", 0.9),
        _Span(20, 28, "PHONE", 0.8),
        _Span(28, 30, "TAX_ID", 0.9),
    ]


def test_request_window_limit_is_enforced_before_inference() -> None:
    current = detector([], [], max_windows_per_request=1)
    text = " ".join(f"word{index}" for index in range(600))
    with pytest.raises(ValueError, match="max_windows_per_request"):
        current.detect_request({"version": 1, "texts": [{"id": 0, "text": text}]})


def test_detector_rejects_invalid_model_outputs() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        RampartDetector(
            RampartSettings(),
            FakeTokenizer(),
            FakeSession([], []),
            {0: "O", 2: "B-GIVEN_NAME"},
        )

    current = RampartDetector(
        RampartSettings(),
        FakeTokenizer(),
        NonFiniteSession([], []),
        {0: "O", 1: "B-GIVEN_NAME", 2: "I-GIVEN_NAME", 3: "B-CITY", 4: "I-CITY"},
    )
    with pytest.raises(ValueError, match="finite"):
        current.detect_request({"version": 1, "texts": [{"id": 0, "text": "hello"}]})
