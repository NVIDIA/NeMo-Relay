# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small direct ONNX runner for the Rampart token-classification model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

from ._detectors import TokenClassifier

DEFAULT_MODEL_ID = "nationaldesignstudio/rampart"
DEFAULT_MODEL_REVISION = "b1993e4e68b082835b80ffc65acc03325ea2e501"
_MODEL_FILE = "onnx/model_q4.onnx"
_TOKENIZER_FILE = "tokenizer.json"
_CONFIG_FILE = "config.json"
_MAX_SEQUENCE_LENGTH = 512
_TOKEN_STRIDE = 64


class _RawDetection(TypedDict):
    start: int
    end: int
    entity_group: str
    score: float


def _download_snapshot(
    model_id: str,
    model_revision: str,
    allow_network: bool,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "Rampart worker dependencies are missing; register the plugin with 'nemo-relay plugins add'"
        ) from error

    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            allow_patterns=[_MODEL_FILE, _TOKENIZER_FILE, _CONFIG_FILE],
            local_files_only=not allow_network,
        )
    )


class _OnnxTokenClassifier:
    def __init__(self, model_path: str, tokenizer_path: str, config_path: str) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "Rampart worker dependencies are missing; register the plugin with 'nemo-relay plugins add'"
            ) from error

        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        id2label = config.get("id2label")
        if not isinstance(id2label, dict):
            raise RuntimeError("Rampart config.json does not contain an id2label mapping")

        self._np = np
        self._labels = {int(index): str(label) for index, label in id2label.items()}
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(
            max_length=_MAX_SEQUENCE_LENGTH,
            stride=_TOKEN_STRIDE,
        )
        session_options = ort.SessionOptions()
        self._session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {model_input.name for model_input in self._session.get_inputs()}

    def _infer_encoding(self, encoding: Any) -> list[_RawDetection]:
        np = self._np
        inputs = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
        if "token_type_ids" in self._input_names:
            inputs["token_type_ids"] = np.asarray([encoding.type_ids], dtype=np.int64)

        logits = cast(Any, self._session.run(None, inputs)[0])[0]
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        label_ids = np.argmax(probabilities, axis=-1)
        scores = np.max(probabilities, axis=-1)

        detections: list[_RawDetection] = []
        current: dict[str, int | float | str] | None = None

        def flush() -> None:
            nonlocal current
            if current is None:
                return
            detections.append(
                {
                    "start": cast(int, current["start"]),
                    "end": cast(int, current["end"]),
                    "entity_group": cast(str, current["entity_group"]),
                    "score": cast(float, current["score_total"]) / cast(int, current["token_count"]),
                }
            )
            current = None

        for token_index, (start, end) in enumerate(encoding.offsets):
            if start == end:
                continue
            raw_label = self._labels[int(label_ids[token_index])]
            if raw_label == "O":
                flush()
                continue

            prefix, _, label = raw_label.partition("-")
            score = float(scores[token_index])
            is_subword = encoding.tokens[token_index].startswith("##")
            continues = current is not None and current["entity_group"] == label and (prefix != "B" or is_subword)
            if not continues:
                flush()
                current = {
                    "start": start,
                    "end": end,
                    "entity_group": label,
                    "score_total": score,
                    "token_count": 1,
                }
            else:
                assert current is not None
                current["end"] = end
                current["score_total"] = cast(float, current["score_total"]) + score
                current["token_count"] = cast(int, current["token_count"]) + 1

        flush()
        return detections

    def __call__(self, text: str) -> list[Mapping[str, object]]:
        encoding = self._tokenizer.encode(text)
        encodings = [encoding, *encoding.overflowing]
        detections = [
            detection for current_encoding in encodings for detection in self._infer_encoding(current_encoding)
        ]
        unique: dict[tuple[int, int, str], _RawDetection] = {}
        for detection in detections:
            key = (
                detection["start"],
                detection["end"],
                detection["entity_group"],
            )
            previous = unique.get(key)
            if previous is None or detection["score"] > previous["score"]:
                unique[key] = detection
        return list(unique.values())


@lru_cache(maxsize=4)
def load_classifier(
    model_id: str,
    model_revision: str,
    allow_network: bool,
) -> TokenClassifier:
    """Load one cached CPU classifier, downloading artifacts only when allowed."""
    snapshot = _download_snapshot(model_id, model_revision, allow_network)
    return _OnnxTokenClassifier(
        str(snapshot / _MODEL_FILE),
        str(snapshot / _TOKENIZER_FILE),
        str(snapshot / _CONFIG_FILE),
    )
