# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic and model-backed detection for the Rampart integration."""

from __future__ import annotations

import ipaddress
import math
import re
import unicodedata
from dataclasses import dataclass
from numbers import Real
from time import monotonic
from typing import Callable, Mapping, Protocol, Sequence

DEFAULT_MIN_SCORE = 0.4
DEFAULT_MAX_LATENCY_MS = 250
DEFAULT_MAX_CONTENT_CHARS = 8_192
MAX_CONTENT_CHARS = 65_536
FAILURE_REPLACEMENT = "[REDACTED:PII_DETECTION_FAILURE]"

_KEEP_LABELS = frozenset({"CITY", "STATE", "ZIP_CODE", "O"})
_MODEL_LABELS = frozenset(
    {
        "BANK_ACCOUNT",
        "BUILDING_NUMBER",
        "CITY",
        "DRIVERS_LICENSE",
        "EMAIL",
        "GIVEN_NAME",
        "GOVERNMENT_ID",
        "PASSPORT",
        "PHONE",
        "ROUTING_NUMBER",
        "SECONDARY_ADDRESS",
        "STATE",
        "STREET_NAME",
        "SURNAME",
        "TAX_ID",
        "URL",
        "ZIP_CODE",
    }
)
_MODEL_LABEL_ALIASES = {
    "GIVENNAME": "GIVEN_NAME",
    "LASTNAME": "SURNAME",
    "SECADDRESS": "SECONDARY_ADDRESS",
}
_EXTEND_SCORE = 0.15
_PERSON_LABELS = frozenset({"GIVEN_NAME", "SURNAME"})
_CONNECTOR_CHARS = frozenset({"'", "\u2019", ".", "-"})
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)[^\s<>'\"\\)\]}]+", re.IGNORECASE)
_IP_CANDIDATE_PATTERN = re.compile(r"(?<![\w:])(?:[0-9A-Fa-f:.]{3,})(?![\w:])")
_MAC_PATTERN = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")
_DIGIT_RUN_PATTERN = re.compile(r"(?<!\d)\d(?:[ .-]?\d)*(?!\d)")
_UUID_PATTERN = re.compile(
    r"(?<![0-9A-Fa-f])"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[1-8][0-9A-Fa-f]{3}-"
    r"[89ABab][0-9A-Fa-f]{3}-[0-9A-Fa-f]{12}"
    r"(?![0-9A-Fa-f])"
)
_TRACE_ID_PATTERN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{16,64}(?![0-9A-Fa-f])")
_OPERATIONAL_ID_CONTEXT_PATTERN = re.compile(
    r"(?:"
    r"(?:trace|span|request|correlation)(?:[ _.-]?id)?"
    r"|commit|sha(?:1|256)?|checksum|digest"
    r")\s*[:=]?\s*$",
    re.IGNORECASE,
)
_CLOUD_REGION_PATTERN = re.compile(
    r"\b(?:af|ap|ca|cn|eu|il|me|mx|sa|us(?:-gov|-iso|-isob)?)-"
    r"(?:central|east|north|northeast|northwest|south|southeast|southwest|west)-\d\b"
)
_HTTP_STATUS_PATTERN = re.compile(
    r"\b(?:HTTP(?:/\d(?:\.\d)?)?(?:\s+status(?:\s+code)?)?|status(?:\s+code)?)"
    r"\s*[:=]?\s+[1-5]\d{2}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Detection:
    """One character-offset detection."""

    start: int
    end: int
    label: str
    score: float = 1.0
    deterministic: bool = False


class TokenClassifier(Protocol):
    """Character-offset token-classification interface used by the sanitizer."""

    def __call__(self, text: str) -> Sequence[Mapping[str, object]]: ...


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        value = int(character)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _ssn_valid(digits: str) -> bool:
    if len(digits) != 9:
        return False
    area = int(digits[:3])
    return area not in {0, 666} and area < 900 and digits[3:5] != "00" and digits[5:] != "0000"


def _deterministic_detections(text: str) -> list[Detection]:
    detections = [
        Detection(match.start(), match.end(), "EMAIL", deterministic=True) for match in _EMAIL_PATTERN.finditer(text)
    ]
    detections.extend(
        Detection(match.start(), match.end(), "URL", deterministic=True) for match in _URL_PATTERN.finditer(text)
    )
    detections.extend(
        Detection(match.start(), match.end(), "IP_ADDRESS", deterministic=True) for match in _MAC_PATTERN.finditer(text)
    )

    for match in _IP_CANDIDATE_PATTERN.finditer(text):
        candidate = match.group(0).rstrip(".")
        if not candidate:
            continue
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        detections.append(
            Detection(
                match.start(),
                match.start() + len(candidate),
                "IP_ADDRESS",
                deterministic=True,
            )
        )

    for match in _DIGIT_RUN_PATTERN.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) in {14, 15, 16} and _luhn_valid(digits):
            detections.append(Detection(match.start(), match.end(), "CREDIT_CARD", deterministic=True))
        elif _ssn_valid(digits):
            detections.append(Detection(match.start(), match.end(), "SSN", deterministic=True))

    return _prefer_non_overlapping(detections)


def _operational_spans(text: str) -> list[Detection]:
    spans: list[Detection] = []
    for pattern in (_CLOUD_REGION_PATTERN, _HTTP_STATUS_PATTERN):
        spans.extend(
            Detection(match.start(), match.end(), "OPERATIONAL_ID", deterministic=True)
            for match in pattern.finditer(text)
        )
    for pattern in (_UUID_PATTERN, _TRACE_ID_PATTERN):
        for match in pattern.finditer(text):
            if not _has_operational_context(text, match.start()):
                continue
            spans.append(
                Detection(
                    match.start(),
                    match.end(),
                    "OPERATIONAL_ID",
                    deterministic=True,
                )
            )
    return _prefer_non_overlapping(spans)


def _has_operational_context(text: str, start: int) -> bool:
    context = text[max(0, start - 32) : start]
    return _OPERATIONAL_ID_CONTEXT_PATTERN.search(context) is not None


def _private_identifier_detections(text: str) -> list[Detection]:
    detections = []
    for pattern in (_UUID_PATTERN, _TRACE_ID_PATTERN):
        detections.extend(
            Detection(
                match.start(),
                match.end(),
                "IDENTIFIER",
                deterministic=True,
            )
            for match in pattern.finditer(text)
            if not _has_operational_context(text, match.start())
        )
    return _prefer_non_overlapping(detections)


def _preferred(first: Detection, second: Detection) -> Detection:
    first_key = (
        first.score,
        first.end - first.start,
        int(first.deterministic),
    )
    second_key = (
        second.score,
        second.end - second.start,
        int(second.deterministic),
    )
    return first if first_key >= second_key else second


def _prefer_non_overlapping(detections: Sequence[Detection]) -> list[Detection]:
    """Merge detections without exposing bytes from a partial overlap."""
    ordered = sorted(
        (detection for detection in detections if detection.end > detection.start),
        key=lambda detection: (detection.start, -detection.end),
    )
    accepted: list[Detection] = []
    for candidate in ordered:
        if not accepted or candidate.start >= accepted[-1].end:
            accepted.append(candidate)
            continue

        previous = accepted[-1]
        winner = _preferred(previous, candidate)
        accepted[-1] = Detection(
            min(previous.start, candidate.start),
            max(previous.end, candidate.end),
            winner.label,
            winner.score,
            winner.deterministic,
        )
    return accepted


def _premask(text: str, detections: Sequence[Detection]) -> tuple[str, list[tuple[int, int]]]:
    masked_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0

    for detection in _prefer_non_overlapping(detections):
        for index in range(cursor, detection.start):
            masked_parts.append(text[index])
            offsets.append((index, index + 1))
        sentinel = f"[{detection.label}]"
        masked_parts.append(sentinel)
        offsets.extend([(detection.start, detection.end)] * len(sentinel))
        cursor = detection.end

    for index in range(cursor, len(text)):
        masked_parts.append(text[index])
        offsets.append((index, index + 1))

    return "".join(masked_parts), offsets


def _model_detection(raw: Mapping[str, object], offsets: Sequence[tuple[int, int]]) -> Detection | None:
    start = raw.get("start")
    end = raw.get("end")
    score = raw.get("score", 0.0)
    label = raw.get("entity_group", raw.get("entity"))
    if not isinstance(start, int) or isinstance(start, bool):
        raise ValueError("Rampart detection start offset must be an integer")
    if not isinstance(end, int) or isinstance(end, bool):
        raise ValueError("Rampart detection end offset must be an integer")
    if not isinstance(score, Real) or isinstance(score, bool):
        raise ValueError("Rampart detection score must be numeric")
    if not math.isfinite(float(score)):
        raise ValueError("Rampart detection score must be finite")
    if not isinstance(label, str):
        raise ValueError("Rampart detection label must be a string")
    if start < 0 or end <= start or end > len(offsets):
        raise ValueError("Rampart detection offsets are outside the classified text")

    normalized_label = label.removeprefix("B-").removeprefix("I-").upper()
    normalized_label = _MODEL_LABEL_ALIASES.get(normalized_label, normalized_label)
    if normalized_label not in _MODEL_LABELS:
        raise ValueError(f"Rampart classifier returned an unknown label: {normalized_label}")
    if normalized_label in _KEEP_LABELS or float(score) < _EXTEND_SCORE:
        return None
    raw_start = offsets[start][0]
    raw_end = offsets[end - 1][1]
    if raw_end <= raw_start:
        return None
    return Detection(raw_start, raw_end, normalized_label, float(score))


def _is_connector(value: str) -> bool:
    return all(character.isspace() or character in _CONNECTOR_CHARS for character in value)


def _is_initial(text: str, index: int) -> bool:
    character = text[index] if 0 <= index < len(text) else ""
    if not character or not character.isalpha() or not character.isupper():
        return False
    previous = text[index - 1] if index > 0 else ""
    return not previous.isalpha()


def _can_bridge(text: str, first: Detection, second: Detection) -> bool:
    if first.label != second.label:
        return False
    left, right = (first, second) if first.start <= second.start else (second, first)
    gap = text[left.end : right.start]
    if not _is_connector(gap):
        return False
    return "." not in gap or _is_initial(text, left.end - 1)


def _merge_adjacent(text: str, detections: Sequence[Detection]) -> list[Detection]:
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: (item.start, item.end)):
        if merged and _can_bridge(text, merged[-1], detection):
            previous = merged[-1]
            merged[-1] = Detection(
                previous.start,
                max(previous.end, detection.end),
                previous.label,
                max(previous.score, detection.score),
            )
        else:
            merged.append(detection)
    return merged


def _is_name_particle(value: str) -> bool:
    if not 1 <= len(value) <= 4 or not value[0].isalpha() or not value[0].isupper():
        return False
    return all(
        character.isalpha() or unicodedata.category(character).startswith("M") or character in {"'", "\u2019"}
        for character in value[1:]
    )


def _left_particle(text: str, start: int, lower_bound: int) -> tuple[int, str, str] | None:
    connector_end = start
    connector_start = connector_end
    while (
        connector_start > lower_bound
        and connector_end - connector_start < 3
        and _is_connector(text[connector_start - 1])
    ):
        connector_start -= 1
    connector = text[connector_start:connector_end]
    if not connector:
        return None

    for length in range(1, 5):
        particle_start = connector_start - length
        if particle_start < lower_bound:
            break
        particle = text[particle_start:connector_start]
        if _is_name_particle(particle):
            return particle_start, particle, connector
    return None


def _right_particle(text: str, end: int, upper_bound: int) -> tuple[int, str, str] | None:
    connector_start = end
    connector_end = connector_start
    while connector_end < upper_bound and connector_end - connector_start < 3 and _is_connector(text[connector_end]):
        connector_end += 1
    connector = text[connector_start:connector_end]
    if not connector:
        return None

    for length in range(1, 5):
        particle_end = connector_end + length
        if particle_end > upper_bound:
            break
        particle = text[connector_end:particle_end]
        if _is_name_particle(particle):
            return particle_end, particle, connector
    return None


def _rescue_capitalized_particles(
    text: str,
    detection: Detection,
    all_detections: Sequence[Detection],
    index: int,
) -> Detection:
    if detection.label not in _PERSON_LABELS:
        return detection

    lower_bound = 0
    upper_bound = len(text)
    for current_index, other in enumerate(all_detections):
        if current_index == index:
            continue
        if other.end <= detection.start:
            lower_bound = max(lower_bound, other.end)
        if other.start >= detection.end:
            upper_bound = min(upper_bound, other.start)

    start = detection.start
    end = detection.end
    left = _left_particle(text, start, lower_bound)
    if left is not None:
        particle_start, particle, connector = left
        if "." not in connector or len(particle) == 1:
            start = particle_start
    right = _right_particle(text, end, upper_bound)
    if right is not None:
        particle_end, _particle, connector = right
        if "." not in connector:
            end = particle_end
    if start == detection.start and end == detection.end:
        return detection
    return Detection(start, end, detection.label, detection.score)


def _repair_model_detections(text: str, detections: Sequence[Detection]) -> list[Detection]:
    kept = [detection for detection in detections if detection.score >= DEFAULT_MIN_SCORE]
    candidates = [detection for detection in detections if _EXTEND_SCORE <= detection.score < DEFAULT_MIN_SCORE]

    for _iteration in range(32):
        changed = False
        remaining: list[Detection] = []
        for candidate in candidates:
            if any(_can_bridge(text, candidate, anchor) for anchor in kept):
                kept.append(candidate)
                changed = True
            else:
                remaining.append(candidate)
        candidates = remaining

        merged = _merge_adjacent(text, kept)
        if merged != kept:
            kept = merged
            changed = True

        repaired = [_rescue_capitalized_particles(text, detection, kept, index) for index, detection in enumerate(kept)]
        if repaired != kept:
            kept = repaired
            changed = True
        if not changed:
            break

    return sorted(kept, key=lambda detection: (detection.start, detection.end))


class RampartSanitizer:
    """Runs Rampart with deterministic pre-masking and bounded failure behavior."""

    def __init__(
        self,
        classifier: TokenClassifier,
        *,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        max_latency_ms: int = DEFAULT_MAX_LATENCY_MS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not 1 <= max_content_chars <= MAX_CONTENT_CHARS:
            raise ValueError(f"max_content_chars must be between 1 and {MAX_CONTENT_CHARS}")
        if max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be greater than zero")
        self._classifier = classifier
        self._max_content_chars = max_content_chars
        self._max_latency_ms = max_latency_ms
        self._clock = clock

    @property
    def max_content_chars(self) -> int:
        """Maximum aggregate content size accepted by this sanitizer."""
        return self._max_content_chars

    def sanitize(self, text: str) -> str:
        """Return an observability-safe copy of one semantic content string."""
        if not text:
            return text
        if len(text) > self._max_content_chars:
            return FAILURE_REPLACEMENT

        try:
            started_at = self._clock()
            deterministic = [
                *_deterministic_detections(text),
                *_private_identifier_detections(text),
            ]
            operational = _operational_spans(text)
            masked, offsets = _premask(text, [*deterministic, *operational])
            # Rampart trains and serves with hyphens folded to spaces. The
            # replacement is one-to-one, so classifier offsets remain valid.
            raw_model_detections = self._classifier(masked.replace("-", " "))
            elapsed_ms = (self._clock() - started_at) * 1_000
            if elapsed_ms > self._max_latency_ms:
                return FAILURE_REPLACEMENT

            model_detections: list[Detection] = []
            for raw in raw_model_detections:
                if not isinstance(raw, Mapping):
                    raise ValueError("Rampart classifier returned a non-object detection")
                detection = _model_detection(raw, offsets)
                if detection is None:
                    continue
                if any(
                    detection.start < protected.end and protected.start < detection.end for protected in operational
                ):
                    continue
                model_detections.append(detection)
            repaired_model_detections = _repair_model_detections(text, model_detections)
            detections = _prefer_non_overlapping([*deterministic, *repaired_model_detections])
            if not detections:
                return text

            output = text
            for detection in reversed(detections):
                output = f"{output[: detection.start]}[REDACTED:{detection.label}]{output[detection.end :]}"
            return output
        except Exception:
            return FAILURE_REPLACEMENT
