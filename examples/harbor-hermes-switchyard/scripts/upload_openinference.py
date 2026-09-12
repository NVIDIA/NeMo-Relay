# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream bounded OTLP JSON batches into a Phoenix HTTP receiver."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from google.protobuf import json_format
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2


def _serialize(resource_spans: list[dict[str, Any]]) -> bytes:
    request = trace_service_pb2.ExportTraceServiceRequest()
    json_format.ParseDict({"resourceSpans": resource_spans}, request)
    return request.SerializeToString()


def _hex_id_to_base64(span: dict[str, Any], field: str, width: int) -> None:
    value = span.get(field)
    if isinstance(value, str) and re.fullmatch(rf"[0-9a-fA-F]{{{width}}}", value):
        span[field] = base64.b64encode(bytes.fromhex(value)).decode("ascii")


def _normalize_ids(payload: dict[str, Any]) -> None:
    for resource_span in payload.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                _hex_id_to_base64(span, "traceId", 32)
                _hex_id_to_base64(span, "spanId", 16)
                _hex_id_to_base64(span, "parentSpanId", 16)


def _set_project(resource_span: dict[str, Any], project: str) -> None:
    attributes = resource_span.setdefault("resource", {}).setdefault("attributes", [])
    for attribute in attributes:
        if attribute.get("key") == "openinference.project.name":
            attribute["value"] = {"stringValue": project}
            return
    attributes.append(
        {
            "key": "openinference.project.name",
            "value": {"stringValue": project},
        }
    )


def _post(endpoint: str, body: bytes, timeout: float, attempts: int) -> int:
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={"content-type": "application/x-protobuf"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"Phoenix returned HTTP {response.status}")
            return attempt - 1
        except (TimeoutError, urllib.error.URLError):
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openinference", type=Path, required=True)
    parser.add_argument("--phoenix-url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_batch_bytes < 1 or args.max_attempts < 1:
        raise ValueError("batch and retry limits must be positive")

    endpoint = args.phoenix_url.rstrip("/") + "/v1/traces"
    pending: list[dict[str, Any]] = []
    pending_documents = 0
    uploaded_documents = 0
    uploaded_batches = 0
    uploaded_spans = 0
    retries = 0

    def upload(items: list[dict[str, Any]]) -> None:
        nonlocal uploaded_batches, retries
        retries += _post(
            endpoint,
            _serialize(items),
            args.timeout_seconds,
            args.max_attempts,
        )
        uploaded_batches += 1

    with args.openinference.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload: dict[str, Any] = json.loads(line)
            _normalize_ids(payload)
            resource_spans = list(payload.get("resourceSpans", []))
            for resource_span in resource_spans:
                _set_project(resource_span, args.project)
                uploaded_spans += sum(len(scope.get("spans", [])) for scope in resource_span.get("scopeSpans", []))
            candidate = pending + resource_spans
            if pending and (pending_documents >= args.batch_size or len(_serialize(candidate)) > args.max_batch_bytes):
                upload(pending)
                pending = resource_spans
                pending_documents = 1
            else:
                pending = candidate
                pending_documents += 1
            uploaded_documents += 1
    if pending:
        upload(pending)
    if uploaded_documents == 0 or uploaded_spans == 0:
        raise RuntimeError("OpenInference artifact did not contain any spans")

    result = {
        "status": "passed",
        "project": args.project,
        "endpoint": endpoint,
        "uploaded_documents": uploaded_documents,
        "uploaded_batches": uploaded_batches,
        "upload_retries": retries,
        "uploaded_spans": uploaded_spans,
    }
    args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
