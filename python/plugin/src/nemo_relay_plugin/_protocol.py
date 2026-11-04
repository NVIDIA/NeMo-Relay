# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private typed protocol boundary; callback dictionaries are never JSON envelopes."""

from __future__ import annotations

import json
import re
from typing import Any

import protovalidate
from google.protobuf.message_factory import GetMessageClass

from ._proto.buf.validate import validate_pb2

PACKAGE = "nvidia.nemo.relay.worker.v1"
VERSION = "1.0.0"
DIAGNOSTIC_SNAPSHOT_CONTRACT_VERSION = "1.0.0"
DIAGNOSTIC_PROVENANCE_CONTRACT_VERSION = "1.0.0"
_RENAMES = {
    "time": "timestamp",
    "observe_time": "observed_at",
    "capture_time": "captured_at",
    "receive_time": "received_at",
    "last_failure_time": "last_failure_at",
    "start_time": "started_at",
    "last_observe_time": "last_observed_at",
    "register_time": "registered_at",
    "close_time": "closed_at",
    "complete_time": "completed_at",
    "time_to_first_chunk_seconds": "time_to_first_chunk",
    "producer_support_entries": "producer_support",
    "call_type": "type",
    "choice_type": "type",
}
# The enclosing variant determines aliases; the other names remain valid extras.
_CONTENT_PART_FIELDS = {
    "image": {"value": "image"},
    "audio": {"value": "audio"},
    "file": {"value": "file"},
    "refusal": {"text": "refusal"},
}
_TAGS = {
    "Event": "kind",
    "LlmMessage": "role",
    "ContentPart": "type",
    "ToolDefinition": "type",
}
_ARRAY_WRAPPERS = {
    "StringList",
    "ContentPartList",
    "ToolCallList",
    "ToolDefinitionList",
    "ResponseToolCallList",
    "JsonArray",
}
# These native event fields serialize explicit nulls rather than omitting them.
_NULLABLE_EVENT_FIELDS = {
    "BaseEvent": {"parent_uuid", "data", "data_schema", "metadata"},
    "ScopeEvent": {"category_profile"},
    "MarkEvent": {"category", "category_profile"},
}


# Native diagnostic JSON uses decimal strings for counters and explicit nulls.
# Derive transport facts from protobuf; list only native-format exceptions here.
_DIAGNOSTIC_FILES = {
    f"{PACKAGE.replace('.', '/')}/{name}.proto"
    for name in ("diagnostics", "diagnostic_read", "diagnostic_memory", "diagnostic_resource")
}
_NUMERIC_DIAGNOSTICS = {"DiagnosticDiscovery", "DiagnosticObservation", "DiagnosticReceipt", "ResourceMemoryNumber"}
_OMITTED_DIAGNOSTIC_FIELDS = {
    ("DiagnosticClassification", "failure_class"),
    ("DiagnosticCorrelation", "session_id"),
    ("DiagnosticCorrelation", "request_id"),
    ("DiagnosticEvent", "remote"),
    ("DiagnosticQuery", "ownership"),
}


def _diagnostic_nullable(field: Any) -> bool:
    return (
        field.containing_type.file.name in _DIAGNOSTIC_FILES
        and field.containing_type.name != "DiagnosticObservation"
        and field.has_presence
        and (field.containing_oneof is None or field.containing_oneof.name == "_" + field.name)
        and not field.GetOptions().Extensions[validate_pb2.field].required
        and (field.containing_type.name, field.name) not in _OMITTED_DIAGNOSTIC_FIELDS
    )


def validate(value: Any) -> None:
    """Enforce annotations and JSON syntax before dispatch."""
    protovalidate.validate(value)
    validate_json(value)


def validate_json(value: Any) -> None:
    name = value.DESCRIPTOR.full_name
    if name == f"{PACKAGE}.JsonValue":
        json.loads(value.json.decode("utf-8"), parse_constant=_invalid_constant)
    elif name == "google.protobuf.Timestamp":
        if not -62135596800 <= value.seconds <= 253402300799 or not 0 <= value.nanos < 1_000_000_000:
            raise ValueError("invalid google.protobuf.Timestamp")
    elif name == "google.protobuf.Duration":
        if (
            not -315576000000 <= value.seconds <= 315576000000
            or not -999999999 <= value.nanos <= 999999999
            or value.seconds * value.nanos < 0
        ):
            raise ValueError("invalid google.protobuf.Duration")
    for field, item in value.ListFields():
        if field.message_type is None:
            continue
        children = (
            item.values() if field.message_type.GetOptions().map_entry else item if field.is_repeated else (item,)
        )
        for child in children:
            if not hasattr(child, "DESCRIPTOR"):
                continue
            if child.DESCRIPTOR.full_name == f"{PACKAGE}.JsonValue":
                json.loads(child.json.decode("utf-8"), parse_constant=_invalid_constant)
            else:
                validate_json(child)


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def validate_json_object_keys(value: Any, ancestors: set[int] | None = None) -> None:
    """Reject key coercion and cycles before serializing application JSON."""
    if not isinstance(value, (dict, list, tuple)):
        return
    if ancestors is None:
        ancestors = set()
    object_id = id(value)
    if object_id in ancestors:
        raise ValueError("JSON payload must not contain circular references")
    ancestors.add(object_id)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                validate_json_object_keys(item, ancestors)
        else:
            for item in value:
                validate_json_object_keys(item, ancestors)
    finally:
        ancestors.remove(object_id)


def _shape(cls: Any, value: Any) -> dict[str, Any]:
    name = cls.DESCRIPTOR.name
    if name == "MessageContent":
        if not isinstance(value, (str, list, tuple)):
            raise ValueError("MessageContent must be a string or array")
    elif name in _ARRAY_WRAPPERS:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be an array")
    elif name in {"FinishReason", "ToolChoice"}:
        if not isinstance(value, (str, dict)):
            raise ValueError(f"{name} must be a string or object")
    elif not (name == "MemoryLimitEventSlot" and value is None) and not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    if name == "MessageContent":
        return {"text": value} if isinstance(value, str) else {"parts": value}
    if name in _TAGS:
        body = dict(value)
        if not isinstance(body.get(_TAGS[name]), str):
            raise ValueError(f"{name}.{_TAGS[name]} must be a string")
        tag = body.pop(_TAGS[name])
        return {tag: body}
    if name == "FinishReason":
        return {value: {}} if isinstance(value, str) else dict(value)
    if name == "ToolChoice":
        if isinstance(value, str):
            return {value: {}}
        return {"provider_native" if "provider" in value else "specific": value}
    if name == "JsonObject":
        return {"fields": value}
    if name == "ToolExecutionInterceptOutcome":
        return {
            "value": {
                "result": value["result"],
                **({"annotation": value["annotation"]} if value.get("annotation") is not None else {}),
            },
            "pending_marks": value.get("pending_marks", []),
        }
    if name in _ARRAY_WRAPPERS:
        return {"values": value}
    if name == "MemoryLimitEventSlot":
        return {"absent": {}} if value is None else {"value": value}
    if name in {"ResourceMemoryNumber", "ResourceMemoryUnit"}:
        return (
            {value["kind"]: value.get("value", {})} if name.endswith("Number") else {value["dimension"]: value["name"]}
        )
    body = dict(value)
    if name == "CostEstimate":
        body.setdefault("currency", "USD")
    if name in {"AnnotatedLlmRequest", "AnnotatedLlmResponse"} and body.get("api_specific") is None:
        body.pop("api_specific", None)
    if name in {"PendingMarkSpec", "EventSanitizeFields"}:
        body = {key: item for key, item in body.items() if item is not None}
    if name == "CategoryProfile" and body.get("tool_result_annotation") is None:
        body.pop("tool_result_annotation", None)
    if name in {"ScopeEvent", "MarkEvent"}:
        base_fields = cls.DESCRIPTOR.fields_by_name["base"].message_type.fields
        base = {
            f.name: body.pop(_RENAMES.get(f.name, f.name)) for f in base_fields if _RENAMES.get(f.name, f.name) in body
        }
        body["base"] = base
    if name in {"EntryMemoryUsage", "ComponentMemoryUsage"}:
        body["footprint"] = {
            f.name: body.pop(f.name)
            for f in cls.DESCRIPTOR.fields_by_name["footprint"].message_type.fields
            if f.name in body
        }
    return body


def pack(cls: Any, value: Any, *, _field_names: dict[str, str] | None = None) -> Any:
    """Encode domain data into a concrete typed protobuf message."""
    name = cls.DESCRIPTOR.name
    if name == "JsonValue":
        validate_json_object_keys(value)
        return cls(json=json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    body = _shape(cls, value)
    field_names = _RENAMES if _field_names is None else _RENAMES | _field_names
    fields = cls.DESCRIPTOR.fields_by_name
    if "extra" in fields:
        known = {field_names.get(f.name, f.name) for f in fields.values() if f.name != "extra"}
        extra = {k: body.pop(k) for k in list(body) if k not in known}
        body["extra"] = extra
    result = cls()
    for field in fields.values():
        source_name = field.name if field.name in body else field_names.get(field.name, field.name)
        if source_name not in body:
            continue
        item = body.pop(source_name)
        if (
            item is None
            and field.has_presence
            and (field.message_type is None or field.message_type.name != "JsonValue")
        ):
            continue
        target = getattr(result, field.name)
        if field.message_type and field.message_type.GetOptions().map_entry:
            if not isinstance(item, dict):
                raise ValueError(f"{name}.{source_name} must be an object")
            value_field = field.message_type.fields_by_name["value"]
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("JSON object keys must be strings")
                if value_field.message_type:
                    target[key].CopyFrom(pack(GetMessageClass(value_field.message_type), child))
                else:
                    target[key] = child
        elif field.is_repeated:
            if not isinstance(item, (list, tuple)):
                raise ValueError(f"{name}.{source_name} must be an array")
            if field.message_type:
                target.extend(pack(GetMessageClass(field.message_type), child) for child in item)
            elif field.enum_type:
                prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", field.enum_type.name).upper() + "_"
                for child in item:
                    if not isinstance(child, str):
                        raise ValueError(f"{name}.{source_name} elements must be strings")
                    member = field.enum_type.values_by_name.get(prefix + child.upper())
                    if member is None:
                        raise ValueError(f"{name}.{source_name} has unknown value {child!r}")
                    target.append(member.number)
            else:
                if field.type == field.TYPE_STRING and any(not isinstance(child, str) for child in item):
                    raise ValueError(f"{name}.{source_name} elements must be strings")
                target.extend(item)
        elif field.message_type:
            if field.message_type.full_name == "google.protobuf.Timestamp":
                target.FromJsonString(item)
            elif field.message_type.full_name == "google.protobuf.Empty":
                target.SetInParent()
            else:
                target.CopyFrom(
                    pack(
                        GetMessageClass(field.message_type),
                        item,
                        _field_names=_CONTENT_PART_FIELDS.get(field.name) if name == "ContentPart" else None,
                    )
                )
        elif field.enum_type:
            if not isinstance(item, str):
                raise ValueError(f"{name}.{source_name} must be a string")
            prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", field.enum_type.name).upper() + "_"
            member = field.enum_type.values_by_name.get(prefix + item.upper())
            if member is None:
                raise ValueError(f"{name}.{source_name} has unknown value {item!r}")
            setattr(result, field.name, member.number)
        elif field.type in (field.TYPE_UINT64, field.TYPE_INT64, field.TYPE_UINT32, field.TYPE_INT32):
            decimal_string = field.type == field.TYPE_UINT64 and (
                (field.containing_type.file.name in _DIAGNOSTIC_FILES and name not in _NUMERIC_DIAGNOSTICS)
                or (name == "ResourceMemoryNumber" and field.name == "integer")
            )
            if type(item) is not int:
                if decimal_string and isinstance(item, str) and re.fullmatch(r"0|[1-9][0-9]*", item):
                    item = int(item)
                else:
                    raise ValueError(f"{name}.{source_name} must be an integer")
            setattr(result, field.name, item)
        elif field.type == field.TYPE_BOOL:
            if type(item) is not bool:
                raise ValueError(f"{name}.{source_name} must be a boolean")
            setattr(result, field.name, item)
        elif field.type in (field.TYPE_DOUBLE, field.TYPE_FLOAT):
            if type(item) not in (int, float):
                raise ValueError(f"{name}.{source_name} must be a number")
            setattr(result, field.name, item)
        elif field.type == field.TYPE_STRING:
            if not isinstance(item, str):
                raise ValueError(f"{name}.{source_name} must be a string")
            setattr(result, field.name, item)
        else:
            setattr(result, field.name, item)
    if body:
        raise ValueError(f"unknown {name} fields: {sorted(body)}")
    return result


def unpack(value: Any, *, _field_names: dict[str, str] | None = None) -> Any:
    """Decode typed protobuf values into the callback's domain representation."""
    name = value.DESCRIPTOR.name
    if name == "JsonValue":
        return json.loads(value.json.decode("utf-8"), parse_constant=_invalid_constant)
    if value.DESCRIPTOR.full_name == "google.protobuf.Timestamp":
        return value.ToJsonString()
    if value.DESCRIPTOR.full_name == "google.protobuf.Empty":
        return {}
    field_names = _RENAMES if _field_names is None else _RENAMES | _field_names
    body: dict[str, Any] = {}
    for field in value.DESCRIPTOR.fields:
        if field.has_presence and not value.HasField(field.name):
            if field.name in _NULLABLE_EVENT_FIELDS.get(name, ()) or _diagnostic_nullable(field):
                body[field_names.get(field.name, field.name)] = None
            continue
        item = getattr(value, field.name)
        if field.message_type and field.message_type.GetOptions().map_entry:
            item = {k: unpack(v) if hasattr(v, "DESCRIPTOR") else v for k, v in item.items()}
        elif field.is_repeated:
            if field.enum_type:
                prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", field.enum_type.name).upper() + "_"
                converted = []
                for raw in item:
                    member = field.enum_type.values_by_number.get(raw)
                    if member is None:
                        raise ValueError(f"unknown {field.enum_type.name}")
                    converted.append(member.name.removeprefix(prefix).lower())
                item = converted
            else:
                item = [unpack(v) if hasattr(v, "DESCRIPTOR") else v for v in item]
        elif field.message_type:
            item = unpack(item, _field_names=_CONTENT_PART_FIELDS.get(field.name) if name == "ContentPart" else None)
        elif field.enum_type:
            member = field.enum_type.values_by_number.get(item)
            if member is None:
                raise ValueError(f"unknown {field.enum_type.name}")
            prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", field.enum_type.name).upper() + "_"
            item = member.name.removeprefix(prefix).lower()
        if (
            field.containing_type.file.name in _DIAGNOSTIC_FILES
            and name not in _NUMERIC_DIAGNOSTICS
            and field.type == field.TYPE_UINT64
        ):
            item = str(item)
        body[field_names.get(field.name, field.name)] = item
    extra = body.pop("extra", {})
    if extra and set(extra) & {
        field_names.get(field.name, field.name) for field in value.DESCRIPTOR.fields if field.name != "extra"
    }:
        raise ValueError("extension fields shadow modeled fields")
    body.update(extra)
    if name in _TAGS:
        selected = value.WhichOneof(next(iter(value.DESCRIPTOR.oneofs_by_name)))
        if selected is None:
            raise ValueError(f"missing {name} alternative")
        inner = body[selected]
        tag = _TAGS[name]
        if tag in inner:
            raise ValueError("extension fields shadow modeled fields")
        return {tag: selected, **inner}
    if name == "MessageContent":
        selected = value.WhichOneof("content")
        if selected is None:
            raise ValueError("missing content alternative")
        return body[selected]
    if name in {"FinishReason", "ToolChoice"}:
        selected = value.WhichOneof(next(iter(value.DESCRIPTOR.oneofs_by_name)))
        return (
            selected
            if selected in {"complete", "length", "tool_use", "content_filter", "auto", "none", "required"}
            else body[selected]
            if selected in {"specific", "provider_native"}
            else {selected: body[selected]}
        )
    if name == "JsonObject":
        return body["fields"]
    if name.endswith("List") or name == "JsonArray":
        return body["values"]
    if name in {"ScopeEvent", "MarkEvent"}:
        base = body.pop("base")
        return {**base, **body}
    if name == "ToolExecutionInterceptOutcome":
        return {**body.pop("value"), **body}
    if name == "MemoryLimitEventSlot":
        return body.get("value")
    if name in {"EntryMemoryUsage", "ComponentMemoryUsage"}:
        body.update(body.pop("footprint"))
    if name == "ResourceMemoryNumber":
        selected = value.WhichOneof("kind")
        return (
            {"kind": selected}
            if selected == "non_finite"
            else {"kind": selected, "value": str(body[selected]) if selected == "integer" else body[selected]}
        )
    if name == "ResourceMemoryUnit":
        selected = value.WhichOneof("dimension")
        return {"dimension": selected, "name": body[selected]}
    if name == "LlmRequestInterceptOutcome":
        body.setdefault("annotated_request", None)
    if name == "EventSanitizeFields":
        for field in ("data", "metadata", "category_profile"):
            body.setdefault(field, None)
    if name == "FunctionMessage":
        body.setdefault("content", None)
    return body
