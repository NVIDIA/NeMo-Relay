# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize task results and Switchyard 0.3.2 ATOF marks without applying prices."""

import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path


def read_events(path):
    if not path.exists():
        return []
    marks = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as error:
            raise ValueError(f"Malformed event at {path}:{number}: {error}") from None

        # Repeated delivery of one mark must not inflate token or call totals.
        identity = event.get("uuid")
        identity = (identity, event.get("kind"), event.get("scope_category")) if identity else None
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        marks.append(event)
    return marks


def summarize(output):
    results = json.loads((output / "results.json").read_text())
    targets = tomllib.loads((output / "routes.toml").read_text())["targets"]
    efficient_model, capable_model = targets["efficient"]["id"], targets["capable"]["id"]
    rows = []
    for arm in ("baseline", "efficient", "routed"):
        tasks = [task for task in results if task["arm"] == arm]
        events = read_events(output / arm / "logs/events.jsonl")
        marks = [event for event in events if event.get("kind") == "mark"]
        truncated = sum(
            event.get("kind") == "scope"
            and event.get("category") == "llm"
            and event.get("scope_category") == "end"
            and ((event.get("category_profile") or {}).get("annotated_response") or {}).get("finish_reason")
            in {"length", "max_tokens"}
            for event in events
        )
        calls = [m.get("data") or {} for m in marks if m.get("name") in {"switchyard.routing.llm_call"}]
        decisions = [m.get("data") or {} for m in marks if m.get("name") == "switchyard.routing.decision"]
        by_model = Counter(c.get("selected_model") for c in calls if c.get("call_role") == "answer")
        recoveries = {}
        for mark in marks:
            data = mark.get("data") or {}
            evidence = data.get("evidence") or {}
            # 0.3.2 uses both spellings for classifier recovery evidence.
            if evidence.get("source") in {"fail_open", "fall_open"}:
                recoveries[data.get("outcome_id") or mark.get("uuid")] = evidence.get("reason_code", "unspecified")
        tokens = {}
        for role in ("answer", "routing"):
            observed = [c for c in calls if c.get("call_role") == role]
            totals = []
            for mark in marks:
                schema = mark.get("data_schema") or {}
                if not isinstance(schema, dict) or schema.get("name") != "nemo.relay.metric_measurements":
                    continue
                for metric in (mark.get("data") or {}).get("measurements", []):
                    attrs = metric.get("attributes") or {}
                    if (
                        metric.get("name") == "switchyard.routing.llm_tokens"
                        and attrs.get("call_role") == role
                        and attrs.get("token_type") == "total"
                    ):
                        value = metric.get("value")
                        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                            totals.append(value)
            # Report partial metrics as unknown, not a misleading subtotal.
            tokens[role] = sum(totals) if observed and len(totals) == len(observed) else None
        row = {
            "arm": arm,
            "accepted": sum(t["accepted"] for t in tasks),
            "tasks": len(tasks),
            "truncated_calls": truncated,
            "seconds": round(sum(t["seconds"] for t in tasks), 3),
            "answer_attempts": dict(by_model),
            "decisions": len(decisions),
            "served_models": dict(Counter(d.get("served_model") or "unknown" for d in decisions)),
            "judge_errors": sum(c.get("outcome") == "error" for c in calls if c.get("call_role") == "routing"),
            "classifier_recoveries": len(recoveries),
            "recovery_reasons": dict(Counter(recoveries.values())),
            "fallbacks": sum(d.get("fallback_used") is True for d in decisions),
            "unknown_fallbacks": sum(d.get("fallback_used") is None for d in decisions),
            "tokens": tokens,
        }
        rows.append(row)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(
        "arm | accepted | seconds | efficient attempts | capable attempts | "
        "judge errors / fail-open / answer fallbacks | truncated calls | answer tokens | routing tokens"
    )
    for row in rows:
        counts = row["answer_attempts"]
        # Read the recorded configuration, never the caller's current model environment.
        efficient = counts.get(efficient_model, 0) if counts else "unknown"
        capable = counts.get(capable_model, 0) if counts else "unknown"
        if not row["decisions"]:
            efficient = capable = "unknown"

        def display(value):
            return "unknown" if value is None else str(value)

        print(
            f"{row['arm']} | {row['accepted']}/{row['tasks']} | {row['seconds']} | {efficient} | {capable} | "
            f"{row['judge_errors']}/{row['classifier_recoveries']}/{row['fallbacks']} "
            f"(unknown: {row['unknown_fallbacks']}) | "
            f"{row['truncated_calls']} | {display(row['tokens']['answer'])} | {display(row['tokens']['routing'])}"
        )
    print("Answer attempts are selected targets; served_models is in summary.json. Missing usage stays unknown.")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        summarize(args.output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"Cannot summarize: {error}. Check results.json and arm logs.") from None
