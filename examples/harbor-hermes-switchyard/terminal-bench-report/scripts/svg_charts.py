#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Dependency-free SVG charts for Terminal-Bench reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

COLORS = ["#76B900", "#1F6FEB", "#A371F7", "#F0883E", "#DB6D92", "#2DA44E"]
INK = "#17212B"
MUTED = "#667085"
GRID = "#D0D5DD"
BACKGROUND = "#FFFFFF"


def _text(x: float, y: float, value: Any, *, size: int = 20, anchor: str = "start", weight: int = 400) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans,Arial,sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{INK}">'
        f"{html.escape(str(value))}</text>"
    )


def _rotated_text(
    x: float, y: float, value: Any, *, angle: int = -52, size: int = 14, anchor: str = "end", weight: int = 600
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate({angle} {x:.1f} {y:.1f})" '
        f'font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="{INK}">{html.escape(str(value))}</text>'
    )


def _document(width: int, height: int, body: list[str], title: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{html.escape(title)}</title>",
            f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def grouped_bars(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[float | None]]],
    *,
    percent: bool = False,
    currency: bool = False,
) -> None:
    width, height = 1400, 800
    dense_labels = len(labels) > 8
    left, right, top, bottom = 150, 60, 100, 230 if dense_labels else 150
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for _, items in series for value in items if value is not None]
    maximum = max(values, default=1.0)
    if percent:
        maximum = 1.0
    elif maximum > 0:
        maximum *= 1.12
    body = [_text(width / 2, 48, title, size=30, anchor="middle", weight=700)]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>')
        label = f"{value * 100:.0f}%" if percent else f"${value:.2f}" if currency else f"{value:.2f}"
        body.append(_text(left - 18, y + 7, label, size=17, anchor="end"))
    group_w = plot_w / max(1, len(labels))
    bar_w = min(110, group_w * 0.72 / max(1, len(series)))
    for group_index, label in enumerate(labels):
        center = left + group_w * (group_index + 0.5)
        if dense_labels:
            body.append(_rotated_text(center + 5, top + plot_h + 35, label))
        else:
            body.append(_text(center, top + plot_h + 42, label, size=18, anchor="middle", weight=600))
        for series_index, (series_name, items) in enumerate(series):
            value = items[group_index] if group_index < len(items) else None
            if value is None:
                continue
            x = center - len(series) * bar_w / 2 + series_index * bar_w
            bar_h = plot_h * max(0.0, value) / maximum if maximum else 0
            y = top + plot_h - bar_h
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" fill="{COLORS[series_index % len(COLORS)]}"/>'
            )
            value_label = f"{value * 100:.1f}%" if percent else f"${value:.2f}" if currency else f"{value:.2f}"
            body.append(_text(x + (bar_w - 4) / 2, y - 10, value_label, size=15, anchor="middle"))
    legend_y = height - 45
    start_x = width / 2 - len(series) * 150
    for index, (name, _) in enumerate(series):
        x = start_x + index * 300
        body.append(
            f'<rect x="{x:.1f}" y="{legend_y - 17}" width="22" height="22" fill="{COLORS[index % len(COLORS)]}"/>'
        )
        body.append(_text(x + 32, legend_y, name, size=17))
    path.write_text(_document(width, height, body, title), encoding="utf-8")


def mean_bars_with_points(
    path: Path,
    title: str,
    labels: list[str],
    means: list[float | None],
    sample_sds: list[float | None],
    observations: list[list[float]],
    *,
    percent: bool = False,
    currency: bool = False,
) -> None:
    """Plot group means, sample-SD error bars, and every run observation."""

    width, height = 1400, 800
    left, right, top, bottom = 150, 60, 100, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    candidates = [
        float(value) + float(sd or 0) for value, sd in zip(means, sample_sds, strict=True) if value is not None
    ]
    maximum = max(candidates, default=1.0)
    if percent:
        maximum = 1.0
    elif maximum > 0:
        maximum *= 1.12
    body = [_text(width / 2, 48, title, size=30, anchor="middle", weight=700)]
    for tick in range(6):
        value = maximum * tick / 5
        y = top + plot_h - plot_h * tick / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID}"/>')
        label = f"{value * 100:.0f}%" if percent else f"${value:.2f}" if currency else f"{value:.2f}"
        body.append(_text(left - 18, y + 7, label, size=17, anchor="end"))
    group_w = plot_w / max(1, len(labels))
    bar_w = min(150, group_w * 0.56)
    for index, label in enumerate(labels):
        mean = means[index] if index < len(means) else None
        if mean is None:
            continue
        center = left + group_w * (index + 0.5)
        color = COLORS[index % len(COLORS)]
        bar_h = plot_h * max(0.0, mean) / maximum if maximum else 0
        y = top + plot_h - bar_h
        body.append(
            f'<rect x="{center - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.82"/>'
        )
        value_label = f"{mean * 100:.2f}%" if percent else f"${mean:.4f}" if currency else f"{mean:.3f}"
        body.append(_text(center, y - 28, value_label, size=16, anchor="middle", weight=700))
        sd = sample_sds[index] if index < len(sample_sds) else None
        if sd is not None:
            high = min(maximum, mean + sd)
            low = max(0.0, mean - sd)
            high_y = top + plot_h - plot_h * high / maximum
            low_y = top + plot_h - plot_h * low / maximum
            body.append(
                f'<line x1="{center}" y1="{high_y:.1f}" x2="{center}" y2="{low_y:.1f}" stroke="{INK}" stroke-width="3"/>'
            )
            body.append(
                f'<line x1="{center - 18}" y1="{high_y:.1f}" x2="{center + 18}" y2="{high_y:.1f}" stroke="{INK}" stroke-width="3"/>'
            )
            body.append(
                f'<line x1="{center - 18}" y1="{low_y:.1f}" x2="{center + 18}" y2="{low_y:.1f}" stroke="{INK}" stroke-width="3"/>'
            )
        points = observations[index] if index < len(observations) else []
        for point_index, point in enumerate(points):
            offset = (point_index - (len(points) - 1) / 2) * 22
            point_y = top + plot_h - plot_h * max(0.0, point) / maximum if maximum else top + plot_h
            body.append(
                f'<circle cx="{center + offset:.1f}" cy="{point_y:.1f}" r="7" fill="{BACKGROUND}" stroke="{INK}" stroke-width="3"/>'
            )
        body.append(_text(center, top + plot_h + 42, label, size=18, anchor="middle", weight=600))
        body.append(_text(center, top + plot_h + 70, f"N={len(points)}", size=15, anchor="middle"))
    body.append(
        _text(
            width / 2,
            height - 35,
            "Bar = mean; whisker = sample SD; circles = independent runs",
            size=16,
            anchor="middle",
        )
    )
    path.write_text(_document(width, height, body, title), encoding="utf-8")


def stacked_model_bars(path: Path, title: str, runs: list[dict[str, Any]]) -> None:
    models = sorted({model for run in runs for model in run.get("model_counts", {})})
    labels = [str(run["label"]) for run in runs]
    width, height = 1400, 900
    left, right, top, bottom = 150, 80, 110, 310
    plot_w, plot_h = width - left - right, height - top - bottom
    body = [_text(width / 2, 48, title, size=30, anchor="middle", weight=700)]
    group_w = plot_w / max(1, len(runs))
    for index, run in enumerate(runs):
        total = sum(int(value) for value in run.get("model_counts", {}).values())
        x = left + group_w * (index + 0.2)
        bar_w = group_w * 0.6
        y = top + plot_h
        for model_index, model in enumerate(models):
            value = int(run.get("model_counts", {}).get(model, 0))
            height_part = plot_h * value / total if total else 0
            y -= height_part
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height_part:.1f}" fill="{COLORS[model_index % len(COLORS)]}"/>'
            )
            if height_part >= 34:
                body.append(_text(x + bar_w / 2, y + height_part / 2 + 6, value, size=16, anchor="middle", weight=700))
        label_x = x + bar_w / 2 + 5
        body.append(_rotated_text(label_x, top + plot_h + 35, labels[index], angle=-60, size=13))
        body.append(_text(x + bar_w / 2, top - 16, f"n={total}", size=16, anchor="middle"))
    legend_y = height - 110
    for index, model in enumerate(models):
        y = legend_y + index * 30
        body.append(f'<rect x="{left}" y="{y - 17}" width="22" height="22" fill="{COLORS[index % len(COLORS)]}"/>')
        body.append(_text(left + 32, y, model, size=16))
    path.write_text(_document(width, height, body, title), encoding="utf-8")


def outcome_matrix(path: Path, title: str, tasks: list[dict[str, Any]], run_labels: list[str]) -> None:
    by_task: dict[str, dict[str, bool | None]] = {}
    for task in tasks:
        by_task.setdefault(str(task["task_name"]), {})[str(task["run_label"])] = task["benchmark_passed"]
    names = sorted(by_task)
    width = 1450
    height = 160 + len(run_labels) * 70
    left, top = 260, 85
    matrix_w = width - left - 50
    step = matrix_w / max(1, len(names))
    body = [_text(width / 2, 42, title, size=28, anchor="middle", weight=700)]
    colors = {True: "#76B900", False: "#D73A49", None: "#D0D5DD"}
    for row, label in enumerate(run_labels):
        y = top + row * 70
        body.append(_text(left - 20, y + 20, label, size=18, anchor="end", weight=600))
        for column, name in enumerate(names):
            value = by_task[name].get(label)
            x = left + column * step
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(3, step - 1):.1f}" height="36" fill="{colors[value]}"/>'
            )
    body.append(_text(left, height - 28, "Tasks in lexical order (green=pass, red=nonpass, gray=missing)", size=16))
    path.write_text(_document(width, height, body, title), encoding="utf-8")
