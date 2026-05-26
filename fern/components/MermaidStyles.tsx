/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const mermaidCss = `
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) {
  --surface: #ffffff;
  --border: #e0e0e0;
  --line: #757575;
  --text: #000000;
  --edge-label-background: #ffffff;
  background-color: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 1.25rem 0;
  overflow: auto;
  padding: 1rem;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg {
  background-color: var(--surface) !important;
  color: var(--text);
  display: block;
  height: auto;
  margin: 0 auto;
  max-width: 100%;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .node :is(rect, polygon, circle, ellipse, path),
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .cluster rect {
  stroke-width: 2px !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgePath .path,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .flowchart-link {
  stroke: var(--line) !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .marker,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .marker.cross,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .arrowheadPath {
  fill: var(--line) !important;
  stroke: var(--line) !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgeLabel rect,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .labelBkg {
  background-color: var(--edge-label-background) !important;
  fill: var(--edge-label-background) !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgeLabel p {
  background-color: var(--edge-label-background) !important;
  color: var(--text) !important;
}

.mermaid-container svg .edgeLabel p,
.mermaid-container svg .edgeLabel span.edgeLabel,
.mermaid-container svg .edgeLabel .labelBkg,
.mermaid-container-expanded svg .edgeLabel p,
.mermaid-container-expanded svg .edgeLabel span.edgeLabel,
.mermaid-container-expanded svg .edgeLabel .labelBkg,
.mermaid svg .edgeLabel p {
  background-color: var(--edge-label-background) !important;
  color: var(--text) !important;
}

.mermaid svg .edgeLabel span.edgeLabel,
.mermaid svg .edgeLabel .labelBkg {
  background-color: var(--edge-label-background) !important;
  color: var(--text) !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .label text,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .label span,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .label tspan,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .nodeLabel,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .nodeLabel p,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .cluster-label text,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .cluster-label span,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .cluster-label tspan,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgeLabel text,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgeLabel span,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgeLabel tspan,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .edgeLabel p {
  color: var(--text) !important;
  fill: var(--text) !important;
}

.mermaid-container svg .edgeLabel :is(text, span, tspan, p),
.mermaid-container-expanded svg .edgeLabel :is(text, span, tspan, p),
.mermaid svg .edgeLabel :is(text, span, tspan, p) {
  color: var(--text) !important;
  fill: var(--text) !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .grey-lightest,
:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .grey-hint {
  --mermaid-fill: #e0e0e0;
  --mermaid-stroke: #a7a7a7;
  --mermaid-text: #000000;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .green-lightest {
  --mermaid-fill: #cfff40;
  --mermaid-stroke: #76b900;
  --mermaid-text: #000000;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .yellow-lightest {
  --mermaid-fill: #feeeb2;
  --mermaid-stroke: #fcde7b;
  --mermaid-text: #000000;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .red-lightest {
  --mermaid-fill: #ffd7d7;
  --mermaid-stroke: #ff8181;
  --mermaid-text: #000000;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .magenta-lightest {
  --mermaid-fill: #ffd3f2;
  --mermaid-stroke: #fc79ca;
  --mermaid-text: #000000;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .teal-lightest {
  --mermaid-fill: #adfcf8;
  --mermaid-stroke: #9aefe5;
  --mermaid-text: #000000;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg :is(
  .grey-lightest, .grey-hint, .green-lightest, .yellow-lightest,
  .red-lightest, .magenta-lightest, .teal-lightest
) > * {
  color: var(--mermaid-text) !important;
  fill: var(--mermaid-fill) !important;
  stroke: var(--mermaid-stroke) !important;
}

:where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg :is(
  .grey-lightest, .grey-hint, .green-lightest, .yellow-lightest,
  .red-lightest, .magenta-lightest, .teal-lightest
) :is(text, tspan, span, p, div, foreignObject) {
  color: var(--mermaid-text) !important;
  fill: var(--mermaid-text) !important;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) {
  --surface: #000000;
  --border: #5f5f5f;
  --line: #a7a7a7;
  --text: #f7f7f7;
  --edge-label-background: #000000;
  background-color: var(--surface) !important;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .grey-lightest,
:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .grey-hint {
  --mermaid-fill: #1f1f1f;
  --mermaid-stroke: #5f5f5f;
  --mermaid-text: #f7f7f7;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .green-lightest {
  --mermaid-fill: #265600;
  --mermaid-stroke: #76b900;
  --mermaid-text: #ffffff;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .yellow-lightest {
  --mermaid-fill: #4b2d00;
  --mermaid-stroke: #ef9100;
  --mermaid-text: #ffffff;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .red-lightest {
  --mermaid-fill: #650b0b;
  --mermaid-stroke: #ff8181;
  --mermaid-text: #ffffff;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .magenta-lightest {
  --mermaid-fill: #5d1337;
  --mermaid-stroke: #fc79ca;
  --mermaid-text: #ffffff;
}

:where(.dark, html[data-theme="dark"], html[data-mode="dark"]) :where(.mermaid-container, .mermaid-container-expanded, .mermaid) svg .teal-lightest {
  --mermaid-fill: #04554b;
  --mermaid-stroke: #1dbba4;
  --mermaid-text: #ffffff;
}
`;

export function MermaidStyles() {
  return <style dangerouslySetInnerHTML={{ __html: mermaidCss }} />;
}
