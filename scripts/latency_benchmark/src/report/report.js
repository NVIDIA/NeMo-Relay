/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

"use strict";

const benchmark = JSON.parse(document.getElementById("benchmark-data").textContent);
const svgNamespace = "http://www.w3.org/2000/svg";
const colors = ["#5f8f00", "#006e9c", "#b45f06", "#7042a6", "#b3261e", "#007c70", "#725f00", "#4554a3"];

function byId(id) {
  return document.getElementById(id);
}

function addElement(parent, tag, attributes = {}, text = null) {
  const element = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  if (text !== null) {
    element.textContent = text;
  }
  parent.appendChild(element);
  return element;
}

function addSvg(parent, tag, attributes = {}, text = null) {
  const element = document.createElementNS(svgNamespace, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  if (text !== null) {
    element.textContent = text;
  }
  parent.appendChild(element);
  return element;
}

function labelName(value) {
  const names = {
    "direct": "Direct",
    "relay-minimal": "Relay minimal",
    "relay-file": "ATOF file exporter",
    "relay-otlp": "OTLP exporter",
    "relay-minimal_vs_direct": "Minimal − direct",
    "relay-file_vs_direct": "ATOF file exporter − direct",
    "relay-otlp_vs_direct": "OTLP exporter − direct",
    "file_exporter_vs_minimal": "ATOF file exporter − minimal",
    "otlp_exporter_vs_minimal": "OTLP exporter − minimal",
    "process_baseline": "Process baseline",
  };
  if (names[value]) {
    return names[value];
  }
  if (value.endsWith("_vs_direct")) {
    return `${labelName(value.slice(0, -"_vs_direct".length))} − direct`;
  }
  if (value.endsWith("_vs_minimal")) {
    return `${labelName(value.slice(0, -"_vs_minimal".length))} − minimal`;
  }
  if (value.endsWith("_vs_process_baseline")) {
    return `${labelName(value.slice(0, -"_vs_process_baseline".length))} − process baseline`;
  }
  if (value.startsWith("relay-")) {
    return `Relay ${value.slice("relay-".length).replaceAll("-", " ")}`;
  }
  const label = value.replaceAll("_", " ").replaceAll("-", " ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function formatMs(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const magnitude = Math.abs(value);
  let digits = 3;
  if (magnitude >= 100) {
    digits = 1;
  } else if (magnitude >= 10) {
    digits = 2;
  }
  return `${value.toFixed(digits)} ms`;
}

function formatBytes(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  const units = ["B", "KiB", "MiB", "GiB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024;
    unit += 1;
  }
  const digits = amount >= 10 || unit === 0 ? 0 : 1;
  return `${amount.toFixed(digits)} ${units[unit]}`;
}

function formatValue(value) {
  if (Array.isArray(value)) {
    return value.map((item) => formatValue(item)).join("; ");
  }
  if (value !== null && typeof value === "object") {
    return Object.entries(value)
      .map(([key, item]) => `${key}: ${formatValue(item)}`)
      .join("; ");
  }
  return String(value);
}

function summaryCard(parent, label, value) {
  const card = addElement(parent, "div", { class: "summary-card" });
  addElement(card, "span", { class: "label" }, label);
  addElement(card, "strong", { class: "value" }, value);
}

function renderKeyValueTable(parent, values) {
  const rows = Object.entries(values).map(([key, value]) => [labelName(key), formatValue(value)]);
  renderTable(parent, ["Field", "Value"], rows);
}

function renderTable(parent, headers, rows, numericColumns = []) {
  parent.replaceChildren();
  const table = addElement(parent, "table");
  const head = addElement(table, "thead");
  const headRow = addElement(head, "tr");
  headers.forEach((header) => addElement(headRow, "th", { scope: "col" }, header));
  const body = addElement(table, "tbody");
  for (const row of rows) {
    const tableRow = addElement(body, "tr");
    row.forEach((value, index) => {
      addElement(tableRow, "td", numericColumns.includes(index) ? { class: "numeric" } : {}, value);
    });
  }
}

function renderOverview() {
  const environment = benchmark.environment || {};
  const parameters = benchmark.parameters || {};
  const tests = parameters.tests || [];

  const cards = byId("summary-cards");
  summaryCard(cards, "Suites", tests.join(", ") || "None");
  summaryCard(cards, "Gateway scenarios", String((benchmark.gateway || []).length));
  summaryCard(cards, "Relay", environment.relay_version || "unknown");
  summaryCard(cards, "Working tree", environment.git_dirty ? "Dirty" : "Clean");
  summaryCard(cards, "Platform", environment.platform || "unknown");

  renderKeyValueTable(byId("environment-table"), environment);
  renderKeyValueTable(byId("parameters-table"), parameters);
}

function populateSelect(select, entries) {
  select.replaceChildren();
  for (const [value, label] of entries) {
    addElement(select, "option", { value }, label);
  }
}

function uniqueSorted(values, numeric = false) {
  const unique = [...new Set(values)];
  return unique.sort(numeric ? (left, right) => left - right : undefined);
}

function graphRange(values, includeZero) {
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (includeZero) {
    minimum = Math.min(minimum, 0);
    maximum = Math.max(maximum, 0);
  }
  if (minimum === maximum) {
    const padding = Math.max(Math.abs(minimum) * 0.15, 0.1);
    minimum -= padding;
    maximum += padding;
  } else {
    const padding = (maximum - minimum) * 0.12;
    minimum -= padding;
    maximum += padding;
  }
  return [minimum, maximum];
}

function drawLineChart(svg, series, payloads, includeZero) {
  svg.replaceChildren();
  const width = 920;
  const height = 390;
  const margin = { top: 28, right: 24, bottom: 66, left: 82 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const values = series.flatMap((item) => item.values.map((point) => point.value));
  const [minimum, maximum] = graphRange(values, includeZero);
  const x = (index) => margin.left + (payloads.length === 1 ? innerWidth / 2 : (index / (payloads.length - 1)) * innerWidth);
  const y = (value) => margin.top + ((maximum - value) / (maximum - minimum)) * innerHeight;

  for (let index = 0; index <= 5; index += 1) {
    const value = minimum + ((maximum - minimum) * index) / 5;
    const position = y(value);
    addSvg(svg, "line", {
      class: Math.abs(value) < (maximum - minimum) / 1000 ? "zero-line" : "grid-line",
      x1: margin.left,
      y1: position,
      x2: width - margin.right,
      y2: position,
    });
    addSvg(svg, "text", { class: "chart-label", x: margin.left - 12, y: position + 4, "text-anchor": "end" }, formatMs(value));
  }

  addSvg(svg, "line", {
    class: "axis",
    x1: margin.left,
    y1: margin.top + innerHeight,
    x2: width - margin.right,
    y2: margin.top + innerHeight,
  });
  payloads.forEach((payload, index) => {
    const position = x(index);
    addSvg(svg, "line", {
      class: "axis",
      x1: position,
      y1: margin.top + innerHeight,
      x2: position,
      y2: margin.top + innerHeight + 6,
    });
    addSvg(
      svg,
      "text",
      { class: "chart-label", x: position, y: height - 30, "text-anchor": "middle" },
      formatBytes(payload),
    );
  });
  addSvg(svg, "text", { class: "chart-label", x: margin.left + innerWidth / 2, y: height - 6, "text-anchor": "middle" }, "Request content size");

  series.forEach((item, seriesIndex) => {
    const color = colors[seriesIndex % colors.length];
    const points = item.values.map((point, index) => `${x(index)},${y(point.value)}`).join(" ");
    addSvg(svg, "polyline", {
      class: "line-series",
      points,
      stroke: color,
      "stroke-dasharray": strokeDasharray(seriesIndex),
    });
    item.values.forEach((point, index) => {
      const circle = addSvg(svg, "circle", {
        class: "chart-point",
        cx: x(index),
        cy: y(point.value),
        r: 5,
        fill: color,
      });
      addSvg(circle, "title", {}, `${item.label}, ${formatBytes(point.payload)}: ${formatMs(point.value)}`);
    });
  });
}

function strokeDasharray(seriesIndex) {
  if (seriesIndex === 1) {
    return "9 5";
  }
  if (seriesIndex === 2) {
    return "3 5";
  }
  return "none";
}

function renderLegend(series) {
  const legend = byId("gateway-legend");
  legend.replaceChildren();
  series.forEach((item, index) => {
    const entry = addElement(legend, "span", { class: "legend-item" });
    addElement(entry, "span", {
      class: "legend-swatch",
      style: `border-color: ${colors[index % colors.length]}`,
      "aria-hidden": "true",
    });
    addElement(entry, "span", {}, item.label);
  });
}

function gatewaySeries(scenarios, view, metric, statistic) {
  const selected = scenarios[0];
  const definitions = view === "absolute"
    ? Object.keys(selected.absolute)
    : Object.keys(selected.comparisons).filter((name) => name.endsWith(view === "minimal" ? "_vs_minimal" : "_vs_direct"));
  return definitions.map((name) => ({
    name,
    label: labelName(name),
    values: scenarios.map((scenario) => {
      const collection = view === "absolute" ? scenario.absolute : scenario.comparisons;
      return {
        payload: scenario.payload_bytes,
        value: collection[name][metric][statistic],
        summary: collection[name][metric],
      };
    }),
  }));
}

function selectedGatewayScenarios() {
  const provider = byId("gateway-provider").value;
  const mode = byId("gateway-mode").value;
  const concurrency = Number(byId("gateway-concurrency").value);
  return benchmark.gateway
    .filter((scenario) => scenario.provider === provider && scenario.mode === mode && scenario.concurrency === concurrency)
    .sort((left, right) => left.payload_bytes - right.payload_bytes);
}

function syncMetricOptions() {
  const metric = byId("gateway-metric");
  const previous = metric.value;
  const options = byId("gateway-mode").value === "streaming"
    ? [["first_content", "Time to first content"], ["total", "Total response"]]
    : [["total", "Total response"]];
  populateSelect(metric, options);
  if (options.some(([value]) => value === previous)) {
    metric.value = previous;
  }
}

function renderGateway() {
  const scenarios = selectedGatewayScenarios();
  if (!scenarios.length) {
    return;
  }
  const view = byId("gateway-view").value;
  const metric = byId("gateway-metric").value;
  const statistic = byId("gateway-percentile").value;
  const series = gatewaySeries(scenarios, view, metric, statistic);
  const payloads = scenarios.map((scenario) => scenario.payload_bytes);
  const isDelta = view !== "absolute";
  drawLineChart(byId("gateway-chart"), series, payloads, isDelta);
  renderLegend(series);

  const metricLabel = metric === "first_content" ? "time to first content" : "total response time";
  let viewLabel = "Relay overhead relative to direct provider calls";
  if (view === "absolute") {
    viewLabel = "absolute latency";
  } else if (view === "minimal") {
    viewLabel = "variant overhead relative to minimal Relay";
  }
  byId("gateway-chart-description").textContent =
    `${statistic.replace("_ms", "")} ${metricLabel}; ${viewLabel}. ` +
    `Provider ${byId("gateway-provider").value}, ${byId("gateway-mode").value}, ` +
    `concurrency ${byId("gateway-concurrency").value}.`;

  const rows = [];
  for (const item of series) {
    for (const point of item.values) {
      const interval = point.summary.median_ci95_ms;
      rows.push([
        item.label,
        formatBytes(point.payload),
        formatMs(point.summary.p50_ms),
        formatMs(point.summary.p95_ms),
        formatMs(point.summary.p99_ms),
        interval ? `${formatMs(interval[0])} to ${formatMs(interval[1])}` : "—",
        String(point.summary.samples),
      ]);
    }
  }
  renderTable(
    byId("gateway-table"),
    ["Path or comparison", "Payload", "p50", "p95", "p99", "Median 95% CI", "Samples"],
    rows,
    [2, 3, 4, 6],
  );
}

function initializeGateway() {
  if (!benchmark.gateway?.length) {
    byId("gateway-section").hidden = true;
    return;
  }
  populateSelect(
    byId("gateway-provider"),
    uniqueSorted(benchmark.gateway.map((scenario) => scenario.provider)).map((value) => [value, value]),
  );
  populateSelect(
    byId("gateway-mode"),
    uniqueSorted(benchmark.gateway.map((scenario) => scenario.mode)).map((value) => [value, value]),
  );
  populateSelect(
    byId("gateway-concurrency"),
    uniqueSorted(benchmark.gateway.map((scenario) => scenario.concurrency), true).map((value) => [String(value), String(value)]),
  );
  populateSelect(byId("gateway-percentile"), [["p50_ms", "p50"], ["p95_ms", "p95"], ["p99_ms", "p99"]]);
  populateSelect(byId("gateway-view"), [
    ["direct", "Paired delta vs direct"],
    ["minimal", "Paired delta vs minimal"],
    ["absolute", "Absolute latency"],
  ]);
  syncMetricOptions();
  for (const id of [
    "gateway-provider",
    "gateway-mode",
    "gateway-concurrency",
    "gateway-metric",
    "gateway-percentile",
    "gateway-view",
  ]) {
    byId(id).addEventListener("change", () => {
      if (id === "gateway-mode") {
        syncMetricOptions();
      }
      renderGateway();
    });
  }
  renderGateway();
}

function drawBarChart(svg, entries) {
  svg.replaceChildren();
  const width = 920;
  const rowHeight = 38;
  const margin = { top: 16, right: 90, bottom: 28, left: 220 };
  const innerWidth = width - margin.left - margin.right;
  const height = margin.top + margin.bottom + entries.length * rowHeight;
  const maximum = Math.max(...entries.map((entry) => entry.value), 0.001) * 1.08;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  entries.forEach((entry, index) => {
    const y = margin.top + index * rowHeight;
    const barWidth = Math.max((entry.value / maximum) * innerWidth, 1);
    addSvg(svg, "text", { class: "bar-label", x: margin.left - 12, y: y + 20, "text-anchor": "end" }, entry.label);
    const bar = addSvg(svg, "rect", {
      x: margin.left,
      y: y + 5,
      width: barWidth,
      height: 22,
      rx: 4,
      fill: colors[index % colors.length],
    });
    addSvg(bar, "title", {}, `${entry.label}: ${formatMs(entry.value)}`);
    addSvg(svg, "text", { class: "bar-value", x: margin.left + barWidth + 8, y: y + 20 }, formatMs(entry.value));
  });
}

function summaryRows(collection) {
  return Object.entries(collection).map(([name, summary]) => {
    const interval = summary.median_ci95_ms;
    return [
      labelName(name),
      formatMs(summary.p50_ms),
      formatMs(summary.p95_ms),
      formatMs(summary.p99_ms),
      interval ? `${formatMs(interval[0])} to ${formatMs(interval[1])}` : "—",
      String(summary.samples),
    ];
  });
}

function renderProcessSuite(name) {
  const result = benchmark[name];
  if (!result) {
    byId(`${name}-section`).hidden = true;
    return;
  }
  const entries = Object.entries(result.absolute).map(([path, summary]) => ({
    label: labelName(path),
    value: summary.p50_ms,
  }));
  drawBarChart(byId(`${name}-chart`), entries);
  const headers = ["Path", "p50", "p95", "p99", "Median 95% CI", "Samples"];
  renderTable(byId(`${name}-absolute-table`), headers, summaryRows(result.absolute), [1, 2, 3, 5]);
  renderTable(byId(`${name}-comparison-table`), headers, summaryRows(result.comparisons), [1, 2, 3, 5]);
}

function renderDelivery() {
  if (!benchmark.exporter_delivery) {
    byId("delivery-section").hidden = true;
    return;
  }
  const cards = byId("delivery-cards");
  summaryCard(cards, "ATOF written", formatBytes(benchmark.exporter_delivery.atof_bytes));
  summaryCard(cards, "OTLP requests", Number(benchmark.exporter_delivery.otlp_requests).toLocaleString());
}

renderOverview();
initializeGateway();
renderProcessSuite("hooks");
renderProcessSuite("startup");
renderDelivery();
