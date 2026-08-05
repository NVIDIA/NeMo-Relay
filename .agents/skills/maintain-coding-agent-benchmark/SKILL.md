---
name: maintain-coding-agent-benchmark
description: Run, configure, troubleshoot, maintain, or expand the NeMo Relay coding-agent latency benchmark fixture. Use for changes under scripts/latency_benchmark, custom middleware benchmark variants, new benchmark suites or matrix axes, static provider and Relay fixtures, result reporting, or coding-agent benchmark documentation.
---

# Maintain The Coding-Agent Latency Benchmark

## Companion Guidance

Use `karpathy-guidelines` for implementation and `validate-change` to select
final checks. Keep benchmark changes isolated from runtime behavior.

## Understand The Layout

- Use `python -m scripts.latency_benchmark.src` as the direct entry
  point and `just latency-benchmark` as the normal wrapper.
- Keep the human run guide in
  `scripts/latency_benchmark/README.md` current with CLI and
  configuration changes.
- Read `scripts/latency_benchmark/config/default.toml` before a run.
  A custom TOML file overlays these defaults, then CLI arguments take final
  precedence.
- Keep all runtime Python modules under `src/`. Change config parsing and
  validation in `src/config.py`.
- Keep OpenAI and Anthropic payload shapes in `src/protocol.py`.
- Keep loopback provider and OTLP behavior in `src/servers.py`.
- Keep temporary Relay and coding-agent process lifecycle in `src/processes.py`.
- Add measurement logic to `src/benchmarks.py`, orchestration to `src/cli.py`,
  and terminal presentation to `src/reporting.py`.
- Keep HTML report assembly in `src/html_report.py` and its static template,
  CSS, and JavaScript under `src/report/`. Keep the report self-contained and
  offline.
- Put TOML assets under `config/` and platform scripts under `data/`. Load or
  render those assets through `src/fixtures.py`; do not embed them in executable
  modules.
- Use `config/plugins-pii-redaction.toml` as the self-contained real-middleware
  smoke fixture. It uses the built-in email detector and requires no external
  service.
- Treat `data/mock-codex.*` as the transparent-mode lifecycle stub, not as the
  Codex hook implementation. Hook measurements call `hook-forward` directly for
  both Codex and Claude Code, so they do not require a mock Claude executable.

## Run The Fixture

List every config override without building Relay:

```bash
just latency-benchmark --help
```

Run a small functional check before a statistically meaningful run:

```bash
just latency-benchmark \
  --tests gateway \
  --providers openai \
  --modes buffered \
  --payload-sizes 4096 \
  --concurrency 1 \
  --samples 5 \
  --warmup 1 \
  --response-bytes 1024
```

Treat a small run only as a correctness check. Run the default matrix with
`just latency-benchmark` when collecting performance data. The
default file-exporter matrix can write tens of gigabytes temporarily, so check
free disk space first. With the current defaults, 11,860 file-exporter gateway
requests contain about 12.3 GiB of request content and are expected to produce
about 25 GiB of ATOF JSON. Reserve at least 30 GiB; treat these values as an
estimate because event serialization can change.

The benchmark writes this large ATOF output and other ephemeral Relay files to
the operating system's temporary directory in a folder named
`nemo-relay-latency-*` by default. This directory is typically under `/tmp`;
macOS can use a private per-user temporary directory instead. A normal run
removes the directory and its large output. The JSON result and HTML report are
smaller persistent artifacts in the configured result directory.

Use a partial TOML config for repeatable experiments:

```toml
tests = ["gateway"]
providers = ["openai"]
modes = ["streaming"]
samples = 50
payload_sizes = [4096, 65536]
concurrency = [1, 4]
```

```bash
just latency-benchmark --config /path/to/benchmark.toml
```

Keep the minimal, file, and OTLP variants enabled on every run. Add opt-in
middleware variants through `[[middleware]]` benchmark config tables:

```toml
[[middleware]]
name = "pii-redaction"
plugin_config = "./plugins-pii-redaction.toml"
```

Run the bundled PII-redaction middleware through a small gateway matrix:

```bash
just latency-benchmark \
  --tests gateway \
  --providers openai \
  --modes buffered \
  --payload-sizes 4096 \
  --concurrency 1 \
  --samples 5 \
  --warmup 1 \
  --response-bytes 1024 \
  --middleware pii-redaction=scripts/latency_benchmark/config/plugins-pii-redaction.toml
```

Treat this command as a middleware lifecycle and reporting check. The static
payload contains no email address, so it does not verify redaction correctness.

Resolve TOML plugin paths relative to the benchmark config. Use repeatable
`--middleware NAME=PATH` options for one-off CLI variants; CLI middleware
entries replace those from the TOML file. Compare each custom gateway variant
with direct calls and minimal Relay, and include it in selected hook and startup
suites.

Find the JSON result and self-contained HTML report at
`target/benchmark-results/nemo-relay-latency-report.json` and
`target/benchmark-results/nemo-relay-latency-report.html` unless `output_dir`
was overridden. Use `--report` only when the HTML path must differ from the
JSON path. Compare added milliseconds and paired confidence intervals; do not
draw performance conclusions from a smoke run.

## Interpret Suites And Metrics

- Treat gateway `total` as request start through buffered-body completion or
  streaming end-of-stream. Treat streaming `first_content` as request start
  through the first content-delta event.
- Use gateway Relay-versus-direct paired deltas for total Relay overhead. Use
  file-versus-minimal and OTLP-versus-minimal deltas to isolate exporter
  overhead.
- Treat hook absolute values as complete `hook-forward` subprocess wall time.
  Hook paired deltas subtract a `nemo-relay --version` process measurement from
  the same cycle.
- Treat startup absolute values as process launch through healthy gateway
  readiness. Startup paired deltas subtract the same process baseline.
- Read p50 as the median and p95/p99 as tail percentiles. Min and max are
  observed extremes. `median_ci95_ms` is a bootstrap uncertainty interval for
  the median paired delta, not an interval containing 95% of observations.
- Treat exporter-delivery bytes and request counts as correctness checks, not
  latency metrics.

## Maintain Measurement Integrity

- Keep providers on loopback and deterministic. Do not add model-service or
  Internet latency to the core fixture.
- Compare variants within the same measurement cycle and retain the rotated or
  randomized execution order to reduce ordering bias.
- Warm persistent connections before recording gateway samples.
- Keep streaming time-to-first-content separate from total stream time.
- Preserve exporter-delivery checks when gateway or hook traffic is measured.
- Keep the three default variants when adding custom middleware. Validate
  custom names and plugin paths before starting subprocesses.
- Record all resolved matrix values in the JSON result so another engineer can
  reproduce the run.
- Keep temporary state isolated from the developer's home and Relay config.

## Expand The Fixture

To add a test suite:

1. Add its name to `AVAILABLE_TESTS` in `src/config.py`.
2. Implement the measurement in `src/benchmarks.py`.
3. Dispatch it conditionally in `src/cli.py` and report it conditionally in
   `src/reporting.py` and the HTML report.
4. Add config and selection tests in
   `scripts/latency_benchmark/tests/test_config.py`.
5. Update `scripts/README.md` and `docs/reference/performance.mdx`.

To add a provider, mode, or matrix axis, update config validation, the protocol
fixture, the loopback server, orchestration, result parameters, and tests
together. Add static fixture files when the change introduces fixed text.

To change middleware variant behavior, update config parsing, fixture config
selection, all selected suite loops, terminal output, HTML series discovery,
tests, and the human README together. Do not hard-code a custom middleware name
in reporting.

## Validate Changes

Format and test the focused surface first:

```bash
uv run ruff format scripts/latency_benchmark
uv run ruff check scripts/latency_benchmark
uv run python -m unittest \
  scripts.latency_benchmark.tests.test_config \
  scripts.latency_benchmark.tests.test_html_report \
  scripts.latency_benchmark.tests.test_processes
```

Run the small functional command above after lifecycle, protocol, server,
fixture, or orchestration changes. It requires permission to bind loopback
ports in restricted environments. Run `just docs` when the performance page
changes and `uv run pre-commit run --all-files` before review.
