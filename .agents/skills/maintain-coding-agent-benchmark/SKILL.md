---
name: maintain-coding-agent-benchmark
description: Run, configure, troubleshoot, maintain, or expand the NeMo Relay coding-agent latency benchmark fixture. Use for changes under scripts/benchmark_coding_agent_latency, new benchmark suites or matrix axes, static provider and Relay fixtures, result reporting, or coding-agent benchmark documentation.
---

# Maintain The Coding-Agent Latency Benchmark

## Companion Guidance

Use `karpathy-guidelines` for implementation and `validate-change` to select
final checks. Keep benchmark changes isolated from runtime behavior.

## Understand The Layout

- Use `scripts/benchmark-coding-agent-latency.py` as the stable entry point.
- Read `scripts/benchmark_coding_agent_latency/data/default.toml` before a run.
  A custom TOML file overlays these defaults, then CLI arguments take final
  precedence.
- Change config parsing and validation in `config.py`.
- Keep OpenAI and Anthropic payload shapes in `protocol.py`.
- Keep loopback provider and OTLP behavior in `servers.py`.
- Keep temporary Relay and coding-agent process lifecycle in `processes.py`.
- Add measurement logic to `benchmarks.py`, orchestration to `cli.py`, and
  terminal presentation to `reporting.py`.
- Put multi-line configs, scripts, and other fixed fixture text under `data/`.
  Load or render those assets through `fixtures.py`; do not embed them in
  executable modules.

## Run The Fixture

List every config override without building Relay:

```bash
uv run python scripts/benchmark-coding-agent-latency.py --help
```

Run a small functional check before a statistically meaningful run:

```bash
just benchmark-coding-agent-latency \
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
`just benchmark-coding-agent-latency` when collecting performance data. The
default file-exporter matrix can write tens of gigabytes temporarily, so check
free disk space first.

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
just benchmark-coding-agent-latency --config /path/to/benchmark.toml
```

Find the JSON report at
`target/benchmark-results/coding-agent-latency.json` unless `output_dir` was
overridden. Compare added milliseconds and paired confidence intervals; do not
draw performance conclusions from a smoke run.

## Maintain Measurement Integrity

- Keep providers on loopback and deterministic. Do not add model-service or
  Internet latency to the core fixture.
- Compare variants within the same measurement cycle and retain the rotated or
  randomized execution order to reduce ordering bias.
- Warm persistent connections before recording gateway samples.
- Keep streaming time-to-first-content separate from total stream time.
- Preserve exporter-delivery checks when gateway or hook traffic is measured.
- Record all resolved matrix values in the JSON result so another engineer can
  reproduce the run.
- Keep temporary state isolated from the developer's home and Relay config.

## Expand The Fixture

To add a test suite:

1. Add its name to `AVAILABLE_TESTS` in `config.py`.
2. Implement the measurement in `benchmarks.py`.
3. Dispatch it conditionally in `cli.py` and report it conditionally in
   `reporting.py`.
4. Add config and selection tests in
   `scripts/tests/test_benchmark_coding_agent_latency.py`.
5. Update `scripts/README.md` and `docs/reference/performance.mdx`.

To add a provider, mode, or matrix axis, update config validation, the protocol
fixture, the loopback server, orchestration, result parameters, and tests
together. Add static fixture files when the change introduces fixed text.

## Validate Changes

Format and test the focused surface first:

```bash
uv run ruff format scripts/benchmark-coding-agent-latency.py \
  scripts/benchmark_coding_agent_latency \
  scripts/tests/test_benchmark_coding_agent_latency.py
uv run ruff check scripts/benchmark-coding-agent-latency.py \
  scripts/benchmark_coding_agent_latency \
  scripts/tests/test_benchmark_coding_agent_latency.py
uv run python -m unittest scripts.tests.test_benchmark_coding_agent_latency
```

Run the small functional command above after lifecycle, protocol, server,
fixture, or orchestration changes. It requires permission to bind loopback
ports in restricted environments. Run `just docs` when the performance page
changes and `uv run pre-commit run --all-files` before review.
