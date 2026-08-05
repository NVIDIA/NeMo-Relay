# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and command-line overrides for the benchmark."""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = BENCHMARK_ROOT / "data"
CONFIG_ROOT = BENCHMARK_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "default.toml"

AVAILABLE_TESTS = ("gateway", "hooks", "startup")
AVAILABLE_PROVIDERS = ("openai", "anthropic")
AVAILABLE_MODES = ("buffered", "streaming")
RESERVED_MIDDLEWARE_NAMES = {"direct", "minimal", "file", "otlp"}
MIDDLEWARE_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]*")
CONFIG_KEYS = {
    "tests",
    "providers",
    "modes",
    "samples",
    "hook_samples",
    "startup_samples",
    "warmup",
    "payload_sizes",
    "concurrency",
    "response_bytes",
    "stream_chunks",
    "models",
    "content",
    "middleware",
}
TABLE_KEYS = {
    "models": {"openai", "anthropic"},
    "content": {"request_fill", "response_fill"},
}
MIDDLEWARE_KEYS = {"name", "plugin_config"}


@dataclass(frozen=True)
class MiddlewareVariant:
    """One opt-in Relay plugin configuration benchmarked as an extra variant."""

    name: str
    plugin_config: Path

    @property
    def relay_name(self) -> str:
        """Return the variant key stored in benchmark results."""
        return f"relay-{self.name}"

    def parameters(self) -> dict[str, str]:
        """Return the serializable configuration recorded with results."""
        return {"name": self.name, "plugin_config": str(self.plugin_config)}


@dataclass(frozen=True)
class BenchmarkConfig:
    """Validated benchmark matrix and sample settings."""

    tests: tuple[str, ...]
    providers: tuple[str, ...]
    modes: tuple[str, ...]
    samples: int
    hook_samples: int
    startup_samples: int
    warmup: int
    payload_sizes: tuple[int, ...]
    concurrency: tuple[int, ...]
    response_bytes: int
    stream_chunks: int
    openai_model: str
    anthropic_model: str
    request_fill: str
    response_fill: str
    middleware: tuple[MiddlewareVariant, ...]

    def parameters(self) -> dict[str, Any]:
        """Return the configuration embedded in the JSON result."""
        return {
            "tests": self.tests,
            "providers": self.providers,
            "modes": self.modes,
            "samples": self.samples,
            "hook_samples": self.hook_samples,
            "startup_samples": self.startup_samples,
            "warmup": self.warmup,
            "payload_sizes": self.payload_sizes,
            "concurrency": self.concurrency,
            "response_bytes": self.response_bytes,
            "stream_chunks": self.stream_chunks,
            "models": {
                "openai": self.openai_model,
                "anthropic": self.anthropic_model,
            },
            "content": {
                "request_fill": self.request_fill,
                "response_fill": self.response_fill,
            },
            "middleware": [variant.parameters() for variant in self.middleware],
        }


@dataclass(frozen=True)
class CliOptions:
    """File paths and resolved benchmark configuration."""

    relay_bin: Path
    output: Path
    report: Path
    config: BenchmarkConfig


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    unknown = set(override) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}")
    merged = dict(base)
    for key, value in override.items():
        if key in {"models", "content"}:
            if not isinstance(value, dict):
                raise ValueError(f"[{key}] must be a TOML table")
            unknown_nested = set(value) - TABLE_KEYS[key]
            if unknown_nested:
                raise ValueError(f"unknown [{key}] key(s): {', '.join(sorted(unknown_nested))}")
            merged[key] = dict(merged.get(key, {})) | value
        else:
            merged[key] = value
    return merged


def _string_tuple(value: Any, name: str, available: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    values = tuple(value)
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise ValueError(f"unknown {name}: {', '.join(invalid)}; choose from {', '.join(available)}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of positive integers")
    values = tuple(_positive_int(item, name) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _middleware_variants(value: Any, base_directory: Path) -> tuple[MiddlewareVariant, ...]:
    if not isinstance(value, list):
        raise ValueError("middleware must be a list of tables")
    variants = []
    names = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"middleware[{index}] must be a TOML table")
        item_mapping = cast(dict[str, Any], item)
        unknown = set(item_mapping) - MIDDLEWARE_KEYS
        if unknown:
            raise ValueError(f"unknown middleware[{index}] key(s): {', '.join(sorted(unknown))}")
        name = _nonempty_string(item_mapping.get("name"), f"middleware[{index}].name")
        if MIDDLEWARE_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError(f"middleware[{index}].name must contain only lowercase letters, digits, and hyphens")
        if name in RESERVED_MIDDLEWARE_NAMES:
            raise ValueError(f"middleware name is reserved by a default variant: {name}")
        if name in names:
            raise ValueError(f"middleware names must not contain duplicates: {name}")
        names.add(name)
        raw_path = _nonempty_string(item_mapping.get("plugin_config"), f"middleware[{index}].plugin_config")
        plugin_config = Path(raw_path)
        if not plugin_config.is_absolute():
            plugin_config = base_directory / plugin_config
        plugin_config = plugin_config.resolve()
        if not plugin_config.is_file():
            raise ValueError(f"middleware plugin config does not exist: {plugin_config}")
        variants.append(MiddlewareVariant(name=name, plugin_config=plugin_config))
    return tuple(variants)


def _config_from_mapping(value: dict[str, Any], *, middleware_base: Path) -> BenchmarkConfig:
    unknown = set(value) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}")
    models = value.get("models")
    content = value.get("content")
    if not isinstance(models, dict) or not isinstance(content, dict):
        raise ValueError("config must contain [models] and [content] tables")
    warmup = value.get("warmup")
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    request_fill = _nonempty_string(content.get("request_fill"), "content.request_fill")
    response_fill = _nonempty_string(content.get("response_fill"), "content.response_fill")
    if len(request_fill) != 1 or len(response_fill) != 1 or not request_fill.isascii() or not response_fill.isascii():
        raise ValueError("content fill values must each contain exactly one ASCII character")
    config = BenchmarkConfig(
        tests=_string_tuple(value.get("tests"), "tests", AVAILABLE_TESTS),
        providers=_string_tuple(value.get("providers"), "providers", AVAILABLE_PROVIDERS),
        modes=_string_tuple(value.get("modes"), "modes", AVAILABLE_MODES),
        samples=_positive_int(value.get("samples"), "samples"),
        hook_samples=_positive_int(value.get("hook_samples"), "hook_samples"),
        startup_samples=_positive_int(value.get("startup_samples"), "startup_samples"),
        warmup=warmup,
        payload_sizes=_positive_int_tuple(value.get("payload_sizes"), "payload_sizes"),
        concurrency=_positive_int_tuple(value.get("concurrency"), "concurrency"),
        response_bytes=_positive_int(value.get("response_bytes"), "response_bytes"),
        stream_chunks=_positive_int(value.get("stream_chunks"), "stream_chunks"),
        openai_model=_nonempty_string(models.get("openai"), "models.openai"),
        anthropic_model=_nonempty_string(models.get("anthropic"), "models.anthropic"),
        request_fill=request_fill,
        response_fill=response_fill,
        middleware=_middleware_variants(value.get("middleware"), middleware_base),
    )
    if "gateway" in config.tests and max(config.concurrency) > config.samples:
        raise ValueError("samples must be greater than or equal to every gateway concurrency value")
    return config


def load_config(path: Path) -> BenchmarkConfig:
    """Load the defaults and overlay a possibly partial user config."""
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    resolved_path = path.resolve()
    middleware_base = DEFAULT_CONFIG_PATH.parent
    if resolved_path != DEFAULT_CONFIG_PATH.resolve():
        override = _read_toml(resolved_path)
        defaults = _merge_config(defaults, override)
        if "middleware" in override:
            middleware_base = resolved_path.parent
    return _config_from_mapping(defaults, middleware_base=middleware_base)


def _csv_strings(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _arg_positive_int(value: str) -> int:
    try:
        return _positive_int(int(value), "value")
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error


def _arg_nonnegative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from error
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


def _middleware_arg(value: str) -> dict[str, str]:
    name, separator, plugin_config = value.partition("=")
    if not separator or not name or not plugin_config:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return {"name": name, "plugin_config": plugin_config}


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface for the benchmark."""
    parser = argparse.ArgumentParser(
        prog="just latency-benchmark",
        usage="just latency-benchmark [options]",
        description=(
            "Measure the local latency that NeMo Relay adds to coding-agent gateway, hook, and startup paths."
        ),
        epilog=(
            "Configuration precedence: built-in defaults, then --config, then CLI overrides.\n"
            "Example smoke test: just latency-benchmark --tests gateway "
            "--providers openai --modes buffered --payload-sizes 4096 "
            "--concurrency 1 --samples 5 --warmup 1"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    outputs = parser.add_argument_group("output and configuration")
    outputs.add_argument(
        "--relay-bin",
        type=Path,
        required=True,
        metavar="PATH",
        help="Relay CLI executable to benchmark (the just recipe supplies the release binary).",
    )
    outputs.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="PATH",
        help="write machine-readable JSON results to PATH (the just recipe supplies a default).",
    )
    outputs.add_argument(
        "--report",
        type=Path,
        metavar="PATH",
        help="write the self-contained HTML report to PATH (default: the JSON path with an .html suffix).",
    )
    outputs.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        metavar="PATH",
        help="overlay benchmark settings from a TOML file (default: bundled config/default.toml).",
    )

    matrix = parser.add_argument_group("suite and matrix selection")
    matrix.add_argument(
        "--tests",
        type=_csv_strings,
        metavar="LIST",
        help=f"run comma-separated suites; choices: {','.join(AVAILABLE_TESTS)}.",
    )
    matrix.add_argument(
        "--providers",
        type=_csv_strings,
        metavar="LIST",
        help=f"benchmark comma-separated mock provider protocols; choices: {','.join(AVAILABLE_PROVIDERS)}.",
    )
    matrix.add_argument(
        "--modes",
        type=_csv_strings,
        metavar="LIST",
        help=f"benchmark comma-separated response modes; choices: {','.join(AVAILABLE_MODES)}.",
    )
    matrix.add_argument(
        "--payload-sizes",
        type=_csv_ints,
        metavar="BYTES",
        help="benchmark comma-separated request-content sizes in bytes.",
    )
    matrix.add_argument(
        "--concurrency",
        type=_csv_ints,
        metavar="COUNTS",
        help="benchmark comma-separated in-flight request counts; each value must not exceed --samples.",
    )
    matrix.add_argument(
        "--middleware",
        action="append",
        type=_middleware_arg,
        metavar="NAME=PATH",
        help=(
            "add a named Relay middleware plugin config as an extra variant; repeat for multiple variants "
            "and use this option to replace middleware entries from --config."
        ),
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--samples",
        type=_arg_positive_int,
        metavar="COUNT",
        help="record COUNT gateway measurement cycles per scenario.",
    )
    sampling.add_argument(
        "--hook-samples",
        type=_arg_positive_int,
        metavar="COUNT",
        help="record COUNT subprocess measurements for every hook path.",
    )
    sampling.add_argument(
        "--startup-samples",
        type=_arg_positive_int,
        metavar="COUNT",
        help="record COUNT cold-start measurements for every Relay variant.",
    )
    sampling.add_argument(
        "--warmup",
        type=_arg_nonnegative_int,
        metavar="COUNT",
        help="run COUNT unrecorded warmup cycles before each measured workload.",
    )

    response = parser.add_argument_group("mock provider response")
    response.add_argument(
        "--response-bytes",
        type=_arg_positive_int,
        metavar="BYTES",
        help="return approximately BYTES of deterministic content from the mock provider.",
    )
    response.add_argument(
        "--stream-chunks",
        type=_arg_positive_int,
        metavar="COUNT",
        help="split streaming mock responses into COUNT content-delta events.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> CliOptions:
    """Parse CLI options, applying them after values from the config file."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        values = config.parameters()
        values.update(
            {
                name: getattr(args, name)
                for name in (
                    "tests",
                    "providers",
                    "modes",
                    "samples",
                    "hook_samples",
                    "startup_samples",
                    "warmup",
                    "payload_sizes",
                    "concurrency",
                    "response_bytes",
                    "stream_chunks",
                    "middleware",
                )
                if getattr(args, name) is not None
            }
        )
        config = _config_from_mapping(
            {
                **values,
                "tests": list(values["tests"]),
                "providers": list(values["providers"]),
                "modes": list(values["modes"]),
                "payload_sizes": list(values["payload_sizes"]),
                "concurrency": list(values["concurrency"]),
            },
            middleware_base=Path.cwd(),
        )
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        parser.error(f"invalid benchmark config: {error}")

    relay_bin = args.relay_bin.resolve()
    if not relay_bin.is_file():
        parser.error(f"Relay binary does not exist: {relay_bin}")
    output = args.output.resolve()
    report = args.report.resolve() if args.report is not None else output.with_suffix(".html")
    return CliOptions(
        relay_bin=relay_bin,
        output=output,
        report=report,
        config=config,
    )
