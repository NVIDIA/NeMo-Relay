# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading and command-line overrides for the benchmark."""

from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data"
DEFAULT_CONFIG_PATH = DATA_ROOT / "default.toml"

AVAILABLE_TESTS = ("gateway", "hooks", "startup")
AVAILABLE_PROVIDERS = ("openai", "anthropic")
AVAILABLE_MODES = ("buffered", "streaming")
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
}
TABLE_KEYS = {
    "models": {"openai", "anthropic"},
    "content": {"request_fill", "response_fill"},
}


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
        }


@dataclass(frozen=True)
class CliOptions:
    """File paths and resolved benchmark configuration."""

    relay_bin: Path
    output: Path
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


def _config_from_mapping(value: dict[str, Any]) -> BenchmarkConfig:
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
    )
    if "gateway" in config.tests and max(config.concurrency) > config.samples:
        raise ValueError("samples must be greater than or equal to every gateway concurrency value")
    return config


def load_config(path: Path) -> BenchmarkConfig:
    """Load the defaults and overlay a possibly partial user config."""
    defaults = _read_toml(DEFAULT_CONFIG_PATH)
    if path.resolve() != DEFAULT_CONFIG_PATH.resolve():
        defaults = _merge_config(defaults, _read_toml(path))
    return _config_from_mapping(defaults)


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


def parse_args(argv: list[str] | None = None) -> CliOptions:
    """Parse CLI options, applying them after values from the config file."""
    parser = argparse.ArgumentParser(description="Measure local coding-agent gateway latency.")
    parser.add_argument("--relay-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="TOML overrides for the benchmark")
    parser.add_argument("--tests", type=_csv_strings, help=f"override test suites: {','.join(AVAILABLE_TESTS)}")
    parser.add_argument("--providers", type=_csv_strings, help=f"override providers: {','.join(AVAILABLE_PROVIDERS)}")
    parser.add_argument("--modes", type=_csv_strings, help=f"override response modes: {','.join(AVAILABLE_MODES)}")
    parser.add_argument("--samples", type=_arg_positive_int)
    parser.add_argument("--hook-samples", type=_arg_positive_int)
    parser.add_argument("--startup-samples", type=_arg_positive_int)
    parser.add_argument("--warmup", type=_arg_nonnegative_int)
    parser.add_argument("--payload-sizes", type=_csv_ints, help="override comma-separated request payload sizes")
    parser.add_argument("--concurrency", type=_csv_ints, help="override comma-separated in-flight request counts")
    parser.add_argument("--response-bytes", type=_arg_positive_int)
    parser.add_argument("--stream-chunks", type=_arg_positive_int)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        overrides = {
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
            )
            if getattr(args, name) is not None
        }
        config = replace(config, **overrides)
        # Reuse the same validation for values supplied by argparse.
        config = _config_from_mapping(
            {
                **config.parameters(),
                "tests": list(config.tests),
                "providers": list(config.providers),
                "modes": list(config.modes),
                "payload_sizes": list(config.payload_sizes),
                "concurrency": list(config.concurrency),
            }
        )
    except (OSError, tomllib.TOMLDecodeError, ValueError) as error:
        parser.error(f"invalid benchmark config: {error}")

    relay_bin = args.relay_bin.resolve()
    if not relay_bin.is_file():
        parser.error(f"Relay binary does not exist: {relay_bin}")
    return CliOptions(
        relay_bin=relay_bin,
        output=args.output.resolve(),
        config=config,
    )
