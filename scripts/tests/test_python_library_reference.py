# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep source-only dataclass constructors in generated Python reference pages."""

import importlib
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "docs"), *sys.path]):
    REFERENCE = importlib.import_module("generate_python_library_reference")


class PythonReferenceTests(unittest.TestCase):
    def render_source(self, source: str, stub: str | None = None) -> str:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_root = root / "source"
            package_root.mkdir()
            (package_root / "config.py").write_text(textwrap.dedent(source), encoding="utf-8")
            if stub is not None:
                (package_root / "config.pyi").write_text(textwrap.dedent(stub), encoding="utf-8")
            package = REFERENCE.Package("sample-package", "sample", package_root, "Sample package.")
            modules = REFERENCE._discover_modules(package, package_root)
            self.assertEqual(len(modules), 1)
            output = root / "generated"
            REFERENCE._write_module(output, modules[0], 1, modules)
            return (output / "sample-package/config.mdx").read_text(encoding="utf-8")

    def test_source_fields_preserve_required_defaults_and_factories_without_execution(self) -> None:
        rendered = self.render_source(
            """
            from dataclasses import dataclass, field

            raise AssertionError("Documentation must not import this module")

            def create_tags():
                raise AssertionError("Documentation must not execute a default factory")

            @dataclass(slots=True)
            class Options:
                amount: float
                description: str | None = None
                tags: list[str] = field(default_factory=create_tags)
                hidden: bool = field(default=True, init=False)
            """
        )

        self.assertIn(
            "def Options(amount: float, description: str | None = None, tags: list[str] = ...) -> None", rendered
        )
        self.assertNotIn("hidden:", rendered)
        self.assertNotIn("create_tags)", rendered)

    def test_plain_and_qualified_decorators_preserve_empty_and_required_constructors(self) -> None:
        rendered = self.render_source(
            """
            import dataclasses
            from dataclasses import dataclass

            @dataclass
            class Empty:
                pass

            @dataclasses.dataclass()
            class Options:
                amount: float
            """
        )

        self.assertIn("def Empty() -> None", rendered)
        self.assertIn("def Options(amount: float) -> None", rendered)

    def test_keyword_only_class_setting_allows_field_override(self) -> None:
        rendered = self.render_source(
            """
            from dataclasses import dataclass, field

            @dataclass(kw_only=True)
            class Options:
                name: str
                amount: float = field(kw_only=False)
                enabled: bool = True
            """
        )

        self.assertIn("def Options(amount: float, *, name: str, enabled: bool = True) -> None", rendered)

    def test_keyword_only_marker_and_field_setting_preserve_argument_order(self) -> None:
        rendered = self.render_source(
            """
            from dataclasses import KW_ONLY, dataclass, field

            @dataclass
            class Options:
                name: str
                _: KW_ONLY
                amount: float
                enabled: bool = field(default=True, kw_only=False)

            @dataclass
            class FieldOptions:
                name: str
                amount: float = field(kw_only=True)
            """
        )

        self.assertIn("def Options(name: str, enabled: bool = True, *, amount: float) -> None", rendered)
        self.assertIn("def FieldOptions(name: str, *, amount: float) -> None", rendered)
        self.assertNotIn("_: KW_ONLY", rendered)

    def test_class_variables_are_excluded_from_constructor(self) -> None:
        rendered = self.render_source(
            """
            import typing
            from dataclasses import dataclass
            from typing import ClassVar

            @dataclass
            class Options:
                name: str
                kind: ClassVar[str] = "pricing"
                version: typing.ClassVar[int] = 1
            """
        )

        self.assertIn("def Options(name: str) -> None", rendered)
        self.assertNotIn("kind:", rendered)
        self.assertNotIn("version:", rendered)

    def test_unpacked_dataclass_options_do_not_guess_constructor(self) -> None:
        for decorator in (
            "dataclass(**options)",
            "dataclass(slots=True, **options)",
            "dataclasses.dataclass(**options)",
            "dataclasses.dataclass(slots=True, **options)",
        ):
            with self.subTest(decorator=decorator):
                rendered = self.render_source(
                    f"""
                    import dataclasses
                    from dataclasses import dataclass

                    options = {{"kw_only": True}}

                    @{decorator}
                    class Options:
                        amount: float
                    """
                )

                self.assertIn("### `Options`", rendered)
                self.assertNotIn("def Options(", rendered)

    def test_unpacked_field_options_do_not_guess_constructor(self) -> None:
        for field_call in (
            "field(**options)",
            "field(default=1.0, **options)",
            "dataclasses.field(**options)",
            "dataclasses.field(default=1.0, **options)",
        ):
            with self.subTest(field_call=field_call):
                rendered = self.render_source(
                    f"""
                    import dataclasses
                    from dataclasses import dataclass, field

                    options = {{"init": False}}

                    @dataclass
                    class Options:
                        name: str
                        amount: float = {field_call}
                    """
                )

                self.assertIn("### `Options`", rendered)
                self.assertNotIn("def Options(", rendered)

    def test_explicit_constructor_takes_precedence_over_dataclass_fields(self) -> None:
        rendered = self.render_source(
            """
            from dataclasses import dataclass

            @dataclass(init=False)
            class Options:
                amount: float

                def __init__(self, value: int = 1) -> None:
                    self.amount = float(value)
            """
        )

        self.assertIn("def Options(value: int = 1) -> None", rendered)
        self.assertNotIn("def Options(amount:", rendered)

    def test_disabled_and_inherited_constructors_are_not_guessed(self) -> None:
        rendered = self.render_source(
            """
            from dataclasses import dataclass

            @dataclass(init=False)
            class Disabled:
                amount: float

            @dataclass
            class Parent:
                name: str

            @dataclass
            class Child(Parent):
                amount: float

            class Ordinary:
                amount: float
            """
        )

        self.assertIn("def Parent(name: str) -> None", rendered)
        for class_name in ("Disabled", "Child", "Ordinary"):
            with self.subTest(class_name=class_name):
                self.assertNotIn(f"def {class_name}(", rendered)

    def test_existing_stub_constructor_remains_authoritative(self) -> None:
        rendered = self.render_source(
            """
            from dataclasses import dataclass

            @dataclass
            class Options:
                amount: float
            """,
            """
            class Options:
                def __init__(self, value: int) -> None: ...
            """,
        )

        self.assertIn("def Options(value: int) -> None", rendered)
        self.assertNotIn("def Options(amount:", rendered)

    def test_pricing_reference_preserves_all_ten_source_constructors(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        package = REFERENCE.PACKAGES[0]
        package_root = repository_root / package.source_root
        self.assertFalse((package_root / "model_pricing.pyi").exists())
        modules = REFERENCE._discover_modules(package, package_root)
        pricing_module = next(module for module in modules if module.name == "nemo_relay.model_pricing")
        self.assertEqual(pricing_module.source, package_root / "model_pricing.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            REFERENCE._write_module(output, pricing_module, 1, modules)
            rendered = (output / "nemo-relay/model-pricing.mdx").read_text(encoding="utf-8")

        constructors = (
            "def TokenPricingRates(input_per_million: float, output_per_million: float, "
            "cache_read_per_million: float | None = None, cache_write_per_million: float | None = None) -> None",
            "def PromptCachePricing(read_accounting: Literal['included_in_prompt_tokens', 'separate'] "
            "= 'included_in_prompt_tokens') -> None",
            "def TokenRateTier(rates: TokenPricingRates | JsonObject, min_prompt_tokens: int | None = None, "
            "max_prompt_tokens: int | None = None) -> None",
            "def PromptTokenThresholdRateSchedule(tiers: list[TokenRateTier | JsonObject] = ..., "
            "applies_to: Literal['full_request'] = 'full_request') -> None",
            "def ModelPricing(provider: str, model_id: str, pricing_as_of: str, pricing_source: str, "
            "aliases: list[str] = ..., currency: str = 'USD', "
            "unit: Literal['per_token', 'per_request', 'per_second', 'gpu_hour'] = 'per_token', "
            "rates: TokenPricingRates | JsonObject | None = None, "
            "rate_schedule: PromptTokenThresholdRateSchedule | JsonObject | None = None, "
            "prompt_cache: PromptCachePricing | JsonObject = ...) -> None",
            "def PricingCatalog(entries: list[ModelPricing | JsonObject] = ..., version: int = 1) -> None",
            "def InlineSource(catalog: PricingCatalog | JsonObject) -> None",
            "def FileSource(path: str) -> None",
            "def PricingConfig(sources: list[InlineSource | FileSource | JsonObject] = ...) -> None",
            "def ComponentSpec(config: PricingConfig | JsonObject, enabled: bool = True) -> None",
        )
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                self.assertIn(constructor, rendered)
        self.assertEqual(rendered.count("Create an instance."), len(constructors))


if __name__ == "__main__":
    unittest.main()
