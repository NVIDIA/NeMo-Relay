// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed configuration editor metadata.
//!
//! This module provides a small compile-time reflection surface for interactive
//! configuration editors. Config structs use `editor_config!` to expose
//! ordered field metadata without making editor UIs depend on JSON Schema.

use serde_json::Value as Json;

/// Editor control shape for one configuration field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditorFieldKind {
    /// Boolean toggle.
    Boolean,
    /// String-like value, including paths.
    String,
    /// Integer value.
    Integer,
    /// Floating-point number value.
    Float,
    /// String enum with a fixed set of allowed values.
    Enum,
    /// Object with string keys and string values.
    StringMap,
    /// Arbitrary JSON value.
    Json,
    /// A collection whose entries are edited recursively.
    List,
    /// A tagged object whose variant is selected from a discriminator field.
    TaggedUnion,
    /// Nested configuration section.
    Section,
}

/// Static editor metadata for one configuration field.
#[derive(Clone, Copy)]
pub struct EditorFieldSpec {
    /// Serialized field name.
    pub name: &'static str,
    /// Human-readable label.
    pub label: &'static str,
    /// Editor control shape.
    pub kind: EditorFieldKind,
    /// Allowed string enum values, when [`EditorFieldKind::Enum`] is used.
    pub enum_values: &'static [&'static str],
    /// Whether the field is represented as an `Option<T>` in Rust.
    pub optional: bool,
    /// Nested editor schema for section fields.
    pub nested_schema: Option<fn() -> &'static EditorSchema>,
    /// Default value for a nested section.
    pub nested_default: Option<fn() -> Json>,
    /// Description of list entries, when [`EditorFieldKind::List`] is used.
    pub list_item: Option<&'static EditorListItemSpec>,
    /// Variant metadata, when [`EditorFieldKind::TaggedUnion`] is used.
    pub tagged_union: Option<&'static EditorTaggedUnionSpec>,
}

/// Metadata used to edit a tagged union value.
#[derive(Clone, Copy)]
pub struct EditorTaggedUnionSpec {
    /// Field that selects the active variant.
    pub discriminator: &'static str,
    /// Available tagged-union variants.
    pub variants: &'static [EditorVariantSpec],
}

/// Recursive metadata for one list entry.
#[derive(Clone, Copy)]
pub struct EditorListItemSpec {
    /// Shape of each list entry.
    pub kind: EditorFieldKind,
    /// Schema used for object entries.
    pub schema: Option<fn() -> &'static EditorSchema>,
    /// Default value used for non-union entries.
    pub default: Option<fn() -> Json>,
    /// Variant metadata when this list entry is a tagged union.
    pub tagged_union: Option<&'static EditorTaggedUnionSpec>,
    /// Nested list entry description when this item is itself a list.
    pub list_item: Option<&'static EditorListItemSpec>,
}

/// One selectable variant of a tagged list entry.
#[derive(Clone, Copy)]
pub struct EditorVariantSpec {
    /// Human-readable variant label.
    pub label: &'static str,
    /// Serialized discriminator value.
    pub tag: &'static str,
    /// Schema for the variant object.
    pub schema: fn() -> &'static EditorSchema,
    /// Initial object value for the variant.
    pub default: fn() -> Json,
}

/// Default value for a newly added string list item.
pub fn default_string_list_item_value() -> Json {
    Json::String(String::new())
}

/// Default value for a newly added arbitrary JSON list item.
pub fn default_json_list_item_value() -> Json {
    Json::Null
}

/// Reusable metadata for a list of strings.
pub static STRING_LIST_ITEM: EditorListItemSpec = EditorListItemSpec {
    kind: EditorFieldKind::String,
    schema: None,
    default: Some(default_string_list_item_value),
    tagged_union: None,
    list_item: None,
};

/// Reusable metadata for a list of arbitrary JSON values.
pub static JSON_LIST_ITEM: EditorListItemSpec = EditorListItemSpec {
    kind: EditorFieldKind::Json,
    schema: None,
    default: Some(default_json_list_item_value),
    tagged_union: None,
    list_item: None,
};

impl EditorFieldSpec {
    /// Returns the nested schema for this field, if it is a section.
    pub fn schema(self) -> Option<&'static EditorSchema> {
        self.nested_schema.map(|schema| schema())
    }

    /// Returns the typed default value for this field's nested section.
    pub fn default_value(self) -> Option<Json> {
        self.nested_default.map(|default_value| default_value())
    }
}

/// Static editor metadata for one configuration struct.
#[derive(Clone, Copy)]
pub struct EditorSchema {
    /// Ordered editor fields.
    pub fields: &'static [EditorFieldSpec],
}

impl EditorSchema {
    /// Finds a field by serialized name.
    pub fn field(self, name: &str) -> Option<EditorFieldSpec> {
        self.fields.iter().copied().find(|field| field.name == name)
    }
}

/// Trait implemented by configuration structs that expose editor metadata.
pub trait EditorConfig {
    /// Returns the static editor schema for this config type.
    fn editor_schema() -> &'static EditorSchema;
}

/// Implements [`EditorConfig`] for a configuration type.
///
/// This macro intentionally keeps editor metadata next to the Rust config type
/// while avoiding proc-macro reflection. Field order is declaration order inside
/// the macro invocation.
#[macro_export]
macro_rules! editor_config {
    (
        impl $ty:ty {
            $(
                $field:ident => {
                    label: $label:literal,
                    kind: $kind:ident
                    $(, values: [$($value:literal),* $(,)?])?
                    $(, optional: $optional:literal)?
                    $(, nested: $nested:ty)?
                    $(, default: $default:ty)?
                    $(, list: $list:expr)?
                    $(, tagged_union: $tagged_union:expr)?
                    $(,)?
                }
            ),* $(,)?
        }
    ) => {
        const _: fn(&$ty) = |value: &$ty| {
            $(
                let _ = &value.$field;
            )*
        };

        impl $crate::config_editor::EditorConfig for $ty {
            fn editor_schema() -> &'static $crate::config_editor::EditorSchema {
                static SCHEMA: $crate::config_editor::EditorSchema = $crate::config_editor::EditorSchema {
                    fields: &[
                        $(
                            $crate::config_editor::EditorFieldSpec {
                                name: stringify!($field),
                                label: $label,
                                kind: $crate::editor_config!(@kind $kind),
                                enum_values: $crate::editor_config!(@values $($($value),*)?),
                                optional: $crate::editor_config!(@optional $($optional)?),
                                nested_schema: $crate::editor_config!(@nested $($nested)?),
                                nested_default: $crate::editor_config!(@default $($default)?),
                                list_item: $crate::editor_config!(@list $($list)?),
                                tagged_union: $crate::editor_config!(@tagged_union $($tagged_union)?),
                            }
                        ),*
                    ],
                };
                &SCHEMA
            }
        }
    };

    (@kind Boolean) => { $crate::config_editor::EditorFieldKind::Boolean };
    (@kind String) => { $crate::config_editor::EditorFieldKind::String };
    (@kind Integer) => { $crate::config_editor::EditorFieldKind::Integer };
    (@kind Float) => { $crate::config_editor::EditorFieldKind::Float };
    (@kind Enum) => { $crate::config_editor::EditorFieldKind::Enum };
    (@kind StringMap) => { $crate::config_editor::EditorFieldKind::StringMap };
    (@kind Json) => { $crate::config_editor::EditorFieldKind::Json };
    (@kind List) => { $crate::config_editor::EditorFieldKind::List };
    (@kind TaggedUnion) => { $crate::config_editor::EditorFieldKind::TaggedUnion };
    (@kind Section) => { $crate::config_editor::EditorFieldKind::Section };

    (@values) => { &[] };
    (@values $($value:literal),*) => { &[$($value),*] };

    (@optional) => { false };
    (@optional $optional:literal) => { $optional };

    (@nested) => { None };
    (@nested $nested:ty) => {
        Some(<$nested as $crate::config_editor::EditorConfig>::editor_schema)
    };

    (@default) => { None };
    (@default $default:ty) => {
        Some(|| {
            serde_json::to_value(<$default as Default>::default())
                .expect("editor default value should serialize")
        })
    };

    (@list) => { None };
    (@list $list:expr) => { Some($list) };

    (@tagged_union) => { None };
    (@tagged_union $tagged_union:expr) => { Some($tagged_union) };
}
