// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use nemo_relay_runtime::config_editor::{
    EditorFieldKind, EditorFieldSpec, EditorListItemSpec, EditorSchema, EditorTaggedUnionSpec,
    EditorVariantSpec,
};

fn field(kind: EditorFieldKind) -> EditorFieldSpec {
    EditorFieldSpec {
        name: "value",
        label: "Value",
        kind,
        enum_values: &[],
        optional: true,
        nested_schema: None,
        nested_default: None,
        list_item: None,
        tagged_union: None,
    }
}

fn item(kind: EditorFieldKind) -> EditorListItemSpec {
    EditorListItemSpec {
        kind,
        schema: None,
        default: None,
        tagged_union: None,
        list_item: None,
    }
}

fn empty_schema() -> &'static EditorSchema {
    static SCHEMA: EditorSchema = EditorSchema { fields: &[] };
    &SCHEMA
}

fn string_schema() -> &'static EditorSchema {
    static FIELDS: [EditorFieldSpec; 1] = [EditorFieldSpec {
        name: "value",
        label: "Value",
        kind: EditorFieldKind::String,
        enum_values: &[],
        optional: true,
        nested_schema: None,
        nested_default: None,
        list_item: None,
        tagged_union: None,
    }];
    static SCHEMA: EditorSchema = EditorSchema { fields: &FIELDS };
    &SCHEMA
}

fn known_variant_default() -> Value {
    json!({"kind": "known"})
}

static KNOWN_VARIANTS: [EditorVariantSpec; 1] = [EditorVariantSpec {
    label: "Known",
    tag: "known",
    schema: empty_schema,
    default: known_variant_default,
}];

static KNOWN_UNION: EditorTaggedUnionSpec = EditorTaggedUnionSpec {
    discriminator: "kind",
    variants: &KNOWN_VARIANTS,
};

fn assert_config_error(result: Result<impl Sized, CliError>, expected: &str) {
    let error = result
        .err()
        .expect("operation should reject malformed metadata");
    assert!(error.to_string().contains(expected), "{error}");
}

#[test]
fn malformed_structured_field_metadata_fails_before_terminal_input() {
    let theme = ColorfulTheme::default();
    let mut config = json!({});

    assert_config_error(
        edit_section(&theme, &mut config, field(EditorFieldKind::Section)),
        "not an editable section",
    );
    for (kind, expected) in [
        (EditorFieldKind::List, "does not describe its list entries"),
        (EditorFieldKind::Map, "does not describe its map values"),
        (
            EditorFieldKind::TaggedUnion,
            "does not describe its variants",
        ),
    ] {
        assert_config_error(
            edit_config_field(&theme, &mut config, field(kind)),
            expected,
        );
    }
    assert_config_error(
        edit_config_field(&theme, &mut config, field(EditorFieldKind::Section)),
        "not an editable section",
    );
}

#[test]
fn malformed_nested_value_metadata_fails_before_terminal_input() {
    let theme = ColorfulTheme::default();
    let mut value = json!({});
    let schema = empty_schema();

    for (kind, expected) in [
        (EditorFieldKind::Section, "not an editable section"),
        (EditorFieldKind::List, "does not describe its list entries"),
        (EditorFieldKind::Map, "does not describe its map values"),
        (
            EditorFieldKind::TaggedUnion,
            "does not describe its variants",
        ),
        (
            EditorFieldKind::DiscriminatedSection,
            "does not describe its variants",
        ),
    ] {
        assert_config_error(
            edit_value_field(&theme, "root", &mut value, schema, field(kind), None),
            expected,
        );
    }
}

#[test]
fn malformed_collection_items_and_tagged_values_are_rejected_without_prompting() {
    let theme = ColorfulTheme::default();
    for (kind, expected) in [
        (EditorFieldKind::Section, "list item has no schema"),
        (
            EditorFieldKind::List,
            "nested list item has no entry description",
        ),
        (
            EditorFieldKind::Map,
            "nested map item has no entry description",
        ),
    ] {
        let mut value = json!({});
        assert_config_error(
            edit_editor_item(&theme, "item", &mut value, &item(kind)),
            expected,
        );
    }

    static EMPTY_UNION: EditorTaggedUnionSpec = EditorTaggedUnionSpec {
        discriminator: "kind",
        variants: &[],
    };
    assert_config_error(
        select_tagged_union_variant(&theme, &EMPTY_UNION),
        "tagged union has no variants",
    );
    assert_config_error(
        new_tagged_union_value(&theme, &EMPTY_UNION),
        "tagged union has no variants",
    );

    let mut missing = json!({});
    assert_config_error(
        edit_tagged_union_payload(&theme, "backend", &mut missing, &KNOWN_UNION),
        "tagged union has no discriminator value",
    );
    let mut unknown = json!({"kind": "unknown"});
    assert_config_error(
        edit_tagged_union_payload(&theme, "backend", &mut unknown, &KNOWN_UNION),
        "unknown tagged union type",
    );
}

#[test]
fn selection_dispatch_resets_values_and_ignores_out_of_range_actions() {
    let theme = ColorfulTheme::default();
    let mut value = json!({"value": "changed"});
    let schema = string_schema();
    let default = json!({"value": "default"});

    assert!(
        edit_selected_value_item(
            &theme,
            "root",
            &mut value,
            schema,
            Some(&default),
            schema.fields.len(),
        )
        .unwrap()
    );
    assert_eq!(value, default);
    assert!(
        edit_selected_value_item(
            &theme,
            "root",
            &mut value,
            schema,
            None,
            schema.fields.len(),
        )
        .unwrap()
    );
    assert_eq!(value, json!({}));
    assert!(
        !edit_selected_value_item(
            &theme,
            "root",
            &mut value,
            schema,
            None,
            schema.fields.len() + 1,
        )
        .unwrap()
    );
}

#[test]
fn scalar_prompt_rejects_structured_kinds_without_reading_the_terminal() {
    let theme = ColorfulTheme::default();
    for kind in [
        EditorFieldKind::Section,
        EditorFieldKind::List,
        EditorFieldKind::Map,
        EditorFieldKind::TaggedUnion,
        EditorFieldKind::DiscriminatedSection,
    ] {
        assert_config_error(
            prompt_value(&theme, &field(kind), None),
            if kind == EditorFieldKind::Section {
                "nested section"
            } else {
                "structured value"
            },
        );
    }
}

#[test]
fn menu_and_editor_io_errors_preserve_cancellation_and_failure_context() {
    for kind in [
        std::io::ErrorKind::Interrupted,
        std::io::ErrorKind::UnexpectedEof,
    ] {
        for error in [
            menu_error(std::io::Error::from(kind)),
            editor_error(dialoguer::Error::IO(std::io::Error::from(kind))),
        ] {
            assert!(matches!(
                error,
                CliError::Config(ref message) if message == PLUGIN_EDIT_CANCELLED_MESSAGE
            ));
        }
    }
    assert!(
        menu_error(std::io::Error::other("terminal failed"))
            .to_string()
            .contains("terminal failed")
    );
    assert!(
        editor_error(dialoguer::Error::IO(std::io::Error::other("editor failed")))
            .to_string()
            .contains("editor failed")
    );
}
