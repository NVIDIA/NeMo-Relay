// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static JSON Schema loading and editor metadata for dynamic plugins.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::Read;
use std::path::Path;

use jsonschema::{Draft, Validator};
use percent_encoding::{AsciiSet, CONTROLS, percent_decode_str, utf8_percent_encode};
use serde_json::{Map, Value};

use crate::error::CliError;

const DRAFT_7_URI: &str = "http://json-schema.org/draft-07/schema";
const DRAFT_7_HTTPS_URI: &str = "https://json-schema.org/draft-07/schema";
const DRAFT_2020_12_URI: &str = "https://json-schema.org/draft/2020-12/schema";
const DRAFT_2020_12_HTTP_URI: &str = "http://json-schema.org/draft/2020-12/schema";
const REDACTED: &str = "<redacted>";
const EDIT_REDACTED_PREFIX: &str = "<redacted:nemo-relay:";
const MAX_CONFIG_SCHEMA_BYTES: u64 = 1024 * 1024;
const URI_FRAGMENT_ENCODE_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'<')
    .add(b'>')
    .add(b'`')
    .add(b'%');

pub(super) type SecretEditValues = BTreeMap<String, SecretEditValue>;

#[derive(Debug, Clone)]
pub(super) struct SecretEditValue {
    value: Value,
    pattern: SecretPattern,
}

/// Supported JSON Schema dialects for dynamic plugin configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConfigSchemaDraft {
    Draft7,
    Draft202012,
}

impl ConfigSchemaDraft {
    fn validator_draft(self) -> Draft {
        match self {
            Self::Draft7 => Draft::Draft7,
            Self::Draft202012 => Draft::Draft202012,
        }
    }
}

/// Editor representation of a dynamic plugin's object-root configuration schema.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DynamicConfigEditorSchema {
    pub(super) title: Option<String>,
    pub(super) description: Option<String>,
    pub(super) fields: Vec<DynamicConfigField>,
}

/// One named property in a dynamic plugin configuration schema.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DynamicConfigField {
    pub(super) key: String,
    pub(super) title: String,
    pub(super) description: Option<String>,
    pub(super) default: Option<Value>,
    pub(super) required: bool,
    pub(super) kind: DynamicConfigFieldKind,
}

/// Native editor control selected for a configuration property.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum DynamicConfigFieldKind {
    Boolean,
    String { secret: bool },
    Integer,
    Number,
    StringEnum { options: Vec<String>, secret: bool },
    Object { fields: Vec<DynamicConfigField> },
    StringMap,
    RawJson,
}

/// A loaded and compiled dynamic plugin configuration schema.
#[derive(Debug, Clone)]
pub(super) struct PluginConfigSchema {
    plugin_id: String,
    validator: Validator,
    editor: DynamicConfigEditorSchema,
    secret_patterns: Vec<SecretPattern>,
}

impl PluginConfigSchema {
    /// Reads, validates, and compiles a plugin's static configuration schema.
    pub(super) fn load(
        plugin_id: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Self, CliError> {
        let plugin_id = plugin_id.into();
        let path = path.as_ref().to_path_buf();
        let contents = read_schema_file(&plugin_id, &path)?;
        let mut source: Value = serde_json::from_slice(&contents).map_err(|error| {
            schema_error(
                &plugin_id,
                &path,
                format!("schema is not valid JSON: {error}"),
            )
        })?;

        let draft = parse_draft(&plugin_id, &path, &source)?;
        normalize_local_references(&plugin_id, &path, &mut source)?;
        validate_schema_security(&plugin_id, &path, &source)?;
        validate_schema_document(&plugin_id, &path, draft, &source)?;
        let validator = jsonschema::options()
            .with_draft(draft.validator_draft())
            .build(&source)
            .map_err(|error| {
                schema_error(
                    &plugin_id,
                    &path,
                    format!("failed to compile schema: {error}"),
                )
            })?;

        let mut root_references = HashSet::new();
        let resolved_root =
            resolve_schema(&source, &source, &mut root_references).map_err(|error| {
                schema_error(
                    &plugin_id,
                    &path,
                    format!("root schema cannot be resolved: {error}"),
                )
            })?;
        if schema_type(resolved_root) != Some("object") {
            return Err(schema_error(
                &plugin_id,
                &path,
                "root schema must resolve to type 'object'",
            ));
        }

        let fields =
            build_object_fields(&plugin_id, &path, &source, resolved_root, &root_references)?;
        let editor = DynamicConfigEditorSchema {
            title: annotation_string(&source, resolved_root, "title"),
            description: annotation_string(&source, resolved_root, "description"),
            fields,
        };

        let mut secret_patterns = Vec::new();
        discover_secret_patterns(&source, &source, &[], &HashSet::new(), &mut secret_patterns)
            .map_err(|error| {
                schema_error(
                    &plugin_id,
                    &path,
                    format!("secret schema patterns are invalid: {error}"),
                )
            })?;
        secret_patterns.sort();
        secret_patterns.dedup();

        Ok(Self {
            plugin_id,
            validator,
            editor,
            secret_patterns,
        })
    }

    pub(super) fn editor(&self) -> &DynamicConfigEditorSchema {
        &self.editor
    }

    pub(super) fn fields(&self) -> &[DynamicConfigField] {
        &self.editor.fields
    }

    /// Validates one object configuration and reports the first failing instance pointer.
    pub(super) fn validate(&self, config: &Value) -> Result<(), CliError> {
        self.validator.validate(config).map_err(|error| {
            let pointer = error.instance_path().to_string();
            let masked = error.masked_with(REDACTED);
            CliError::Config(format!(
                "dynamic plugin '{}' configuration at JSON pointer '{}' is invalid: {masked}",
                self.plugin_id, pointer
            ))
        })
    }

    /// Clones a configuration and masks every schema-declared secret present in it.
    pub(super) fn redact(&self, config: &Value) -> Value {
        let mut redacted = config.clone();
        for pattern in &self.secret_patterns {
            pattern.redact(&mut redacted, 0);
        }
        redacted
    }

    pub(super) fn has_secrets(&self) -> bool {
        !self.secret_patterns.is_empty()
    }

    pub(super) fn has_secrets_at(&self, path: &[String]) -> bool {
        self.secret_patterns
            .iter()
            .any(|pattern| pattern.applies_below(path))
    }

    /// Redacts secrets with per-value tokens so raw JSON editing can safely reorder values.
    pub(super) fn redact_for_edit(&self, config: &Value) -> (Value, SecretEditValues) {
        let mut redacted = config.clone();
        let mut secrets = BTreeMap::new();
        let mut occupied = HashSet::new();
        collect_string_values(config, &mut occupied);
        let mut next_token = 0;
        for pattern in &self.secret_patterns {
            pattern.redact_for_edit(&mut redacted, 0, &mut secrets, &occupied, &mut next_token);
        }
        (redacted, secrets)
    }

    /// Restores redaction tokens only at their schema-declared secret locations.
    pub(super) fn restore_edit_secrets(
        &self,
        edited: &Value,
        secrets: &SecretEditValues,
    ) -> Result<Value, CliError> {
        restore_secret_tokens(edited, secrets).map_err(|error| {
            CliError::Config(format!(
                "dynamic plugin '{}' configuration contains an invalid secret redaction token: {error}",
                self.plugin_id
            ))
        })
    }
}

fn read_schema_file(plugin_id: &str, path: &Path) -> Result<Vec<u8>, CliError> {
    let path_metadata = fs::metadata(path).map_err(|error| {
        schema_error(plugin_id, path, format!("failed to read schema: {error}"))
    })?;
    validate_schema_file_metadata(plugin_id, path, &path_metadata)?;

    let file = fs::File::open(path).map_err(|error| {
        schema_error(plugin_id, path, format!("failed to read schema: {error}"))
    })?;
    let file_metadata = file.metadata().map_err(|error| {
        schema_error(
            plugin_id,
            path,
            format!("failed to inspect open schema: {error}"),
        )
    })?;
    validate_schema_file_metadata(plugin_id, path, &file_metadata)?;

    let mut contents = Vec::with_capacity(file_metadata.len() as usize);
    file.take(MAX_CONFIG_SCHEMA_BYTES + 1)
        .read_to_end(&mut contents)
        .map_err(|error| {
            schema_error(plugin_id, path, format!("failed to read schema: {error}"))
        })?;
    if contents.len() as u64 > MAX_CONFIG_SCHEMA_BYTES {
        return Err(schema_error(
            plugin_id,
            path,
            "schema exceeds the 1 MiB size limit",
        ));
    }
    Ok(contents)
}

fn validate_schema_file_metadata(
    plugin_id: &str,
    path: &Path,
    metadata: &fs::Metadata,
) -> Result<(), CliError> {
    if !metadata.is_file() {
        return Err(schema_error(
            plugin_id,
            path,
            "schema path must identify a regular file",
        ));
    }
    if metadata.len() > MAX_CONFIG_SCHEMA_BYTES {
        return Err(schema_error(
            plugin_id,
            path,
            "schema exceeds the 1 MiB size limit",
        ));
    }
    Ok(())
}

fn schema_error(plugin_id: &str, path: &Path, message: impl std::fmt::Display) -> CliError {
    CliError::Config(format!(
        "dynamic plugin '{plugin_id}' config schema '{}': {message}",
        path.display()
    ))
}

fn parse_draft(
    plugin_id: &str,
    path: &Path,
    schema: &Value,
) -> Result<ConfigSchemaDraft, CliError> {
    let Some(schema_object) = schema.as_object() else {
        return Err(schema_error(
            plugin_id,
            path,
            "schema document must be a JSON object",
        ));
    };
    let Some(dialect) = schema_object.get("$schema") else {
        return Err(schema_error(
            plugin_id,
            path,
            "schema document must declare '$schema' as Draft 7 or Draft 2020-12",
        ));
    };
    let Some(dialect) = dialect.as_str() else {
        return Err(schema_error(plugin_id, path, "'$schema' must be a string"));
    };
    match dialect.trim_end_matches('#') {
        DRAFT_7_URI | DRAFT_7_HTTPS_URI => Ok(ConfigSchemaDraft::Draft7),
        DRAFT_2020_12_URI | DRAFT_2020_12_HTTP_URI => Ok(ConfigSchemaDraft::Draft202012),
        _ => Err(schema_error(
            plugin_id,
            path,
            format!(
                "unsupported '$schema' value '{dialect}'; expected JSON Schema Draft 7 or Draft 2020-12"
            ),
        )),
    }
}

fn normalize_local_references(
    plugin_id: &str,
    path: &Path,
    schema: &mut Value,
) -> Result<(), CliError> {
    normalize_schema_references(plugin_id, path, schema, "")
}

fn normalize_schema_references(
    plugin_id: &str,
    file_path: &Path,
    schema: &mut Value,
    pointer: &str,
) -> Result<(), CliError> {
    let Some(object) = schema.as_object_mut() else {
        return Ok(());
    };
    normalize_schema_reference(plugin_id, file_path, object, pointer)?;
    normalize_schema_value_children(plugin_id, file_path, object, pointer)?;
    normalize_schema_items(plugin_id, file_path, object, pointer)?;
    normalize_schema_array_children(plugin_id, file_path, object, pointer)?;
    normalize_schema_object_children(plugin_id, file_path, object, pointer)?;
    normalize_schema_dependencies(plugin_id, file_path, object, pointer)
}

fn normalize_schema_reference(
    plugin_id: &str,
    file_path: &Path,
    object: &mut Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    if let Some(Value::String(reference)) = object.get_mut("$ref")
        && reference.starts_with('#')
    {
        *reference = canonicalize_local_reference(reference).map_err(|error| {
            schema_error(
                plugin_id,
                file_path,
                format!(
                    "invalid local $ref at JSON pointer '{}': {error}",
                    push_pointer(pointer, "$ref")
                ),
            )
        })?;
    }
    Ok(())
}

fn normalize_schema_value_children(
    plugin_id: &str,
    file_path: &Path,
    object: &mut Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    for key in [
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    ] {
        if let Some(child) = object.get_mut(key) {
            normalize_schema_references(plugin_id, file_path, child, &push_pointer(pointer, key))?;
        }
    }
    Ok(())
}

fn normalize_schema_items(
    plugin_id: &str,
    file_path: &Path,
    object: &mut Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    let Some(items) = object.get_mut("items") else {
        return Ok(());
    };
    let items_pointer = push_pointer(pointer, "items");
    if let Some(items) = items.as_array_mut() {
        for (index, child) in items.iter_mut().enumerate() {
            normalize_schema_references(
                plugin_id,
                file_path,
                child,
                &push_pointer(&items_pointer, &index.to_string()),
            )?;
        }
    } else {
        normalize_schema_references(plugin_id, file_path, items, &items_pointer)?;
    }
    Ok(())
}

fn normalize_schema_array_children(
    plugin_id: &str,
    file_path: &Path,
    object: &mut Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        let Some(children) = object.get_mut(key).and_then(Value::as_array_mut) else {
            continue;
        };
        let children_pointer = push_pointer(pointer, key);
        for (index, child) in children.iter_mut().enumerate() {
            normalize_schema_references(
                plugin_id,
                file_path,
                child,
                &push_pointer(&children_pointer, &index.to_string()),
            )?;
        }
    }
    Ok(())
}

fn normalize_schema_object_children(
    plugin_id: &str,
    file_path: &Path,
    object: &mut Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    for key in [
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    ] {
        let Some(children) = object.get_mut(key).and_then(Value::as_object_mut) else {
            continue;
        };
        let children_pointer = push_pointer(pointer, key);
        for (name, child) in children {
            normalize_schema_references(
                plugin_id,
                file_path,
                child,
                &push_pointer(&children_pointer, name),
            )?;
        }
    }
    Ok(())
}

fn normalize_schema_dependencies(
    plugin_id: &str,
    file_path: &Path,
    object: &mut Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    let Some(dependencies) = object
        .get_mut("dependencies")
        .and_then(Value::as_object_mut)
    else {
        return Ok(());
    };
    let dependencies_pointer = push_pointer(pointer, "dependencies");
    for (name, dependency) in dependencies {
        if dependency.is_object() || dependency.is_boolean() {
            normalize_schema_references(
                plugin_id,
                file_path,
                dependency,
                &push_pointer(&dependencies_pointer, name),
            )?;
        }
    }
    Ok(())
}

fn validate_schema_security(plugin_id: &str, path: &Path, schema: &Value) -> Result<(), CliError> {
    validate_schema_security_node(plugin_id, path, schema, schema, "", None)
}

fn validate_schema_security_node(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    schema: &Value,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    validate_schema_reference_security(plugin_id, file_path, object, pointer)?;
    validate_schema_write_only_security(
        plugin_id,
        file_path,
        root,
        schema,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_value_children(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_context_children(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_items(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_array_children(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_object_children(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_dependent_schemas(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )?;
    validate_schema_dependencies(
        plugin_id,
        file_path,
        root,
        object,
        pointer,
        unsupported_secret_context,
    )
}

fn validate_schema_reference_security(
    plugin_id: &str,
    file_path: &Path,
    object: &Map<String, Value>,
    pointer: &str,
) -> Result<(), CliError> {
    if object.contains_key("$dynamicRef") || object.contains_key("$dynamicAnchor") {
        let keyword = if object.contains_key("$dynamicRef") {
            "$dynamicRef"
        } else {
            "$dynamicAnchor"
        };
        return Err(schema_error(
            plugin_id,
            file_path,
            format!(
                "Draft 2020-12 dynamic references are not supported at JSON pointer '{}'",
                push_pointer(pointer, keyword)
            ),
        ));
    }
    if let Some(reference) = object.get("$ref").and_then(Value::as_str)
        && !reference.starts_with('#')
    {
        return Err(schema_error(
            plugin_id,
            file_path,
            format!(
                "$ref at JSON pointer '{}' must be a local fragment reference beginning with '#', got '{reference}'",
                push_pointer(pointer, "$ref")
            ),
        ));
    }
    Ok(())
}

fn validate_schema_write_only_security(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    schema: &Value,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    if schema.get("writeOnly").and_then(Value::as_bool) != Some(true) {
        return Ok(());
    }
    if let Some(keyword) = unsupported_secret_context {
        return Err(schema_error(
            plugin_id,
            file_path,
            format!(
                "writeOnly at JSON pointer '{}' is not supported under '{keyword}'",
                push_pointer(pointer, "writeOnly")
            ),
        ));
    }
    let mut references = HashSet::new();
    let mut reference_chain = Vec::new();
    resolve_schema_chain(root, schema, &mut references, &mut reference_chain).map_err(|error| {
        schema_error(
            plugin_id,
            file_path,
            format!(
                "writeOnly schema at JSON pointer '{}' cannot be resolved: {error}",
                pointer
            ),
        )
    })?;
    classify_write_only_chain(&reference_chain).map_err(|error| {
        schema_error(
            plugin_id,
            file_path,
            format!("unsupported writeOnly shape at JSON pointer '{pointer}': {error}"),
        )
    })?;
    Ok(())
}

fn validate_schema_value_children(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    for key in ["additionalItems", "additionalProperties", "contains"] {
        if let Some(child) = object.get(key) {
            validate_schema_security_node(
                plugin_id,
                file_path,
                root,
                child,
                &push_pointer(pointer, key),
                unsupported_secret_context,
            )?;
        }
    }
    Ok(())
}

fn validate_schema_context_children(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    for key in [
        "contentSchema",
        "else",
        "if",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    ] {
        if let Some(child) = object.get(key) {
            validate_schema_security_node(
                plugin_id,
                file_path,
                root,
                child,
                &push_pointer(pointer, key),
                unsupported_secret_context.or(Some(key)),
            )?;
        }
    }
    Ok(())
}

fn validate_schema_items(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    let Some(items) = object.get("items") else {
        return Ok(());
    };
    let items_pointer = push_pointer(pointer, "items");
    if let Some(items) = items.as_array() {
        for (index, child) in items.iter().enumerate() {
            validate_schema_security_node(
                plugin_id,
                file_path,
                root,
                child,
                &push_pointer(&items_pointer, &index.to_string()),
                unsupported_secret_context,
            )?;
        }
    } else {
        validate_schema_security_node(
            plugin_id,
            file_path,
            root,
            items,
            &items_pointer,
            unsupported_secret_context,
        )?;
    }
    Ok(())
}

fn validate_schema_array_children(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        let Some(children) = object.get(key).and_then(Value::as_array) else {
            continue;
        };
        let child_context = match key {
            "anyOf" | "oneOf" => unsupported_secret_context.or(Some(key)),
            _ => unsupported_secret_context,
        };
        let children_pointer = push_pointer(pointer, key);
        for (index, child) in children.iter().enumerate() {
            validate_schema_security_node(
                plugin_id,
                file_path,
                root,
                child,
                &push_pointer(&children_pointer, &index.to_string()),
                child_context,
            )?;
        }
    }
    Ok(())
}

fn validate_schema_object_children(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    for key in ["$defs", "definitions", "patternProperties", "properties"] {
        let Some(children) = object.get(key).and_then(Value::as_object) else {
            continue;
        };
        let children_pointer = push_pointer(pointer, key);
        for (name, child) in children {
            validate_schema_security_node(
                plugin_id,
                file_path,
                root,
                child,
                &push_pointer(&children_pointer, name),
                unsupported_secret_context,
            )?;
        }
    }
    Ok(())
}

fn validate_schema_dependent_schemas(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    let Some(children) = object.get("dependentSchemas").and_then(Value::as_object) else {
        return Ok(());
    };
    let children_pointer = push_pointer(pointer, "dependentSchemas");
    for (name, child) in children {
        validate_schema_security_node(
            plugin_id,
            file_path,
            root,
            child,
            &push_pointer(&children_pointer, name),
            unsupported_secret_context.or(Some("dependentSchemas")),
        )?;
    }
    Ok(())
}

fn validate_schema_dependencies(
    plugin_id: &str,
    file_path: &Path,
    root: &Value,
    object: &Map<String, Value>,
    pointer: &str,
    unsupported_secret_context: Option<&str>,
) -> Result<(), CliError> {
    let Some(dependencies) = object.get("dependencies").and_then(Value::as_object) else {
        return Ok(());
    };
    let dependencies_pointer = push_pointer(pointer, "dependencies");
    for (name, dependency) in dependencies {
        if dependency.is_object() || dependency.is_boolean() {
            validate_schema_security_node(
                plugin_id,
                file_path,
                root,
                dependency,
                &push_pointer(&dependencies_pointer, name),
                unsupported_secret_context.or(Some("dependencies")),
            )?;
        }
    }
    Ok(())
}

fn validate_schema_document(
    plugin_id: &str,
    path: &Path,
    draft: ConfigSchemaDraft,
    schema: &Value,
) -> Result<(), CliError> {
    let result = match draft {
        ConfigSchemaDraft::Draft7 => jsonschema::draft7::meta::validate(schema),
        ConfigSchemaDraft::Draft202012 => jsonschema::draft202012::meta::validate(schema),
    };
    result.map_err(|error| {
        schema_error(
            plugin_id,
            path,
            format!(
                "schema is invalid at JSON pointer '{}': {error}",
                error.instance_path()
            ),
        )
    })
}

fn build_object_fields(
    plugin_id: &str,
    path: &Path,
    root: &Value,
    schema: &Value,
    reference_stack: &HashSet<String>,
) -> Result<Vec<DynamicConfigField>, CliError> {
    let Some(object) = schema.as_object() else {
        return Ok(Vec::new());
    };
    let Some(properties) = object.get("properties") else {
        return Ok(Vec::new());
    };
    let Some(properties) = properties.as_object() else {
        return Err(schema_error(
            plugin_id,
            path,
            "'properties' must be an object",
        ));
    };
    let required = required_properties(object);
    let property_names = ordered_property_names(plugin_id, path, object, properties)?;
    property_names
        .into_iter()
        .map(|key| {
            build_field(
                plugin_id,
                path,
                root,
                &key,
                &properties[&key],
                required.contains(key.as_str()),
                reference_stack,
            )
        })
        .collect()
}

fn required_properties(object: &Map<String, Value>) -> HashSet<&str> {
    object
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

fn ordered_property_names(
    plugin_id: &str,
    path: &Path,
    schema: &Map<String, Value>,
    properties: &Map<String, Value>,
) -> Result<Vec<String>, CliError> {
    let mut alphabetical: Vec<String> = properties.keys().cloned().collect();
    alphabetical.sort();
    let Some(order) = schema.get("x-nemo-relay-order") else {
        return Ok(alphabetical);
    };
    let Some(order) = order.as_array() else {
        return Err(schema_error(
            plugin_id,
            path,
            "'x-nemo-relay-order' must be an array of unique property names",
        ));
    };

    let mut seen = HashSet::new();
    let mut ordered = Vec::with_capacity(properties.len());
    for value in order {
        let Some(key) = value.as_str() else {
            return Err(schema_error(
                plugin_id,
                path,
                "'x-nemo-relay-order' must contain only strings",
            ));
        };
        if !properties.contains_key(key) {
            return Err(schema_error(
                plugin_id,
                path,
                format!("'x-nemo-relay-order' names unknown property '{key}'"),
            ));
        }
        if !seen.insert(key) {
            return Err(schema_error(
                plugin_id,
                path,
                format!("'x-nemo-relay-order' contains duplicate property '{key}'"),
            ));
        }
        ordered.push(key.to_owned());
    }
    ordered.extend(
        alphabetical
            .into_iter()
            .filter(|key| !seen.contains(key.as_str())),
    );
    Ok(ordered)
}

fn build_field(
    plugin_id: &str,
    path: &Path,
    root: &Value,
    key: &str,
    schema: &Value,
    required: bool,
    reference_stack: &HashSet<String>,
) -> Result<DynamicConfigField, CliError> {
    let mut references = reference_stack.clone();
    let mut reference_chain = Vec::new();
    let resolved = match resolve_schema_chain(root, schema, &mut references, &mut reference_chain) {
        Ok(schema) => schema,
        Err(ResolveError::Cycle(_)) => {
            return Ok(DynamicConfigField {
                key: key.to_owned(),
                title: annotation_string(schema, schema, "title").unwrap_or_else(|| key.to_owned()),
                description: annotation_string(schema, schema, "description"),
                default: schema.get("default").cloned(),
                required,
                kind: DynamicConfigFieldKind::RawJson,
            });
        }
        Err(error) => {
            return Err(schema_error(
                plugin_id,
                path,
                format!("property '{key}' cannot be resolved: {error}"),
            ));
        }
    };
    let title = annotation_string(schema, resolved, "title").unwrap_or_else(|| key.to_owned());
    let description = annotation_string(schema, resolved, "description");
    let default = schema
        .get("default")
        .or_else(|| resolved.get("default"))
        .cloned();
    let secret = classify_write_only_chain(&reference_chain).map_err(|error| {
        schema_error(
            plugin_id,
            path,
            format!("property '{key}' has an unsupported writeOnly shape: {error}"),
        )
    })?;
    let kind = build_field_kind(plugin_id, path, root, resolved, secret, &references)?;

    Ok(DynamicConfigField {
        key: key.to_owned(),
        title,
        description,
        default,
        required,
        kind,
    })
}

fn build_field_kind(
    plugin_id: &str,
    path: &Path,
    root: &Value,
    schema: &Value,
    secret: bool,
    reference_stack: &HashSet<String>,
) -> Result<DynamicConfigFieldKind, CliError> {
    let Some(object) = schema.as_object() else {
        return Ok(DynamicConfigFieldKind::RawJson);
    };
    if has_unsupported_shape_keywords(object) {
        return Ok(DynamicConfigFieldKind::RawJson);
    }
    match schema_type(schema) {
        Some("boolean") => Ok(DynamicConfigFieldKind::Boolean),
        Some("string") => match object.get("enum") {
            None => Ok(DynamicConfigFieldKind::String { secret }),
            Some(Value::Array(values)) if values.iter().all(Value::is_string) => {
                Ok(DynamicConfigFieldKind::StringEnum {
                    options: values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect(),
                    secret,
                })
            }
            Some(_) => Ok(DynamicConfigFieldKind::RawJson),
        },
        Some("integer") => Ok(DynamicConfigFieldKind::Integer),
        Some("number") => Ok(DynamicConfigFieldKind::Number),
        Some("object") => {
            if object
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|properties| !properties.is_empty())
            {
                let fields = build_object_fields(plugin_id, path, root, schema, reference_stack)?;
                Ok(DynamicConfigFieldKind::Object { fields })
            } else if is_string_map(root, object, reference_stack) {
                Ok(DynamicConfigFieldKind::StringMap)
            } else {
                Ok(DynamicConfigFieldKind::RawJson)
            }
        }
        _ => Ok(DynamicConfigFieldKind::RawJson),
    }
}

fn has_unsupported_shape_keywords(schema: &Map<String, Value>) -> bool {
    [
        "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "const",
    ]
    .iter()
    .any(|keyword| schema.contains_key(*keyword))
}

fn is_string_map(
    root: &Value,
    schema: &Map<String, Value>,
    reference_stack: &HashSet<String>,
) -> bool {
    if schema.contains_key("patternProperties")
        || schema.contains_key("propertyNames")
        || schema
            .get("properties")
            .and_then(Value::as_object)
            .is_some_and(|properties| !properties.is_empty())
    {
        return false;
    }
    let Some(additional) = schema.get("additionalProperties") else {
        return false;
    };
    let mut references = reference_stack.clone();
    resolve_schema(root, additional, &mut references)
        .is_ok_and(|resolved| schema_type(resolved) == Some("string"))
}

fn schema_type(schema: &Value) -> Option<&str> {
    schema.get("type").and_then(Value::as_str)
}

fn annotation_string(primary: &Value, resolved: &Value, key: &str) -> Option<String> {
    primary
        .get(key)
        .or_else(|| resolved.get(key))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn classify_write_only_chain(reference_chain: &[&Value]) -> Result<bool, String> {
    if !reference_chain
        .iter()
        .any(|schema| schema.get("writeOnly").and_then(Value::as_bool) == Some(true))
    {
        return Ok(false);
    }
    if reference_chain
        .iter()
        .any(|schema| supports_string_secret(schema))
    {
        return Ok(true);
    }
    Err(
        "writeOnly must resolve directly or through local $ref values to type 'string' or \
         ['string', 'null']; annotations split across applicators are not supported"
            .to_owned(),
    )
}

fn supports_string_secret(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => kind == "string",
        Some(Value::Array(kinds)) => {
            let mut has_string = false;
            for kind in kinds {
                match kind.as_str() {
                    Some("string") => has_string = true,
                    Some("null") => {}
                    _ => return false,
                }
            }
            has_string
        }
        _ => false,
    }
}

#[derive(Debug)]
enum ResolveError {
    Missing(String),
    Cycle(String),
    InvalidFragment { reference: String, reason: String },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(reference) => {
                write!(formatter, "local reference '{reference}' was not found")
            }
            Self::Cycle(reference) => write!(formatter, "local reference '{reference}' is cyclic"),
            Self::InvalidFragment { reference, reason } => {
                write!(
                    formatter,
                    "local reference '{reference}' has an invalid fragment: {reason}"
                )
            }
        }
    }
}

fn resolve_schema<'a>(
    root: &'a Value,
    schema: &'a Value,
    references: &mut HashSet<String>,
) -> Result<&'a Value, ResolveError> {
    let mut reference_chain = Vec::new();
    resolve_schema_chain(root, schema, references, &mut reference_chain)
}

fn resolve_schema_chain<'a>(
    root: &'a Value,
    schema: &'a Value,
    references: &mut HashSet<String>,
    reference_chain: &mut Vec<&'a Value>,
) -> Result<&'a Value, ResolveError> {
    reference_chain.push(schema);
    let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
        return Ok(schema);
    };
    let fragment = decode_reference_fragment(reference)?;
    let canonical_reference = format!("#{fragment}");
    if !references.insert(canonical_reference.clone()) {
        return Err(ResolveError::Cycle(canonical_reference));
    }
    let target = resolve_fragment(root, &fragment)
        .ok_or_else(|| ResolveError::Missing(reference.to_owned()))?;
    resolve_schema_chain(root, target, references, reference_chain)
}

fn decode_reference_fragment(reference: &str) -> Result<String, ResolveError> {
    let Some(fragment) = reference.strip_prefix('#') else {
        return Err(ResolveError::InvalidFragment {
            reference: reference.to_owned(),
            reason: "reference must begin with '#'".to_owned(),
        });
    };
    let bytes = fragment.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit()
            {
                return Err(ResolveError::InvalidFragment {
                    reference: reference.to_owned(),
                    reason: "percent escapes must contain two hexadecimal digits".to_owned(),
                });
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    percent_decode_str(fragment)
        .decode_utf8()
        .map(|decoded| decoded.into_owned())
        .map_err(|error| ResolveError::InvalidFragment {
            reference: reference.to_owned(),
            reason: format!("decoded fragment is not UTF-8: {error}"),
        })
}

fn canonicalize_local_reference(reference: &str) -> Result<String, ResolveError> {
    let fragment = decode_reference_fragment(reference)?;
    Ok(format!(
        "#{}",
        utf8_percent_encode(&fragment, URI_FRAGMENT_ENCODE_SET)
    ))
}

fn resolve_fragment<'a>(root: &'a Value, fragment: &str) -> Option<&'a Value> {
    if fragment.is_empty() {
        return Some(root);
    }
    if fragment.starts_with('/') {
        return root.pointer(fragment);
    }
    find_anchor(root, fragment)
}

fn find_anchor<'a>(schema: &'a Value, anchor: &str) -> Option<&'a Value> {
    match schema {
        Value::Object(object) => {
            if object.get("$anchor").and_then(Value::as_str) == Some(anchor)
                || object.get("$id").and_then(Value::as_str) == Some(&format!("#{anchor}"))
            {
                return Some(schema);
            }
            object.values().find_map(|child| find_anchor(child, anchor))
        }
        Value::Array(values) => values.iter().find_map(|child| find_anchor(child, anchor)),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SecretSegment {
    Property(String),
    Any,
    Pattern(SecretPropertyPattern),
    UnmatchedProperties(SecretUnmatchedProperties),
    Index(usize),
    Tail(usize),
}

#[derive(Debug, Clone)]
struct SecretPropertyPattern {
    source: String,
    matcher: regex::Regex,
}

impl PartialEq for SecretPropertyPattern {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for SecretPropertyPattern {}

impl PartialOrd for SecretPropertyPattern {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SecretPropertyPattern {
    fn cmp(&self, other: &Self) -> Ordering {
        self.source.cmp(&other.source)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SecretUnmatchedProperties {
    properties: Vec<String>,
    patterns: Vec<SecretPropertyPattern>,
}

impl SecretUnmatchedProperties {
    fn matches(&self, property: &str) -> bool {
        self.properties
            .binary_search_by(|candidate| candidate.as_str().cmp(property))
            .is_err()
            && !self
                .patterns
                .iter()
                .any(|pattern| pattern_matches(pattern, property))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SecretPattern(Vec<SecretSegment>);

impl SecretPattern {
    fn redact(&self, value: &mut Value, offset: usize) {
        self.visit_matching_values(value, offset, &mut |value| {
            // A configuration can contain a schema-invalid value before validation. Once the
            // schema marks this path as secret, its runtime type must not determine whether it
            // is safe to display. Null remains visible because it represents an unset nullable
            // secret and carries no payload.
            if !value.is_null() {
                *value = Value::String(REDACTED.to_owned());
            }
        });
    }

    fn redact_for_edit(
        &self,
        value: &mut Value,
        offset: usize,
        secrets: &mut SecretEditValues,
        occupied: &HashSet<String>,
        next_token: &mut usize,
    ) {
        self.visit_matching_values(value, offset, &mut |value| {
            // Tokenize invalid values too, both to keep raw editing safe and to preserve the
            // original value if the user leaves it unchanged.
            if value.is_null()
                || value
                    .as_str()
                    .is_some_and(|candidate| secrets.contains_key(candidate))
            {
                return;
            }
            let token = next_secret_token(secrets, occupied, next_token);
            secrets.insert(
                token.clone(),
                SecretEditValue {
                    value: value.clone(),
                    pattern: self.clone(),
                },
            );
            *value = Value::String(token);
        });
    }

    fn visit_matching_values(
        &self,
        value: &mut Value,
        offset: usize,
        visit: &mut impl FnMut(&mut Value),
    ) {
        if offset == self.0.len() {
            visit(value);
            return;
        }
        match &self.0[offset] {
            SecretSegment::Property(property) => {
                if let Some(child) = value.get_mut(property) {
                    self.visit_matching_values(child, offset + 1, visit);
                }
            }
            SecretSegment::Any => match value {
                Value::Object(object) => {
                    for child in object.values_mut() {
                        self.visit_matching_values(child, offset + 1, visit);
                    }
                }
                Value::Array(values) => {
                    for child in values {
                        self.visit_matching_values(child, offset + 1, visit);
                    }
                }
                _ => {}
            },
            SecretSegment::Pattern(pattern) => {
                if let Value::Object(object) = value {
                    for (key, child) in object {
                        if pattern_matches(pattern, key) {
                            self.visit_matching_values(child, offset + 1, visit);
                        }
                    }
                }
            }
            SecretSegment::UnmatchedProperties(selector) => {
                if let Value::Object(object) = value {
                    for (key, child) in object {
                        if selector.matches(key) {
                            self.visit_matching_values(child, offset + 1, visit);
                        }
                    }
                }
            }
            SecretSegment::Index(index) => {
                if let Some(child) = value.get_mut(*index) {
                    self.visit_matching_values(child, offset + 1, visit);
                }
            }
            SecretSegment::Tail(start) => {
                if let Value::Array(values) = value {
                    for child in values.iter_mut().skip(*start) {
                        self.visit_matching_values(child, offset + 1, visit);
                    }
                }
            }
        }
    }

    fn applies_below(&self, path: &[String]) -> bool {
        self.0.len() >= path.len()
            && self
                .0
                .iter()
                .zip(path)
                .all(|(segment, property)| match segment {
                    SecretSegment::Property(expected) => expected == property,
                    SecretSegment::Any => true,
                    SecretSegment::Pattern(pattern) => pattern_matches(pattern, property),
                    SecretSegment::UnmatchedProperties(selector) => selector.matches(property),
                    SecretSegment::Index(index) => property.parse::<usize>() == Ok(*index),
                    SecretSegment::Tail(start) => {
                        property.parse::<usize>().is_ok_and(|index| index >= *start)
                    }
                })
    }

    fn matches_instance_path(&self, path: &[SecretInstanceSegment]) -> bool {
        self.0.len() == path.len()
            && self
                .0
                .iter()
                .zip(path)
                .all(|(pattern, instance)| match (pattern, instance) {
                    (
                        SecretSegment::Property(expected),
                        SecretInstanceSegment::Property(actual),
                    ) => expected == actual,
                    (SecretSegment::Any, _) => true,
                    (SecretSegment::Pattern(pattern), SecretInstanceSegment::Property(actual)) => {
                        pattern_matches(pattern, actual)
                    }
                    (
                        SecretSegment::UnmatchedProperties(selector),
                        SecretInstanceSegment::Property(actual),
                    ) => selector.matches(actual),
                    (SecretSegment::Index(expected), SecretInstanceSegment::Index(actual)) => {
                        expected == actual
                    }
                    (SecretSegment::Tail(start), SecretInstanceSegment::Index(actual)) => {
                        actual >= start
                    }
                    _ => false,
                })
    }
}

#[derive(Debug, Clone)]
enum SecretInstanceSegment {
    Property(String),
    Index(usize),
}

fn pattern_matches(pattern: &SecretPropertyPattern, property: &str) -> bool {
    pattern.matcher.is_match(property)
}

fn collect_string_values(value: &Value, output: &mut HashSet<String>) {
    match value {
        Value::String(value) => {
            output.insert(value.clone());
        }
        Value::Array(values) => {
            for value in values {
                collect_string_values(value, output);
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                collect_string_values(value, output);
            }
        }
        _ => {}
    }
}

fn next_secret_token(
    secrets: &SecretEditValues,
    occupied: &HashSet<String>,
    next_token: &mut usize,
) -> String {
    loop {
        let token = format!("{EDIT_REDACTED_PREFIX}{}>", *next_token);
        *next_token += 1;
        if !secrets.contains_key(&token) && !occupied.contains(&token) {
            return token;
        }
    }
}

fn restore_secret_tokens(value: &Value, secrets: &SecretEditValues) -> Result<Value, String> {
    fn restore(
        value: &Value,
        secrets: &SecretEditValues,
        path: &mut Vec<SecretInstanceSegment>,
        used_tokens: &mut HashSet<String>,
    ) -> Result<Value, String> {
        match value {
            Value::String(value) => match secrets.get(value) {
                None => Ok(Value::String(value.clone())),
                Some(secret) if !secret.pattern.matches_instance_path(path) => Err(format!(
                    "token '{value}' may only appear at its original schema-declared secret location"
                )),
                Some(_) if !used_tokens.insert(value.clone()) => {
                    Err(format!("token '{value}' may only appear once"))
                }
                Some(secret) => Ok(secret.value.clone()),
            },
            Value::Array(values) => {
                let mut restored = Vec::with_capacity(values.len());
                for (index, value) in values.iter().enumerate() {
                    path.push(SecretInstanceSegment::Index(index));
                    restored.push(restore(value, secrets, path, used_tokens)?);
                    path.pop();
                }
                Ok(Value::Array(restored))
            }
            Value::Object(values) => {
                let mut restored = Map::with_capacity(values.len());
                for (key, value) in values {
                    path.push(SecretInstanceSegment::Property(key.clone()));
                    restored.insert(key.clone(), restore(value, secrets, path, used_tokens)?);
                    path.pop();
                }
                Ok(Value::Object(restored))
            }
            value => Ok(value.clone()),
        }
    }

    restore(value, secrets, &mut Vec::new(), &mut HashSet::new())
}

fn discover_secret_patterns(
    root: &Value,
    schema: &Value,
    instance_path: &[SecretSegment],
    reference_stack: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let mut references = reference_stack.clone();
    let mut reference_chain = Vec::new();
    resolve_schema_chain(root, schema, &mut references, &mut reference_chain)
        .map_err(|error| format!("secret schema reference could not be resolved: {error}"))?;
    if classify_write_only_chain(&reference_chain)? {
        output.push(SecretPattern(instance_path.to_vec()));
        return Ok(());
    }

    // Draft 2020-12 treats `$ref` as an applicator, so sibling keywords remain active. Walk
    // every node recorded during resolution instead of only the final target; otherwise a
    // sibling `properties` subtree can contain writeOnly fields that never get redacted.
    for effective_schema in reference_chain {
        if let Some(object) = effective_schema.as_object() {
            discover_secret_patterns_in_object(root, object, instance_path, &references, output)?;
        }
    }
    Ok(())
}

fn discover_secret_patterns_in_object(
    root: &Value,
    object: &Map<String, Value>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let properties = object.get("properties").and_then(Value::as_object);
    discover_named_secret_patterns(root, properties, instance_path, references, output)?;
    let pattern_schemas = collect_secret_pattern_schemas(object)?;
    discover_additional_secret_patterns(
        root,
        object,
        properties,
        &pattern_schemas,
        instance_path,
        references,
        output,
    )?;
    discover_pattern_property_secret_patterns(
        root,
        pattern_schemas,
        instance_path,
        references,
        output,
    )?;
    discover_item_secret_patterns(root, object, instance_path, references, output)?;
    discover_prefix_item_secret_patterns(root, object, instance_path, references, output)?;
    discover_all_of_secret_patterns(root, object, instance_path, references, output)?;
    reject_array_applicator_secret_patterns(root, object, instance_path, references)?;
    reject_value_applicator_secret_patterns(
        root,
        object,
        &["if", "then", "else", "not"],
        instance_path,
        references,
    )?;
    discover_contains_secret_patterns(root, object, instance_path, references, output)?;
    reject_value_applicator_secret_patterns(
        root,
        object,
        &["unevaluatedProperties", "unevaluatedItems"],
        instance_path,
        references,
    )?;
    reject_object_applicator_secret_patterns(
        root,
        object,
        &["dependentSchemas", "dependencies"],
        instance_path,
        references,
    )?;
    Ok(())
}

fn discover_named_secret_patterns(
    root: &Value,
    properties: Option<&Map<String, Value>>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let Some(properties) = properties else {
        return Ok(());
    };
    for (property, child_schema) in properties {
        let mut child_path = instance_path.to_vec();
        child_path.push(SecretSegment::Property(property.clone()));
        discover_secret_patterns(root, child_schema, &child_path, references, output)?;
    }
    Ok(())
}

fn collect_secret_pattern_schemas(
    object: &Map<String, Value>,
) -> Result<Vec<(SecretPropertyPattern, &Value)>, String> {
    let Some(patterns) = object.get("patternProperties").and_then(Value::as_object) else {
        return Ok(Vec::new());
    };
    let mut pattern_schemas = Vec::new();
    for (pattern, child_schema) in patterns {
        let matcher = regex::Regex::new(pattern).map_err(|error| {
            format!("unsupported patternProperties expression {pattern:?}: {error}")
        })?;
        pattern_schemas.push((
            SecretPropertyPattern {
                source: pattern.clone(),
                matcher,
            },
            child_schema,
        ));
    }
    pattern_schemas.sort_by(|(left, _), (right, _)| left.cmp(right));
    Ok(pattern_schemas)
}

fn discover_additional_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    properties: Option<&Map<String, Value>>,
    pattern_schemas: &[(SecretPropertyPattern, &Value)],
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let Some(additional) = object.get("additionalProperties") else {
        return Ok(());
    };
    if !additional.is_object() {
        return Ok(());
    }
    let mut excluded_properties = properties
        .into_iter()
        .flat_map(|properties| properties.keys().cloned())
        .collect::<Vec<_>>();
    excluded_properties.sort();
    let mut child_path = instance_path.to_vec();
    child_path.push(SecretSegment::UnmatchedProperties(
        SecretUnmatchedProperties {
            properties: excluded_properties,
            patterns: pattern_schemas
                .iter()
                .map(|(pattern, _)| pattern.clone())
                .collect(),
        },
    ));
    discover_secret_patterns(root, additional, &child_path, references, output)
}

fn discover_pattern_property_secret_patterns(
    root: &Value,
    pattern_schemas: Vec<(SecretPropertyPattern, &Value)>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    for (pattern, child_schema) in pattern_schemas {
        let mut child_path = instance_path.to_vec();
        child_path.push(SecretSegment::Pattern(pattern));
        discover_secret_patterns(root, child_schema, &child_path, references, output)?;
    }
    Ok(())
}

fn discover_item_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let Some(items) = object.get("items") else {
        return Ok(());
    };
    if items.is_object() {
        let mut child_path = instance_path.to_vec();
        let segment = object
            .get("prefixItems")
            .and_then(Value::as_array)
            .map_or(SecretSegment::Any, |prefix_items| {
                SecretSegment::Tail(prefix_items.len())
            });
        child_path.push(segment);
        return discover_secret_patterns(root, items, &child_path, references, output);
    }
    let Some(tuple_items) = items.as_array() else {
        return Ok(());
    };
    for (index, child_schema) in tuple_items.iter().enumerate() {
        let mut child_path = instance_path.to_vec();
        child_path.push(SecretSegment::Index(index));
        discover_secret_patterns(root, child_schema, &child_path, references, output)?;
    }
    if let Some(additional_items) = object.get("additionalItems")
        && additional_items.is_object()
    {
        let mut child_path = instance_path.to_vec();
        child_path.push(SecretSegment::Tail(tuple_items.len()));
        discover_secret_patterns(root, additional_items, &child_path, references, output)?;
    }
    Ok(())
}

fn discover_prefix_item_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let Some(prefix_items) = object.get("prefixItems").and_then(Value::as_array) else {
        return Ok(());
    };
    for (index, child_schema) in prefix_items.iter().enumerate() {
        let mut child_path = instance_path.to_vec();
        child_path.push(SecretSegment::Index(index));
        discover_secret_patterns(root, child_schema, &child_path, references, output)?;
    }
    Ok(())
}

fn discover_all_of_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let Some(branches) = object.get("allOf").and_then(Value::as_array) else {
        return Ok(());
    };
    for branch in branches {
        discover_secret_patterns(root, branch, instance_path, references, output)?;
    }
    Ok(())
}

fn reject_array_applicator_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
) -> Result<(), String> {
    for keyword in ["anyOf", "oneOf"] {
        let Some(branches) = object.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        for branch in branches {
            reject_write_only_under_applicator(root, keyword, branch, instance_path, references)?;
        }
    }
    Ok(())
}

fn reject_value_applicator_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    keywords: &[&str],
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
) -> Result<(), String> {
    for keyword in keywords {
        if let Some(branch) = object.get(*keyword)
            && branch.is_object()
        {
            reject_write_only_under_applicator(root, keyword, branch, instance_path, references)?;
        }
    }
    Ok(())
}

fn discover_contains_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
    output: &mut Vec<SecretPattern>,
) -> Result<(), String> {
    let Some(contains) = object.get("contains") else {
        return Ok(());
    };
    if !contains.is_object() {
        return Ok(());
    }
    let mut child_path = instance_path.to_vec();
    child_path.push(SecretSegment::Any);
    discover_secret_patterns(root, contains, &child_path, references, output)
}

fn reject_object_applicator_secret_patterns(
    root: &Value,
    object: &Map<String, Value>,
    keywords: &[&str],
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
) -> Result<(), String> {
    for keyword in keywords {
        let Some(branches) = object.get(*keyword).and_then(Value::as_object) else {
            continue;
        };
        for branch in branches.values().filter(|branch| branch.is_object()) {
            reject_write_only_under_applicator(root, keyword, branch, instance_path, references)?;
        }
    }
    Ok(())
}

fn reject_write_only_under_applicator(
    root: &Value,
    keyword: &str,
    schema: &Value,
    instance_path: &[SecretSegment],
    references: &HashSet<String>,
) -> Result<(), String> {
    let mut nested_patterns = Vec::new();
    discover_secret_patterns(
        root,
        schema,
        instance_path,
        references,
        &mut nested_patterns,
    )?;
    if nested_patterns.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "writeOnly fields under '{keyword}' are not supported for secret redaction"
        ))
    }
}

fn push_pointer(pointer: &str, segment: &str) -> String {
    format!("{pointer}/{}", escape_pointer(segment))
}

fn escape_pointer(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/plugins_schema_tests.rs"]
mod tests;
