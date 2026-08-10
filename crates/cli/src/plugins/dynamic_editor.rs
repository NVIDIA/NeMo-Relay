// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interactive editor state and controls for dynamic plugin host configuration.

use std::collections::HashSet;

use dialoguer::theme::ColorfulTheme;
use nemo_relay::plugin::dynamic::DynamicPluginManifest;
use serde_json::{Map, Value};

use crate::error::CliError;

use super::config_io::{DynamicPluginConfigEntry, PluginConfigDocument};
use super::schema::{
    DynamicConfigField, DynamicConfigFieldKind, PluginConfigSchema, SecretEditValues,
};
use super::{
    MenuItem, MenuResponse, MenuShortcut, configured_label, editor_error, menu_response_index,
    prompt_menu, shortcut_label,
};

const REDACTED: &str = "<redacted>";

mod prompt;

#[derive(Debug)]
pub(super) struct DynamicPluginEditorState {
    document_index: usize,
    plugin_id: String,
    label: String,
    editor_title: Option<String>,
    description: Option<String>,
    original_config: Option<Map<String, Value>>,
    config: Option<Map<String, Value>>,
    schema: Option<PluginConfigSchema>,
    touched: bool,
}

impl DynamicPluginEditorState {
    pub(super) fn label(&self) -> &str {
        &self.label
    }

    pub(super) fn menu_summary(&self) -> String {
        let config = match &self.config {
            None => "config absent",
            Some(config) if config.is_empty() => "explicit empty config",
            Some(_) => "configured",
        };
        let editor = match &self.schema {
            Some(schema) if schema.fields().is_empty() => "schema-validated JSON",
            Some(_) => "schema fields",
            None => "raw JSON object",
        };
        format!("dynamic; {config}; {editor}")
    }

    pub(super) fn validate(&self) -> Result<(), CliError> {
        if let Some(schema) = &self.schema {
            schema.validate(&Value::Object(self.config.clone().unwrap_or_default()))?;
        }
        Ok(())
    }

    pub(super) fn has_persisted_secrets(&self) -> bool {
        self.config.as_ref().is_some_and(|config| {
            self.schema
                .as_ref()
                .is_some_and(|schema| schema.has_persisted_secrets(&Value::Object(config.clone())))
        })
    }

    pub(super) fn apply_to_document(
        &self,
        document: &mut PluginConfigDocument,
        redact_secrets: bool,
    ) -> Result<(), CliError> {
        let needs_preview_redaction = redact_secrets
            && self
                .schema
                .as_ref()
                .is_some_and(PluginConfigSchema::has_secrets);
        if !self.touched && !needs_preview_redaction {
            return Ok(());
        }
        match &self.config {
            None => document.patch_dynamic_config(
                self.document_index,
                self.original_config.as_ref(),
                None,
            ),
            Some(config) => {
                let config = if redact_secrets {
                    self.schema
                        .as_ref()
                        .map(|schema| schema.redact(&Value::Object(config.clone())))
                        .unwrap_or_else(|| Value::Object(config.clone()))
                } else {
                    Value::Object(config.clone())
                };
                let config = config.as_object().cloned().ok_or_else(|| {
                    CliError::Config(format!(
                        "dynamic plugin '{}' configuration must be a JSON object",
                        self.plugin_id
                    ))
                })?;
                document.patch_dynamic_config(
                    self.document_index,
                    self.original_config.as_ref(),
                    Some(config),
                )
            }
        }
    }

    fn redacted_config(&self) -> Option<Map<String, Value>> {
        self.config.as_ref().map(|config| {
            self.schema
                .as_ref()
                .map(|schema| schema.redact(&Value::Object(config.clone())))
                .unwrap_or_else(|| Value::Object(config.clone()))
                .as_object()
                .cloned()
                .unwrap_or_default()
        })
    }

    pub(super) fn reset(&mut self) {
        self.config = None;
        self.touched = true;
    }

    #[cfg(test)]
    pub(super) fn config(&self) -> Option<&Map<String, Value>> {
        self.config.as_ref()
    }

    #[cfg(test)]
    pub(super) fn top_level_field_labels(&self) -> Vec<String> {
        self.schema
            .as_ref()
            .map(|schema| {
                dynamic_field_menu_items(self, schema.fields(), &[])
                    .0
                    .into_iter()
                    .map(|item| console::strip_ansi_codes(&item.label).into_owned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(super) fn editor_fields(&self) -> &[DynamicConfigField] {
        self.schema.as_ref().map_or(&[], |schema| schema.fields())
    }

    #[cfg(test)]
    pub(super) fn reset_top_level_field(&mut self, key: &str) -> Result<(), CliError> {
        let field = self
            .schema
            .as_ref()
            .and_then(|schema| schema.fields().iter().find(|field| field.key == key))
            .cloned()
            .ok_or_else(|| CliError::Config(format!("unknown dynamic config field '{key}'")))?;
        self.reset_field(&[key.to_owned()], &field);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn clear_top_level_field(&mut self, key: &str) {
        self.remove_field(&[key.to_owned()]);
    }

    #[cfg(test)]
    pub(super) fn top_level_field_uses_hidden_json(&self, key: &str) -> bool {
        self.field_has_secrets(&[key.to_owned()])
    }

    fn set_raw_config(&mut self, config: Map<String, Value>) {
        self.config = Some(config);
        self.touched = true;
    }

    fn field_value(&self, path: &[String]) -> Option<&Value> {
        value_at_path(self.config.as_ref(), path)
    }

    fn display_field_value(&self, path: &[String]) -> Option<Value> {
        let redacted = self.redacted_config();
        value_at_path(redacted.as_ref(), path).cloned()
    }

    fn field_has_secrets(&self, path: &[String]) -> bool {
        self.schema
            .as_ref()
            .is_some_and(|schema| schema.has_secrets_at(path))
    }

    fn field_value_for_raw_edit(
        &self,
        path: &[String],
    ) -> (
        Option<Value>,
        Option<Map<String, Value>>,
        SecretEditValues,
        bool,
    ) {
        let original = Value::Object(self.config.clone().unwrap_or_default());
        let Some(schema) = &self.schema else {
            return (
                self.field_value(path).cloned(),
                None,
                SecretEditValues::new(),
                false,
            );
        };
        let (redacted, secrets) = schema.redact_for_edit(&original);
        let value = value_at_path(redacted.as_object(), path).cloned();
        (
            value,
            redacted.as_object().cloned(),
            secrets,
            schema.has_secrets_at(path),
        )
    }

    fn restore_raw_field_edit(
        &self,
        path: &[String],
        value: Value,
        redacted_config: Option<Map<String, Value>>,
        secrets: &SecretEditValues,
    ) -> Result<Value, CliError> {
        let Some(schema) = &self.schema else {
            return Ok(value);
        };
        let mut config = redacted_config;
        set_value_at_path(&mut config, path, value);
        let restored =
            schema.restore_edit_secrets(&Value::Object(config.unwrap_or_default()), secrets)?;
        value_at_path(restored.as_object(), path)
            .cloned()
            .ok_or_else(|| {
                CliError::Config(format!(
                    "dynamic plugin '{}' raw field '{}' could not be restored",
                    self.plugin_id,
                    path.join(".")
                ))
            })
    }

    #[cfg(test)]
    pub(super) fn restore_raw_field_for_test(&self, path: &[String]) -> Result<Value, CliError> {
        let (value, redacted_config, secrets, _) = self.field_value_for_raw_edit(path);
        self.restore_raw_field_edit(
            path,
            value.unwrap_or(Value::Null),
            redacted_config,
            &secrets,
        )
    }

    fn set_field(&mut self, path: &[String], value: Value) {
        set_value_at_path(&mut self.config, path, value);
        self.touched = true;
    }

    fn remove_field(&mut self, path: &[String]) {
        if let Some(config) = &mut self.config {
            remove_value_at_path(config, path);
            self.touched = true;
        }
    }

    fn reset_field(&mut self, path: &[String], field: &DynamicConfigField) {
        match &field.default {
            Some(default) => self.set_field(path, default.clone()),
            None => self.remove_field(path),
        }
    }
}

pub(super) fn load_dynamic_plugin_states(
    document: &PluginConfigDocument,
) -> Result<Vec<DynamicPluginEditorState>, CliError> {
    let entries = document.dynamic_entries()?;
    let mut plugin_ids = HashSet::new();
    entries
        .into_iter()
        .map(|entry| load_dynamic_plugin_state(entry, &mut plugin_ids))
        .collect()
}

fn load_dynamic_plugin_state(
    entry: DynamicPluginConfigEntry,
    plugin_ids: &mut HashSet<String>,
) -> Result<DynamicPluginEditorState, CliError> {
    let (manifest, manifest_ref) = crate::configuration::load_bounded_dynamic_plugin_manifest(
        &entry.manifest_path,
    )
    .map_err(|error| {
        CliError::Config(format!(
            "could not load dynamic plugin manifest '{}' for editing: {error}",
            entry.manifest
        ))
    })?;
    let plugin_id = manifest.plugin.id.trim().to_owned();
    if !plugin_ids.insert(plugin_id.clone()) {
        return Err(CliError::Config(format!(
            "dynamic plugin '{}' is declared more than once in {}",
            plugin_id,
            entry.manifest_path.display()
        )));
    }
    let schema = load_config_schema(&manifest, &manifest_ref)?;
    let label = manifest
        .plugin
        .name
        .as_deref()
        .filter(|name| *name != plugin_id)
        .map(|name| format!("{name} ({plugin_id})"))
        .unwrap_or_else(|| plugin_id.clone());
    let description = schema
        .as_ref()
        .and_then(|schema| schema.editor().description.clone())
        .or(manifest.description);
    let editor_title = schema
        .as_ref()
        .and_then(|schema| schema.editor().title.clone());

    let original_config = entry.config.clone();
    Ok(DynamicPluginEditorState {
        document_index: entry.index,
        plugin_id,
        label,
        editor_title,
        description,
        original_config,
        config: entry.config,
        schema,
        touched: false,
    })
}

fn load_config_schema(
    manifest: &DynamicPluginManifest,
    manifest_ref: &str,
) -> Result<Option<PluginConfigSchema>, CliError> {
    manifest
        .resolve_config_schema_path(manifest_ref)
        .map_err(|error| {
            CliError::Config(format!(
                "dynamic plugin '{}' config schema path could not be resolved from '{}': {error}",
                manifest.plugin.id, manifest_ref
            ))
        })?
        .map(|path| PluginConfigSchema::load(manifest.plugin.id.trim(), path))
        .transpose()
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DynamicMenuAction {
    EditField(usize),
    EditRawConfig,
    ResetPlugin,
    Back,
}

pub(super) fn edit_dynamic_plugin(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
) -> Result<(), CliError> {
    prompt::edit_dynamic_plugin(theme, state)
}

pub(super) fn dynamic_root_menu_items(
    state: &DynamicPluginEditorState,
    fields: &[DynamicConfigField],
) -> (Vec<MenuItem>, Vec<DynamicMenuAction>) {
    let mut items = Vec::new();
    let mut actions = Vec::new();
    if fields.is_empty() {
        items.push(MenuItem::new(configured_label(
            state.config.is_some(),
            "Edit configuration as JSON object",
        )));
        actions.push(DynamicMenuAction::EditRawConfig);
    }
    items.push(MenuItem::new(shortcut_label(
        "Reset plugin configuration",
        "r",
    )));
    actions.push(DynamicMenuAction::ResetPlugin);
    items.push(MenuItem::new(shortcut_label("Back", "q")));
    actions.push(DynamicMenuAction::Back);
    (items, actions)
}

pub(super) fn dynamic_field_menu_items(
    state: &DynamicPluginEditorState,
    fields: &[DynamicConfigField],
    parent_path: &[String],
) -> (Vec<MenuItem>, Vec<DynamicMenuAction>) {
    let mut items = Vec::with_capacity(fields.len() + 2);
    let mut actions = Vec::with_capacity(fields.len() + 2);
    for (index, field) in fields.iter().enumerate() {
        let path = field_path(parent_path, field);
        let configured = state.field_value(&path).is_some();
        let value = state
            .display_field_value(&path)
            .map(|value| display_dynamic_value(&value, &field.kind))
            .or_else(|| {
                field.default.as_ref().map(|default| {
                    if field_is_secret(field) || state.field_has_secrets(&path) {
                        format!("{REDACTED} (default)")
                    } else {
                        format!("{} (default)", display_dynamic_value(default, &field.kind))
                    }
                })
            })
            .unwrap_or_else(|| "(unset)".to_owned());
        let required = if field.required { " [required]" } else { "" };
        items.push(MenuItem::new(format!(
            "{}{} = {value}",
            configured_label(configured, &field.title),
            required
        )));
        actions.push(DynamicMenuAction::EditField(index));
    }
    if parent_path.is_empty() {
        items.push(MenuItem::new(shortcut_label(
            "Reset plugin configuration",
            "r",
        )));
        actions.push(DynamicMenuAction::ResetPlugin);
    }
    items.push(MenuItem::new(shortcut_label("Back", "q")));
    actions.push(DynamicMenuAction::Back);
    (items, actions)
}

pub(super) fn reset_dynamic_selection(
    state: &mut DynamicPluginEditorState,
    fields: &[DynamicConfigField],
    parent_path: &[String],
    actions: &[DynamicMenuAction],
    selected: usize,
) {
    match actions.get(selected).copied() {
        Some(DynamicMenuAction::EditField(index)) => {
            let field = &fields[index];
            state.reset_field(&field_path(parent_path, field), field);
        }
        Some(DynamicMenuAction::ResetPlugin) => state.reset(),
        _ => println!("  Select a setting to reset."),
    }
}

pub(super) fn clear_dynamic_selection(
    state: &mut DynamicPluginEditorState,
    fields: &[DynamicConfigField],
    parent_path: &[String],
    actions: &[DynamicMenuAction],
    selected: usize,
) {
    match actions.get(selected).copied() {
        Some(DynamicMenuAction::EditField(index)) => {
            state.remove_field(&field_path(parent_path, &fields[index]));
        }
        _ => println!("  Select a field to clear."),
    }
}

pub(super) fn field_path(parent_path: &[String], field: &DynamicConfigField) -> Vec<String> {
    let mut path = parent_path.to_vec();
    path.push(field.key.clone());
    path
}

fn field_is_secret(field: &DynamicConfigField) -> bool {
    matches!(
        field.kind,
        DynamicConfigFieldKind::String { secret: true }
            | DynamicConfigFieldKind::StringEnum { secret: true, .. }
    )
}

pub(super) fn value_at_path<'a>(
    config: Option<&'a Map<String, Value>>,
    path: &[String],
) -> Option<&'a Value> {
    let (first, rest) = path.split_first()?;
    let mut value = config?.get(first)?;
    for segment in rest {
        value = value.as_object()?.get(segment)?;
    }
    Some(value)
}

pub(super) fn set_value_at_path(
    config: &mut Option<Map<String, Value>>,
    path: &[String],
    value: Value,
) {
    let Some((last, parents)) = path.split_last() else {
        return;
    };
    let mut object = config.get_or_insert_with(Map::new);
    for segment in parents {
        let entry = object
            .entry(segment.clone())
            .or_insert_with(|| Value::Object(Map::new()));
        if !entry.is_object() {
            *entry = Value::Object(Map::new());
        }
        object = entry
            .as_object_mut()
            .expect("newly inserted path segment is an object");
    }
    object.insert(last.clone(), value);
}

pub(super) fn remove_value_at_path(config: &mut Map<String, Value>, path: &[String]) -> bool {
    let Some((first, rest)) = path.split_first() else {
        return config.is_empty();
    };
    if rest.is_empty() {
        config.remove(first);
        return config.is_empty();
    }
    let remove_parent = config
        .get_mut(first)
        .and_then(Value::as_object_mut)
        .is_some_and(|object| remove_value_at_path(object, rest));
    if remove_parent {
        config.remove(first);
    }
    config.is_empty()
}

fn display_dynamic_value(value: &Value, kind: &DynamicConfigFieldKind) -> String {
    if matches!(kind, DynamicConfigFieldKind::Object { .. }) {
        return "{…}".to_owned();
    }
    json_text(value)
}

fn json_text(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid JSON>".to_owned())
}
