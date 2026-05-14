// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable Observability plugin editor state helpers.

use nemo_flow::config_editor::{EditorConfig, EditorFieldKind, EditorFieldSpec};
use nemo_flow::observability::plugin_component::{OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig};
use nemo_flow::plugin::{PluginComponentSpec, PluginConfig};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::error::CliError;

pub(super) const POLICY_SECTION: &str = "policy";

pub(super) fn ensure_observability_component(config: &mut PluginConfig) -> Result<(), CliError> {
    if !config
        .components
        .iter()
        .any(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
    {
        config.components.push(PluginComponentSpec {
            kind: OBSERVABILITY_PLUGIN_KIND.to_string(),
            enabled: true,
            config: observability_config_map(&ObservabilityConfig::default())?,
        });
    }
    Ok(())
}

pub(super) fn component_enabled(config: &PluginConfig) -> bool {
    observability_component(config)
        .map(|component| component.enabled)
        .unwrap_or(true)
}

pub(super) fn set_component_enabled(config: &mut PluginConfig, enabled: bool) {
    if let Some(component) = observability_component_mut(config) {
        component.enabled = enabled;
    }
}

pub(super) fn component_observability_config(
    config: &PluginConfig,
) -> Result<ObservabilityConfig, CliError> {
    observability_component(config)
        .map(|component| serde_json::from_value(Value::Object(component.config.clone())))
        .transpose()
        .map_err(|error| CliError::Config(format!("invalid observability plugin config: {error}")))?
        .ok_or_else(|| CliError::Config("observability plugin component is missing".into()))
}

pub(super) fn config_with_observability(
    config: &PluginConfig,
    observability: &ObservabilityConfig,
) -> Result<PluginConfig, CliError> {
    let mut config = config.clone();
    store_observability_config(&mut config, observability)?;
    Ok(config)
}

pub(super) fn store_observability_config(
    config: &mut PluginConfig,
    observability: &ObservabilityConfig,
) -> Result<(), CliError> {
    if let Some(component) = observability_component_mut(config) {
        merge_observability_editor_config(
            &mut component.config,
            observability_config_map(observability)?,
        );
    }
    Ok(())
}

pub(super) fn ensure_section(config: &mut ObservabilityConfig, section: EditorFieldSpec) {
    if let Ok(Some(Value::Object(_))) = section_value(config, section) {
        return;
    }
    let Some(default) = section.default_value() else {
        return;
    };
    let _ = set_struct_field(config, section.name, default);
}

pub(super) fn toggle_section(config: &mut ObservabilityConfig, section: EditorFieldSpec) {
    ensure_section(config, section);
    let enabled = section_enabled(config, section).unwrap_or(false);
    let _ = set_section_field(config, section, "enabled", json!(!enabled));
}

pub(super) fn reset_section(config: &mut ObservabilityConfig, section: EditorFieldSpec) {
    let value = section.default_value().unwrap_or_else(|| json!({}));
    let _ = set_struct_field(config, section.name, value);
}

pub(super) fn reset_selected_field(
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
    selected: usize,
) -> Result<bool, CliError> {
    let offset = usize::from(section_has_enabled_toggle(section));
    let Some(index) = selected.checked_sub(offset) else {
        return Ok(false);
    };
    let Some(field) = fields.get(index) else {
        return Ok(false);
    };
    remove_section_field(config, section, field.name)?;
    Ok(true)
}

pub(super) fn section_has_enabled_toggle(section: EditorFieldSpec) -> bool {
    section.name != POLICY_SECTION
        && section
            .schema()
            .and_then(|schema| schema.field("enabled"))
            .is_some_and(|field| field.kind == EditorFieldKind::Boolean)
}

pub(super) fn section_enabled(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
) -> Option<bool> {
    section_value(config, section)
        .ok()
        .flatten()
        .and_then(|section| section.get("enabled").cloned())
        .and_then(|enabled| enabled.as_bool())
}

pub(super) fn section_configured(config: &ObservabilityConfig, section: EditorFieldSpec) -> bool {
    let Ok(Some(value)) = section_value(config, section) else {
        return false;
    };
    if section.optional {
        return true;
    }
    section
        .default_value()
        .as_ref()
        .is_none_or(|default| default != &value)
}

pub(super) fn section_field_configured(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
    field: EditorFieldSpec,
) -> Result<bool, CliError> {
    let Some(value) = section_field_value(config, section, field.name)? else {
        return Ok(false);
    };
    if field.optional {
        return Ok(true);
    }
    Ok(default_field_value(section, field)
        .as_ref()
        .is_none_or(|default| default != &value))
}

pub(super) fn section_field_value(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
    field: &str,
) -> Result<Option<Value>, CliError> {
    Ok(section_value(config, section)?
        .and_then(|section| section.as_object().cloned())
        .and_then(|section| section.get(field).cloned()))
}

pub(super) fn section_value(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
) -> Result<Option<Value>, CliError> {
    let value = serde_json::to_value(config).map_err(serde_error)?;
    Ok(value
        .as_object()
        .and_then(|config| config.get(section.name))
        .filter(|section| !section.is_null())
        .cloned())
}

pub(super) fn set_section_field(
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
    field: &str,
    value: Value,
) -> Result<(), CliError> {
    ensure_section(config, section);
    let mut object = serde_json::to_value(&*config).map_err(serde_error)?;
    let config_object = ensure_object(&mut object);
    let section_object = config_object
        .entry(section.name)
        .or_insert_with(|| section.default_value().unwrap_or_else(|| json!({})));
    ensure_object(section_object).insert(field.to_string(), value);
    *config = serde_json::from_value(object).map_err(serde_error)?;
    Ok(())
}

pub(super) fn remove_section_field(
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
    field: &str,
) -> Result<(), CliError> {
    let mut object = serde_json::to_value(&*config).map_err(serde_error)?;
    if let Some(section_object) = object
        .as_object_mut()
        .and_then(|config| config.get_mut(section.name))
        .and_then(Value::as_object_mut)
    {
        section_object.remove(field);
    }
    *config = serde_json::from_value(object).map_err(serde_error)?;
    Ok(())
}

pub(super) fn set_struct_field<T>(target: &mut T, field: &str, value: Value) -> Result<(), CliError>
where
    T: Serialize + DeserializeOwned,
{
    let mut object = serde_json::to_value(&*target).map_err(serde_error)?;
    ensure_object(&mut object).insert(field.to_string(), value);
    *target = serde_json::from_value(object).map_err(serde_error)?;
    Ok(())
}

pub(super) fn observability_component(config: &PluginConfig) -> Option<&PluginComponentSpec> {
    config
        .components
        .iter()
        .find(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
}

pub(super) fn observability_component_mut(
    config: &mut PluginConfig,
) -> Option<&mut PluginComponentSpec> {
    config
        .components
        .iter_mut()
        .find(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
}

pub(super) fn ensure_object(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = json!({});
    }
    value.as_object_mut().expect("value initialized as object")
}

pub(super) fn observability_config_map(
    config: &ObservabilityConfig,
) -> Result<Map<String, Value>, CliError> {
    let value = serde_json::to_value(config).map_err(serde_error)?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(CliError::Config(
            "observability config must serialize to an object".into(),
        )),
    }
}

pub(super) fn merge_observability_editor_config(
    existing: &mut Map<String, Value>,
    edited: Map<String, Value>,
) {
    merge_known_editor_object(
        existing,
        edited,
        &observability_editor_fields_with_version(),
        ObservabilityConfig::editor_schema(),
    );
}

pub(super) fn merge_known_editor_object(
    existing: &mut Map<String, Value>,
    edited: Map<String, Value>,
    known_keys: &[&str],
    schema: &nemo_flow::config_editor::EditorSchema,
) {
    for key in known_keys {
        let Some(edited_value) = edited.get(*key) else {
            existing.remove(*key);
            continue;
        };
        if let Some(field) = schema.field(key)
            && field.kind == EditorFieldKind::Section
            && let Some(nested_schema) = field.schema()
            && let (Some(existing_object), Some(edited_object)) = (
                existing.get_mut(*key).and_then(Value::as_object_mut),
                edited_value.as_object(),
            )
        {
            merge_known_editor_object(
                existing_object,
                edited_object.clone(),
                &nested_editor_keys(nested_schema),
                nested_schema,
            );
            continue;
        }
        existing.insert((*key).to_string(), edited_value.clone());
    }
}

pub(super) fn observability_editor_fields_with_version() -> Vec<&'static str> {
    let mut keys = vec!["version"];
    keys.extend(
        ObservabilityConfig::editor_schema()
            .fields
            .iter()
            .map(|field| field.name),
    );
    keys
}

pub(super) fn nested_editor_keys(
    schema: &nemo_flow::config_editor::EditorSchema,
) -> Vec<&'static str> {
    schema.fields.iter().map(|field| field.name).collect()
}

pub(super) fn serde_error(error: serde_json::Error) -> CliError {
    CliError::Config(format!("invalid plugin editor value: {error}"))
}

pub(super) fn display_field_value(
    section: EditorFieldSpec,
    field: EditorFieldSpec,
    value: &Value,
) -> String {
    if default_field_value(section, field)
        .as_ref()
        .is_some_and(|default| default == value)
    {
        format!("{} (default)", display_value(value))
    } else {
        display_value(value)
    }
}

pub(super) fn default_field_value(
    section: EditorFieldSpec,
    field: EditorFieldSpec,
) -> Option<Value> {
    section
        .default_value()
        .and_then(|section| section.as_object().cloned())
        .and_then(|section| section.get(field.name).cloned())
}

pub(super) fn display_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "<invalid>".to_string()),
    }
}

pub(super) fn observability_summary(
    config: &PluginConfig,
    observability: &ObservabilityConfig,
) -> String {
    let enabled_sections = ObservabilityConfig::editor_schema()
        .fields
        .iter()
        .filter(|section| section.name != POLICY_SECTION)
        .filter(|section| section_enabled(observability, **section).unwrap_or(false))
        .map(|section| section.label)
        .collect::<Vec<_>>();
    format!(
        "component {}, sections {}",
        if component_enabled(config) {
            "enabled"
        } else {
            "disabled"
        },
        if enabled_sections.is_empty() {
            "none".into()
        } else {
            enabled_sections.join(", ")
        }
    )
}
