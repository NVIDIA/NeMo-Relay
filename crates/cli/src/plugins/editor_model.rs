// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Testable plugin editor state helpers.

use std::path::Path;

use nemo_relay::config_editor::{EditorConfig, EditorFieldKind, EditorFieldSpec};
use nemo_relay::observability::plugin_component::{OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig};
use nemo_relay::plugin::{PluginComponentSpec, PluginConfig};
use nemo_relay::plugins::nemo_guardrails::component::{
    NEMO_GUARDRAILS_PLUGIN_KIND, NeMoGuardrailsConfig,
};
use nemo_relay_adaptive::AdaptiveConfig;
use nemo_relay_adaptive::plugin_component::ADAPTIVE_PLUGIN_KIND;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::error::CliError;

pub(super) const POLICY_SECTION: &str = "policy";

#[derive(Debug, Clone)]
pub(super) struct ComponentEditorState<T> {
    pub(super) config: T,
    pub(super) enabled: bool,
    default_enabled: bool,
    existing: bool,
    touched: bool,
    config_touched: bool,
}

#[derive(Debug)]
pub(super) enum EditableComponent {
    Observability(Box<ComponentEditorState<ObservabilityConfig>>),
    Adaptive(Box<ComponentEditorState<AdaptiveConfig>>),
    NemoGuardrails(Box<ComponentEditorState<NeMoGuardrailsConfig>>),
}

impl EditableComponent {
    pub(super) fn label(&self) -> &'static str {
        match self {
            Self::Observability(_) => "Observability",
            Self::Adaptive(_) => "Adaptive",
            Self::NemoGuardrails(_) => "NeMo Guardrails",
        }
    }

    pub(super) fn fields(&self) -> &'static [EditorFieldSpec] {
        match self {
            Self::Observability(_) => ObservabilityConfig::editor_schema().fields,
            Self::Adaptive(_) => AdaptiveConfig::editor_schema().fields,
            Self::NemoGuardrails(_) => NeMoGuardrailsConfig::editor_schema().fields,
        }
    }

    pub(super) fn enabled(&self) -> bool {
        match self {
            Self::Observability(state) => state.enabled,
            Self::Adaptive(state) => state.enabled,
            Self::NemoGuardrails(state) => state.enabled,
        }
    }

    pub(super) fn toggle_enabled(&mut self) {
        match self {
            Self::Observability(state) => state.toggle_enabled(),
            Self::Adaptive(state) => state.toggle_enabled(),
            Self::NemoGuardrails(state) => state.toggle_enabled(),
        }
    }

    pub(super) fn set_enabled(&mut self, enabled: bool) {
        match self {
            Self::Observability(state) => state.set_enabled(enabled),
            Self::Adaptive(state) => state.set_enabled(enabled),
            Self::NemoGuardrails(state) => state.set_enabled(enabled),
        }
    }

    pub(super) fn reset_enabled(&mut self) {
        match self {
            Self::Observability(state) => state.reset_enabled(),
            Self::Adaptive(state) => state.reset_enabled(),
            Self::NemoGuardrails(state) => state.reset_enabled(),
        }
    }

    pub(super) fn summary(&self) -> String {
        match self {
            Self::Observability(state) => observability_summary(state),
            Self::Adaptive(state) => adaptive_summary(state),
            Self::NemoGuardrails(state) => nemo_guardrails_summary(state),
        }
    }

    pub(super) fn field_configured(&self, field: EditorFieldSpec) -> bool {
        match self {
            Self::Observability(state) => section_configured(&state.config, field),
            Self::Adaptive(state) => config_field_configured(&state.config, field).unwrap_or(false),
            Self::NemoGuardrails(state) => {
                config_field_configured(&state.config, field).unwrap_or(false)
            }
        }
    }

    pub(super) fn reset_field(&mut self, field: EditorFieldSpec) -> Result<(), CliError> {
        match self {
            Self::Observability(state) => {
                reset_config_field(&mut state.config, field)?;
                state.mark_config_touched();
            }
            Self::Adaptive(state) => {
                reset_config_field(&mut state.config, field)?;
                state.mark_config_touched();
            }
            Self::NemoGuardrails(state) => {
                reset_config_field(&mut state.config, field)?;
                state.mark_config_touched();
            }
        }
        Ok(())
    }

    pub(super) fn store(&self, config: &mut PluginConfig) -> Result<(), CliError> {
        match self {
            Self::Observability(state) => store_observability_state(config, state),
            Self::Adaptive(state) => store_adaptive_state(config, state),
            Self::NemoGuardrails(state) => store_nemo_guardrails_state(config, state),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum MenuAction {
    ToggleComponent(usize),
    EditField {
        component_index: usize,
        field_index: usize,
    },
    Preview,
    Save,
    Cancel,
}

pub(super) fn editable_components(
    config: &PluginConfig,
) -> Result<Vec<EditableComponent>, CliError> {
    Ok(vec![
        EditableComponent::Observability(Box::new(component_observability_state(config)?)),
        EditableComponent::Adaptive(Box::new(component_adaptive_state(config)?)),
        EditableComponent::NemoGuardrails(Box::new(component_nemo_guardrails_state(config)?)),
    ])
}

pub(super) fn plugin_menu_items(
    components: &[EditableComponent],
    path: &Path,
) -> (Vec<super::MenuItem>, Vec<MenuAction>) {
    let mut items = Vec::new();
    let mut actions = Vec::new();
    for (component_index, component) in components.iter().enumerate() {
        items.push(super::MenuItem::new(format!(
            "Toggle {} component [{}]",
            component.label(),
            super::status_label(component.enabled())
        )));
        actions.push(MenuAction::ToggleComponent(component_index));

        items.extend(component.fields().iter().map(|field| {
            super::MenuItem::new(super::configured_label(
                component.field_configured(*field),
                format!("Edit {} {}", component.label(), field.label),
            ))
        }));
        actions.extend(
            component
                .fields()
                .iter()
                .enumerate()
                .map(|(field_index, _)| MenuAction::EditField {
                    component_index,
                    field_index,
                }),
        );
    }

    items.push(super::MenuItem::new(super::shortcut_label(
        "Preview TOML",
        "p",
    )));
    actions.push(MenuAction::Preview);
    items.push(super::MenuItem::new(super::shortcut_label(
        format!("Save to {}", path.display()),
        "s",
    )));
    actions.push(MenuAction::Save);
    items.push(super::MenuItem::new(super::shortcut_label("Cancel", "q")));
    actions.push(MenuAction::Cancel);

    (items, actions)
}

pub(super) fn config_with_editable_components(
    config: &PluginConfig,
    components: &[EditableComponent],
) -> Result<PluginConfig, CliError> {
    let mut config = config.clone();
    store_editable_components(&mut config, components)?;
    Ok(config)
}

pub(super) fn store_editable_components(
    config: &mut PluginConfig,
    components: &[EditableComponent],
) -> Result<(), CliError> {
    for component in components {
        component.store(config)?;
    }
    Ok(())
}

impl<T> ComponentEditorState<T> {
    pub(super) fn mark_touched(&mut self) {
        self.touched = true;
    }

    pub(super) fn mark_config_touched(&mut self) {
        self.config_touched = true;
        self.mark_touched();
    }

    pub(super) fn toggle_enabled(&mut self) {
        self.enabled = !self.enabled;
        self.mark_touched();
    }

    pub(super) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.mark_touched();
    }

    pub(super) fn reset_enabled(&mut self) {
        self.enabled = self.default_enabled;
        self.mark_touched();
    }

    pub(super) fn should_store(&self, configured: bool) -> bool {
        self.existing || (self.touched && (self.enabled || configured))
    }
}

fn component_editor_state<T>(
    config: &PluginConfig,
    kind: &str,
    default_enabled: bool,
) -> Result<ComponentEditorState<T>, CliError>
where
    T: Default + DeserializeOwned,
{
    if let Some(component) = config
        .components
        .iter()
        .find(|component| component.kind == kind)
    {
        let config = serde_json::from_value(Value::Object(component.config.clone()))
            .map_err(|error| CliError::Config(format!("invalid {kind} plugin config: {error}")))?;
        return Ok(ComponentEditorState {
            config,
            enabled: component.enabled,
            default_enabled,
            existing: true,
            touched: false,
            config_touched: false,
        });
    }

    Ok(ComponentEditorState {
        config: T::default(),
        enabled: default_enabled,
        default_enabled,
        existing: false,
        touched: false,
        config_touched: false,
    })
}

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

pub(super) fn ensure_adaptive_component(config: &mut PluginConfig) -> Result<(), CliError> {
    if !config
        .components
        .iter()
        .any(|component| component.kind == ADAPTIVE_PLUGIN_KIND)
    {
        config.components.push(PluginComponentSpec {
            kind: ADAPTIVE_PLUGIN_KIND.to_string(),
            enabled: false,
            config: adaptive_config_map(&AdaptiveConfig::default())?,
        });
    }
    Ok(())
}

pub(super) fn component_observability_state(
    config: &PluginConfig,
) -> Result<ComponentEditorState<ObservabilityConfig>, CliError> {
    component_editor_state(config, OBSERVABILITY_PLUGIN_KIND, true)
}

pub(super) fn component_adaptive_state(
    config: &PluginConfig,
) -> Result<ComponentEditorState<AdaptiveConfig>, CliError> {
    component_editor_state(config, ADAPTIVE_PLUGIN_KIND, false)
}

pub(super) fn component_nemo_guardrails_state(
    config: &PluginConfig,
) -> Result<ComponentEditorState<NeMoGuardrailsConfig>, CliError> {
    component_editor_state(config, NEMO_GUARDRAILS_PLUGIN_KIND, false)
}

pub(super) fn store_observability_state(
    config: &mut PluginConfig,
    state: &ComponentEditorState<ObservabilityConfig>,
) -> Result<(), CliError> {
    if state.should_store(true) {
        store_component_editor_config(
            config,
            OBSERVABILITY_PLUGIN_KIND,
            state.enabled,
            observability_config_map(&state.config)?,
            merge_observability_editor_config,
        );
    }
    Ok(())
}

pub(super) fn store_adaptive_state(
    config: &mut PluginConfig,
    state: &ComponentEditorState<AdaptiveConfig>,
) -> Result<(), CliError> {
    if state.should_store(true) {
        store_component_editor_config(
            config,
            ADAPTIVE_PLUGIN_KIND,
            state.enabled,
            adaptive_config_map(&state.config)?,
            merge_adaptive_editor_config,
        );
    }
    Ok(())
}

pub(super) fn store_nemo_guardrails_state(
    config: &mut PluginConfig,
    state: &ComponentEditorState<NeMoGuardrailsConfig>,
) -> Result<(), CliError> {
    if state.should_store(state.config_touched || nemo_guardrails_configured(&state.config)) {
        store_component_editor_config(
            config,
            NEMO_GUARDRAILS_PLUGIN_KIND,
            state.enabled,
            nemo_guardrails_config_map(&state.config)?,
            merge_nemo_guardrails_editor_config,
        );
    }
    Ok(())
}

fn store_component_editor_config(
    config: &mut PluginConfig,
    kind: &str,
    enabled: bool,
    edited: Map<String, Value>,
    merge: fn(&mut Map<String, Value>, Map<String, Value>),
) {
    if let Some(component) = config
        .components
        .iter_mut()
        .find(|component| component.kind == kind)
    {
        component.enabled = enabled;
        merge(&mut component.config, edited);
    } else {
        config.components.push(PluginComponentSpec {
            kind: kind.to_string(),
            enabled,
            config: edited,
        });
    }
}

pub(super) fn ensure_section<T>(config: &mut T, section: EditorFieldSpec)
where
    T: Serialize + DeserializeOwned,
{
    if let Ok(Some(Value::Object(_))) = section_value(config, section) {
        return;
    }
    let Some(default) = section.default_value() else {
        return;
    };
    let _ = set_struct_field(config, section.name, default);
}

pub(super) fn toggle_section<T>(config: &mut T, section: EditorFieldSpec)
where
    T: Serialize + DeserializeOwned,
{
    ensure_section(config, section);
    let enabled = section_enabled(config, section).unwrap_or(false);
    let _ = set_section_field(config, section, "enabled", json!(!enabled));
}

pub(super) fn reset_section<T>(config: &mut T, section: EditorFieldSpec)
where
    T: Serialize + DeserializeOwned,
{
    if let Some(value) = section.default_value() {
        let _ = set_struct_field(config, section.name, value);
    } else if section.optional {
        let _ = remove_struct_field(config, section.name);
    } else {
        let _ = set_struct_field(config, section.name, json!({}));
    }
}

pub(super) fn should_clear_empty_section(field: EditorFieldSpec, value: &Value) -> bool {
    field.kind == EditorFieldKind::Section
        && field.optional
        && field.default_value().is_none()
        && value.as_object().is_some_and(|object| object.is_empty())
}

pub(super) fn reset_selected_field<T>(
    config: &mut T,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
    selected: usize,
) -> Result<bool, CliError>
where
    T: Serialize + DeserializeOwned,
{
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

pub(super) fn section_enabled<T>(config: &T, section: EditorFieldSpec) -> Option<bool>
where
    T: Serialize,
{
    section_value(config, section)
        .ok()
        .flatten()
        .and_then(|section| section.get("enabled").cloned())
        .and_then(|enabled| enabled.as_bool())
}

pub(super) fn section_configured<T>(config: &T, section: EditorFieldSpec) -> bool
where
    T: Serialize,
{
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

pub(super) fn section_field_configured<T>(
    config: &T,
    section: EditorFieldSpec,
    field: EditorFieldSpec,
) -> Result<bool, CliError>
where
    T: Serialize,
{
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

pub(super) fn section_field_value<T>(
    config: &T,
    section: EditorFieldSpec,
    field: &str,
) -> Result<Option<Value>, CliError>
where
    T: Serialize,
{
    Ok(section_value(config, section)?
        .and_then(|section| section.as_object().cloned())
        .and_then(|section| section.get(field).cloned()))
}

pub(super) fn section_value<T>(
    config: &T,
    section: EditorFieldSpec,
) -> Result<Option<Value>, CliError>
where
    T: Serialize,
{
    let value = serde_json::to_value(config).map_err(serde_error)?;
    Ok(value
        .as_object()
        .and_then(|config| config.get(section.name))
        .filter(|section| !section.is_null())
        .cloned())
}

pub(super) fn set_section_field<T>(
    config: &mut T,
    section: EditorFieldSpec,
    field: &str,
    value: Value,
) -> Result<(), CliError>
where
    T: Serialize + DeserializeOwned,
{
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

pub(super) fn remove_section_field<T>(
    config: &mut T,
    section: EditorFieldSpec,
    field: &str,
) -> Result<(), CliError>
where
    T: Serialize + DeserializeOwned,
{
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

pub(super) fn remove_struct_field<T>(target: &mut T, field: &str) -> Result<(), CliError>
where
    T: Serialize + DeserializeOwned,
{
    let mut object = serde_json::to_value(&*target).map_err(serde_error)?;
    if let Some(object) = object.as_object_mut() {
        object.remove(field);
    }
    *target = serde_json::from_value(object).map_err(serde_error)?;
    Ok(())
}

pub(super) fn config_field_value<T>(config: &T, field: &str) -> Result<Option<Value>, CliError>
where
    T: Serialize,
{
    let value = serde_json::to_value(config).map_err(serde_error)?;
    Ok(value
        .as_object()
        .and_then(|config| config.get(field))
        .filter(|value| !value.is_null())
        .cloned())
}

pub(super) fn config_field_configured<T>(
    config: &T,
    field: EditorFieldSpec,
) -> Result<bool, CliError>
where
    T: Default + Serialize,
{
    let Some(value) = config_field_value(config, field.name)? else {
        return Ok(false);
    };
    if field.optional {
        return Ok(true);
    }
    Ok(default_config_field_value::<T>(field)
        .as_ref()
        .is_none_or(|default| default != &value))
}

pub(super) fn reset_config_field<T>(config: &mut T, field: EditorFieldSpec) -> Result<(), CliError>
where
    T: Default + Serialize + DeserializeOwned,
{
    if let Some(default) = default_config_field_value::<T>(field) {
        set_struct_field(config, field.name, default)
    } else {
        remove_struct_field(config, field.name)
    }
}

pub(super) fn default_config_field_value<T>(field: EditorFieldSpec) -> Option<Value>
where
    T: Default + Serialize,
{
    serde_json::to_value(T::default())
        .ok()
        .and_then(|value| value.as_object().cloned())
        .and_then(|config| config.get(field.name).cloned())
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

pub(super) fn adaptive_config_map(config: &AdaptiveConfig) -> Result<Map<String, Value>, CliError> {
    let value = serde_json::to_value(config).map_err(serde_error)?;
    match value {
        Value::Object(mut map) => {
            if is_version_one(map.get("version")) {
                map.remove("version");
            }
            Ok(map)
        }
        _ => Err(CliError::Config(
            "adaptive config must serialize to an object".into(),
        )),
    }
}

pub(super) fn nemo_guardrails_config_map(
    config: &NeMoGuardrailsConfig,
) -> Result<Map<String, Value>, CliError> {
    let value = serde_json::to_value(config).map_err(serde_error)?;
    match value {
        Value::Object(mut map) => {
            if is_version_one(map.get("version")) {
                map.remove("version");
            }
            Ok(map)
        }
        _ => Err(CliError::Config(
            "nemo_guardrails config must serialize to an object".into(),
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

pub(super) fn merge_adaptive_editor_config(
    existing: &mut Map<String, Value>,
    edited: Map<String, Value>,
) {
    if is_version_one(existing.get("version")) {
        existing.remove("version");
    }
    merge_known_editor_object(
        existing,
        edited,
        &nested_editor_keys(AdaptiveConfig::editor_schema()),
        AdaptiveConfig::editor_schema(),
    );
}

pub(super) fn merge_nemo_guardrails_editor_config(
    existing: &mut Map<String, Value>,
    edited: Map<String, Value>,
) {
    if is_version_one(existing.get("version")) {
        existing.remove("version");
    }
    merge_known_editor_object(
        existing,
        edited,
        &nested_editor_keys(NeMoGuardrailsConfig::editor_schema()),
        NeMoGuardrailsConfig::editor_schema(),
    );
}

fn is_version_one(value: Option<&Value>) -> bool {
    value.and_then(Value::as_u64) == Some(1)
}

pub(super) fn merge_known_editor_object(
    existing: &mut Map<String, Value>,
    edited: Map<String, Value>,
    known_keys: &[&str],
    schema: &nemo_relay::config_editor::EditorSchema,
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
    schema: &nemo_relay::config_editor::EditorSchema,
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

pub(super) fn observability_summary(state: &ComponentEditorState<ObservabilityConfig>) -> String {
    let enabled_sections = ObservabilityConfig::editor_schema()
        .fields
        .iter()
        .filter(|section| section.name != POLICY_SECTION)
        .filter(|section| section_enabled(&state.config, **section).unwrap_or(false))
        .map(|section| section.label)
        .collect::<Vec<_>>();
    format!(
        "component {}, sections {}",
        if state.enabled { "enabled" } else { "disabled" },
        if enabled_sections.is_empty() {
            "none".into()
        } else {
            enabled_sections.join(", ")
        }
    )
}

pub(super) fn adaptive_summary(state: &ComponentEditorState<AdaptiveConfig>) -> String {
    let configured_fields = AdaptiveConfig::editor_schema()
        .fields
        .iter()
        .filter(|field| field.name != POLICY_SECTION)
        .filter(|field| config_field_configured(&state.config, **field).unwrap_or(false))
        .map(|field| field.label)
        .collect::<Vec<_>>();
    format!(
        "component {}, fields {}",
        if state.enabled { "enabled" } else { "disabled" },
        if configured_fields.is_empty() {
            "none".into()
        } else {
            configured_fields.join(", ")
        }
    )
}

pub(super) fn nemo_guardrails_configured(config: &NeMoGuardrailsConfig) -> bool {
    NeMoGuardrailsConfig::editor_schema()
        .fields
        .iter()
        .filter(|field| field.name != POLICY_SECTION)
        .any(|field| config_field_configured(config, *field).unwrap_or(false))
}

pub(super) fn nemo_guardrails_summary(
    state: &ComponentEditorState<NeMoGuardrailsConfig>,
) -> String {
    let configured_fields = NeMoGuardrailsConfig::editor_schema()
        .fields
        .iter()
        .filter(|field| field.name != POLICY_SECTION)
        .filter(|field| config_field_configured(&state.config, **field).unwrap_or(false))
        .map(|field| field.label)
        .collect::<Vec<_>>();
    format!(
        "component {}, fields {}",
        if state.enabled { "enabled" } else { "disabled" },
        if configured_fields.is_empty() {
            "none".into()
        } else {
            configured_fields.join(", ")
        }
    )
}
