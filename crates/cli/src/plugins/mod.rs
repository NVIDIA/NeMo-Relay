// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Plugin configuration state and deterministic editor behavior.
//!
//! Terminal-only interaction lives in `plugins/prompt.rs`.

use std::path::{Path, PathBuf};

use console::{Key, style, truncate_str};
use dialoguer::theme::ColorfulTheme;
use nemo_relay::config_editor::{EditorFieldKind, EditorFieldSpec};
use serde_json::{Value, json};

use crate::error::CliError;

pub(crate) mod config_io;
mod dynamic_editor;
mod editor_model;
pub(crate) mod lifecycle;
pub(crate) mod policy;
pub(crate) mod pricing;
mod prompt;
pub(crate) mod schema;
mod types;

pub(crate) use types::*;

use self::config_io::*;
use self::dynamic_editor::*;
use self::editor_model::*;
use self::prompt::{editor_error, print_editor_help, prompt_menu};

#[cfg(test)]
use self::prompt::menu_error;

const PLUGIN_EDIT_CANCELLED_MESSAGE: &str = "plugin edit cancelled; no plugin changes saved";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuShortcut {
    Preview,
    Save,
    Help,
    Reset,
    Clear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuResponse {
    Selected(usize),
    Shortcut(MenuShortcut, usize),
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditLoopControl {
    Continue,
    Finish,
}

#[derive(Debug)]
struct MenuItem {
    label: String,
}

impl MenuItem {
    fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
        }
    }
}

fn status_label(enabled: bool) -> String {
    if enabled {
        style("on").green().to_string()
    } else {
        style("off").red().to_string()
    }
}

fn shortcut_label(label: impl AsRef<str>, shortcut: &str) -> String {
    format!(
        "{} {}",
        label.as_ref(),
        style(format!("[{shortcut}]")).black().bright()
    )
}

fn configured_label(configured: bool, label: impl AsRef<str>) -> String {
    if configured {
        format!("{} {}", style("✓").green(), label.as_ref())
    } else {
        format!("  {}", label.as_ref())
    }
}

fn print_save_success(path: &Path) {
    println!(
        "  {} Saved {}",
        style("✔").green(),
        single_line_text(&path.display().to_string())
    );
}

pub(crate) fn edit(command: PluginsEditRequest) -> Result<(), CliError> {
    prompt::edit(command)
}

pub(crate) fn resolve_edit_target(
    command: PluginsEditRequest,
) -> Result<(TargetScope, PathBuf), CliError> {
    let scope = target_scope(&command.scope)?;
    let path = match command.explicit_path {
        Some(path) => path,
        None => target_path(scope)?,
    };
    Ok((scope, path))
}

fn preview_document(
    document: &PluginConfigDocument,
    components: &[EditableComponent],
    dynamic_plugins: &[DynamicPluginEditorState],
) -> Result<(), CliError> {
    let mut preview = document.clone();
    preview.set_config(config_with_editable_components(
        document.config(),
        components,
    )?);
    for plugin in dynamic_plugins {
        plugin.apply_to_document(&mut preview, true)?;
    }
    print_document_preview(&preview)
}

fn save_document(
    document: &mut PluginConfigDocument,
    components: &[EditableComponent],
    dynamic_plugins: &[DynamicPluginEditorState],
    scope: TargetScope,
) -> Result<EditLoopControl, CliError> {
    store_editable_components(document.config_mut(), components)?;
    validate_config(document.config())?;
    for plugin in dynamic_plugins {
        plugin.validate()?;
    }
    if scope == TargetScope::Global
        && dynamic_plugins
            .iter()
            .any(DynamicPluginEditorState::has_persisted_secrets)
    {
        return Err(CliError::Config(
            "global plugin configuration cannot contain schema-declared secret values; use a user plugin config".into(),
        ));
    }
    for plugin in dynamic_plugins {
        plugin.apply_to_document(document, false)?;
    }
    document.write_for_scope(scope)?;
    print_save_success(document.path());
    Ok(EditLoopControl::Finish)
}

fn handle_reset_or_clear_shortcut(
    components: &mut [EditableComponent],
    action: Option<MenuAction>,
    shortcut: MenuShortcut,
) -> Result<EditLoopControl, CliError> {
    let _ = (components, action, shortcut);
    println!("  Open a plugin to reset or clear its settings.");
    Ok(EditLoopControl::Continue)
}

fn reset_component_menu_item(
    component: &mut EditableComponent,
    action: Option<ComponentMenuAction>,
) -> Result<(), CliError> {
    match action {
        Some(ComponentMenuAction::Toggle) => component.reset_enabled(),
        Some(ComponentMenuAction::EditField(field_index)) => {
            if let Some(field) = component.fields().get(field_index) {
                component.reset_field(*field)?;
            }
        }
        Some(ComponentMenuAction::Back) | None => {
            println!("  Select a component setting to reset.");
        }
    }
    Ok(())
}

fn clear_component_menu_item(
    component: &mut EditableComponent,
    action: Option<ComponentMenuAction>,
) -> Result<(), CliError> {
    match action {
        Some(ComponentMenuAction::Toggle) => component.set_enabled(false),
        Some(ComponentMenuAction::EditField(field_index)) => {
            if let Some(field) = component.fields().get(field_index)
                && !component.clear_field(*field)?
            {
                println!(
                    "  {} is required; use reset to restore its default.",
                    field.label
                );
            }
        }
        Some(ComponentMenuAction::Back) | None => {
            println!("  Select a component setting to clear.");
        }
    }
    Ok(())
}

fn cancelled_error() -> CliError {
    CliError::Config(PLUGIN_EDIT_CANCELLED_MESSAGE.into())
}

fn menu_response_index(response: &MenuResponse) -> Option<usize> {
    match response {
        MenuResponse::Selected(index)
        | MenuResponse::Shortcut(
            MenuShortcut::Preview
            | MenuShortcut::Save
            | MenuShortcut::Help
            | MenuShortcut::Reset
            | MenuShortcut::Clear,
            index,
        ) => Some(*index),
        MenuResponse::Cancel => None,
    }
}

fn menu_response_for_key(key: &Key, selected: usize) -> Option<MenuResponse> {
    match key {
        Key::Enter | Key::Char(' ') => Some(MenuResponse::Selected(selected)),
        Key::Char('p') => Some(MenuResponse::Shortcut(MenuShortcut::Preview, selected)),
        Key::Char('s') => Some(MenuResponse::Shortcut(MenuShortcut::Save, selected)),
        Key::Char('r') => Some(MenuResponse::Shortcut(MenuShortcut::Reset, selected)),
        Key::Backspace | Key::Del => Some(MenuResponse::Shortcut(MenuShortcut::Clear, selected)),
        Key::Char('?') => Some(MenuResponse::Shortcut(MenuShortcut::Help, selected)),
        Key::Escape | Key::CtrlC | Key::Char('q') => Some(MenuResponse::Cancel),
        _ => None,
    }
}

fn menu_selection_after_key(
    key: &Key,
    selected: usize,
    item_count: usize,
    page_size: usize,
) -> Option<usize> {
    if item_count == 0 {
        return None;
    }

    let last = item_count - 1;
    let selected = selected.min(last);
    match key {
        Key::ArrowUp | Key::Char('k') => Some(if selected == 0 { last } else { selected - 1 }),
        Key::ArrowDown | Key::Char('j') => Some((selected + 1) % item_count),
        Key::PageUp => Some(selected.saturating_sub(page_size)),
        Key::PageDown => Some(selected.saturating_add(page_size).min(last)),
        Key::Home => Some(0),
        Key::End => Some(last),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MenuViewport {
    start: usize,
    end: usize,
    page_size: usize,
    indicators: bool,
}

fn menu_viewport(item_count: usize, selected: usize, terminal_rows: usize) -> MenuViewport {
    let terminal_rows = terminal_rows.max(1);
    let header_rows = menu_header_rows(terminal_rows);
    let available_without_indicators = terminal_rows.saturating_sub(header_rows).max(1);
    if item_count <= available_without_indicators {
        return MenuViewport {
            start: 0,
            end: item_count,
            page_size: available_without_indicators,
            indicators: false,
        };
    }

    let indicator_rows = usize::from(terminal_rows.saturating_sub(header_rows) >= 3) * 2;
    let page_size = terminal_rows
        .saturating_sub(header_rows + indicator_rows)
        .max(1);
    let selected = selected.min(item_count.saturating_sub(1));
    let start = selected
        .saturating_sub(page_size.saturating_sub(1))
        .min(item_count.saturating_sub(page_size));
    MenuViewport {
        start,
        end: (start + page_size).min(item_count),
        page_size,
        indicators: indicator_rows > 0,
    }
}

fn menu_header_rows(terminal_rows: usize) -> usize {
    match terminal_rows {
        0 | 1 => 0,
        2..=4 => 1,
        _ => 2,
    }
}

fn render_menu_for_size(
    theme: &ColorfulTheme,
    prompt: &str,
    items: &[MenuItem],
    selected: usize,
    terminal_rows: usize,
    terminal_columns: usize,
) -> Vec<String> {
    let terminal_rows = terminal_rows.max(1);
    let viewport = menu_viewport(items.len(), selected, terminal_rows);
    let mut lines = Vec::with_capacity(viewport.page_size + 4);
    let header_rows = menu_header_rows(terminal_rows);
    if header_rows >= 1 {
        lines.push(format!(
            "{} {} {}",
            theme.prompt_prefix,
            theme.prompt_style.apply_to(single_line_text(prompt)),
            theme.prompt_suffix
        ));
    }
    if header_rows >= 2 {
        lines.push(
            theme
                .hint_style
                .apply_to("  ↑/↓ or j/k move, PgUp/PgDn page, Home/End jump, Enter/Space select, p preview, s save, r reset, Backspace/Delete clear, ? help, q cancel.")
                .to_string(),
        );
    }
    if viewport.indicators {
        lines.push(if viewport.start > 0 {
            theme
                .hint_style
                .apply_to(format!("  ↑ {} more", viewport.start))
                .to_string()
        } else {
            String::new()
        });
    }
    lines.extend(
        items[viewport.start..viewport.end]
            .iter()
            .enumerate()
            .map(|(offset, item)| {
                let index = viewport.start + offset;
                let label = single_line_text(&item.label);
                if index == selected {
                    format!(
                        "{} {}",
                        theme.active_item_prefix,
                        theme.active_item_style.apply_to(label)
                    )
                } else {
                    format!(
                        "{} {}",
                        theme.inactive_item_prefix,
                        theme.inactive_item_style.apply_to(label)
                    )
                }
            }),
    );
    if viewport.indicators {
        lines.push(if viewport.end < items.len() {
            theme
                .hint_style
                .apply_to(format!("  ↓ {} more", items.len() - viewport.end))
                .to_string()
        } else {
            String::new()
        });
    }
    let width = terminal_columns.max(1);
    lines
        .into_iter()
        .map(|line| truncate_str(&line, width, "…").into_owned())
        .collect()
}

fn single_line_text(value: &str) -> String {
    console::strip_ansi_codes(value)
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect()
}

fn section_menu_items<T>(
    config: &T,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
) -> Result<Vec<MenuItem>, CliError>
where
    T: serde::Serialize,
{
    let mut items = Vec::new();
    if section_has_enabled_toggle(section) {
        let enabled = section_enabled(config, section).unwrap_or(false);
        items.push(MenuItem::new(format!(
            "Toggle section [{}]",
            status_label(enabled)
        )));
    }
    for field in fields {
        items.push(section_field_menu_item(config, section, *field)?);
    }
    items.push(MenuItem::new(shortcut_label("Reset section", "r")));
    items.push(MenuItem::new(shortcut_label("Back", "q")));
    Ok(items)
}

fn section_field_menu_item<T>(
    config: &T,
    section: EditorFieldSpec,
    field: EditorFieldSpec,
) -> Result<MenuItem, CliError>
where
    T: serde::Serialize,
{
    let configured = section_field_configured(config, section, field)?;
    let value = section_field_value(config, section, field.name)?
        .map(|value| display_field_value(section, field, &value))
        .or_else(|| {
            default_field_value(section, field)
                .map(|value| format!("{} (default)", display_value(&value)))
        })
        .unwrap_or_else(|| "(default)".to_string());
    Ok(MenuItem::new(format!(
        "{} = {}",
        configured_label(configured, field.name),
        value
    )))
}

fn selected_field_index(section: EditorFieldSpec, selected: usize) -> usize {
    selected - usize::from(section_has_enabled_toggle(section))
}

fn reset_section_index(section: EditorFieldSpec, fields: &[EditorFieldSpec]) -> usize {
    usize::from(section_has_enabled_toggle(section)) + fields.len()
}

fn reset_selected_item<T>(
    config: &mut T,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
    selected: usize,
) -> Result<(), CliError>
where
    T: SerializeConfig,
{
    if reset_selected_field(config, section, fields, selected)? {
        return Ok(());
    }
    if selected == reset_section_index(section, fields) {
        reset_section(config, section);
    }
    Ok(())
}

fn string_map_entry_exists(value: &Value, key: &str) -> bool {
    value
        .as_object()
        .is_some_and(|entries| entries.contains_key(key.trim()))
}

fn editor_item_label(
    value: &Value,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> String {
    if let Some(tagged_union) = item.tagged_union {
        return value
            .get(tagged_union.discriminator)
            .and_then(Value::as_str)
            .unwrap_or("invalid")
            .to_string();
    }
    display_value(value)
}

fn tagged_union_variant_value(
    tagged_union: &nemo_relay::config_editor::EditorTaggedUnionSpec,
    selected: usize,
) -> Result<Value, CliError> {
    tagged_union
        .variants
        .get(selected)
        .map(|variant| (variant.default)())
        .ok_or_else(|| CliError::Config("tagged union variant does not exist".into()))
}

#[derive(Debug, PartialEq)]
enum TaggedUnionFieldEdit {
    Set(Value),
    Reset,
    Unchanged,
}

struct TaggedUnionFieldState {
    baseline: Value,
    value: Value,
}

impl TaggedUnionFieldState {
    fn new(current: Option<Value>, default: Option<Value>) -> Self {
        let baseline = current.or(default).unwrap_or(Value::Null);
        Self {
            value: baseline.clone(),
            baseline,
        }
    }

    fn value(&self) -> &Value {
        &self.value
    }

    fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }

    fn change_variant(
        &mut self,
        tagged_union: &nemo_relay::config_editor::EditorTaggedUnionSpec,
        selected: usize,
    ) -> Result<(), CliError> {
        self.value = tagged_union_variant_value(tagged_union, selected)?;
        Ok(())
    }

    fn reset(self) -> TaggedUnionFieldEdit {
        TaggedUnionFieldEdit::Reset
    }

    fn finish(self) -> TaggedUnionFieldEdit {
        if self.value == self.baseline {
            TaggedUnionFieldEdit::Unchanged
        } else {
            TaggedUnionFieldEdit::Set(self.value)
        }
    }
}

fn collection_shortcut_value(
    default: Option<&Value>,
    empty: Value,
    shortcut: MenuShortcut,
) -> Value {
    match shortcut {
        MenuShortcut::Reset => default
            .filter(|value| {
                value.is_array() == empty.is_array() && value.is_object() == empty.is_object()
            })
            .cloned()
            .unwrap_or(empty),
        MenuShortcut::Clear => empty,
        _ => unreachable!("only reset and clear shortcuts are collection shortcuts"),
    }
}

fn section_field_default(section: EditorFieldSpec, field: EditorFieldSpec) -> Option<Value> {
    default_field_value(section, field).or_else(|| field.default_value())
}

fn store_edited_config_section<T>(
    config: &mut T,
    field: EditorFieldSpec,
    value: Value,
) -> Result<(), CliError>
where
    T: SerializeConfig,
{
    if should_clear_empty_section(field, &value) {
        remove_struct_field(config, field.name)
    } else {
        set_struct_field(config, field.name, value)
    }
}

fn store_edited_section_field<T>(
    config: &mut T,
    section: EditorFieldSpec,
    field: EditorFieldSpec,
    value: Value,
) -> Result<(), CliError>
where
    T: SerializeConfig,
{
    if should_clear_empty_section(field, &value) {
        remove_section_field(config, section, field.name)
    } else {
        set_section_field(config, section, field.name, value)
    }
}

fn value_section_menu_items(
    value: &Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    default: Option<&Value>,
) -> Result<Vec<MenuItem>, CliError> {
    let mut items = schema
        .fields
        .iter()
        .map(|field| value_field_menu_item(value, *field, default))
        .collect::<Result<Vec<_>, _>>()?;
    items.push(MenuItem::new(shortcut_label("Reset section", "r")));
    items.push(MenuItem::new(shortcut_label("Back", "q")));
    Ok(items)
}

fn value_field_menu_item(
    value: &Value,
    field: EditorFieldSpec,
    default: Option<&Value>,
) -> Result<MenuItem, CliError> {
    let configured = value_field_configured(value, field, default);
    let rendered = value_field_value(value, field.name)
        .map(|value| display_value_with_default(&value, value_field_default(default, field)))
        .or_else(|| {
            value_field_default(default, field)
                .map(|value| format!("{} (default)", display_value(&value)))
        })
        .unwrap_or_else(|| "(default)".to_string());
    Ok(MenuItem::new(format!(
        "{} = {}",
        configured_label(configured, field.name),
        rendered
    )))
}

fn reset_value_section_item(
    value: &mut Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    default: Option<&Value>,
    selected: usize,
) {
    if let Some(field) = schema.fields.get(selected) {
        reset_value_field(value, *field, default);
    } else if selected == schema.fields.len() {
        *value = default.cloned().unwrap_or_else(|| json!({}));
        ensure_object(value);
    }
}

fn clear_value_field(
    value: &mut Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    selected: usize,
) -> bool {
    let Some(field) = schema.fields.get(selected) else {
        return false;
    };
    if !field.optional {
        return false;
    }
    remove_value_field(value, field.name);
    true
}

fn value_field_configured(value: &Value, field: EditorFieldSpec, default: Option<&Value>) -> bool {
    let Some(current) = value_field_value(value, field.name) else {
        return false;
    };
    if field.optional {
        return true;
    }
    value_field_default(default, field)
        .as_ref()
        .is_none_or(|default| default != &current)
}

fn value_field_value(value: &Value, field: &str) -> Option<Value> {
    value
        .as_object()
        .and_then(|object| object.get(field))
        .filter(|value| !value.is_null())
        .cloned()
}

fn default_object_field_value(default: Option<&Value>, field: EditorFieldSpec) -> Option<Value> {
    default
        .and_then(Value::as_object)
        .and_then(|object| object.get(field.name))
        .filter(|value| !value.is_null())
        .cloned()
}

fn value_field_default(default: Option<&Value>, field: EditorFieldSpec) -> Option<Value> {
    default_object_field_value(default, field).or_else(|| field.default_value())
}

fn set_value_field(target: &mut Value, field: &str, field_value: Value) {
    ensure_object(target).insert(field.to_string(), field_value);
}

fn store_edited_value_section(target: &mut Value, field: EditorFieldSpec, field_value: Value) {
    if should_clear_empty_section(field, &field_value) {
        remove_value_field(target, field.name);
    } else {
        set_value_field(target, field.name, field_value);
    }
}

fn remove_value_field(target: &mut Value, field: &str) {
    if let Some(object) = target.as_object_mut() {
        object.remove(field);
    }
}

fn reset_value_field(value: &mut Value, field: EditorFieldSpec, default: Option<&Value>) {
    if let Some(default) = value_field_default(default, field) {
        set_value_field(value, field.name, default);
    } else {
        remove_value_field(value, field.name);
    }
}

fn display_value_with_default(value: &Value, default: Option<Value>) -> String {
    if default.as_ref().is_some_and(|default| default == value) {
        format!("{} (default)", display_value(value))
    } else {
        display_value(value)
    }
}

trait SerializeConfig: serde::Serialize + serde::de::DeserializeOwned {}

impl<T> SerializeConfig for T where T: serde::Serialize + serde::de::DeserializeOwned {}

fn parse_float_value(field: &EditorFieldSpec, value: &str) -> Result<Value, CliError> {
    let value = value.trim();
    let parsed = value
        .parse::<f64>()
        .map_err(|error| CliError::Config(format!("{} must be a number: {error}", field.name)))?;
    if !parsed.is_finite() {
        return Err(CliError::Config(format!(
            "{} must be a finite number: {value}",
            field.name
        )));
    }
    Ok(json!(parsed))
}

fn editor_enum_default_index(field: &EditorFieldSpec, current: Option<&Value>) -> usize {
    current
        .map(display_value)
        .and_then(|value| {
            field
                .enum_values
                .iter()
                .position(|candidate| *candidate == value)
        })
        .unwrap_or(0)
}

fn editor_enum_value(field: &EditorFieldSpec, selected: usize) -> Value {
    let value = field.enum_values[selected];
    match field.kind {
        EditorFieldKind::IntegerEnum => json!(
            value
                .parse::<i64>()
                .expect("integer editor enum values must be valid i64 values")
        ),
        EditorFieldKind::Enum => json!(value),
        _ => unreachable!("editor enum value requested for a non-enum field"),
    }
}

#[cfg(test)]
#[path = "../../tests/coverage/shared/plugins_tests.rs"]
mod tests;
