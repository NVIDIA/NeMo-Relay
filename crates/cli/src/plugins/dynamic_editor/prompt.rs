// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Terminal-only prompt adapter for dynamic plugin configuration.

use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Password, Select};
use serde_json::{Map, Number, Value};

use super::*;
use crate::error::CliError;
use crate::plugins::{print_editor_help, single_line_text};

pub(super) fn edit_dynamic_plugin(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
) -> Result<(), CliError> {
    if let Some(description) = &state.description {
        println!("  {}", single_line_text(description));
    }
    let fields = state
        .schema
        .as_ref()
        .map(|schema| schema.fields().to_vec())
        .unwrap_or_default();
    if state.schema.is_none() || fields.is_empty() {
        edit_dynamic_root_menu(theme, state, &fields)
    } else {
        let prompt = state
            .editor_title
            .clone()
            .unwrap_or_else(|| state.label.clone());
        edit_dynamic_fields_menu(theme, state, &fields, &[], prompt)
    }
}

fn edit_dynamic_root_menu(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
    fields: &[DynamicConfigField],
) -> Result<(), CliError> {
    let mut selected_index = 0;
    loop {
        let (items, actions) = dynamic_root_menu_items(state, fields);

        let selection = prompt_menu(theme, state.label(), &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        if handle_dynamic_root_menu_response(theme, state, &actions, selection)? {
            return Ok(());
        }
    }
}

fn handle_dynamic_root_menu_response(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
    actions: &[DynamicMenuAction],
    selection: MenuResponse,
) -> Result<bool, CliError> {
    match selection {
        MenuResponse::Selected(selected) => match actions.get(selected).copied() {
            Some(DynamicMenuAction::EditRawConfig) => {
                prompt_raw_config(theme, state)?;
                Ok(false)
            }
            Some(DynamicMenuAction::ResetPlugin) => {
                state.reset();
                Ok(false)
            }
            Some(DynamicMenuAction::Back) | None => Ok(true),
            Some(DynamicMenuAction::EditField(_)) => {
                println!("  Select Edit raw configuration to modify settings.");
                Ok(false)
            }
        },
        MenuResponse::Shortcut(MenuShortcut::Reset, selected) => {
            if matches!(actions.get(selected), Some(DynamicMenuAction::ResetPlugin)) {
                state.reset();
            } else {
                println!("  Select Reset plugin configuration to remove config.");
            }
            Ok(false)
        }
        MenuResponse::Shortcut(MenuShortcut::Clear, selected) => {
            if matches!(
                actions.get(selected),
                Some(DynamicMenuAction::EditRawConfig)
            ) {
                state.set_raw_config(Map::new());
            }
            Ok(false)
        }
        MenuResponse::Shortcut(MenuShortcut::Help, _) => {
            print_editor_help();
            Ok(false)
        }
        MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
            println!("  Preview and save are available from the main plugins.toml menu.");
            Ok(false)
        }
        MenuResponse::Cancel => Ok(true),
    }
}

fn edit_dynamic_fields_menu(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
    fields: &[DynamicConfigField],
    parent_path: &[String],
    prompt: String,
) -> Result<(), CliError> {
    let mut selected_index = 0;
    loop {
        let (items, actions) = dynamic_field_menu_items(state, fields, parent_path);
        let selection = prompt_menu(theme, &prompt, &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        match selection {
            MenuResponse::Selected(selected) => match actions.get(selected).copied() {
                Some(DynamicMenuAction::EditField(index)) => {
                    edit_dynamic_field(theme, state, &fields[index], parent_path)?;
                }
                Some(DynamicMenuAction::ResetPlugin) => state.reset(),
                Some(DynamicMenuAction::Back) | None => return Ok(()),
                Some(DynamicMenuAction::EditRawConfig) => unreachable!(),
            },
            MenuResponse::Shortcut(MenuShortcut::Reset, selected) => {
                reset_dynamic_selection(state, fields, parent_path, &actions, selected);
            }
            MenuResponse::Shortcut(MenuShortcut::Clear, selected) => {
                clear_dynamic_selection(state, fields, parent_path, &actions, selected);
            }
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
            }
            MenuResponse::Cancel => return Ok(()),
        }
    }
}

fn edit_dynamic_field(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
    field: &DynamicConfigField,
    parent_path: &[String],
) -> Result<(), CliError> {
    if let Some(description) = &field.description {
        println!("  {}", single_line_text(description));
    }
    let path = field_path(parent_path, field);
    if let DynamicConfigFieldKind::Object { fields } = &field.kind {
        return edit_dynamic_fields_menu(theme, state, fields, &path, field.title.clone());
    }
    if let Some(value) = prompt_dynamic_value(theme, state, field, &path)? {
        state.set_field(&path, value);
    }
    Ok(())
}

fn prompt_dynamic_value(
    theme: &ColorfulTheme,
    state: &DynamicPluginEditorState,
    field: &DynamicConfigField,
    path: &[String],
) -> Result<Option<Value>, CliError> {
    let current = state.field_value(path);
    match &field.kind {
        DynamicConfigFieldKind::Boolean => {
            let values = ["false", "true"];
            let default = current
                .and_then(Value::as_bool)
                .or_else(|| field.default.as_ref().and_then(Value::as_bool))
                .map(usize::from)
                .unwrap_or(0);
            let selected = Select::with_theme(theme)
                .with_prompt(single_line_text(&field.title))
                .items(&values)
                .default(default)
                .interact()
                .map_err(editor_error)?;
            Ok(Some(Value::Bool(selected == 1)))
        }
        DynamicConfigFieldKind::String { secret } => {
            prompt_dynamic_string(theme, field, current, *secret, None)
        }
        DynamicConfigFieldKind::StringEnum { options, secret } => {
            if *secret {
                prompt_dynamic_string(theme, field, current, true, Some(options))
            } else {
                let default = current
                    .and_then(Value::as_str)
                    .or_else(|| field.default.as_ref().and_then(Value::as_str))
                    .and_then(|value| options.iter().position(|option| option == value))
                    .unwrap_or(0);
                let selected = Select::with_theme(theme)
                    .with_prompt(single_line_text(&field.title))
                    .items(options)
                    .default(default)
                    .interact()
                    .map_err(editor_error)?;
                Ok(Some(Value::String(options[selected].clone())))
            }
        }
        DynamicConfigFieldKind::Integer => {
            let initial = current
                .or(field.default.as_ref())
                .map(json_text)
                .unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(single_line_text(&field.title))
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            let value = value.trim().parse::<i64>().map_err(|error| {
                CliError::Config(format!("{} must be an integer: {error}", field.key))
            })?;
            Ok(Some(Value::Number(value.into())))
        }
        DynamicConfigFieldKind::Number => {
            let initial = current
                .or(field.default.as_ref())
                .map(json_text)
                .unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(single_line_text(&field.title))
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            let parsed = value.trim().parse::<f64>().map_err(|error| {
                CliError::Config(format!("{} must be a number: {error}", field.key))
            })?;
            let number = Number::from_f64(parsed).ok_or_else(|| {
                CliError::Config(format!("{} must be a finite number", field.key))
            })?;
            Ok(Some(Value::Number(number)))
        }
        DynamicConfigFieldKind::StringMap => {
            let (current, redacted_config, secrets, hidden) = state.field_value_for_raw_edit(path);
            let Some(value) = prompt_json_value(
                theme,
                field,
                current.as_ref(),
                Value::Object(Map::new()),
                hidden,
            )?
            else {
                return Ok(None);
            };
            let value = state.restore_raw_field_edit(path, value, redacted_config, &secrets)?;
            let object = value
                .as_object()
                .ok_or_else(|| CliError::Config(format!("{} must be a JSON object", field.key)))?;
            if object.values().any(|value| !value.is_string()) {
                return Err(CliError::Config(format!(
                    "{} must contain only string values",
                    field.key
                )));
            }
            Ok(Some(value))
        }
        DynamicConfigFieldKind::RawJson => {
            let fallback = field.default.clone().unwrap_or(Value::Null);
            let (current, redacted_config, secrets, hidden) = state.field_value_for_raw_edit(path);
            let Some(value) = prompt_json_value(theme, field, current.as_ref(), fallback, hidden)?
            else {
                return Ok(None);
            };
            let value = state.restore_raw_field_edit(path, value, redacted_config, &secrets)?;
            Ok(Some(value))
        }
        DynamicConfigFieldKind::Object { .. } => unreachable!(),
    }
}

fn prompt_dynamic_string(
    theme: &ColorfulTheme,
    field: &DynamicConfigField,
    current: Option<&Value>,
    secret: bool,
    options: Option<&[String]>,
) -> Result<Option<Value>, CliError> {
    if secret {
        let title = single_line_text(&field.title);
        let value = Password::with_theme(theme)
            .with_prompt(format!("New {} (blank preserves the current value)", title))
            .allow_empty_password(true)
            .report(false)
            .interact()
            .map_err(editor_error)?;
        if value.is_empty() {
            return Ok(None);
        }
        if options.is_some_and(|options| !options.iter().any(|option| option == &value)) {
            return Err(CliError::Config(format!(
                "{} must be one of the schema enum values",
                field.key
            )));
        }
        return Ok(Some(Value::String(value)));
    }
    let initial = current
        .and_then(Value::as_str)
        .or_else(|| field.default.as_ref().and_then(Value::as_str))
        .unwrap_or_default();
    let value: String = Input::with_theme(theme)
        .with_prompt(single_line_text(&field.title))
        .with_initial_text(initial)
        .interact_text()
        .map_err(editor_error)?;
    Ok(Some(Value::String(value)))
}

fn prompt_json_value(
    theme: &ColorfulTheme,
    field: &DynamicConfigField,
    current: Option<&Value>,
    fallback: Value,
    hidden: bool,
) -> Result<Option<Value>, CliError> {
    let initial = current.or(field.default.as_ref()).unwrap_or(&fallback);
    let prompt = format!("{} as JSON", single_line_text(&field.title));
    let value = if hidden {
        if current.is_some() {
            println!("  Current redacted JSON: {}", json_text(initial));
        }
        let value = Password::with_theme(theme)
            .with_prompt(format!("New {prompt} (blank preserves the current value)"))
            .allow_empty_password(true)
            .report(false)
            .interact()
            .map_err(editor_error)?;
        if value.is_empty() {
            return Ok(None);
        }
        value
    } else {
        Input::with_theme(theme)
            .with_prompt(prompt)
            .with_initial_text(json_text(initial))
            .interact_text()
            .map_err(editor_error)?
    };
    serde_json::from_str(value.trim())
        .map_err(|error| CliError::Config(format!("invalid JSON for {}: {error}", field.key)))
        .map(Some)
}

fn prompt_raw_config(
    theme: &ColorfulTheme,
    state: &mut DynamicPluginEditorState,
) -> Result<(), CliError> {
    let original = Value::Object(state.config.clone().unwrap_or_default());
    let (initial, secrets, hidden) = state
        .schema
        .as_ref()
        .map(|schema| {
            let (redacted, secrets) = schema.redact_for_edit(&original);
            (redacted, secrets, schema.has_secrets())
        })
        .unwrap_or_else(|| (original, SecretEditValues::new(), false));
    let value = if hidden {
        println!("  Current redacted JSON: {}", json_text(&initial));
        let value = Password::with_theme(theme)
            .with_prompt("New configuration as JSON object (blank preserves the current value)")
            .allow_empty_password(true)
            .report(false)
            .interact()
            .map_err(editor_error)?;
        if value.is_empty() {
            return Ok(());
        }
        value
    } else {
        Input::with_theme(theme)
            .with_prompt("Configuration as JSON object")
            .with_initial_text(json_text(&initial))
            .interact_text()
            .map_err(editor_error)?
    };
    let value: Value = serde_json::from_str(value.trim())
        .map_err(|error| CliError::Config(format!("invalid JSON configuration: {error}")))?;
    let value = match &state.schema {
        Some(schema) => schema.restore_edit_secrets(&value, &secrets)?,
        None => value,
    };
    let object = value.as_object().cloned().ok_or_else(|| {
        CliError::Config(format!(
            "dynamic plugin '{}' configuration must be a JSON object",
            state.plugin_id
        ))
    })?;
    if let Some(schema) = &state.schema {
        schema.validate(&value)?;
    }
    state.set_raw_config(object);
    Ok(())
}
