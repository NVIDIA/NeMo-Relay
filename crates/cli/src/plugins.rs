// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interactive plugin configuration editing.
//!
//! Keep this module focused on TTY and `dialoguer` orchestration. New testable plugin config
//! behavior should live in `plugins/config_io.rs` or `plugins/editor_model.rs`, with focused unit
//! tests, so Codecov does not depend on exercising interactive prompt loops.

use std::io::IsTerminal;
use std::path::Path;

use console::{Key, Term, style};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Select};
use nemo_flow::config_editor::{EditorConfig, EditorFieldKind, EditorFieldSpec};
use nemo_flow::observability::plugin_component::ObservabilityConfig;
use serde_json::{Value, json};

use crate::config::PluginsEditCommand;
use crate::error::CliError;

mod config_io;
mod editor_model;

use self::config_io::*;
use self::editor_model::*;

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
    println!("  {} Saved {}", style("✔").green(), path.display());
}

pub(crate) fn edit(command: PluginsEditCommand) -> Result<(), CliError> {
    ensure_tty()?;
    let scope = target_scope(&command)?;
    let path = target_path(scope)?;
    let mut config = read_plugin_config(&path)?;
    ensure_observability_component(&mut config)?;
    let mut observability = component_observability_config(&config)?;

    let theme = ColorfulTheme::default();
    crate::banner::print_intro();
    println!(
        "  Editing Observability plugin config at {}",
        path.display()
    );
    println!("  Tip: ↑/↓ or j/k to move, SPACE/ENTER to select, p to preview, s to save.");
    println!();
    loop {
        let summary = observability_summary(&config, &observability);
        let section_fields = ObservabilityConfig::editor_schema().fields;
        let mut items = vec![MenuItem::new(format!(
            "Toggle Observability component [{}]",
            status_label(component_enabled(&config))
        ))];
        items.extend(section_fields.iter().map(|section| {
            MenuItem::new(configured_label(
                section_configured(&observability, *section),
                format!("Edit {}", section.label),
            ))
        }));
        items.push(MenuItem::new(shortcut_label("Preview TOML", "p")));
        items.push(MenuItem::new(shortcut_label(
            format!("Save to {}", path.display()),
            "s",
        )));
        items.push(MenuItem::new(shortcut_label("Cancel", "q")));
        println!();
        println!("Observability: {summary}");
        let preview_index = section_fields.len() + 1;
        let save_index = section_fields.len() + 2;
        let cancel_index = section_fields.len() + 3;
        let selection = prompt_menu(&theme, "plugins.toml", &items, 0)?;
        match selection {
            MenuResponse::Selected(0) => {
                let enabled = !component_enabled(&config);
                set_component_enabled(&mut config, enabled);
            }
            MenuResponse::Selected(selection)
                if (1..=section_fields.len()).contains(&selection) =>
            {
                edit_section(&theme, &mut observability, section_fields[selection - 1])?
            }
            MenuResponse::Selected(selection) if selection == preview_index => {
                let preview_config = config_with_observability(&config, &observability)?;
                print_preview(&preview_config)?;
            }
            MenuResponse::Selected(selection) if selection == save_index => {
                store_observability_config(&mut config, &observability)?;
                validate_config(&config)?;
                write_plugin_config(&path, &config)?;
                print_save_success(&path);
                return Ok(());
            }
            MenuResponse::Selected(selection) if selection == cancel_index => {
                return Err(CliError::Config(
                    "plugin edit cancelled; no config saved".into(),
                ));
            }
            MenuResponse::Shortcut(MenuShortcut::Preview, _) => {
                let preview_config = config_with_observability(&config, &observability)?;
                print_preview(&preview_config)?;
            }
            MenuResponse::Shortcut(MenuShortcut::Save, _) => {
                store_observability_config(&mut config, &observability)?;
                validate_config(&config)?;
                write_plugin_config(&path, &config)?;
                print_save_success(&path);
                return Ok(());
            }
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(MenuShortcut::Reset | MenuShortcut::Clear, _) => {
                println!("  Select a section first, then use reset or clear on a field.");
            }
            MenuResponse::Cancel | MenuResponse::Selected(_) => {
                return Err(CliError::Config(
                    "plugin edit cancelled; no config saved".into(),
                ));
            }
        }
    }
}

fn prompt_menu(
    theme: &ColorfulTheme,
    prompt: &str,
    items: &[MenuItem],
    default: usize,
) -> Result<MenuResponse, CliError> {
    if items.is_empty() {
        return Err(CliError::Config(format!("{prompt} menu has no items")));
    }
    let term = Term::stderr();
    let mut selected = default.min(items.len() - 1);
    let mut rendered_lines = 0;
    loop {
        if rendered_lines > 0 {
            term.clear_last_lines(rendered_lines).map_err(menu_error)?;
        }
        let lines = render_menu(theme, prompt, items, selected);
        rendered_lines = lines.len();
        for line in &lines {
            term.write_line(line).map_err(menu_error)?;
        }
        term.flush().map_err(menu_error)?;
        match term.read_key().map_err(menu_error)? {
            Key::ArrowUp | Key::Char('k') => {
                selected = if selected == 0 {
                    items.len() - 1
                } else {
                    selected - 1
                };
            }
            Key::ArrowDown | Key::Char('j') => {
                selected = (selected + 1) % items.len();
            }
            Key::Enter | Key::Char(' ') => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Selected(selected));
            }
            Key::Char('p') => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Shortcut(MenuShortcut::Preview, selected));
            }
            Key::Char('s') => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Shortcut(MenuShortcut::Save, selected));
            }
            Key::Char('r') => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Shortcut(MenuShortcut::Reset, selected));
            }
            Key::Backspace | Key::Del => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Shortcut(MenuShortcut::Clear, selected));
            }
            Key::Char('?') => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Shortcut(MenuShortcut::Help, selected));
            }
            Key::Escape | Key::CtrlC | Key::Char('q') => {
                clear_menu(&term, rendered_lines)?;
                return Ok(MenuResponse::Cancel);
            }
            _ => {}
        }
    }
}

fn render_menu(
    theme: &ColorfulTheme,
    prompt: &str,
    items: &[MenuItem],
    selected: usize,
) -> Vec<String> {
    let mut lines = Vec::with_capacity(items.len() + 2);
    lines.push(format!(
        "{} {} {}",
        theme.prompt_prefix,
        theme.prompt_style.apply_to(prompt),
        theme.prompt_suffix
    ));
    lines.push(
        theme
            .hint_style
            .apply_to("  ↑/↓ or j/k move, Enter/Space select, p preview, s save, r reset, Backspace/Delete clear, ? help, q cancel.")
            .to_string(),
    );
    lines.extend(items.iter().enumerate().map(|(index, item)| {
        if index == selected {
            format!(
                "{} {}",
                theme.active_item_prefix,
                theme.active_item_style.apply_to(&item.label)
            )
        } else {
            format!(
                "{} {}",
                theme.inactive_item_prefix,
                theme.inactive_item_style.apply_to(&item.label)
            )
        }
    }));
    lines
}

fn clear_menu(term: &Term, rendered_lines: usize) -> Result<(), CliError> {
    if rendered_lines > 0 {
        term.clear_last_lines(rendered_lines).map_err(menu_error)?;
    }
    Ok(())
}

fn menu_error(error: std::io::Error) -> CliError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
    ) {
        CliError::Config("plugin edit cancelled; no config saved".into())
    } else {
        CliError::Config(format!("plugin editor terminal error: {error}"))
    }
}

fn print_editor_help() {
    println!();
    println!(
        "{} {}",
        style("?").yellow(),
        style("Plugin editor keys").bold()
    );
    println!("  {}  move", style("↑/↓ or j/k").cyan());
    println!(
        "  {} select/toggle the highlighted item",
        style("Enter/Space").cyan()
    );
    println!(
        "  {}             reset the highlighted field or section",
        style("r").cyan()
    );
    println!(
        "  {} clear the highlighted optional field",
        style("Backspace/Del").cyan()
    );
    println!(
        "  {}             preview TOML from the main menu",
        style("p").cyan()
    );
    println!(
        "  {}             save from the main menu",
        style("s").cyan()
    );
    println!("  {}      go back/cancel", style("q or Esc").cyan());
}

fn ensure_tty() -> Result<(), CliError> {
    if !std::io::stdin().is_terminal()
        || !std::io::stdout().is_terminal()
        || !std::io::stderr().is_terminal()
    {
        return Err(CliError::Config(
            "interactive plugin editing requires a TTY".into(),
        ));
    }
    Ok(())
}

fn edit_section(
    theme: &ColorfulTheme,
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
) -> Result<(), CliError> {
    ensure_section(config, section);
    let fields = section
        .schema()
        .ok_or_else(|| CliError::Config(format!("{} is not an editable section", section.name)))?
        .fields;
    loop {
        let items = section_menu_items(config, section, fields)?;
        let selection = prompt_menu(theme, section.name, &items, 0)?;
        let selection = match selection {
            MenuResponse::Selected(selection) => selection,
            MenuResponse::Shortcut(MenuShortcut::Help, _) => {
                print_editor_help();
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Reset, selected) => {
                reset_selected_item(config, section, fields, selected)?;
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Clear, selected) => {
                if reset_selected_field(config, section, fields, selected)? {
                    continue;
                }
                println!("  Select a field to clear.");
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
                continue;
            }
            MenuResponse::Cancel => return Ok(()),
        };
        if !edit_selected_section_item(theme, config, section, fields, selection)? {
            return Ok(());
        }
    }
}

fn section_menu_items(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
) -> Result<Vec<MenuItem>, CliError> {
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

fn section_field_menu_item(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
    field: EditorFieldSpec,
) -> Result<MenuItem, CliError> {
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

fn reset_selected_item(
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
    selected: usize,
) -> Result<(), CliError> {
    if reset_selected_field(config, section, fields, selected)? {
        return Ok(());
    }
    if selected == reset_section_index(section, fields) {
        reset_section(config, section);
    }
    Ok(())
}

fn edit_selected_section_item(
    theme: &ColorfulTheme,
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
    selection: usize,
) -> Result<bool, CliError> {
    if section_has_enabled_toggle(section) && selection == 0 {
        toggle_section(config, section);
        return Ok(true);
    }
    let index = selected_field_index(section, selection);
    if let Some(field) = fields.get(index) {
        edit_field(theme, config, section, field)?;
        return Ok(true);
    }
    if index == fields.len() {
        reset_section(config, section);
        return Ok(true);
    }
    Ok(false)
}

fn edit_field(
    theme: &ColorfulTheme,
    config: &mut ObservabilityConfig,
    section: EditorFieldSpec,
    field: &EditorFieldSpec,
) -> Result<(), CliError> {
    let current = section_field_value(config, section, field.name)?;
    let actions = [
        MenuItem::new("Set value"),
        MenuItem::new(shortcut_label(
            "Reset to default/none",
            "r, Backspace, Delete",
        )),
        MenuItem::new(shortcut_label("Back", "q")),
    ];
    let action = prompt_menu(
        theme,
        &format!(
            "{}.{}, current {}",
            section.name,
            field.name,
            current
                .as_ref()
                .map(|value| display_field_value(section, *field, value))
                .unwrap_or_else(|| "(default)".to_string())
        ),
        &actions,
        0,
    )?;
    match action {
        MenuResponse::Selected(0) => {
            let value = prompt_value(theme, field, current.as_ref())?;
            set_section_field(config, section, field.name, value)?;
        }
        MenuResponse::Selected(1)
        | MenuResponse::Shortcut(MenuShortcut::Reset | MenuShortcut::Clear, _) => {
            remove_section_field(config, section, field.name)?
        }
        MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
        MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
            println!("  Preview and save are available from the main plugins.toml menu.");
        }
        _ => {}
    }
    Ok(())
}

fn prompt_value(
    theme: &ColorfulTheme,
    field: &EditorFieldSpec,
    current: Option<&Value>,
) -> Result<Value, CliError> {
    match field.kind {
        EditorFieldKind::Boolean => {
            let values = ["false", "true"];
            let default_idx = current
                .and_then(Value::as_bool)
                .map(usize::from)
                .unwrap_or(0);
            let idx = Select::with_theme(theme)
                .with_prompt(field.name)
                .items(&values)
                .default(default_idx)
                .interact()
                .map_err(editor_error)?;
            Ok(json!(idx == 1))
        }
        EditorFieldKind::Integer => {
            let initial = current.map(display_value).unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(field.name)
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            let parsed = value.trim().parse::<u64>().map_err(|error| {
                CliError::Config(format!("{} must be an integer: {error}", field.name))
            })?;
            Ok(json!(parsed))
        }
        EditorFieldKind::StringMap | EditorFieldKind::Json => {
            let initial = current.map(display_value).unwrap_or_else(|| {
                if field.name == "tool_definitions" {
                    "[]".to_string()
                } else {
                    "{}".to_string()
                }
            });
            let value: String = Input::with_theme(theme)
                .with_prompt(format!("{} as JSON", field.name))
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            serde_json::from_str(value.trim()).map_err(|error| {
                CliError::Config(format!("invalid JSON for {}: {error}", field.name))
            })
        }
        EditorFieldKind::Enum => {
            let values = field.enum_values;
            let default_idx = current
                .and_then(Value::as_str)
                .and_then(|value| values.iter().position(|candidate| *candidate == value))
                .unwrap_or(0);
            let idx = Select::with_theme(theme)
                .with_prompt(field.name)
                .items(values)
                .default(default_idx)
                .interact()
                .map_err(editor_error)?;
            Ok(json!(values[idx]))
        }
        EditorFieldKind::String => {
            let initial = current.and_then(Value::as_str).unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(field.name)
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            Ok(json!(value))
        }
        EditorFieldKind::Section => Err(CliError::Config(format!(
            "{} is a nested section and cannot be edited as a scalar",
            field.name
        ))),
    }
}

fn editor_error(err: dialoguer::Error) -> CliError {
    match err {
        dialoguer::Error::IO(io_err)
            if matches!(
                io_err.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
            ) =>
        {
            CliError::Config("plugin edit cancelled; no config saved".into())
        }
        other => CliError::Config(format!("plugin edit error: {other}")),
    }
}

#[cfg(test)]
#[path = "../tests/coverage/plugins_tests.rs"]
mod tests;
