// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Interactive plugin configuration editing.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};

use console::{Key, Term, style};
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Select};
use nemo_flow::config_editor::{EditorConfig, EditorFieldKind, EditorFieldSpec};
use nemo_flow::observability::plugin_component::{OBSERVABILITY_PLUGIN_KIND, ObservabilityConfig};
use nemo_flow::plugin::{ConfigPolicy, PluginComponentSpec, PluginConfig, validate_plugin_config};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::config::{
    PluginsEditCommand, global_plugin_config_path, project_plugin_config_path,
    user_plugin_config_path,
};
use crate::error::CliError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetScope {
    User,
    Project,
    Global,
}

const POLICY_SECTION: &str = "policy";

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
        items.extend(
            section_fields
                .iter()
                .map(|section| MenuItem::new(format!("Edit {}", section.label))),
        );
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
    if !std::io::stdin().is_terminal() {
        return Err(CliError::Config(
            "interactive plugin editing requires a TTY".into(),
        ));
    }
    Ok(())
}

fn target_scope(command: &PluginsEditCommand) -> Result<TargetScope, CliError> {
    let selected = [command.user, command.project, command.global]
        .into_iter()
        .filter(|selected| *selected)
        .count();
    if selected > 1 {
        return Err(CliError::Config(
            "choose only one of --user, --project, or --global".into(),
        ));
    }
    if command.project {
        Ok(TargetScope::Project)
    } else if command.global {
        Ok(TargetScope::Global)
    } else {
        Ok(TargetScope::User)
    }
}

fn target_path(scope: TargetScope) -> Result<PathBuf, CliError> {
    match scope {
        TargetScope::User => user_plugin_config_path().ok_or_else(|| {
            CliError::Config(
                "cannot determine user config directory; set HOME or XDG_CONFIG_HOME".into(),
            )
        }),
        TargetScope::Project => {
            let cwd = std::env::current_dir()?;
            Ok(project_plugin_config_path(&cwd))
        }
        TargetScope::Global => Ok(global_plugin_config_path()),
    }
}

fn read_plugin_config(path: &Path) -> Result<PluginConfig, CliError> {
    if !path.exists() {
        return Ok(PluginConfig::default());
    }
    let raw = std::fs::read_to_string(path)?;
    let parsed = raw
        .parse::<toml::Table>()
        .map(toml::Value::Table)
        .map_err(|error| {
            CliError::Config(format!(
                "invalid plugin TOML in {}: {error}",
                path.display()
            ))
        })?;
    serde_json::from_value(
        serde_json::to_value(parsed)
            .map_err(|error| CliError::Config(format!("invalid plugin TOML shape: {error}")))?,
    )
    .map_err(|error| CliError::Config(format!("invalid plugin config: {error}")))
}

fn write_plugin_config(path: &Path, config: &PluginConfig) -> Result<(), CliError> {
    let mut value = serde_json::to_value(config)
        .map_err(|error| CliError::Config(format!("could not serialize plugin config: {error}")))?;
    prune_plugin_defaults(&mut value);
    let toml_value: toml::Value = serde_json::from_value(value).map_err(|error| {
        CliError::Config(format!("could not convert plugin config to TOML: {error}"))
    })?;
    let rendered = toml::to_string_pretty(&toml_value)
        .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, rendered)?;
    Ok(())
}

fn print_preview(config: &PluginConfig) -> Result<(), CliError> {
    println!();
    println!(
        "{} {}",
        style("❯").green(),
        style("plugins.toml preview").bold()
    );
    println!("{}", style("─".repeat(58)).black().bright());
    let mut value = serde_json::to_value(config)
        .map_err(|error| CliError::Config(format!("could not serialize plugin config: {error}")))?;
    prune_plugin_defaults(&mut value);
    let toml_value: toml::Value = serde_json::from_value(value).map_err(|error| {
        CliError::Config(format!("could not convert plugin config to TOML: {error}"))
    })?;
    let rendered = toml::to_string_pretty(&toml_value)
        .map_err(|error| CliError::Config(format!("could not render plugin TOML: {error}")))?;
    print!("{rendered}");
    println!("{}", style("─".repeat(58)).black().bright());
    Ok(())
}

fn validate_config(config: &PluginConfig) -> Result<(), CliError> {
    let report = validate_plugin_config(config);
    if report.has_errors() {
        let messages = report
            .diagnostics
            .into_iter()
            .filter(|diagnostic| diagnostic.level == nemo_flow::plugin::DiagnosticLevel::Error)
            .map(|diagnostic| diagnostic.message)
            .collect::<Vec<_>>()
            .join("; ");
        return Err(CliError::Config(format!(
            "plugin validation failed: {messages}"
        )));
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
        let mut items = Vec::new();
        if section_has_enabled_toggle(section) {
            let enabled = section_enabled(config, section).unwrap_or(false);
            items.push(MenuItem::new(format!(
                "Toggle section [{}]",
                status_label(enabled)
            )));
        }
        for field in fields {
            items.push(MenuItem::new(format!(
                "{} = {}",
                field.name,
                section_field_value(config, section, field.name)?
                    .map(|value| display_field_value(section, *field, &value))
                    .or_else(|| default_field_value(section, *field)
                        .map(|value| format!("{} (default)", display_value(&value))))
                    .unwrap_or_else(|| "(default)".to_string())
            )));
        }
        items.push(MenuItem::new(shortcut_label("Reset section", "r")));
        items.push(MenuItem::new(shortcut_label("Back", "q")));
        let selection = prompt_menu(theme, section.name, &items, 0)?;
        let selection = match selection {
            MenuResponse::Selected(selection) => selection,
            MenuResponse::Shortcut(MenuShortcut::Help, _) => {
                print_editor_help();
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Reset, selected) => {
                if reset_selected_field(config, section, fields, selected)? {
                    continue;
                }
                reset_section(config, section);
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
        let mut index = selection;
        if section_has_enabled_toggle(section) {
            if index == 0 {
                toggle_section(config, section);
                continue;
            }
            index -= 1;
        }
        if index < fields.len() {
            edit_field(theme, config, section, &fields[index])?;
        } else if index == fields.len() {
            reset_section(config, section);
        } else {
            return Ok(());
        }
    }
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

fn ensure_observability_component(config: &mut PluginConfig) -> Result<(), CliError> {
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

fn component_enabled(config: &PluginConfig) -> bool {
    observability_component(config)
        .map(|component| component.enabled)
        .unwrap_or(true)
}

fn set_component_enabled(config: &mut PluginConfig, enabled: bool) {
    if let Some(component) = observability_component_mut(config) {
        component.enabled = enabled;
    }
}

fn component_observability_config(config: &PluginConfig) -> Result<ObservabilityConfig, CliError> {
    observability_component(config)
        .map(|component| serde_json::from_value(Value::Object(component.config.clone())))
        .transpose()
        .map_err(|error| CliError::Config(format!("invalid observability plugin config: {error}")))?
        .ok_or_else(|| CliError::Config("observability plugin component is missing".into()))
}

fn config_with_observability(
    config: &PluginConfig,
    observability: &ObservabilityConfig,
) -> Result<PluginConfig, CliError> {
    let mut config = config.clone();
    store_observability_config(&mut config, observability)?;
    Ok(config)
}

fn store_observability_config(
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

fn ensure_section(config: &mut ObservabilityConfig, section: EditorFieldSpec) {
    if let Ok(Some(Value::Object(_))) = section_value(config, section) {
        return;
    }
    let Some(default) = section.default_value() else {
        return;
    };
    let _ = set_struct_field(config, section.name, default);
}

fn toggle_section(config: &mut ObservabilityConfig, section: EditorFieldSpec) {
    ensure_section(config, section);
    let enabled = section_enabled(config, section).unwrap_or(false);
    let _ = set_section_field(config, section, "enabled", json!(!enabled));
}

fn reset_section(config: &mut ObservabilityConfig, section: EditorFieldSpec) {
    let value = section.default_value().unwrap_or_else(|| json!({}));
    let _ = set_struct_field(config, section.name, value);
}

fn reset_selected_field(
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

fn section_has_enabled_toggle(section: EditorFieldSpec) -> bool {
    section.name != POLICY_SECTION
        && section
            .schema()
            .and_then(|schema| schema.field("enabled"))
            .is_some_and(|field| field.kind == EditorFieldKind::Boolean)
}

fn section_enabled(config: &ObservabilityConfig, section: EditorFieldSpec) -> Option<bool> {
    section_value(config, section)
        .ok()
        .flatten()
        .and_then(|section| section.get("enabled").cloned())
        .and_then(|enabled| enabled.as_bool())
}

fn section_field_value(
    config: &ObservabilityConfig,
    section: EditorFieldSpec,
    field: &str,
) -> Result<Option<Value>, CliError> {
    Ok(section_value(config, section)?
        .and_then(|section| section.as_object().cloned())
        .and_then(|section| section.get(field).cloned()))
}

fn section_value(
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

fn set_section_field(
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

fn remove_section_field(
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

fn set_struct_field<T>(target: &mut T, field: &str, value: Value) -> Result<(), CliError>
where
    T: Serialize + DeserializeOwned,
{
    let mut object = serde_json::to_value(&*target).map_err(serde_error)?;
    ensure_object(&mut object).insert(field.to_string(), value);
    *target = serde_json::from_value(object).map_err(serde_error)?;
    Ok(())
}

fn observability_component(config: &PluginConfig) -> Option<&PluginComponentSpec> {
    config
        .components
        .iter()
        .find(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
}

fn observability_component_mut(config: &mut PluginConfig) -> Option<&mut PluginComponentSpec> {
    config
        .components
        .iter_mut()
        .find(|component| component.kind == OBSERVABILITY_PLUGIN_KIND)
}

fn ensure_object(value: &mut Value) -> &mut Map<String, Value> {
    if !value.is_object() {
        *value = json!({});
    }
    value.as_object_mut().expect("value initialized as object")
}

fn observability_config_map(config: &ObservabilityConfig) -> Result<Map<String, Value>, CliError> {
    let value = serde_json::to_value(config).map_err(serde_error)?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(CliError::Config(
            "observability config must serialize to an object".into(),
        )),
    }
}

fn merge_observability_editor_config(
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

fn merge_known_editor_object(
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

fn observability_editor_fields_with_version() -> Vec<&'static str> {
    let mut keys = vec!["version"];
    keys.extend(
        ObservabilityConfig::editor_schema()
            .fields
            .iter()
            .map(|field| field.name),
    );
    keys
}

fn nested_editor_keys(schema: &nemo_flow::config_editor::EditorSchema) -> Vec<&'static str> {
    schema.fields.iter().map(|field| field.name).collect()
}

fn prune_plugin_defaults(value: &mut Value) {
    let Some(object) = value.as_object_mut() else {
        return;
    };
    remove_default_field(
        object,
        "policy",
        serde_json::to_value(ConfigPolicy::default()).expect("policy default serializes"),
    );
    if let Some(components) = object.get_mut("components").and_then(Value::as_array_mut) {
        for component in components {
            if let Some(component) = component.as_object_mut()
                && component.get("enabled") == Some(&Value::Bool(true))
            {
                component.remove("enabled");
            }
        }
    }
}

fn remove_default_field(object: &mut Map<String, Value>, key: &str, default: Value) {
    let Some(value) = object.get_mut(key) else {
        return;
    };
    remove_matching_defaults(value, &default);
    if value == &default || value.as_object().is_some_and(|value| value.is_empty()) {
        object.remove(key);
    }
}

fn remove_matching_defaults(value: &mut Value, default: &Value) {
    let (Some(value), Some(default)) = (value.as_object_mut(), default.as_object()) else {
        return;
    };
    let default_keys = default.keys().cloned().collect::<Vec<_>>();
    for key in default_keys {
        if value.get(&key) == default.get(&key) {
            value.remove(&key);
        }
    }
}

fn serde_error(error: serde_json::Error) -> CliError {
    CliError::Config(format!("invalid plugin editor value: {error}"))
}

fn display_field_value(section: EditorFieldSpec, field: EditorFieldSpec, value: &Value) -> String {
    if default_field_value(section, field)
        .as_ref()
        .is_some_and(|default| default == value)
    {
        format!("{} (default)", display_value(value))
    } else {
        display_value(value)
    }
}

fn default_field_value(section: EditorFieldSpec, field: EditorFieldSpec) -> Option<Value> {
    section
        .default_value()
        .and_then(|section| section.as_object().cloned())
        .and_then(|section| section.get(field.name).cloned())
}

fn display_value(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| "<invalid>".to_string()),
    }
}

fn observability_summary(config: &PluginConfig, observability: &ObservabilityConfig) -> String {
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
