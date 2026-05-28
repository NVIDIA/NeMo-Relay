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
use nemo_relay::config_editor::{EditorFieldKind, EditorFieldSpec};
use nemo_relay::plugin::PluginConfig;
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
    println!("  {} Saved {}", style("✔").green(), path.display());
}

pub(crate) fn edit(command: PluginsEditCommand) -> Result<(), CliError> {
    ensure_tty()?;
    let scope = target_scope(&command)?;
    let path = target_path(scope)?;
    let mut config = read_plugin_config(&path)?;
    ensure_observability_component(&mut config)?;
    ensure_adaptive_component(&mut config)?;
    let mut components = editable_components(&config)?;

    let theme = ColorfulTheme::default();
    crate::banner::print_intro();
    println!("  Editing plugin config at {}", path.display());
    println!("  Tip: ↑/↓ or j/k to move, SPACE/ENTER to select, p to preview, s to save.");
    println!();
    let mut selected_index = 0;
    loop {
        let (items, actions) = plugin_menu_items(&components, &path);
        println!();
        for component in &components {
            println!("{}: {}", component.label(), component.summary());
        }
        let selection = prompt_menu(&theme, "plugins.toml", &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        if handle_menu_response(
            &theme,
            &path,
            &mut config,
            &mut components,
            &actions,
            selection,
        )? == EditLoopControl::Finish
        {
            return Ok(());
        }
    }
}

fn handle_menu_response(
    theme: &ColorfulTheme,
    path: &Path,
    config: &mut PluginConfig,
    components: &mut [EditableComponent],
    actions: &[MenuAction],
    selection: MenuResponse,
) -> Result<EditLoopControl, CliError> {
    match selection {
        MenuResponse::Selected(selection) => handle_menu_action(
            theme,
            path,
            config,
            components,
            actions.get(selection).copied(),
        ),
        MenuResponse::Shortcut(MenuShortcut::Preview, _) => {
            preview_components(config, components)?;
            Ok(EditLoopControl::Continue)
        }
        MenuResponse::Shortcut(MenuShortcut::Save, _) => save_components(path, config, components),
        MenuResponse::Shortcut(MenuShortcut::Help, _) => {
            print_editor_help();
            Ok(EditLoopControl::Continue)
        }
        MenuResponse::Shortcut(
            shortcut @ (MenuShortcut::Reset | MenuShortcut::Clear),
            selected,
        ) => handle_reset_or_clear_shortcut(components, actions.get(selected).copied(), shortcut),
        MenuResponse::Cancel => Err(cancelled_error()),
    }
}

fn handle_menu_action(
    theme: &ColorfulTheme,
    path: &Path,
    config: &mut PluginConfig,
    components: &mut [EditableComponent],
    action: Option<MenuAction>,
) -> Result<EditLoopControl, CliError> {
    match action {
        Some(MenuAction::ToggleComponent(component_index)) => {
            if let Some(component) = components.get_mut(component_index) {
                component.toggle_enabled();
            }
            Ok(EditLoopControl::Continue)
        }
        Some(MenuAction::EditField {
            component_index,
            field_index,
        }) => {
            edit_selected_component_field(theme, components, component_index, field_index)?;
            Ok(EditLoopControl::Continue)
        }
        Some(MenuAction::Preview) => {
            preview_components(config, components)?;
            Ok(EditLoopControl::Continue)
        }
        Some(MenuAction::Save) => save_components(path, config, components),
        Some(MenuAction::Cancel) | None => Err(cancelled_error()),
    }
}

fn edit_selected_component_field(
    theme: &ColorfulTheme,
    components: &mut [EditableComponent],
    component_index: usize,
    field_index: usize,
) -> Result<(), CliError> {
    if let Some(component) = components.get_mut(component_index)
        && let Some(field) = component.fields().get(field_index)
    {
        edit_component_field(theme, component, *field)?;
    }
    Ok(())
}

fn preview_components(
    config: &PluginConfig,
    components: &[EditableComponent],
) -> Result<(), CliError> {
    let preview_config = config_with_editable_components(config, components)?;
    print_preview(&preview_config)
}

fn save_components(
    path: &Path,
    config: &mut PluginConfig,
    components: &[EditableComponent],
) -> Result<EditLoopControl, CliError> {
    store_editable_components(config, components)?;
    validate_config(config)?;
    write_plugin_config(path, config)?;
    print_save_success(path);
    Ok(EditLoopControl::Finish)
}

fn handle_reset_or_clear_shortcut(
    components: &mut [EditableComponent],
    action: Option<MenuAction>,
    shortcut: MenuShortcut,
) -> Result<EditLoopControl, CliError> {
    match action {
        Some(MenuAction::ToggleComponent(component_index)) => {
            if let Some(component) = components.get_mut(component_index) {
                apply_component_enablement_shortcut(component, shortcut);
            }
        }
        Some(MenuAction::EditField {
            component_index,
            field_index,
        }) => reset_selected_component_field(components, component_index, field_index)?,
        _ => println!("  Select a component or editable field to reset or clear."),
    }
    Ok(EditLoopControl::Continue)
}

fn reset_selected_component_field(
    components: &mut [EditableComponent],
    component_index: usize,
    field_index: usize,
) -> Result<(), CliError> {
    if let Some(component) = components.get_mut(component_index)
        && let Some(field) = component.fields().get(field_index)
    {
        component.reset_field(*field)?;
    }
    Ok(())
}

fn cancelled_error() -> CliError {
    CliError::Config("plugin edit cancelled; no config saved".into())
}

fn edit_component_field(
    theme: &ColorfulTheme,
    component: &mut EditableComponent,
    field: EditorFieldSpec,
) -> Result<(), CliError> {
    match component {
        EditableComponent::Observability(state) => {
            edit_section(theme, &mut state.config, field)?;
            state.mark_config_touched();
        }
        EditableComponent::Adaptive(state) => {
            edit_config_field(theme, &mut state.config, field)?;
            state.mark_config_touched();
        }
        EditableComponent::NemoGuardrails(state) => {
            edit_config_field(theme, &mut state.config, field)?;
            state.mark_config_touched();
        }
    }
    Ok(())
}

fn apply_component_enablement_shortcut(component: &mut EditableComponent, shortcut: MenuShortcut) {
    match shortcut {
        MenuShortcut::Reset => component.reset_enabled(),
        MenuShortcut::Clear => component.set_enabled(false),
        _ => {}
    }
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

fn edit_section<T>(
    theme: &ColorfulTheme,
    config: &mut T,
    section: EditorFieldSpec,
) -> Result<(), CliError>
where
    T: SerializeConfig,
{
    let fields = section
        .schema()
        .ok_or_else(|| CliError::Config(format!("{} is not an editable section", section.name)))?
        .fields;
    let mut selected_index = 0;
    loop {
        let items = section_menu_items(config, section, fields)?;
        let selection = prompt_menu(theme, section.name, &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
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

fn edit_selected_section_item<T>(
    theme: &ColorfulTheme,
    config: &mut T,
    section: EditorFieldSpec,
    fields: &[EditorFieldSpec],
    selection: usize,
) -> Result<bool, CliError>
where
    T: SerializeConfig,
{
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

fn edit_field<T>(
    theme: &ColorfulTheme,
    config: &mut T,
    section: EditorFieldSpec,
    field: &EditorFieldSpec,
) -> Result<(), CliError>
where
    T: SerializeConfig,
{
    if field.kind == EditorFieldKind::Section {
        edit_nested_section(theme, config, section, *field)?;
        return Ok(());
    }
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

fn edit_config_field<T>(
    theme: &ColorfulTheme,
    config: &mut T,
    field: EditorFieldSpec,
) -> Result<(), CliError>
where
    T: Default + SerializeConfig,
{
    if field.kind == EditorFieldKind::Section {
        let mut value = config_field_value(config, field.name)?
            .or_else(|| field.default_value())
            .unwrap_or_else(|| json!({}));
        let schema = field.schema().ok_or_else(|| {
            CliError::Config(format!("{} is not an editable section", field.name))
        })?;
        if edit_value_section(theme, field.name, &mut value, schema, field.default_value())? {
            store_edited_config_section(config, field, value)?;
        }
        return Ok(());
    }

    let current = config_field_value(config, field.name)?;
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
            "{}, current {}",
            field.label,
            current
                .as_ref()
                .map(display_value)
                .or_else(|| default_config_field_value::<T>(field)
                    .map(|value| { format!("{} (default)", display_value(&value)) }))
                .unwrap_or_else(|| "(default)".to_string())
        ),
        &actions,
        0,
    )?;
    match action {
        MenuResponse::Selected(0) => {
            let value = prompt_value(theme, &field, current.as_ref())?;
            set_struct_field(config, field.name, value)?;
        }
        MenuResponse::Selected(1)
        | MenuResponse::Shortcut(MenuShortcut::Reset | MenuShortcut::Clear, _) => {
            reset_config_field(config, field)?
        }
        MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
        MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
            println!("  Preview and save are available from the main plugins.toml menu.");
        }
        _ => {}
    }
    Ok(())
}

fn edit_nested_section<T>(
    theme: &ColorfulTheme,
    config: &mut T,
    section: EditorFieldSpec,
    field: EditorFieldSpec,
) -> Result<(), CliError>
where
    T: SerializeConfig,
{
    let mut value = section_field_value(config, section, field.name)?
        .or_else(|| section_field_default(section, field))
        .unwrap_or_else(|| json!({}));
    let schema = field
        .schema()
        .ok_or_else(|| CliError::Config(format!("{} is not an editable section", field.name)))?;
    let default = section_field_default(section, field);
    if edit_value_section(
        theme,
        &format!("{}.{}", section.name, field.name),
        &mut value,
        schema,
        default,
    )? {
        store_edited_section_field(config, section, field, value)?;
    }
    Ok(())
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

fn edit_value_section(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    default: Option<Value>,
) -> Result<bool, CliError> {
    ensure_object(value);
    let original = value.clone();
    let mut selected_index = 0;
    loop {
        let items = value_section_menu_items(value, schema, default.as_ref())?;
        let selection = prompt_menu(theme, prompt, &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        let selection = match selection {
            MenuResponse::Selected(selection) => selection,
            MenuResponse::Shortcut(MenuShortcut::Help, _) => {
                print_editor_help();
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Reset, selected) => {
                reset_value_section_item(value, schema, default.as_ref(), selected);
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Clear, selected) => {
                if clear_value_field(value, schema, selected) {
                    continue;
                }
                println!("  Select a field to clear.");
                continue;
            }
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
                continue;
            }
            MenuResponse::Cancel => return Ok(*value != original),
        };
        if !edit_selected_value_item(theme, prompt, value, schema, default.as_ref(), selection)? {
            return Ok(*value != original);
        }
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

fn edit_selected_value_item(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    default: Option<&Value>,
    selection: usize,
) -> Result<bool, CliError> {
    if let Some(field) = schema.fields.get(selection) {
        edit_value_field(theme, prompt, value, *field, default)?;
        return Ok(true);
    }
    if selection == schema.fields.len() {
        *value = default.cloned().unwrap_or_else(|| json!({}));
        ensure_object(value);
        return Ok(true);
    }
    Ok(false)
}

fn edit_value_field(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    field: EditorFieldSpec,
    default: Option<&Value>,
) -> Result<(), CliError> {
    if field.kind == EditorFieldKind::Section {
        let nested_default = value_field_default(default, field);
        let mut nested_value = value_field_value(value, field.name)
            .or_else(|| nested_default.clone())
            .unwrap_or_else(|| json!({}));
        let nested_schema = field.schema().ok_or_else(|| {
            CliError::Config(format!("{} is not an editable section", field.name))
        })?;
        if edit_value_section(
            theme,
            &format!("{prompt}.{}", field.name),
            &mut nested_value,
            nested_schema,
            nested_default,
        )? {
            store_edited_value_section(value, field, nested_value);
        }
        return Ok(());
    }

    let current = value_field_value(value, field.name);
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
            "{prompt}.{}, current {}",
            field.name,
            current
                .as_ref()
                .map(|value| {
                    display_value_with_default(value, value_field_default(default, field))
                })
                .or_else(|| {
                    value_field_default(default, field)
                        .map(|value| format!("{} (default)", display_value(&value)))
                })
                .unwrap_or_else(|| "(default)".to_string())
        ),
        &actions,
        0,
    )?;
    match action {
        MenuResponse::Selected(0) => {
            let field_value = prompt_value(theme, &field, current.as_ref())?;
            set_value_field(value, field.name, field_value);
        }
        MenuResponse::Selected(1)
        | MenuResponse::Shortcut(MenuShortcut::Reset | MenuShortcut::Clear, _) => {
            reset_value_field(value, field, default)
        }
        MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
        MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
            println!("  Preview and save are available from the main plugins.toml menu.");
        }
        _ => {}
    }
    Ok(())
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
                .with_prompt(field.label)
                .items(&values)
                .default(default_idx)
                .interact()
                .map_err(editor_error)?;
            Ok(json!(idx == 1))
        }
        EditorFieldKind::Integer => {
            let initial = current.map(display_value).unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(field.label)
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            let parsed = value.trim().parse::<i64>().map_err(|error| {
                CliError::Config(format!("{} must be an integer: {error}", field.name))
            })?;
            Ok(json!(parsed))
        }
        EditorFieldKind::Float => {
            let initial = current.map(display_value).unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(field.label)
                .with_initial_text(initial)
                .interact_text()
                .map_err(editor_error)?;
            parse_float_value(field, &value)
        }
        EditorFieldKind::StringMap | EditorFieldKind::Json => {
            let initial = current.map(display_value).unwrap_or_else(|| {
                if matches!(field.name, "tool_definitions" | "learners") {
                    "[]".to_string()
                } else {
                    "{}".to_string()
                }
            });
            let value: String = Input::with_theme(theme)
                .with_prompt(format!("{} as JSON", field.label))
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
                .with_prompt(field.label)
                .items(values)
                .default(default_idx)
                .interact()
                .map_err(editor_error)?;
            Ok(json!(values[idx]))
        }
        EditorFieldKind::String => {
            let initial = current.and_then(Value::as_str).unwrap_or_default();
            let value: String = Input::with_theme(theme)
                .with_prompt(field.label)
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
