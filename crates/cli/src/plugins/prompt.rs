// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Terminal-only prompt adapter for plugin configuration.

use std::io::IsTerminal;

use console::Term;
use dialoguer::theme::ColorfulTheme;
use dialoguer::{Input, Select};

use super::*;

pub(crate) fn edit(command: PluginsEditRequest) -> Result<(), CliError> {
    ensure_tty()?;
    let (scope, path) = resolve_edit_target(command)?;
    let mut document = PluginConfigDocument::read(&path)?;
    ensure_observability_component(document.config_mut())?;
    ensure_adaptive_component(document.config_mut())?;
    let mut components = editable_components(document.config())?;
    let mut dynamic_plugins = load_dynamic_plugin_states(&document)?;

    let theme = ColorfulTheme::default();
    crate::banner::print_intro();
    println!(
        "  Editing plugin config at {}",
        single_line_text(&path.display().to_string())
    );
    println!("  Tip: ↑/↓ or j/k to move, PageUp/PageDown to scroll, SPACE/ENTER to select.");
    println!();
    let mut selected_index = 0;
    loop {
        let dynamic_rows = dynamic_plugins
            .iter()
            .map(|plugin| (plugin.label().to_owned(), plugin.menu_summary()))
            .collect::<Vec<_>>();
        let (items, actions) = plugin_menu_items(&components, &dynamic_rows, &path);
        let selection = prompt_menu(&theme, "plugins.toml", &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        if handle_menu_response(
            &theme,
            &mut document,
            &mut components,
            &mut dynamic_plugins,
            &actions,
            selection,
            scope,
        )? == EditLoopControl::Finish
        {
            return Ok(());
        }
    }
}

fn handle_menu_response(
    theme: &ColorfulTheme,
    document: &mut PluginConfigDocument,
    components: &mut [EditableComponent],
    dynamic_plugins: &mut [DynamicPluginEditorState],
    actions: &[MenuAction],
    selection: MenuResponse,
    scope: TargetScope,
) -> Result<EditLoopControl, CliError> {
    match selection {
        MenuResponse::Selected(selection) => handle_menu_action(
            theme,
            document,
            components,
            dynamic_plugins,
            actions.get(selection).copied(),
            scope,
        ),
        MenuResponse::Shortcut(MenuShortcut::Preview, _) => {
            preview_document(document, components, dynamic_plugins)?;
            Ok(EditLoopControl::Continue)
        }
        MenuResponse::Shortcut(MenuShortcut::Save, _) => {
            save_document(document, components, dynamic_plugins, scope)
        }
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
    document: &mut PluginConfigDocument,
    components: &mut [EditableComponent],
    dynamic_plugins: &mut [DynamicPluginEditorState],
    action: Option<MenuAction>,
    scope: TargetScope,
) -> Result<EditLoopControl, CliError> {
    match action {
        Some(MenuAction::EditComponent(component_index)) => {
            if let Some(component) = components.get_mut(component_index) {
                edit_component(theme, component)?;
            }
            Ok(EditLoopControl::Continue)
        }
        Some(MenuAction::EditDynamic(dynamic_index)) => {
            if let Some(plugin) = dynamic_plugins.get_mut(dynamic_index) {
                edit_dynamic_plugin(theme, plugin)?;
            }
            Ok(EditLoopControl::Continue)
        }
        Some(MenuAction::Preview) => {
            preview_document(document, components, dynamic_plugins)?;
            Ok(EditLoopControl::Continue)
        }
        Some(MenuAction::Save) => save_document(document, components, dynamic_plugins, scope),
        Some(MenuAction::Cancel) | None => Err(cancelled_error()),
    }
}

fn edit_component(
    theme: &ColorfulTheme,
    component: &mut EditableComponent,
) -> Result<(), CliError> {
    let mut selected_index = 0;
    loop {
        let (items, actions) = component_menu_items(component);
        let selection = prompt_menu(theme, component.label(), &items, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        match selection {
            MenuResponse::Selected(selected) => match actions.get(selected).copied() {
                Some(ComponentMenuAction::Toggle) => component.toggle_enabled(),
                Some(ComponentMenuAction::EditField(field_index)) => {
                    if let Some(field) = component.fields().get(field_index) {
                        edit_component_field(theme, component, *field)?;
                    }
                }
                Some(ComponentMenuAction::Back) | None => return Ok(()),
            },
            MenuResponse::Shortcut(MenuShortcut::Reset, selected) => {
                reset_component_menu_item(component, actions.get(selected).copied())?;
            }
            MenuResponse::Shortcut(MenuShortcut::Clear, selected) => {
                clear_component_menu_item(component, actions.get(selected).copied())?;
            }
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
            }
            MenuResponse::Cancel => return Ok(()),
        }
    }
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
        EditableComponent::PiiRedaction(state) => {
            edit_config_field(theme, &mut state.config, field)?;
            state.mark_config_touched();
        }
    }
    Ok(())
}

pub(super) fn prompt_menu(
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
        let (rows, columns) = term.size();
        let viewport = menu_viewport(items.len(), selected, usize::from(rows));
        let lines = render_menu_for_size(
            theme,
            prompt,
            items,
            selected,
            usize::from(rows),
            usize::from(columns),
        );
        rendered_lines = lines.len();
        for line in &lines {
            term.write_line(line).map_err(menu_error)?;
        }
        term.flush().map_err(menu_error)?;
        let key = term.read_key().map_err(menu_error)?;
        if let Some(next) =
            menu_selection_after_key(&key, selected, items.len(), viewport.page_size)
        {
            selected = next;
            continue;
        }
        if let Some(response) = menu_response_for_key(&key, selected) {
            clear_menu(&term, rendered_lines)?;
            return Ok(response);
        }
    }
}

fn clear_menu(term: &Term, rendered_lines: usize) -> Result<(), CliError> {
    if rendered_lines > 0 {
        term.clear_last_lines(rendered_lines).map_err(menu_error)?;
    }
    Ok(())
}

pub(super) fn menu_error(error: std::io::Error) -> CliError {
    if matches!(
        error.kind(),
        std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
    ) {
        CliError::Config(PLUGIN_EDIT_CANCELLED_MESSAGE.into())
    } else {
        CliError::Config(format!("plugin editor terminal error: {error}"))
    }
}

pub(super) fn print_editor_help() {
    println!();
    println!(
        "{} {}",
        style("?").yellow(),
        style("Plugin editor keys").bold()
    );
    println!("  {}  move", style("↑/↓ or j/k").cyan());
    println!(
        "  {} move by page or jump to an end",
        style("PageUp/PageDown, Home/End").cyan()
    );
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
    if field.kind == EditorFieldKind::List {
        let item = field.list_item.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its list entries", field.name))
        })?;
        let default = section_field_default(section, *field);
        let mut items = current
            .or_else(|| default.clone())
            .unwrap_or_else(|| json!([]));
        if edit_list_value(
            theme,
            &format!("{}.{}", section.name, field.name),
            &mut items,
            default,
            item,
        )? {
            set_section_field(config, section, field.name, items)?;
        }
        return Ok(());
    }
    if field.kind == EditorFieldKind::StringMap {
        let default = section_field_default(section, *field);
        let mut entries = current
            .or_else(|| default.clone())
            .unwrap_or_else(|| json!({}));
        if edit_string_map_value(
            theme,
            &format!("{}.{}", section.name, field.name),
            &mut entries,
            default,
        )? {
            set_section_field(config, section, field.name, entries)?;
        }
        return Ok(());
    }
    if field.kind == EditorFieldKind::Map {
        let item = field.list_item.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its map values", field.name))
        })?;
        let default = section_field_default(section, *field);
        let mut entries = current
            .or_else(|| default.clone())
            .unwrap_or_else(|| json!({}));
        if edit_map_value(
            theme,
            &format!("{}.{}", section.name, field.name),
            &mut entries,
            default,
            item,
        )? {
            set_section_field(config, section, field.name, entries)?;
        }
        return Ok(());
    }
    if field.kind == EditorFieldKind::TaggedUnion {
        let tagged_union = field.tagged_union.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its variants", field.name))
        })?;
        let default = section_field_default(section, *field);
        match edit_tagged_union_field(
            theme,
            &format!("{}.{}", section.name, field.name),
            current,
            default,
            tagged_union,
        )? {
            TaggedUnionFieldEdit::Set(value) => {
                set_section_field(config, section, field.name, value)?;
            }
            TaggedUnionFieldEdit::Reset => remove_section_field(config, section, field.name)?,
            TaggedUnionFieldEdit::Unchanged => {}
        }
        return Ok(());
    }
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

fn edit_list_value(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    default: Option<Value>,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> Result<bool, CliError> {
    if !value.is_array() {
        *value = default.clone().unwrap_or_else(|| json!([]));
    }
    let original = value.clone();
    let mut selected_index = 0;
    loop {
        let entries = value.as_array().expect("list value is an array");
        let mut menu = vec![MenuItem::new("Add item")];
        menu.extend(entries.iter().enumerate().map(|(index, entry)| {
            MenuItem::new(format!(
                "Edit item {}: {}",
                index + 1,
                editor_item_label(entry, item)
            ))
        }));
        menu.push(MenuItem::new(shortcut_label("Back", "q")));
        let selection = prompt_menu(theme, prompt, &menu, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        match selection {
            MenuResponse::Selected(0) => {
                let mut entry = new_editor_item(theme, item)?;
                let original_entry = entry.clone();
                edit_editor_item(
                    theme,
                    &format!("{prompt}[{}]", entries.len()),
                    &mut entry,
                    item,
                )?;
                if entry != original_entry {
                    value
                        .as_array_mut()
                        .expect("list value is an array")
                        .push(entry);
                }
            }
            MenuResponse::Selected(index) if index <= entries.len() => {
                edit_existing_list_item(theme, prompt, value, index - 1, item)?;
            }
            MenuResponse::Cancel | MenuResponse::Selected(_) => return Ok(*value != original),
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(shortcut @ (MenuShortcut::Reset | MenuShortcut::Clear), _) => {
                *value = collection_shortcut_value(default.as_ref(), json!([]), shortcut)
            }
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
            }
        }
    }
}

fn edit_string_map_value(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    default: Option<Value>,
) -> Result<bool, CliError> {
    if !value.is_object() {
        *value = default.clone().unwrap_or_else(|| json!({}));
    }
    let original = value.clone();
    let mut selected_index = 0;
    loop {
        let entries = value.as_object().expect("string map value is an object");
        let keys = entries.keys().cloned().collect::<Vec<_>>();
        let mut menu = vec![MenuItem::new("Add entry")];
        menu.extend(keys.iter().map(|key| {
            MenuItem::new(format!(
                "Edit {key}: {}",
                entries.get(key).map(display_value).unwrap_or_default()
            ))
        }));
        menu.push(MenuItem::new(shortcut_label("Back", "q")));
        let selection = prompt_menu(theme, prompt, &menu, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        match selection {
            MenuResponse::Selected(0) => {
                let key: String = Input::with_theme(theme)
                    .with_prompt("Entry key")
                    .interact_text()
                    .map_err(editor_error)?;
                if key.trim().is_empty() {
                    println!("  Entry key must not be empty.");
                    continue;
                }
                let key = key.trim().to_owned();
                if string_map_entry_exists(value, &key) {
                    println!("  Entry already exists; select it to edit.");
                    continue;
                }
                let entry: String = Input::with_theme(theme)
                    .with_prompt("Entry value")
                    .interact_text()
                    .map_err(editor_error)?;
                value
                    .as_object_mut()
                    .expect("string map value is an object")
                    .insert(key, Value::String(entry));
            }
            MenuResponse::Selected(index) if index <= keys.len() => {
                edit_existing_string_map_entry(theme, prompt, value, &keys[index - 1])?;
            }
            MenuResponse::Cancel | MenuResponse::Selected(_) => return Ok(*value != original),
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(shortcut @ (MenuShortcut::Reset | MenuShortcut::Clear), _) => {
                *value = collection_shortcut_value(default.as_ref(), json!({}), shortcut)
            }
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
            }
        }
    }
}

fn edit_map_value(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    default: Option<Value>,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> Result<bool, CliError> {
    if !value.is_object() {
        *value = default.clone().unwrap_or_else(|| json!({}));
    }
    let original = value.clone();
    let mut selected_index = 0;
    loop {
        let entries = value.as_object().expect("map value is an object");
        let keys = entries.keys().cloned().collect::<Vec<_>>();
        let mut menu = vec![MenuItem::new("Add entry")];
        menu.extend(keys.iter().map(|key| {
            MenuItem::new(format!(
                "Edit {key}: {}",
                entries
                    .get(key)
                    .map(|entry| editor_item_label(entry, item))
                    .unwrap_or_default()
            ))
        }));
        menu.push(MenuItem::new(shortcut_label("Back", "q")));
        let selection = prompt_menu(theme, prompt, &menu, selected_index)?;
        if let Some(selected) = menu_response_index(&selection) {
            selected_index = selected;
        }
        match selection {
            MenuResponse::Selected(0) => {
                let key: String = Input::with_theme(theme)
                    .with_prompt("Entry key")
                    .interact_text()
                    .map_err(editor_error)?;
                let key = key.trim();
                if key.is_empty() {
                    println!("  Entry key must not be empty.");
                    continue;
                }
                if string_map_entry_exists(value, key) {
                    println!("  Entry already exists; select it to edit.");
                    continue;
                }
                let mut entry = new_editor_item(theme, item)?;
                edit_editor_item(theme, &format!("{prompt}.{key}"), &mut entry, item)?;
                value
                    .as_object_mut()
                    .expect("map value is an object")
                    .insert(key.to_owned(), entry);
            }
            MenuResponse::Selected(index) if index <= keys.len() => {
                edit_existing_map_entry(theme, prompt, value, &keys[index - 1], item)?;
            }
            MenuResponse::Cancel | MenuResponse::Selected(_) => return Ok(*value != original),
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(shortcut @ (MenuShortcut::Reset | MenuShortcut::Clear), _) => {
                *value = collection_shortcut_value(default.as_ref(), json!({}), shortcut)
            }
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
            }
        }
    }
}

fn edit_existing_map_entry(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    key: &str,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> Result<(), CliError> {
    let actions = [
        MenuItem::new("Edit value"),
        MenuItem::new("Remove entry"),
        MenuItem::new(shortcut_label("Back", "q")),
    ];
    match prompt_menu(theme, &format!("{prompt}.{key}"), &actions, 0)? {
        MenuResponse::Selected(0) => {
            if let Some(entry) = value
                .as_object_mut()
                .and_then(|entries| entries.get_mut(key))
            {
                edit_editor_item(theme, &format!("{prompt}.{key}"), entry, item)?;
            }
        }
        MenuResponse::Selected(1) | MenuResponse::Shortcut(MenuShortcut::Clear, _) => {
            value
                .as_object_mut()
                .expect("map value is an object")
                .remove(key);
        }
        _ => {}
    }
    Ok(())
}

fn edit_existing_string_map_entry(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    key: &str,
) -> Result<(), CliError> {
    let actions = [
        MenuItem::new("Edit value"),
        MenuItem::new("Remove entry"),
        MenuItem::new(shortcut_label("Back", "q")),
    ];
    match prompt_menu(theme, &format!("{prompt}.{key}"), &actions, 0)? {
        MenuResponse::Selected(0) => {
            let current = value
                .as_object()
                .and_then(|entries| entries.get(key))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let entry: String = Input::with_theme(theme)
                .with_prompt("Entry value")
                .with_initial_text(current)
                .interact_text()
                .map_err(editor_error)?;
            value
                .as_object_mut()
                .expect("string map value is an object")
                .insert(key.to_owned(), Value::String(entry));
        }
        MenuResponse::Selected(1) | MenuResponse::Shortcut(MenuShortcut::Clear, _) => {
            value
                .as_object_mut()
                .expect("string map value is an object")
                .remove(key);
        }
        _ => {}
    }
    Ok(())
}

fn edit_existing_list_item(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    index: usize,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> Result<(), CliError> {
    let actions = [
        MenuItem::new("Edit item"),
        MenuItem::new("Remove item"),
        MenuItem::new(shortcut_label("Back", "q")),
    ];
    match prompt_menu(theme, &format!("{prompt}[{}]", index + 1), &actions, 0)? {
        MenuResponse::Selected(0) => {
            if let Some(entry) = value
                .as_array_mut()
                .and_then(|entries| entries.get_mut(index))
            {
                edit_editor_item(theme, &format!("{prompt}[{}]", index + 1), entry, item)?;
            }
        }
        MenuResponse::Selected(1) | MenuResponse::Shortcut(MenuShortcut::Clear, _) => {
            value
                .as_array_mut()
                .expect("list value is an array")
                .remove(index);
        }
        _ => {}
    }
    Ok(())
}

fn new_editor_item(
    theme: &ColorfulTheme,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> Result<Value, CliError> {
    if let Some(tagged_union) = item.tagged_union {
        return new_tagged_union_value(theme, tagged_union);
    }
    Ok(item.default.map(|default| default()).unwrap_or(Value::Null))
}

fn edit_editor_item(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    item: &nemo_relay::config_editor::EditorListItemSpec,
) -> Result<(), CliError> {
    if let Some(tagged_union) = item.tagged_union {
        return edit_tagged_union_payload(theme, prompt, value, tagged_union);
    }

    match item.kind {
        EditorFieldKind::Section => {
            let schema = item
                .schema
                .ok_or_else(|| CliError::Config("list item has no schema".into()))?(
            );
            edit_value_section(theme, prompt, value, schema, None)?;
        }
        EditorFieldKind::List => {
            let nested = item.list_item.ok_or_else(|| {
                CliError::Config("nested list item has no entry description".into())
            })?;
            let _ = edit_list_value(theme, prompt, value, None, nested)?;
        }
        EditorFieldKind::StringMap => {
            let _ = edit_string_map_value(theme, prompt, value, None)?;
        }
        EditorFieldKind::Map => {
            let nested = item.list_item.ok_or_else(|| {
                CliError::Config("nested map item has no entry description".into())
            })?;
            let _ = edit_map_value(theme, prompt, value, None, nested)?;
        }
        kind => {
            let field = EditorFieldSpec {
                name: "item",
                label: "item",
                kind,
                enum_values: &[],
                optional: false,
                nested_schema: None,
                nested_default: None,
                list_item: None,
                tagged_union: None,
            };
            *value = prompt_value(theme, &field, Some(value))?;
        }
    }
    Ok(())
}

fn select_tagged_union_variant(
    theme: &ColorfulTheme,
    tagged_union: &nemo_relay::config_editor::EditorTaggedUnionSpec,
) -> Result<usize, CliError> {
    if tagged_union.variants.is_empty() {
        return Err(CliError::Config("tagged union has no variants".into()));
    }
    Select::with_theme(theme)
        .with_prompt("Variant type")
        .items(
            &tagged_union
                .variants
                .iter()
                .map(|variant| variant.label)
                .collect::<Vec<_>>(),
        )
        .default(0)
        .interact()
        .map_err(editor_error)
}

fn new_tagged_union_value(
    theme: &ColorfulTheme,
    tagged_union: &nemo_relay::config_editor::EditorTaggedUnionSpec,
) -> Result<Value, CliError> {
    tagged_union_variant_value(
        tagged_union,
        select_tagged_union_variant(theme, tagged_union)?,
    )
}

fn edit_tagged_union_payload(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    tagged_union: &nemo_relay::config_editor::EditorTaggedUnionSpec,
) -> Result<(), CliError> {
    if !value.is_object() {
        *value = new_tagged_union_value(theme, tagged_union)?;
    }
    let tag = value
        .get(tagged_union.discriminator)
        .and_then(Value::as_str)
        .ok_or_else(|| CliError::Config("tagged union has no discriminator value".into()))?;
    let variant = tagged_union
        .variants
        .iter()
        .find(|variant| variant.tag == tag)
        .ok_or_else(|| CliError::Config(format!("unknown tagged union type {tag:?}")))?;
    edit_value_section(theme, prompt, value, (variant.schema)(), None)?;
    Ok(())
}

fn edit_tagged_union_field(
    theme: &ColorfulTheme,
    prompt: &str,
    current: Option<Value>,
    default: Option<Value>,
    tagged_union: &nemo_relay::config_editor::EditorTaggedUnionSpec,
) -> Result<TaggedUnionFieldEdit, CliError> {
    let mut state = TaggedUnionFieldState::new(current, default);
    loop {
        let actions = [
            MenuItem::new("Edit fields"),
            MenuItem::new("Change variant"),
            MenuItem::new(shortcut_label(
                "Reset to default/none",
                "r, Backspace, Delete",
            )),
            MenuItem::new(shortcut_label("Back", "q")),
        ];
        match prompt_menu(
            theme,
            &format!("{prompt}, current {}", display_value(state.value())),
            &actions,
            0,
        )? {
            MenuResponse::Selected(0) => {
                edit_tagged_union_payload(theme, prompt, state.value_mut(), tagged_union)?;
            }
            MenuResponse::Selected(1) => {
                state.change_variant(
                    tagged_union,
                    select_tagged_union_variant(theme, tagged_union)?,
                )?;
                edit_tagged_union_payload(theme, prompt, state.value_mut(), tagged_union)?;
            }
            MenuResponse::Selected(2)
            | MenuResponse::Shortcut(MenuShortcut::Reset | MenuShortcut::Clear, _) => {
                return Ok(state.reset());
            }
            MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
            MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
                println!("  Preview and save are available from the main plugins.toml menu.");
            }
            MenuResponse::Cancel | MenuResponse::Selected(_) => {
                return Ok(state.finish());
            }
        }
    }
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

    if field.kind == EditorFieldKind::List {
        let item = field.list_item.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its list entries", field.name))
        })?;
        let default = default_config_field_value::<T>(field).or_else(|| field.default_value());
        let mut items = config_field_value(config, field.name)?
            .or_else(|| default.clone())
            .unwrap_or_else(|| json!([]));
        if edit_list_value(theme, field.name, &mut items, default, item)? {
            set_struct_field(config, field.name, items)?;
        }
        return Ok(());
    }

    if field.kind == EditorFieldKind::StringMap {
        let default = default_config_field_value::<T>(field).or_else(|| field.default_value());
        let mut entries = config_field_value(config, field.name)?
            .or_else(|| default.clone())
            .unwrap_or_else(|| json!({}));
        if edit_string_map_value(theme, field.name, &mut entries, default)? {
            set_struct_field(config, field.name, entries)?;
        }
        return Ok(());
    }

    if field.kind == EditorFieldKind::Map {
        let item = field.list_item.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its map values", field.name))
        })?;
        let default = default_config_field_value::<T>(field).or_else(|| field.default_value());
        let mut entries = config_field_value(config, field.name)?
            .or_else(|| default.clone())
            .unwrap_or_else(|| json!({}));
        if edit_map_value(theme, field.name, &mut entries, default, item)? {
            set_struct_field(config, field.name, entries)?;
        }
        return Ok(());
    }

    if field.kind == EditorFieldKind::TaggedUnion {
        let tagged_union = field.tagged_union.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its variants", field.name))
        })?;
        let default = default_config_field_value::<T>(field).or_else(|| field.default_value());
        match edit_tagged_union_field(
            theme,
            field.name,
            config_field_value(config, field.name)?,
            default,
            tagged_union,
        )? {
            TaggedUnionFieldEdit::Set(value) => set_struct_field(config, field.name, value)?,
            TaggedUnionFieldEdit::Reset => reset_config_field(config, field)?,
            TaggedUnionFieldEdit::Unchanged => {}
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

fn edit_selected_value_item(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    default: Option<&Value>,
    selection: usize,
) -> Result<bool, CliError> {
    if let Some(field) = schema.fields.get(selection) {
        edit_value_field(theme, prompt, value, schema, *field, default)?;
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
    schema: &nemo_relay::config_editor::EditorSchema,
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

    if field.kind == EditorFieldKind::List {
        let item = field.list_item.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its list entries", field.name))
        })?;
        let field_default = value_field_default(default, field);
        let mut items = value_field_value(value, field.name)
            .or_else(|| field_default.clone())
            .unwrap_or_else(|| json!([]));
        if edit_list_value(
            theme,
            &format!("{prompt}.{}", field.name),
            &mut items,
            field_default,
            item,
        )? {
            set_value_field(value, field.name, items);
        }
        return Ok(());
    }

    if field.kind == EditorFieldKind::StringMap {
        let field_default = value_field_default(default, field);
        let mut entries = value_field_value(value, field.name)
            .or_else(|| field_default.clone())
            .unwrap_or_else(|| json!({}));
        if edit_string_map_value(
            theme,
            &format!("{prompt}.{}", field.name),
            &mut entries,
            field_default,
        )? {
            set_value_field(value, field.name, entries);
        }
        return Ok(());
    }

    if field.kind == EditorFieldKind::Map {
        let item = field.list_item.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its map values", field.name))
        })?;
        let field_default = value_field_default(default, field);
        let mut entries = value_field_value(value, field.name)
            .or_else(|| field_default.clone())
            .unwrap_or_else(|| json!({}));
        if edit_map_value(
            theme,
            &format!("{prompt}.{}", field.name),
            &mut entries,
            field_default,
            item,
        )? {
            set_value_field(value, field.name, entries);
        }
        return Ok(());
    }

    if field.kind == EditorFieldKind::DiscriminatedSection {
        edit_discriminated_value_section(theme, prompt, value, field)?;
        return Ok(());
    }

    if field.kind == EditorFieldKind::TaggedUnion {
        let tagged_union = field.tagged_union.ok_or_else(|| {
            CliError::Config(format!("{} does not describe its variants", field.name))
        })?;
        let field_default = value_field_default(default, field);
        match edit_tagged_union_field(
            theme,
            &format!("{prompt}.{}", field.name),
            value_field_value(value, field.name),
            field_default.clone(),
            tagged_union,
        )? {
            TaggedUnionFieldEdit::Set(tagged_value) => {
                set_value_field(value, field.name, tagged_value);
            }
            TaggedUnionFieldEdit::Reset => reset_value_field(value, field, default),
            TaggedUnionFieldEdit::Unchanged => {}
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
            let changed = current.as_ref() != Some(&field_value);
            set_value_field(value, field.name, field_value);
            if changed {
                reset_discriminated_value_sections(value, schema, field.name);
            }
        }
        MenuResponse::Selected(1)
        | MenuResponse::Shortcut(MenuShortcut::Reset | MenuShortcut::Clear, _) => {
            let previous = value_field_value(value, field.name);
            reset_value_field(value, field, default);
            if value_field_value(value, field.name) != previous {
                reset_discriminated_value_sections(value, schema, field.name);
            }
        }
        MenuResponse::Shortcut(MenuShortcut::Help, _) => print_editor_help(),
        MenuResponse::Shortcut(MenuShortcut::Preview | MenuShortcut::Save, _) => {
            println!("  Preview and save are available from the main plugins.toml menu.");
        }
        _ => {}
    }
    Ok(())
}

fn edit_discriminated_value_section(
    theme: &ColorfulTheme,
    prompt: &str,
    value: &mut Value,
    field: EditorFieldSpec,
) -> Result<(), CliError> {
    let tagged_union = field.tagged_union.ok_or_else(|| {
        CliError::Config(format!("{} does not describe its variants", field.name))
    })?;
    let tag = value
        .get(tagged_union.discriminator)
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            tagged_union
                .variants
                .first()
                .map(|variant| variant.tag)
                .unwrap_or("")
        });
    let variant = tagged_union
        .variants
        .iter()
        .find(|variant| variant.tag == tag)
        .ok_or_else(|| CliError::Config(format!("unknown {} variant {tag:?}", field.name)))?;
    let mut nested_value =
        value_field_value(value, field.name).unwrap_or_else(|| (variant.default)());
    if edit_value_section(
        theme,
        &format!("{prompt}.{}", field.name),
        &mut nested_value,
        (variant.schema)(),
        Some((variant.default)()),
    )? {
        set_value_field(value, field.name, nested_value);
    }
    Ok(())
}

fn reset_discriminated_value_sections(
    value: &mut Value,
    schema: &nemo_relay::config_editor::EditorSchema,
    discriminator: &str,
) {
    let tag = value
        .get(discriminator)
        .and_then(Value::as_str)
        .map(str::to_owned);
    for field in schema.fields {
        let Some(tagged_union) = field.tagged_union else {
            continue;
        };
        if field.kind != EditorFieldKind::DiscriminatedSection
            || tagged_union.discriminator != discriminator
        {
            continue;
        }
        if let Some(variant) = tagged_union
            .variants
            .iter()
            .find(|variant| Some(variant.tag) == tag.as_deref())
        {
            set_value_field(value, field.name, (variant.default)());
        }
    }
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
        EditorFieldKind::Enum | EditorFieldKind::IntegerEnum => {
            let values = field.enum_values;
            let default_idx = editor_enum_default_index(field, current);
            let idx = Select::with_theme(theme)
                .with_prompt(field.label)
                .items(values)
                .default(default_idx)
                .interact()
                .map_err(editor_error)?;
            Ok(editor_enum_value(field, idx))
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
        EditorFieldKind::List
        | EditorFieldKind::Map
        | EditorFieldKind::TaggedUnion
        | EditorFieldKind::DiscriminatedSection => Err(CliError::Config(format!(
            "{} is a structured value and cannot be edited as a scalar",
            field.name
        ))),
    }
}

pub(super) fn editor_error(err: dialoguer::Error) -> CliError {
    match err {
        dialoguer::Error::IO(io_err)
            if matches!(
                io_err.kind(),
                std::io::ErrorKind::Interrupted | std::io::ErrorKind::UnexpectedEof
            ) =>
        {
            CliError::Config(PLUGIN_EDIT_CANCELLED_MESSAGE.into())
        }
        other => CliError::Config(format!("plugin edit error: {other}")),
    }
}
