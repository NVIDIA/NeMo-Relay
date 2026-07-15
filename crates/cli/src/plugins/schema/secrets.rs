// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Write-only secret discovery, redaction, and edit-token restoration.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum SecretSegment {
    Property(String),
    Any,
    Pattern(SecretPropertyPattern),
    UnmatchedProperties(SecretUnmatchedProperties),
    Index(usize),
    Tail(usize),
}

#[derive(Debug, Clone)]
pub(super) struct SecretPropertyPattern {
    pub(super) source: String,
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
pub(super) struct SecretUnmatchedProperties {
    properties: Vec<String>,
    patterns: Vec<SecretPropertyPattern>,
}

impl SecretUnmatchedProperties {
    pub(super) fn matches(&self, property: &str) -> bool {
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
pub(super) struct SecretPattern(pub(super) Vec<SecretSegment>);

impl SecretPattern {
    pub(super) fn redact(&self, value: &mut Value, offset: usize) {
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

    pub(super) fn redact_for_edit(
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

    pub(super) fn visit_matching_values(
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
                self.visit_property(value, property, offset, visit)
            }
            SecretSegment::Any => self.visit_any(value, offset, visit),
            SecretSegment::Pattern(pattern) => {
                self.visit_object_matches(value, offset, visit, |key| {
                    pattern_matches(pattern, key)
                });
            }
            SecretSegment::UnmatchedProperties(selector) => {
                self.visit_object_matches(value, offset, visit, |key| selector.matches(key));
            }
            SecretSegment::Index(index) => self.visit_index(value, *index, offset, visit),
            SecretSegment::Tail(start) => self.visit_tail(value, *start, offset, visit),
        }
    }

    fn visit_property(
        &self,
        value: &mut Value,
        property: &str,
        offset: usize,
        visit: &mut impl FnMut(&mut Value),
    ) {
        if let Some(child) = value.get_mut(property) {
            self.visit_matching_values(child, offset + 1, visit);
        }
    }

    fn visit_any(&self, value: &mut Value, offset: usize, visit: &mut impl FnMut(&mut Value)) {
        match value {
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
        }
    }

    fn visit_object_matches(
        &self,
        value: &mut Value,
        offset: usize,
        visit: &mut impl FnMut(&mut Value),
        matches: impl Fn(&str) -> bool,
    ) {
        if let Value::Object(object) = value {
            for (key, child) in object {
                if matches(key) {
                    self.visit_matching_values(child, offset + 1, visit);
                }
            }
        }
    }

    fn visit_index(
        &self,
        value: &mut Value,
        index: usize,
        offset: usize,
        visit: &mut impl FnMut(&mut Value),
    ) {
        if let Some(child) = value.get_mut(index) {
            self.visit_matching_values(child, offset + 1, visit);
        }
    }

    fn visit_tail(
        &self,
        value: &mut Value,
        start: usize,
        offset: usize,
        visit: &mut impl FnMut(&mut Value),
    ) {
        if let Value::Array(values) = value {
            for child in values.iter_mut().skip(start) {
                self.visit_matching_values(child, offset + 1, visit);
            }
        }
    }

    pub(super) fn applies_below(&self, path: &[String]) -> bool {
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

    pub(super) fn matches_instance_path(&self, path: &[SecretInstanceSegment]) -> bool {
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
pub(super) enum SecretInstanceSegment {
    Property(String),
    Index(usize),
}

pub(super) fn pattern_matches(pattern: &SecretPropertyPattern, property: &str) -> bool {
    pattern.matcher.is_match(property)
}

pub(super) fn collect_string_values(value: &Value, output: &mut HashSet<String>) {
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

pub(super) fn next_secret_token(
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

pub(super) fn restore_secret_tokens(
    value: &Value,
    secrets: &SecretEditValues,
) -> Result<Value, String> {
    pub(super) fn restore(
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

pub(super) fn discover_secret_patterns(
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

pub(super) fn discover_secret_patterns_in_object(
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

pub(super) fn discover_named_secret_patterns(
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

pub(super) fn collect_secret_pattern_schemas(
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

pub(super) fn discover_additional_secret_patterns(
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

pub(super) fn discover_pattern_property_secret_patterns(
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

pub(super) fn discover_item_secret_patterns(
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

pub(super) fn discover_prefix_item_secret_patterns(
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

pub(super) fn discover_all_of_secret_patterns(
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

pub(super) fn reject_array_applicator_secret_patterns(
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

pub(super) fn reject_value_applicator_secret_patterns(
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

pub(super) fn discover_contains_secret_patterns(
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

pub(super) fn reject_object_applicator_secret_patterns(
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

pub(super) fn reject_write_only_under_applicator(
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

pub(super) fn push_pointer(pointer: &str, segment: &str) -> String {
    format!("{pointer}/{}", escape_pointer(segment))
}

pub(super) fn escape_pointer(segment: &str) -> String {
    segment.replace('~', "~0").replace('/', "~1")
}
