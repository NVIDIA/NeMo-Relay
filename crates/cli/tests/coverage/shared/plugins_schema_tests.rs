// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::PathBuf};

use serde_json::json;
use tempfile::tempdir;

use super::*;

const DRAFT7: &str = "http://json-schema.org/draft-07/schema#";
const DRAFT2020: &str = "https://json-schema.org/draft/2020-12/schema";

fn write_schema(schema: &Value) -> (tempfile::TempDir, PathBuf) {
    let directory = tempdir().expect("create temp directory");
    let path = directory.path().join("config.schema.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(schema).expect("serialize schema"),
    )
    .expect("write schema");
    (directory, path)
}

fn load(schema: &Value) -> PluginConfigSchema {
    let (_directory, path) = write_schema(schema);
    PluginConfigSchema::load("acme.example", path).expect("load schema")
}

fn secret_paths(schema: &PluginConfigSchema) -> Vec<String> {
    schema
        .secret_patterns
        .iter()
        .map(|pattern| {
            let mut pointer = String::new();
            for segment in &pattern.0 {
                pointer.push('/');
                match segment {
                    SecretSegment::Property(property) => {
                        pointer.push_str(&escape_pointer(property));
                    }
                    SecretSegment::Any => pointer.push('*'),
                    SecretSegment::Pattern(pattern) => {
                        pointer.push_str(&format!("~pattern({})", pattern.source));
                    }
                    SecretSegment::UnmatchedProperties(_) => pointer.push_str("~additional"),
                    SecretSegment::Index(index) => pointer.push_str(&index.to_string()),
                    SecretSegment::Tail(start) => pointer.push_str(&format!("~tail({start})")),
                }
            }
            pointer
        })
        .collect()
}

#[test]
fn loads_supported_drafts_and_requires_object_root() {
    for dialect in [DRAFT7, DRAFT2020] {
        let loaded = load(&json!({"$schema": dialect, "type": "object"}));
        assert_eq!(loaded.plugin_id, "acme.example");
        assert!(loaded.fields().is_empty());
    }

    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT2020,
        "type": "string"
    }));
    let error = PluginConfigSchema::load("acme.bad", &path).expect_err("reject string root");
    let message = error.to_string();
    assert!(message.contains("acme.bad"), "{message}");
    assert!(
        message.contains(path.to_string_lossy().as_ref()),
        "{message}"
    );
    assert!(message.contains("root schema"), "{message}");
}

#[test]
fn requires_supported_explicit_dialect() {
    for schema in [
        json!({"type": "object"}),
        json!({"$schema": 7, "type": "object"}),
        json!({"$schema": "https://json-schema.org/draft/2019-09/schema", "type": "object"}),
    ] {
        let (_directory, path) = write_schema(&schema);
        let error = PluginConfigSchema::load("acme.bad", path).expect_err("reject dialect");
        assert!(error.to_string().contains("$schema"));
    }
}

#[test]
fn rejects_invalid_schema_and_external_references_recursively() {
    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT7,
        "type": 7
    }));
    let error = PluginConfigSchema::load("acme.bad", path).expect_err("reject invalid schema");
    assert!(error.to_string().contains("schema is invalid"));

    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "$defs": {
            "remote": {"$ref": "https://example.com/schema.json"}
        }
    }));
    let error = PluginConfigSchema::load("acme.bad", path).expect_err("reject external ref");
    let message = error.to_string();
    assert!(message.contains("local fragment"), "{message}");
    assert!(message.contains("/$defs/remote/$ref"), "{message}");

    for schema in [
        json!({
            "$schema": DRAFT2020,
            "type": "object",
            "$dynamicRef": "#config"
        }),
        json!({
            "$schema": DRAFT2020,
            "type": "object",
            "$defs": {"config": {"$dynamicAnchor": "config", "type": "object"}}
        }),
    ] {
        let (_directory, path) = write_schema(&schema);
        let error = PluginConfigSchema::load("acme.bad", path)
            .expect_err("reject unsupported dynamic references");
        assert!(error.to_string().contains("dynamic references"));
    }

    load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "default": {"$ref": "https://example.com/literal-data"},
                "examples": [{"$ref": "https://example.com/also-literal"}]
            }
        }
    }));
}

#[test]
fn resolves_local_definitions_for_root_and_fields() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "$ref": "#/$defs/config",
        "$defs": {
            "config": {
                "type": "object",
                "properties": {
                    "endpoint": {"$ref": "#/$defs/nonEmpty"}
                }
            },
            "nonEmpty": {"type": "string", "minLength": 1}
        }
    }));
    assert_eq!(loaded.fields().len(), 1);
    assert!(matches!(
        loaded.fields()[0].kind,
        DynamicConfigFieldKind::String { secret: false }
    ));
    loaded
        .validate(&json!({"endpoint": "relay"}))
        .expect("valid config");
}

#[test]
fn resolves_percent_encoded_local_references() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "$ref": "#%2F$defs%2Fconfig",
        "$defs": {
            "config": {
                "type": "object",
                "properties": {
                    "anchored": {"$ref": "#secret%2Danchor"},
                    "slash": {"$ref": "#/$defs/a%7E1b"},
                    "space": {"$ref": "#/$defs/a%20b"}
                }
            },
            "a b": {"type": "string"},
            "a/b": {"type": "string"},
            "anchored": {"$anchor": "secret-anchor", "type": "string"}
        }
    }));

    assert_eq!(loaded.fields().len(), 3);
    assert!(
        loaded
            .fields()
            .iter()
            .all(|field| matches!(field.kind, DynamicConfigFieldKind::String { secret: false }))
    );
}

#[test]
fn resolves_fragments_within_the_active_nested_schema_resource() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "$defs": {
            "rootValue": {"$anchor": "value", "type": "integer"},
            "child": {
                "$id": "child.json",
                "type": "object",
                "$defs": {
                    "text": {"$anchor": "value", "type": "string"}
                },
                "properties": {
                    "name": {"$ref": "#value"}
                }
            }
        },
        "properties": {
            "child": {"$ref": "#/$defs/child"}
        }
    }));

    let child = loaded
        .fields()
        .iter()
        .find(|field| field.key == "child")
        .expect("child field");
    assert!(matches!(child.kind, DynamicConfigFieldKind::Object { .. }));
    loaded
        .validate(&json!({"child": {"name": "relay"}}))
        .expect("nested anchor resolves within child resource");
}

#[test]
fn canonicalizes_reference_fragments_and_rejects_malformed_encoding() {
    let schema = json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "cycle": {"$ref": "#/$defs/a b"}
        },
        "$defs": {
            "a b": {"$ref": "#/$defs/a%20b"}
        }
    });
    let mut references = HashSet::new();
    let error = resolve_schema(&schema, &schema["properties"]["cycle"], &mut references)
        .expect_err("encoded and decoded references identify the same cycle");
    assert!(error.to_string().contains("cyclic"));

    for reference in ["#/$defs/bad%2", "#/$defs/bad%GG", "#/$defs/bad%FF"] {
        let error = decode_reference_fragment(reference).expect_err("reject malformed fragment");
        assert!(error.to_string().contains("invalid fragment"));
    }
}

#[test]
fn maps_native_nested_map_and_raw_controls() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "required": ["enabled"],
        "properties": {
            "array": {"type": "array", "items": {"type": "string"}},
            "choice": {"type": "string", "enum": ["one", "two"]},
            "count": {"type": "integer"},
            "enabled": {"type": "boolean", "title": "Enabled", "default": true},
            "free": {"type": "object"},
            "labels": {"type": "object", "additionalProperties": {"type": "string"}},
            "nested": {
                "type": "object",
                "properties": {"ratio": {"type": "number", "description": "Weight"}}
            },
            "secret": {"type": "string", "writeOnly": true},
            "union": {"oneOf": [{"type": "string"}, {"type": "number"}]}
        }
    }));
    let field = |key: &str| {
        loaded
            .fields()
            .iter()
            .find(|field| field.key == key)
            .unwrap()
    };

    assert!(matches!(
        field("array").kind,
        DynamicConfigFieldKind::RawJson
    ));
    assert!(matches!(
        field("free").kind,
        DynamicConfigFieldKind::RawJson
    ));
    assert!(matches!(
        field("union").kind,
        DynamicConfigFieldKind::RawJson
    ));
    assert!(matches!(
        field("count").kind,
        DynamicConfigFieldKind::Integer
    ));
    assert!(matches!(
        field("labels").kind,
        DynamicConfigFieldKind::StringMap
    ));
    assert!(matches!(
        field("secret").kind,
        DynamicConfigFieldKind::String { secret: true }
    ));
    assert_eq!(field("enabled").title, "Enabled");
    assert_eq!(field("enabled").default, Some(json!(true)));
    assert!(field("enabled").required);
    assert!(matches!(
        field("choice").kind,
        DynamicConfigFieldKind::StringEnum { ref options, secret: false }
            if options == &["one", "two"]
    ));
    assert!(matches!(
        field("nested").kind,
        DynamicConfigFieldKind::Object { ref fields }
            if fields.len() == 1
                && fields[0].key == "ratio"
                && fields[0].description.as_deref() == Some("Weight")
                && matches!(fields[0].kind, DynamicConfigFieldKind::Number)
    ));
    assert!(loaded.editor().title.is_none());
}

#[test]
fn applies_partial_explicit_order_then_alphabetical_fallback() {
    let loaded = load(&json!({
        "$schema": DRAFT7,
        "type": "object",
        "x-nemo-relay-order": ["zeta", "middle"],
        "properties": {
            "zeta": {"type": "string"},
            "alpha": {"type": "string"},
            "middle": {"type": "string"},
            "beta": {"type": "string"}
        }
    }));
    assert_eq!(
        loaded
            .fields()
            .iter()
            .map(|field| field.key.as_str())
            .collect::<Vec<_>>(),
        ["zeta", "middle", "alpha", "beta"]
    );
}

#[test]
fn rejects_malformed_explicit_order() {
    for order in [
        json!("alpha"),
        json!(["missing"]),
        json!(["alpha", "alpha"]),
        json!([1]),
    ] {
        let (_directory, path) = write_schema(&json!({
            "$schema": DRAFT2020,
            "type": "object",
            "x-nemo-relay-order": order,
            "properties": {"alpha": {"type": "string"}}
        }));
        let error = PluginConfigSchema::load("acme.bad", path).expect_err("reject order");
        assert!(error.to_string().contains("x-nemo-relay-order"));
    }
}

#[test]
fn validation_error_names_plugin_and_instance_pointer() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "service": {
                "type": "object",
                "properties": {"port": {"type": "integer", "minimum": 1}}
            }
        }
    }));
    let error = loaded
        .validate(&json!({"service": {"port": 0}}))
        .expect_err("reject invalid config");
    let message = error.to_string();
    assert!(message.contains("acme.example"), "{message}");
    assert!(message.contains("/service/port"), "{message}");
}

#[test]
fn validation_errors_mask_secret_instance_values() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "token": {
                "type": "string",
                "writeOnly": true,
                "pattern": "^allowed$"
            }
        }
    }));
    let error = loaded
        .validate(&json!({"token": "do-not-print-this-secret"}))
        .expect_err("reject invalid secret");
    let message = error.to_string();

    assert!(message.contains("/token"), "{message}");
    assert!(message.contains(REDACTED), "{message}");
    assert!(!message.contains("do-not-print-this-secret"), "{message}");
}

#[test]
fn recursively_discovers_and_redacts_write_only_strings() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "token": {"$ref": "#/$defs/secret"},
            "nested": {
                "type": "object",
                "properties": {"password": {"type": "string", "writeOnly": true}}
            },
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"key": {"type": "string", "writeOnly": true}}
                }
            }
        },
        "$defs": {"secret": {"type": "string", "writeOnly": true}}
    }));
    assert_eq!(
        secret_paths(&loaded),
        vec![
            "/nested/password".to_owned(),
            "/records/*/key".to_owned(),
            "/token".to_owned()
        ]
    );
    let config = json!({
        "token": "top-secret",
        "nested": {"password": "hunter2", "visible": "ok"},
        "records": [{"key": "one"}, {"key": "two"}]
    });
    assert_eq!(
        loaded.redact(&config),
        json!({
            "token": REDACTED,
            "nested": {"password": REDACTED, "visible": "ok"},
            "records": [{"key": REDACTED}, {"key": REDACTED}]
        })
    );
    assert_eq!(config["token"], "top-secret", "redaction must clone");

    let (redacted, secrets) = loaded.redact_for_edit(&config);
    assert_eq!(
        loaded
            .restore_edit_secrets(&redacted, &secrets)
            .expect("restore original secrets"),
        config
    );

    let mut replacement = redacted;
    replacement["token"] = json!("replacement");
    replacement["nested"]
        .as_object_mut()
        .unwrap()
        .remove("password");
    replacement["records"].as_array_mut().unwrap().swap(0, 1);
    let restored = loaded
        .restore_edit_secrets(&replacement, &secrets)
        .expect("preserve reordered array secrets");
    assert_eq!(restored["token"], json!("replacement"));
    assert!(restored["nested"].get("password").is_none());
    assert_eq!(restored["records"], json!([{"key": "two"}, {"key": "one"}]));

    let (redacted, secrets) = loaded.redact_for_edit(&config);
    let mut moved = redacted.clone();
    let token = redacted["token"].clone();
    moved["visible"] = token;
    let error = loaded
        .restore_edit_secrets(&moved, &secrets)
        .expect_err("reject token copied to a non-secret field");
    assert!(
        error
            .to_string()
            .contains("schema-declared secret location")
    );

    let mut duplicated = redacted;
    duplicated["records"][1]["key"] = duplicated["records"][0]["key"].clone();
    let error = loaded
        .restore_edit_secrets(&duplicated, &secrets)
        .expect_err("reject duplicate secret token");
    assert!(error.to_string().contains("may only appear once"));
}

#[test]
fn discovers_write_only_fields_in_ref_sibling_subschemas() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "$defs": {
            "base": {"type": "object"}
        },
        "properties": {
            "auth": {
                "$ref": "#/$defs/base",
                "properties": {
                    "password": {"type": "string", "writeOnly": true}
                }
            }
        }
    }));

    assert_eq!(secret_paths(&loaded), vec!["/auth/password".to_owned()]);
    assert_eq!(
        loaded.redact(&json!({"auth": {"password": "hidden"}})),
        json!({"auth": {"password": REDACTED}})
    );
}

#[test]
fn redacts_invalid_non_string_values_at_secret_paths() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "token": {"type": "string", "writeOnly": true}
        }
    }));
    let config = json!({"token": {"nested": "must-not-leak"}});

    assert_eq!(loaded.redact(&config), json!({"token": REDACTED}));

    let (redacted, secrets) = loaded.redact_for_edit(&config);
    assert_ne!(redacted["token"], config["token"]);
    assert_eq!(
        loaded
            .restore_edit_secrets(&redacted, &secrets)
            .expect("restore invalid secret value without exposing it"),
        config
    );
}

#[test]
fn discovers_write_only_across_reference_chains_and_nullable_strings() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "nullable": {"type": ["string", "null"], "writeOnly": true},
            "token": {"$ref": "#/$defs/annotated"}
        },
        "$defs": {
            "annotated": {"$ref": "#/$defs/string", "writeOnly": true},
            "string": {"type": "string"}
        }
    }));

    assert_eq!(
        secret_paths(&loaded),
        vec!["/nullable".to_owned(), "/token".to_owned()]
    );
    assert!(matches!(
        loaded
            .fields()
            .iter()
            .find(|field| field.key == "token")
            .unwrap()
            .kind,
        DynamicConfigFieldKind::String { secret: true }
    ));
    assert_eq!(
        loaded.redact(&json!({"nullable": "hidden", "token": "also-hidden"})),
        json!({"nullable": REDACTED, "token": REDACTED})
    );
    assert_eq!(
        loaded.redact(&json!({"nullable": null, "token": "hidden"})),
        json!({"nullable": null, "token": REDACTED})
    );
}

#[test]
fn rejects_split_and_ambiguous_write_only_applicators() {
    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "token": {
                "allOf": [
                    {"type": "string"},
                    {"writeOnly": true}
                ]
            }
        }
    }));
    let error = PluginConfigSchema::load("acme.split", path)
        .expect_err("reject writeOnly annotation split across allOf");
    let message = error.to_string();
    assert!(message.contains("unsupported writeOnly shape"), "{message}");
    assert!(message.contains("/properties/token/allOf/1"), "{message}");

    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "token": {
                "anyOf": [
                    {"$ref": "#/$defs/secret"},
                    {"type": "null"}
                ]
            }
        },
        "$defs": {
            "secret": {"type": "string", "writeOnly": true}
        }
    }));
    let error = PluginConfigSchema::load("acme.ambiguous", path)
        .expect_err("reject conditional writeOnly annotation");
    let message = error.to_string();
    assert!(message.contains("anyOf"), "{message}");
    assert!(message.contains("writeOnly"), "{message}");
}

#[test]
fn secret_discovery_preserves_pattern_prefix_and_contains_selectors() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "patterned": {
                "type": "object",
                "patternProperties": {
                    "^secret_": {"type": "string", "writeOnly": true}
                }
            },
            "tuple": {
                "type": "array",
                "prefixItems": [
                    {"type": "string", "writeOnly": true},
                    {"type": "string"}
                ]
            },
            "contained": {
                "type": "array",
                "contains": {"type": "string", "writeOnly": true}
            }
        }
    }));
    let config = json!({
        "patterned": {"secret_token": "hide", "public": "show"},
        "tuple": ["hide", "show"],
        "contained": ["hide-one", 7, "hide-two"]
    });
    assert_eq!(
        loaded.redact(&config),
        json!({
            "patterned": {"secret_token": REDACTED, "public": "show"},
            "tuple": [REDACTED, "show"],
            "contained": [REDACTED, REDACTED, REDACTED]
        })
    );
    assert!(loaded.has_secrets_at(&["patterned".to_owned()]));
    assert!(loaded.has_secrets_at(&["tuple".to_owned()]));
    assert!(loaded.has_secrets_at(&["contained".to_owned()]));
}

#[test]
fn secret_discovery_limits_additional_properties_and_items_to_unmatched_values() {
    let loaded = load(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {
                    "known": {"type": "string"}
                },
                "patternProperties": {
                    "^public_": {"type": "string"}
                },
                "additionalProperties": {"type": "string", "writeOnly": true}
            },
            "tuple": {
                "type": "array",
                "prefixItems": [
                    {"type": "string"},
                    {"type": "string", "writeOnly": true}
                ],
                "items": {"type": "string", "writeOnly": true}
            }
        }
    }));
    assert_eq!(
        secret_paths(&loaded),
        vec![
            "/metadata/~additional".to_owned(),
            "/tuple/1".to_owned(),
            "/tuple/~tail(2)".to_owned()
        ]
    );

    let config = json!({
        "metadata": {
            "known": "visible-known",
            "public_name": "visible-pattern",
            "token": "hidden-additional"
        },
        "tuple": ["visible-prefix", "hidden-prefix", "hidden-tail"]
    });
    assert_eq!(
        loaded.redact(&config),
        json!({
            "metadata": {
                "known": "visible-known",
                "public_name": "visible-pattern",
                "token": REDACTED
            },
            "tuple": ["visible-prefix", REDACTED, REDACTED]
        })
    );

    let (redacted, secrets) = loaded.redact_for_edit(&config);
    assert_eq!(
        loaded
            .restore_edit_secrets(&redacted, &secrets)
            .expect("restore precisely selected secrets"),
        config
    );
}

#[test]
fn rejects_write_only_under_evaluation_dependent_applicators() {
    let cases = [
        (
            DRAFT2020,
            "dependentSchemas",
            json!({
                "dependentSchemas": {
                    "mode": {
                        "properties": {
                            "token": {"type": "string", "writeOnly": true}
                        }
                    }
                }
            }),
        ),
        (
            DRAFT7,
            "dependencies",
            json!({
                "dependencies": {
                    "mode": {
                        "properties": {
                            "token": {"type": "string", "writeOnly": true}
                        }
                    }
                }
            }),
        ),
        (
            DRAFT2020,
            "unevaluatedProperties",
            json!({
                "unevaluatedProperties": {"$ref": "#/$defs/secret"},
                "$defs": {
                    "secret": {"type": "string", "writeOnly": true}
                }
            }),
        ),
        (
            DRAFT2020,
            "unevaluatedItems",
            json!({
                "properties": {
                    "values": {
                        "type": "array",
                        "prefixItems": [{"type": "string"}],
                        "unevaluatedItems": {"type": "string", "writeOnly": true}
                    }
                }
            }),
        ),
    ];

    for (draft, keyword, body) in cases {
        let mut schema = json!({
            "$schema": draft,
            "type": "object"
        });
        schema
            .as_object_mut()
            .unwrap()
            .extend(body.as_object().unwrap().clone());
        let (_directory, path) = write_schema(&schema);
        let error = PluginConfigSchema::load("acme.unsupported-secret", path)
            .expect_err("reject applicator-dependent writeOnly field");
        let message = error.to_string();
        assert!(message.contains(keyword), "{message}");
        assert!(message.contains("writeOnly"), "{message}");
    }
}

#[test]
fn rejects_recursive_references_that_secret_discovery_cannot_safely_expand() {
    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "properties": {
            "node": {"$ref": "#/$defs/node"}
        },
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "token": {"type": "string", "writeOnly": true},
                    "next": {"$ref": "#/$defs/node"}
                }
            }
        }
    }));

    let error = PluginConfigSchema::load("acme.recursive", path)
        .expect_err("reject recursive secret schema reference");
    let message = error.to_string();
    assert!(message.contains("secret schema reference"), "{message}");
    assert!(message.contains("cyclic"), "{message}");
}

#[test]
fn secret_discovery_handles_draft7_tuple_and_additional_items() {
    let loaded = load(&json!({
        "$schema": DRAFT7,
        "type": "object",
        "properties": {
            "tuple": {
                "type": "array",
                "items": [
                    {"type": "string"},
                    {"type": "string", "writeOnly": true}
                ],
                "additionalItems": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string", "writeOnly": true}
                    }
                }
            }
        }
    }));
    let config = json!({
        "tuple": [
            "visible",
            "tuple-secret",
            {"token": "tail-secret-one", "visible": "keep"},
            {"token": "tail-secret-two"}
        ]
    });

    assert_eq!(
        loaded.redact(&config),
        json!({
            "tuple": [
                "visible",
                REDACTED,
                {"token": REDACTED, "visible": "keep"},
                {"token": REDACTED}
            ]
        })
    );
    let (redacted, secrets) = loaded.redact_for_edit(&config);
    assert_eq!(
        loaded
            .restore_edit_secrets(&redacted, &secrets)
            .expect("restore tuple secrets"),
        config
    );
}

#[test]
fn rejects_pattern_properties_unsupported_by_secret_matcher() {
    let (_directory, path) = write_schema(&json!({
        "$schema": DRAFT2020,
        "type": "object",
        "patternProperties": {
            "(?=secret)": {"type": "string", "writeOnly": true}
        }
    }));

    let error = PluginConfigSchema::load("acme.bad-pattern", path)
        .expect_err("reject unsupported patternProperties expression");
    let message = error.to_string();
    assert!(message.contains("patternProperties"), "{message}");
    assert!(message.contains("look-around"), "{message}");
}

#[test]
fn read_and_json_errors_include_plugin_and_path() {
    let directory = tempdir().expect("create temp directory");
    let missing = directory.path().join("missing.json");
    let error = PluginConfigSchema::load("acme.missing", &missing).expect_err("missing");
    let message = error.to_string();
    assert!(message.contains("acme.missing"), "{message}");
    assert!(
        message.contains(missing.to_string_lossy().as_ref()),
        "{message}"
    );

    let invalid = directory.path().join("invalid.json");
    fs::write(&invalid, "{").expect("write invalid json");
    let error = PluginConfigSchema::load("acme.invalid", &invalid).expect_err("invalid json");
    assert!(error.to_string().contains("not valid JSON"));
}

#[test]
fn schema_reads_require_regular_files_within_the_size_limit() {
    let directory = tempdir().expect("create temp directory");
    let error = PluginConfigSchema::load("acme.directory", directory.path())
        .expect_err("reject directory schema path");
    assert!(error.to_string().contains("regular file"));

    let oversized = directory.path().join("oversized.schema.json");
    fs::File::create(&oversized)
        .expect("create oversized schema")
        .set_len(MAX_CONFIG_SCHEMA_BYTES + 1)
        .expect("size oversized schema");
    let error = PluginConfigSchema::load("acme.oversized", &oversized)
        .expect_err("reject oversized schema");
    assert!(error.to_string().contains("1 MiB size limit"));

    let maximum = directory.path().join("maximum.schema.json");
    let mut source = serde_json::to_vec(&json!({
        "$schema": DRAFT2020,
        "type": "object"
    }))
    .expect("serialize schema");
    source.resize(MAX_CONFIG_SCHEMA_BYTES as usize, b' ');
    fs::write(&maximum, source).expect("write maximum-sized schema");
    PluginConfigSchema::load("acme.maximum", maximum).expect("accept schema at the size limit");
}
