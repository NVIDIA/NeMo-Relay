// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Derive complete contract descriptions from their declarations, not package versions.
use quote::ToTokens;
#[path = "../schema_source.rs"]
mod schema_source;
#[path = "src/contract.rs"]
mod contract;

/// Generate the SDK contract constants and checked schema from the interface sources.
fn main() {
    let root = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let root = if root.join("src/lib.rs").is_file() {
        root
    } else {
        root.join("../../..")
    };
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let (major, minor) = contract::NATIVE_API
        .strip_prefix('v')
        .and_then(|s| s.split_once('.'))
        .expect("native contract must use v<major>.<minor>");
    let major: u32 = major.parse().unwrap();
    let minor: u32 = minor.parse().unwrap();
    assert!(major > 0);
    assert_eq!(contract::NATIVE_API, format!("v{major}.{minor}"));
    println!("cargo:rerun-if-changed=src/contract.rs");
    std::fs::write(output.join("native_api.rs"),format!("/// Minimum native contract required by this SDK.\npub const NATIVE_API_VERSION:NativeApiVersion=NativeApiVersion::new({major},{minor});\n")).unwrap();
    generate_schema(&root, &output);
}

/// Describe wire declarations and compiler-derived layouts without tying them to package versions.
fn generate_schema(root: &std::path::Path, output: &std::path::Path) {
    let mut schema = String::from("relay.native.v1.0; xabi/0.1.2; async-ffi/0.5.1/abi2; transfer/1;\n");
    let mut layouts = String::from("|result:&mut String| {\n");
    for file in [
        "value",
        "version",
        "v1_0",
        "diagnostics",
        "borrow",
        "transfer",
        "values",
        "handles",
        "wire",
        "bootstrap",
        "future",
    ] {
        let source = schema_source::read(root, file);
        let parsed = syn::parse_file(&source).unwrap();
        for item in parsed.items {
            let tokens = match item {
                syn::Item::Struct(mut item) => {
                    if (file == "handles" && item.ident != "LeaseWire")
                        || (file == "transfer"
                            && !["OwnedWire", "OptionalWire", "VectorWire"]
                                .iter()
                                .any(|name| item.ident == *name))
                        || (file == "version" && item.ident != "NativeApiVersion")
                        || (file == "future" && item.ident != "FutureWire")
                    {
                        continue;
                    }

                    item = clean_struct(item);
                    if item.generics.params.is_empty()
                        && matches!(file, "values" | "bootstrap" | "future")
                    {
                        let ty = format!("crate::{file}::{}", item.ident);
                        layouts.push_str(&format!("result.push_str(&format!(\";{}:{{}}/{{}}\",std::mem::size_of::<{ty}>(),std::mem::align_of::<{ty}>()));\n",item.ident));
                    }
                    Some(item.into_token_stream())
                }
                syn::Item::Union(mut item) => {
                    item.attrs.retain(|a| a.path().is_ident("repr"));
                    let name = &item.ident;
                    let ty = format!("crate::{file}::{name}");
                    layouts.push_str(&format!("result.push_str(&format!(\";{name}:{{}}/{{}}\",std::mem::size_of::<{ty}>(),std::mem::align_of::<{ty}>()));\n"));
                    Some(item.into_token_stream())
                }
                syn::Item::Enum(mut item) => {
                    if file == "version" {
                        continue;
                    }
                    item = clean_enum(item);
                    Some(item.into_token_stream())
                }
                syn::Item::Trait(mut item) => {
                    if matches!(file, "transfer" | "future") {
                        continue;
                    }
                    item.attrs.retain(|a| {
                        !a.path().is_ident("doc")
                            && !a.path().is_ident("derive")
                            && !a.path().is_ident("allow")
                    });
                    for i in &mut item.items {
                        if let syn::TraitItem::Fn(f) = i {
                            f.attrs.retain(|a| {
                                !a.path().is_ident("doc")
                                    && !a.path().is_ident("derive")
                                    && !a.path().is_ident("allow")
                            });
                            f.default = None;
                        }
                    }
                    if file == "wire" {
                        let name = format!("crate::wire::XabiV1AbiTrait{}", item.ident);
                        layouts.push_str(&format!("{{let mut items=Vec::new();({name}::XABI_LAYOUT.collect)(&mut items);for item in items {{result.push_str(&format!(\";{{item:?}}\"));}}}}\n"));
                    }
                    Some(item.into_token_stream())
                }
                syn::Item::Impl(item)
                    if item.trait_.as_ref().is_some_and(|(_, path, _)| {
                        path.segments.last().is_some_and(|s| s.ident == "Transfer")
                    }) =>
                {
                    let generics = item.generics;
                    let ty = item.self_ty;
                    let types = item.items.into_iter().filter_map(|item| match item {
                        syn::ImplItem::Type(item) => Some(item),
                        _ => None,
                    });
                    Some(quote::quote!(impl #generics Transfer for #ty { #(#types)* }))
                }
                syn::Item::Type(mut item) => {
                    item.attrs.retain(|a| {
                        !a.path().is_ident("doc")
                            && !a.path().is_ident("derive")
                            && !a.path().is_ident("allow")
                    });
                    Some(item.into_token_stream())
                }
                syn::Item::Macro(item) if item.mac.path.is_ident("record") => {
                    let name = item
                        .mac
                        .tokens
                        .clone()
                        .into_iter()
                        .next()
                        .unwrap()
                        .to_string();
                    layouts.push_str(&format!("crate::values::{name}::append_layout(result);\n"));
                    Some(item.into_token_stream())
                }
                _ => None,
            };
            if let Some(tokens) = tokens {
                schema.push_str(file);
                schema.push(':');
                schema.push_str(&tokens.to_string());
                schema.push('\n');
            }
        }
    }
    layouts.push('}');
    std::fs::write(output.join("native_schema.txt"), schema).unwrap();
    std::fs::write(output.join("native_layout.rs"), layouts).unwrap();
}

/// Remove documentation and derive attributes while preserving the enum contract.
fn clean_enum(mut item: syn::ItemEnum) -> syn::ItemEnum {
    item.attrs.retain(|a| {
        !a.path().is_ident("doc") && !a.path().is_ident("derive") && !a.path().is_ident("allow")
    });
    for v in &mut item.variants {
        v.attrs.retain(|a| {
            !a.path().is_ident("doc") && !a.path().is_ident("derive") && !a.path().is_ident("allow")
        });
        for f in &mut v.fields {
            f.attrs.retain(|a| {
                !a.path().is_ident("doc")
                    && !a.path().is_ident("derive")
                    && !a.path().is_ident("allow")
            });
        }
    }
    item
}

/// Remove documentation and derive attributes while preserving the record contract.
fn clean_struct(mut item: syn::ItemStruct) -> syn::ItemStruct {
    item.attrs.retain(|a| {
        !a.path().is_ident("doc") && !a.path().is_ident("derive") && !a.path().is_ident("allow")
    });
    for f in &mut item.fields {
        f.attrs.retain(|a| {
            !a.path().is_ident("doc") && !a.path().is_ident("derive") && !a.path().is_ident("allow")
        });
    }
    item
}
