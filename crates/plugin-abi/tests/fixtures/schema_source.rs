// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Resolve frozen fixture modules exactly as their crate declarations select them.

/// Read a module for schema generation, including its explicit frozen source path.
pub fn read(root: &std::path::Path, name: &str) -> String {
    let lib = root.join("src/lib.rs");
    println!("cargo:rerun-if-changed={}", lib.display());
    let parsed = syn::parse_file(&std::fs::read_to_string(lib).unwrap()).unwrap();
    let module = parsed
        .items
        .into_iter()
        .find_map(|item| match item {
            syn::Item::Mod(module) if module.ident == name => Some(module),
            _ => None,
        })
        .expect("schema module must be declared in the frozen fixture");
    assert!(
        module.content.is_none(),
        "schema modules must use source files"
    );
    let path = module
        .attrs
        .iter()
        .find(|attr| attr.path().is_ident("path"))
        .map(|attr| {
            let syn::Meta::NameValue(value) = &attr.meta else {
                panic!("invalid module path")
            };
            let syn::Expr::Lit(value) = &value.value else {
                panic!("invalid module path")
            };
            let syn::Lit::Str(value) = &value.lit else {
                panic!("invalid module path")
            };
            value.value()
        })
        .unwrap_or_else(|| format!("{name}.rs"));
    let path = root.join("src").join(path);
    println!("cargo:rerun-if-changed={}", path.display());
    std::fs::read_to_string(path).unwrap()
}
