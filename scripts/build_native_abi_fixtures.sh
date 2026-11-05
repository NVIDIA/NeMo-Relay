#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="$repo_root/crates/plugin-abi/tests/fixtures"
fixture_target="${NEMO_RELAY_NATIVE_ABI_FIXTURE_TARGET_DIR:-$repo_root/target/native-abi-fixtures}"
if command -v cygpath >/dev/null 2>&1; then
    fixture_target="$(cygpath -u "$fixture_target")"
fi

for fixture in baseline future major signature marker reordered missing dependency future-host; do
    manifest="$fixture_root/$fixture/Cargo.toml"
    if [[ "${1:-}" == "--update-lockfiles" ]]; then
        cargo generate-lockfile --manifest-path "$manifest"
    elif [[ "${1:-}" == "--check-lockfiles" ]]; then
        cargo metadata --locked --format-version 1 --manifest-path "$manifest" >/dev/null
    elif [[ -z "${1:-}" ]]; then
        cargo build --locked --manifest-path "$manifest" --target-dir "$fixture_target"
    else
        echo "Usage: $0 [--update-lockfiles|--check-lockfiles]" >&2
        exit 2
    fi
done

if [[ -z "${1:-}" ]]; then
    # These C-callable binaries exercise sidecar loading without adding a Cargo workspace.
    directory="$fixture_target/debug/dependencies"
    mkdir -p "$directory"
    source="$repo_root/crates/plugin-host/tests/fixtures/native_dependencies/library.rs"
    compiler_args=(--edition 2024 --crate-type cdylib -C panic=abort)
    helper_args=()
    consumer_args=()
    case "$(rustc -vV | sed -n 's/^host: //p')" in
        *-windows-*)
            helper="relay_native_dependency.dll"
            consumer="relay_native_consumer.dll"
            ;;
        *-apple-*)
            helper="librelay_native_dependency.dylib"
            consumer="librelay_native_consumer.dylib"
            helper_args+=(-C link-arg=-Wl,-install_name,@rpath/librelay_native_dependency.dylib)
            consumer_args+=(-C link-arg=-Wl,-rpath,@loader_path/deps -C link-arg=-Wl,-rpath,@loader_path)
            ;;
        *)
            helper="librelay_native_dependency.so.1"
            consumer="librelay_native_consumer.so"
            helper_args+=(-C link-arg=-Wl,-soname,librelay_native_dependency.so.1)
            consumer_args+=(-C 'link-arg=-Wl,-rpath,$ORIGIN/deps' -C 'link-arg=-Wl,-rpath,$ORIGIN')
            ;;
    esac
    rustc "$source" "${compiler_args[@]}" --crate-name relay_native_dependency \
        --cfg native_sidecar "${helper_args[@]}" -o "$directory/$helper"
    if [[ "$helper" == *.dll ]]; then
        cp "$directory/relay_native_dependency.dll.lib" "$directory/relay_native_dependency.lib"
    elif [[ "$helper" == *.so.1 ]]; then
        cp "$directory/$helper" "$directory/librelay_native_dependency.so"
    fi
    rustc "$source" "${compiler_args[@]}" --crate-name relay_native_consumer \
        -L "native=$directory" "${consumer_args[@]}" -o "$directory/$consumer"
    if [[ "$helper" == *.so.1 ]]; then
        braced_directory="$directory/braced-origin"
        mkdir -p "$braced_directory"
        rustc "$source" "${compiler_args[@]}" --crate-name relay_native_dependency \
            --cfg native_sidecar -C 'link-arg=-Wl,-soname,${ORIGIN}/librelay_native_dependency.so.1' \
            -o "$braced_directory/$helper"
        cp "$braced_directory/$helper" "$braced_directory/librelay_native_dependency.so"
        rustc "$source" "${compiler_args[@]}" --crate-name relay_native_consumer \
            -L "native=$braced_directory" -o "$braced_directory/$consumer"
    fi
    if [[ "$helper" == *.dll ]]; then
        delayed_directory="$directory/delay-loaded"
        mkdir -p "$delayed_directory"
        rustc "$source" "${compiler_args[@]}" --crate-name relay_delayed_dependency \
            --cfg native_sidecar -o "$delayed_directory/relay_delayed_dependency.dll"
        cp "$delayed_directory/relay_delayed_dependency.dll.lib" "$delayed_directory/relay_delayed_dependency.lib"
        rustc "$source" "${compiler_args[@]}" --crate-name relay_delayed_consumer \
            --cfg native_delay_import -L "native=$delayed_directory" \
            -C link-arg=-DELAYLOAD:relay_delayed_dependency.dll -C link-arg=delayimp.lib \
            -o "$delayed_directory/relay_delayed_consumer.dll"
    fi
    mkdir -p "$directory/alternate"
    rustc "$source" "${compiler_args[@]}" --crate-name relay_native_dependency \
        --cfg native_sidecar --cfg native_alternate "${helper_args[@]}" -o "$directory/alternate/$helper"
    image_directory="$directory/main-image"
    image_helper="${helper/relay_native_dependency/relay_resident_image}"
    image_consumer="${consumer/relay_native_consumer/relay_image_consumer}"
    mkdir -p "$image_directory/alternate"
    image_args=()
    case "$helper" in
        *.dylib) image_args+=(-C "link-arg=-Wl,-install_name,@rpath/$image_helper") ;;
        *.so.1) image_args+=(-C "link-arg=-Wl,-soname,$image_helper") ;;
    esac
    rustc "$source" "${compiler_args[@]}" --crate-name relay_resident_image \
        --cfg native_sidecar "${image_args[@]}" -o "$image_directory/$image_helper"
    if [[ "$helper" == *.dll ]]; then
        cp "$image_directory/relay_resident_image.dll.lib" "$image_directory/relay_resident_image.lib"
    elif [[ "$helper" == *.so.1 ]]; then
        cp "$image_directory/$image_helper" "$image_directory/librelay_resident_image.so"
    fi
    rustc "$source" "${compiler_args[@]}" --crate-name relay_image_consumer --cfg native_main_image \
        -L "native=$image_directory" "${consumer_args[@]}" -o "$image_directory/$image_consumer"
    rustc "$source" "${compiler_args[@]}" --crate-name relay_resident_image \
        --cfg native_sidecar --cfg native_alternate "${image_args[@]}" -o "$image_directory/alternate/$image_helper"
    if [[ "$helper" != *.dll ]]; then
        # A and B share C, which resolves D through its first importer's paths.
        # In the deeper graph A reaches C through X, distinguishing Mach-O's
        # depth-first traversal from ELF's breadth-first traversal.
        search_source="$repo_root/crates/plugin-host/tests/fixtures/native_dependencies/shared_search.rs"
        build_search_image() {
            local search_output="$1" search_name="$2" search_role="$3" search_path="$4"
            shift 4
            local image_flags=() search_file
            case "$helper" in
                *.dylib)
                    search_file="lib${search_name}.dylib"
                    image_flags+=(-C "link-arg=-Wl,-install_name,@rpath/$search_file")
                    if [[ -n "$search_path" ]]; then
                        image_flags+=(-C "link-arg=-Wl,-rpath,@loader_path$search_path")
                    fi
                    ;;
                *)
                    search_file="lib${search_name}.so"
                    image_flags+=(-C "link-arg=-Wl,-soname,$search_file")
                    if [[ -n "$search_path" ]]; then
                        image_flags+=(-C "link-arg=-Wl,--disable-new-dtags,-rpath,\$ORIGIN$search_path")
                    fi
                    ;;
            esac
            rustc "$search_source" "${compiler_args[@]}" --crate-name "$search_name" \
                --cfg "$search_role" "${image_flags[@]}" "$@" \
                -o "$search_output/$search_file"
        }
        for graph in direct deep; do
            search_directory="$directory/shared-search-$graph"
            mkdir -p "$search_directory/a" "$search_directory/b"
            build_search_image "$search_directory/a" relay_search_d search_leaf ""
            build_search_image "$search_directory/b" relay_search_d search_leaf "" --cfg search_alternate
            build_search_image "$search_directory" relay_search_c search_shared "" -L "native=$search_directory/a"
            if [[ "$graph" == deep ]]; then
                build_search_image "$search_directory" relay_search_x search_extra "" -L "native=$search_directory"
            fi
            build_search_image "$search_directory" relay_search_a search_a /a \
                -L "native=$search_directory" --cfg "search_$graph"
            build_search_image "$search_directory" relay_search_b search_b /b -L "native=$search_directory"
            build_search_image "$search_directory" relay_search_main search_main / -L "native=$search_directory"
        done
        if [[ "$helper" == *.dylib ]]; then
            loader_source="$repo_root/crates/plugin-host/tests/fixtures/native_dependencies/loader_paths.c"
            loader_directory="$directory/loader-paths"
            mkdir -p "$loader_directory/actual" "$loader_directory/first" "$loader_directory/second"
            clang -dynamiclib -DFIXTURE_LEAF "$loader_source" \
                -Wl,-install_name,@loader_path/librelay_path_leaf.dylib -o "$loader_directory/actual/librelay_path_leaf.dylib"
            clang -dynamiclib -DFIXTURE_CHILD "$loader_source" -L"$loader_directory/actual" -lrelay_path_leaf \
                -Wl,-install_name,@rpath/librelay_path_child.dylib -o "$loader_directory/actual/librelay_path_child.dylib"
            clang -dynamiclib "$loader_source" -L"$loader_directory/actual" -lrelay_path_child \
                -Wl,-rpath,@loader_path/alias -o "$loader_directory/librelay_path_main.dylib"
            cp "$loader_directory/actual/"*.dylib "$loader_directory/second/"
            case "$(uname -m)" in
                arm64) alternate_arch=x86_64 ;;
                *) alternate_arch=arm64 ;;
            esac
            clang -dynamiclib -target "$alternate_arch-apple-macos11" -DFIXTURE_LEAF "$loader_source" \
                -Wl,-install_name,@rpath/librelay_path_child.dylib -o "$loader_directory/first/librelay_path_child.dylib"
            lipo -create "$loader_directory/first/librelay_path_child.dylib" -output "$loader_directory/first/fat-child.dylib"
            clang -dynamiclib "$loader_source" -L"$loader_directory/second" -lrelay_path_child \
                -Wl,-rpath,@loader_path/first,-rpath,@loader_path/second -o "$loader_directory/librelay_path_arch.dylib"
            cp "$loader_directory/librelay_path_arch.dylib" "$loader_directory/librelay_path_relative.dylib"
            install_name_tool -change @rpath/librelay_path_child.dylib second/librelay_path_child.dylib \
                "$loader_directory/librelay_path_relative.dylib"
            cp "$loader_directory/librelay_path_arch.dylib" "$loader_directory/librelay_path_bare.dylib"
            install_name_tool -change @rpath/librelay_path_child.dylib librelay_path_child.dylib \
                "$loader_directory/librelay_path_bare.dylib"
            clang -dynamiclib "$loader_source" -L"$loader_directory/second" -lrelay_path_child \
                -Wl,-rpath,second -o "$loader_directory/librelay_path_relative_search.dylib"
            cp "$loader_directory/librelay_path_main.dylib" "$loader_directory/librelay_path_parent.dylib"
            install_name_tool -change @rpath/librelay_path_child.dylib @loader_path/alias/../librelay_path_child.dylib \
                "$loader_directory/librelay_path_parent.dylib"
            clang -dynamiclib "$loader_source" -L"$loader_directory/actual" -lrelay_path_child \
                -Wl,-rpath,@loader_path/alias/.. -o "$loader_directory/librelay_path_parent_search.dylib"
            clang -dynamiclib -DFIXTURE_DECOY "$loader_source" \
                -Wl,-install_name,@rpath/librelay_path_child.dylib -o "$loader_directory/librelay_path_child.dylib"
            distinct_directory="$directory/distinct-identities"
            mkdir -p "$distinct_directory/a" "$distinct_directory/b"
            for side in a b; do
                if [[ "$side" == a ]]; then
                    distinct_role=FIXTURE_DISTINCT_A
                else
                    distinct_role=FIXTURE_DISTINCT_B
                fi
                clang -dynamiclib -D"$distinct_role" "$loader_source" \
                    -Wl,-install_name,@rpath/$side/librelay_distinct.dylib \
                    -o "$distinct_directory/$side/librelay_distinct.dylib"
            done
            clang -dynamiclib -DFIXTURE_DISTINCT_MAIN "$loader_source" \
                "$distinct_directory/a/librelay_distinct.dylib" "$distinct_directory/b/librelay_distinct.dylib" \
                -Wl,-rpath,@loader_path -o "$distinct_directory/librelay_distinct_main.dylib"
            case_directory="$directory/case-aliases"
            mkdir -p "$case_directory"
            cp "$directory/shared-search-direct/"*.dylib "$case_directory/"
            cp "$directory/shared-search-direct/a/librelay_search_d.dylib" "$case_directory/"
            install_name_tool -change @rpath/librelay_search_c.dylib @rpath/LIBRELAY_SEARCH_C.dylib \
                "$case_directory/librelay_search_b.dylib"
        fi
        mkdir -p "$directory/absolute"
        case "$helper" in
            *.dylib) absolute_args=(-C "link-arg=-Wl,-install_name,$directory/absolute/$helper") ;;
            *) absolute_args=(-C "link-arg=-Wl,-soname,$directory/absolute/$helper") ;;
        esac
        rustc "$source" "${compiler_args[@]}" --crate-name relay_native_dependency \
            --cfg native_sidecar "${absolute_args[@]}" -o "$directory/absolute/$helper"
        if [[ "$helper" == *.so.1 ]]; then
            cp "$directory/absolute/$helper" "$directory/absolute/librelay_native_dependency.so"
        fi
        rustc "$source" "${compiler_args[@]}" --crate-name relay_native_consumer \
            -L "native=$directory/absolute" -o "$directory/absolute/$consumer"
        rustc "$source" "${compiler_args[@]}" --crate-name relay_native_consumer \
            -L "native=$directory" -C "link-arg=-Wl,-rpath,$directory" -o "$directory/absolute-search-$consumer"
    fi
fi
