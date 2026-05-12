// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn render_frame_settled_contains_figlet_glyphs() {
    let frame = render_frame(false);
    // ANSI Shadow figlet uses filled blocks and box-drawing corners.
    assert!(frame.contains('█'), "frame missing figlet block glyph");
    assert!(
        frame.contains('╗') || frame.contains('╔'),
        "frame missing figlet corners"
    );
}

#[test]
fn render_frame_plain_mode_has_no_ansi_escapes() {
    let frame = render_frame(false);
    assert!(
        !frame.contains('\x1b'),
        "plain mode should emit no ANSI escapes"
    );
}

#[test]
fn render_frame_color_mode_emits_nvidia_green() {
    let frame = render_frame(true);
    assert!(frame.contains("\x1b[38;5;112m"));
    assert!(frame.contains("\x1b[0m"));
}

#[test]
fn docked_frame_has_no_cursor_control_sequences() {
    let frame = render_docked_frame(true);
    assert!(
        !frame.contains("\x1b[?25l") && !frame.contains("\x1b[?25h") && !frame.contains("\x1b7"),
        "static banner should not emit animation cursor control sequences"
    );
}

#[test]
fn frame_is_wrapped_with_rounded_border() {
    let frame = render_frame(false);
    // Four corner glyphs and the side bars must appear.
    assert!(frame.contains('╭'), "missing top-left corner");
    assert!(frame.contains('╮'), "missing top-right corner");
    assert!(frame.contains('╰'), "missing bottom-left corner");
    assert!(frame.contains('╯'), "missing bottom-right corner");
    assert!(frame.contains('│'), "missing vertical border");
    assert!(frame.contains('─'), "missing horizontal border");
}

#[test]
fn docked_frame_includes_version_tag() {
    let frame = render_docked_frame(false);
    let version = env!("CARGO_PKG_VERSION");
    let expected = format!("v{version}");
    assert!(
        frame.contains(&expected),
        "docked frame should include the version tag `{expected}`"
    );
    // No bullet dot before the version — settled state is just the green text label.
    assert!(
        !frame.contains('●'),
        "docked frame should not include a bullet dot before the version"
    );
}
