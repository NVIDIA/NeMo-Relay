// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn render_frame_settled_contains_figlet_glyphs() {
    let frame = render_frame(None, false);
    // ANSI Shadow figlet uses filled blocks and box-drawing corners.
    assert!(frame.contains('█'), "frame missing figlet block glyph");
    assert!(
        frame.contains('╗') || frame.contains('╔'),
        "frame missing figlet corners"
    );
}

#[test]
fn render_frame_plain_mode_has_no_ansi_escapes() {
    let frame = render_frame(None, false);
    assert!(
        !frame.contains('\x1b'),
        "plain mode should emit no ANSI escapes"
    );
}

#[test]
fn render_frame_color_mode_emits_nvidia_green() {
    let frame = render_frame(None, true);
    assert!(frame.contains("\x1b[38;5;112m"));
    assert!(frame.contains("\x1b[0m"));
}

#[test]
fn render_frame_tracer_overlay_inserts_dot_at_position() {
    // Pick a position on the top rail (row 0) that's empty in the static art.
    let frame_with = render_frame(Some((0, 14)), true);
    let frame_without = render_frame(None, true);
    assert!(
        frame_with.contains('●'),
        "tracer should render a `●` head when overlay is active"
    );
    assert!(
        !frame_without.contains('●'),
        "settled frame (no tracer) should not include the dot glyph"
    );
}

#[test]
fn render_frame_tracer_plain_mode_uses_ascii_star() {
    let frame = render_frame(Some((0, 14)), false);
    assert!(
        frame.contains('*'),
        "plain mode tracer head should render as `*` (ASCII star)"
    );
    assert!(
        !frame.contains('●'),
        "plain mode should not emit Unicode dot"
    );
}

#[test]
fn tracer_position_starts_on_top_rail_and_ends_on_bottom_rail() {
    let (r0, _c0) = tracer_position(0).expect("frame 0 should have a position");
    assert_eq!(r0, 0, "tracer starts on the top rail");

    let (r_last, c_last) =
        tracer_position(TRACER_FRAMES - 1).expect("last animated frame should have a position");
    assert!(
        r_last >= 6,
        "tracer should descend to the bottom rail by the last frame"
    );
    assert!(
        c_last >= 80,
        "tracer should travel close to the right edge by the last frame"
    );
}

#[test]
fn tracer_position_is_none_after_animation_ends() {
    assert!(tracer_position(TRACER_FRAMES).is_none());
    assert!(tracer_position(TRACER_FRAMES + 100).is_none());
}

#[test]
fn frame_is_wrapped_with_rounded_border() {
    let frame = render_frame(None, false);
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
