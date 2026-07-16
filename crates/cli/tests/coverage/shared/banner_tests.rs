// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn render_frame_settled_contains_figlet_glyphs() {
    let frame = render_frame(false);
    // ANSI Shadow figlet uses filled blocks and box-drawing corners.
    assert!(frame.contains('█'), "frame missing figlet block glyph");
    assert!(
        !frame.contains("_/=|") && !frame.contains("|=\\_") && !frame.contains("╔██╗██╗"),
        "frame should not include flag columns"
    );
    assert!(
        frame.contains('╗') || frame.contains('╔'),
        "frame missing figlet corners"
    );
}

#[test]
fn figlet_rows_are_centered_as_one_block() {
    let frame = render_frame(false);
    let rows = frame_grid_rows(&frame);
    let figlet_rows = &rows[1..=FIGLET_ROWS];
    let left_edges: Vec<usize> = figlet_rows.iter().map(|row| first_art_col(row)).collect();
    let expected_left_edge = (frame_inner_width(&frame) - banner_art_width()) / 2;

    assert!(
        left_edges.iter().all(|edge| *edge == expected_left_edge),
        "figlet rows should share centered left edge {expected_left_edge}; got {left_edges:?}"
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

#[test]
fn print_headers_fall_back_to_plain_header_when_stdout_is_not_terminal() {
    print_intro();
    print_doctor_header();
}

#[test]
fn banner_support_decision_covers_terminal_environment_branches() {
    assert!(!supports_banner_for(
        false,
        false,
        None,
        None,
        Some(MIN_WIDTH)
    ));
    assert!(!supports_banner_for(
        true,
        true,
        None,
        None,
        Some(MIN_WIDTH)
    ));
    assert!(!supports_banner_for(
        true,
        false,
        Some("true"),
        None,
        Some(MIN_WIDTH)
    ));
    assert!(!supports_banner_for(
        true,
        false,
        Some("1"),
        None,
        Some(MIN_WIDTH)
    ));
    assert!(!supports_banner_for(
        true,
        false,
        None,
        Some("dumb"),
        Some(MIN_WIDTH)
    ));
    assert!(!supports_banner_for(
        true,
        false,
        None,
        None,
        Some(MIN_WIDTH - 1)
    ));
    assert!(supports_banner_for(
        true,
        false,
        Some("false"),
        Some("xterm-256color"),
        Some(MIN_WIDTH)
    ));
}

#[test]
fn terminal_width_helper_covers_default_parse_and_non_terminal_paths() {
    assert_eq!(terminal_width_for(false, Some("200")), None);
    assert_eq!(terminal_width_for(true, Some("200")), Some(200));
    assert_eq!(terminal_width_for(true, Some("not-a-number")), Some(120));
    assert_eq!(terminal_width_for(true, None), Some(120));
}

fn frame_grid_rows(frame: &str) -> Vec<&str> {
    frame
        .lines()
        .filter(|line| line.starts_with('│') && line.ends_with('│'))
        .map(|line| line.trim_matches('│'))
        .collect()
}

fn frame_inner_width(frame: &str) -> usize {
    frame
        .lines()
        .find(|line| line.starts_with('╭'))
        .expect("frame should include a top border")
        .trim_matches(['╭', '╮'])
        .chars()
        .count()
}

fn first_art_col(row: &str) -> usize {
    row.chars()
        .position(|ch| !ch.is_whitespace())
        .expect("figlet row should contain art")
}
