// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slanted ANSI-Shadow "NeMo Relay" banner.
//!
//! Static art: filled block letters in NVIDIA green, each row shifted one column right of the
//! row above for an italic lean. The settled frame includes a small "vX.Y.Z" tag in green at
//! the bottom-right.
//!
//! Three entry points:
//! - [`print_intro`] — wizard intro / bare `nemo-relay`
//! - [`print_doctor_header`] — settled static frame for `doctor`
//! - [`render_frame`] — pure helper for tests

use std::io::IsTerminal;

/// Filled-block NeMo Relay figlet generated with ANSI Shadow. Six content rows; the renderer
/// prepends one blank row above and appends one below for spacing and the docked version tag.
const BANNER_LINES: &[&str] = &[
    "███╗   ██╗███████╗███╗   ███╗ ██████╗     ██████╗ ███████╗██╗      █████╗ ██╗   ██╗",
    "████╗  ██║██╔════╝████╗ ████║██╔═══██╗    ██╔══██╗██╔════╝██║     ██╔══██╗╚██╗ ██╔╝",
    "██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║    ██████╔╝█████╗  ██║     ███████║ ╚████╔╝",
    "██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║    ██╔══██╗██╔══╝  ██║     ██╔══██║  ╚██╔╝",
    "██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝    ██║  ██║███████╗███████╗██║  ██║   ██║",
    "╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝     ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝   ╚═╝",
];

/// Banner geometry (visual rows including the top and bottom spacing rails).
const FIGLET_ROWS: usize = 6;
const BOTTOM_RAIL: usize = FIGLET_ROWS + 1; // row index of the row below the figlet
const TOTAL_ROWS: usize = FIGLET_ROWS + 2; // top rail + 6 figlet rows + bottom rail

/// Version tag position, measured in columns.
const COL_END: usize = 92; // version tag dock

const MIN_WIDTH: usize = 105;

// NVIDIA green on the figlet text and the surrounding border. The settled docked tag at
// bottom-right is dim green to read as a quiet version label without competing with the brand
// mark.
const NVIDIA_GREEN: &str = "\x1b[38;5;112m";
const DOCK_TAG: &str = "\x1b[2;38;5;112m";
const RESET: &str = "\x1b[0m";

// Rounded border glyphs. Drawn in NVIDIA green around the whole banner.
const BORDER_TL: char = '╭';
const BORDER_TR: char = '╮';
const BORDER_BL: char = '╰';
const BORDER_BR: char = '╯';
const BORDER_H: char = '─';
const BORDER_V: char = '│';

#[derive(Clone, Copy)]
struct DockTagSpan {
    row: usize,
    start: usize,
    end: usize,
}

enum CellStyle {
    DockTag,
    Figlet,
    Plain,
}

fn supports_banner() -> bool {
    if !std::io::stdout().is_terminal() {
        return false;
    }
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if std::env::var("CI").is_ok_and(|v| v == "true" || v == "1") {
        return false;
    }
    if std::env::var("TERM").as_deref() == Ok("dumb") {
        return false;
    }
    terminal_width().is_some_and(|w| w >= MIN_WIDTH)
}

fn terminal_width() -> Option<usize> {
    if !std::io::stdout().is_terminal() {
        return None;
    }
    std::env::var("COLUMNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or(Some(120))
}

/// Pure renderer for the static banner. `color=false` strips all ANSI escapes.
#[cfg(test)]
pub(crate) fn render_frame(color: bool) -> String {
    render_frame_inner(color, false)
}

/// Settled frame with a quiet "vX.Y.Z" tag docked at the bottom-right under "Flow". Used by
/// the intro and doctor header.
pub(crate) fn render_docked_frame(color: bool) -> String {
    render_frame_inner(color, true)
}

fn render_frame_inner(color: bool, docked: bool) -> String {
    let mut out = String::with_capacity(BANNER_LINES.iter().map(|l| l.len() + 64).sum());
    out.push('\n');

    let dock_tag = format!(" v{}", env!("CARGO_PKG_VERSION"));
    let max_width = frame_width(&dock_tag);
    let mut grid = build_grid(max_width);
    let dock_tag_span = docked.then(|| overlay_dock_tag(&mut grid, &dock_tag));

    // Top border row.
    push_border_line(&mut out, BORDER_TL, BORDER_TR, max_width, color);

    // Emit the grid with appropriate coloring per cell. Each grid row is wrapped with a
    // vertical border on the left and right, painted in NVIDIA green.
    for (row_idx, row) in grid.iter().enumerate() {
        push_grid_row(&mut out, row_idx, row, dock_tag_span, color);
        out.push('\n');
    }

    // Bottom border row.
    push_border_line(&mut out, BORDER_BL, BORDER_BR, max_width, color);

    out
}

fn frame_width(dock_tag: &str) -> usize {
    let dock_width_needed = COL_END + dock_tag.chars().count() + 2;
    BANNER_LINES
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0)
        .max(dock_width_needed)
}

fn build_grid(width: usize) -> Vec<Vec<char>> {
    // Empty top rail, the 6 figlet rows, and an empty bottom rail. Each cell is a single char
    // because the figlet's block and box glyphs render as one display column in target terminals.
    let mut grid = Vec::with_capacity(TOTAL_ROWS);
    let art_width = banner_art_width();
    let start_col = width.saturating_sub(art_width) / 2;
    grid.push(vec![' '; width]);
    grid.extend(
        BANNER_LINES
            .iter()
            .map(|line| padded_row(line, width, start_col)),
    );
    grid.push(vec![' '; width]);
    grid
}

fn banner_art_width() -> usize {
    BANNER_LINES
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
}

fn padded_row(line: &str, width: usize, start_col: usize) -> Vec<char> {
    let mut row = vec![' '; width];

    for (index, ch) in line.chars().enumerate() {
        if let Some(cell) = row.get_mut(start_col + index) {
            *cell = ch;
        }
    }

    row
}

fn overlay_dock_tag(grid: &mut [Vec<char>], dock_tag: &str) -> DockTagSpan {
    let span = DockTagSpan {
        row: BOTTOM_RAIL,
        start: COL_END,
        end: COL_END + dock_tag.chars().count(),
    };
    for (index, ch) in dock_tag.chars().enumerate() {
        grid[span.row][span.start + index] = ch;
    }
    span
}

fn push_grid_row(
    out: &mut String,
    row_idx: usize,
    row: &[char],
    dock_tag_span: Option<DockTagSpan>,
    color: bool,
) {
    push_vertical_border(out, color);
    for (col_idx, ch) in row.iter().copied().enumerate() {
        push_cell(
            out,
            ch,
            cell_style(ch, row_idx, col_idx, dock_tag_span),
            color,
        );
    }
    push_vertical_border(out, color);
}

fn push_vertical_border(out: &mut String, color: bool) {
    push_styled_char(out, BORDER_V, Some(NVIDIA_GREEN), color);
}

fn push_cell(out: &mut String, ch: char, style: CellStyle, color: bool) {
    match style {
        CellStyle::DockTag => push_styled_char(out, ch, Some(DOCK_TAG), color),
        CellStyle::Figlet => push_styled_char(out, ch, Some(NVIDIA_GREEN), color),
        CellStyle::Plain => out.push(ch),
    }
}

fn push_styled_char(out: &mut String, ch: char, style: Option<&str>, color: bool) {
    if color && let Some(style) = style {
        out.push_str(style);
        out.push(ch);
        out.push_str(RESET);
    } else {
        out.push(ch);
    }
}

fn cell_style(
    ch: char,
    row_idx: usize,
    col_idx: usize,
    dock_tag_span: Option<DockTagSpan>,
) -> CellStyle {
    if dock_tag_span.is_some_and(|span| {
        row_idx == span.row && col_idx >= span.start && col_idx < span.end && ch != ' '
    }) {
        CellStyle::DockTag
    } else if is_figlet_glyph(ch) {
        CellStyle::Figlet
    } else {
        CellStyle::Plain
    }
}

fn push_border_line(out: &mut String, left: char, right: char, inner_width: usize, color: bool) {
    if color {
        out.push_str(NVIDIA_GREEN);
        out.push(left);
        for _ in 0..inner_width {
            out.push(BORDER_H);
        }
        out.push(right);
        out.push_str(RESET);
    } else {
        out.push(left);
        for _ in 0..inner_width {
            out.push(BORDER_H);
        }
        out.push(right);
    }
    out.push('\n');
}

fn is_figlet_glyph(ch: char) -> bool {
    matches!(ch, '█' | '╗' | '╔' | '╝' | '╚' | '═' | '║')
}

pub(crate) fn print_intro() {
    if !supports_banner() {
        print_plain_header();
        return;
    }
    print!("{}", render_docked_frame(true));
}

pub(crate) fn print_doctor_header() {
    if !supports_banner() {
        print_plain_header();
        return;
    }
    print!("{}", render_docked_frame(true));
}

fn print_plain_header() {
    let version = env!("CARGO_PKG_VERSION");
    println!();
    println!("  NeMo Relay v{version}");
    println!();
}

#[cfg(test)]
#[path = "../tests/coverage/banner_tests.rs"]
mod tests;
