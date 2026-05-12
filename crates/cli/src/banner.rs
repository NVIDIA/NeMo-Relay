// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slanted ANSI-Shadow "NeMo Flow" banner.
//!
//! Static art: filled block letters in NVIDIA green, each row shifted one column right of the
//! row above for an italic lean. The settled frame includes a small "vX.Y.Z" tag in green at
//! the bottom-right.
//!
//! Three entry points:
//! - [`print_intro`] — wizard intro / bare `nemo-flow`
//! - [`print_doctor_header`] — settled static frame for `doctor`
//! - [`render_frame`] — pure helper for tests

use std::io::IsTerminal;

/// Filled-block NeMo Flow figlet with a per-row right shift so the letters lean italic. Six
/// content rows; the renderer prepends one blank row above and appends one below for spacing
/// and the docked version tag.
const BANNER_LINES: &[&str] = &[
    "             ███╗   ██╗███████╗███╗   ███╗ ██████╗      ███████╗██╗      ██████╗ ██╗   ██╗",
    "            ████╗  ██║██╔════╝████╗ ████║██╔═══██╗     ██╔════╝██║     ██╔═══██╗██║   ██║",
    "           ██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║     █████╗  ██║     ██║   ██║██║ █╗██║",
    "          ██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║     ██╔══╝  ██║     ██║   ██║██║██║██║",
    "         ██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝     ██║     ███████╗╚██████╔╝╚███╔███╔╝",
    "        ╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝      ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝",
];

/// Banner geometry (visual rows including the top and bottom spacing rails).
const FIGLET_ROWS: usize = 6;
const BOTTOM_RAIL: usize = FIGLET_ROWS + 1; // row index of the row below the figlet
const TOTAL_ROWS: usize = FIGLET_ROWS + 2; // top rail + 6 figlet rows + bottom rail

/// Version tag position, measured in columns.
const COL_END: usize = 92; // right edge below "Flow"

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

    // Build a 2D grid: empty top rail, the 6 figlet rows, empty bottom rail. Each cell is a
    // single char (we treat Unicode block chars as 1 display column wide, which is true for the
    // glyphs the figlet uses).
    let mut grid: Vec<Vec<char>> = Vec::with_capacity(TOTAL_ROWS);
    let dock_tag = format!(" v{}", env!("CARGO_PKG_VERSION"));
    let dock_width_needed = COL_END + dock_tag.chars().count() + 2;
    let max_width = BANNER_LINES
        .iter()
        .map(|l| l.chars().count())
        .max()
        .unwrap_or(0)
        .max(dock_width_needed);

    // Top rail (empty).
    grid.push(vec![' '; max_width]);
    // 6 figlet rows, padded to max_width.
    for line in BANNER_LINES {
        let mut row: Vec<char> = line.chars().collect();
        while row.len() < max_width {
            row.push(' ');
        }
        grid.push(row);
    }
    // Bottom rail (empty).
    grid.push(vec![' '; max_width]);

    // Overlay the docked version tag at bottom-right: just "vX.Y.Z" in dim green. No dot — the
    // version reads as a quiet label below "Flow", letting the brand mark stand on its own.
    let dock_col_start = COL_END;
    let dock_col_end = dock_col_start + dock_tag.chars().count();
    if docked {
        let dock_row = BOTTOM_RAIL;
        for (i, ch) in dock_tag.chars().enumerate() {
            let c = dock_col_start + i;
            if dock_row < grid.len() && c < grid[dock_row].len() {
                grid[dock_row][c] = ch;
            }
        }
    }

    // Top border row.
    push_border_line(&mut out, BORDER_TL, BORDER_TR, max_width, color);

    // Emit the grid with appropriate coloring per cell. Each grid row is wrapped with a
    // vertical border on the left and right, painted in NVIDIA green.
    for (row_idx, row) in grid.iter().enumerate() {
        if color {
            out.push_str(NVIDIA_GREEN);
            out.push(BORDER_V);
            out.push_str(RESET);
        } else {
            out.push(BORDER_V);
        }
        for (col_idx, ch) in row.iter().enumerate() {
            let in_dock_tag = docked
                && row_idx == BOTTOM_RAIL
                && col_idx >= dock_col_start
                && col_idx < dock_col_end;
            if in_dock_tag && *ch != ' ' {
                if color {
                    out.push_str(DOCK_TAG);
                    out.push(*ch);
                    out.push_str(RESET);
                } else {
                    out.push(*ch);
                }
            } else if is_figlet_glyph(*ch) {
                if color {
                    out.push_str(NVIDIA_GREEN);
                    out.push(*ch);
                    out.push_str(RESET);
                } else {
                    out.push(*ch);
                }
            } else {
                out.push(*ch);
            }
        }
        if color {
            out.push_str(NVIDIA_GREEN);
            out.push(BORDER_V);
            out.push_str(RESET);
        } else {
            out.push(BORDER_V);
        }
        out.push('\n');
    }

    // Bottom border row.
    push_border_line(&mut out, BORDER_BL, BORDER_BR, max_width, color);

    out
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
    println!("  NeMo Flow v{version}");
    println!();
}

#[cfg(test)]
#[path = "../tests/coverage/banner_tests.rs"]
mod tests;
