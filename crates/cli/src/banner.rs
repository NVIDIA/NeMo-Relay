// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Slanted ANSI-Shadow "NeMo Flow" banner with a tracer dot that curves over the brand.
//!
//! Static art: filled block letters in NVIDIA green, each row shifted one column right of the
//! row above for an italic lean. Animation: a single bright dot enters from the top-left,
//! glides smoothly horizontally above "NeMo", dips through the gap between "NeMo" and "Flow",
//! glides horizontally below "Flow", and the banner then settles with a small "vX.Y.Z" tag in
//! green at the bottom-right.
//!
//! Three entry points:
//! - [`print_intro`] — wizard intro / bare `nemo-flow` (animated)
//! - [`print_doctor_header`] — settled static frame for `doctor` (no animation)
//! - [`render_frame`] — pure helper for tests

use std::io::{IsTerminal, Write};
use std::time::Duration;

/// Filled-block NeMo Flow figlet with a per-row right shift so the letters lean italic. Six
/// content rows; the renderer prepends one blank row above and appends one below to host the
/// tracer dot's path.
const BANNER_LINES: &[&str] = &[
    "             ███╗   ██╗███████╗███╗   ███╗ ██████╗      ███████╗██╗      ██████╗ ██╗   ██╗",
    "            ████╗  ██║██╔════╝████╗ ████║██╔═══██╗     ██╔════╝██║     ██╔═══██╗██║   ██║",
    "           ██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║     █████╗  ██║     ██║   ██║██║ █╗██║",
    "          ██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║     ██╔══╝  ██║     ██║   ██║██║██║██║",
    "         ██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝     ██║     ███████╗╚██████╔╝╚███╔███╔╝",
    "        ╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝      ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝",
];

/// Banner geometry (visual rows including the dot's top and bottom rails).
const FIGLET_ROWS: usize = 6;
const TOP_RAIL: usize = 0;
const BOTTOM_RAIL: usize = FIGLET_ROWS + 1; // row index of the row below the figlet
const TOTAL_ROWS: usize = FIGLET_ROWS + 2; // top rail + 6 figlet rows + bottom rail

/// Tracer-dot path waypoints — measured in columns. The dot moves linearly in col across
/// frames; its row follows an S-shape (top rail → smooth descent → bottom rail) based on
/// which segment the column falls into.
const COL_START: usize = 13; // above the "N" of NeMo
const COL_END: usize = 92; // right edge below "Flow"
const COL_DIP_START: usize = 44; // start descending after we clear "NeMo"
const COL_DIP_END: usize = 56; // finish descending before we hit "Flow"

const MIN_WIDTH: usize = 105;

// NVIDIA green on the figlet text and the surrounding border. The tracer head is a bright
// mint-green dot. The settled docked tag at bottom-right is dim green to read as a quiet
// version label without competing with the brand mark.
const NVIDIA_GREEN: &str = "\x1b[38;5;112m";
const DOT_HEAD: &str = "\x1b[1;38;5;121m";
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

/// Total animation frames for the tracer dot's traversal. Drives both the timing in
/// `animate_reveal` and the path-step helper used by tests. Higher count = smoother glide.
pub(crate) const TRACER_FRAMES: usize = 160;

/// Returns the tracer dot's `(row, col)` at the given frame. The dot moves linearly in `col`
/// from `COL_START` to `COL_END` and follows an S-shape in `row`: stays on the top rail until
/// it has cleared "NeMo", smoothly descends through the gap, then stays on the bottom rail
/// until it exits below "Flow". `None` when the animation has finished.
pub(crate) fn tracer_position(frame: usize) -> Option<(usize, usize)> {
    if frame >= TRACER_FRAMES {
        return None;
    }
    let t = frame as f32 / (TRACER_FRAMES - 1).max(1) as f32;
    let col = COL_START as f32 + (COL_END - COL_START) as f32 * t;
    let col_usize = col as usize;
    let row = if col_usize <= COL_DIP_START {
        TOP_RAIL as f32
    } else if col_usize >= COL_DIP_END {
        BOTTOM_RAIL as f32
    } else {
        // Smooth ease (smoothstep) between top rail and bottom rail across the dip range.
        let local = (col_usize - COL_DIP_START) as f32 / (COL_DIP_END - COL_DIP_START) as f32;
        let eased = local * local * (3.0 - 2.0 * local);
        TOP_RAIL as f32 + (BOTTOM_RAIL - TOP_RAIL) as f32 * eased
    };
    Some((row.round() as usize, col_usize))
}

/// Pure renderer. `tracer` carries the dot's (row, col) for this frame, or `None` to render
/// the settled static banner. `color=false` strips all ANSI escapes.
pub(crate) fn render_frame(tracer: Option<(usize, usize)>, color: bool) -> String {
    render_frame_inner(tracer, color, false)
}

/// Settled frame with a glowing "● vX.Y.Z" tag docked at the bottom-right under "Flow". Used
/// after the animation finishes and as the static frame for the doctor header.
pub(crate) fn render_docked_frame(color: bool) -> String {
    render_frame_inner(None, color, true)
}

fn render_frame_inner(tracer: Option<(usize, usize)>, color: bool, docked: bool) -> String {
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

    // Overlay the tracer head only — no trail. Smooth motion comes from the higher frame count.
    if let Some((row, col)) = tracer
        && row < grid.len()
        && col < grid[row].len()
    {
        grid[row][col] = '●';
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
            } else if Some((row_idx, col_idx)) == tracer && *ch == '●' {
                if color {
                    out.push_str(DOT_HEAD);
                    out.push(*ch);
                    out.push_str(RESET);
                } else {
                    out.push('*');
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
    animate_reveal();
}

pub(crate) fn print_doctor_header() {
    if !supports_banner() {
        print_plain_header();
        return;
    }
    print!("{}", render_docked_frame(true));
}

fn animate_reveal() {
    // Smoothness strategy:
    // 1. Print the static banner ONCE so the figlet never flickers.
    // 2. Save cursor (DEC ESC 7), then per-frame restore + move-up + move-to-col to repaint
    //    just the dot cell. Erasing + repainting one cell is far cheaper than redrawing the
    //    full banner each frame and reads as continuous motion.
    // 3. Skip frames where the integer column hasn't advanced — we'd just sleep and redraw
    //    the same cell, wasting time and breaking the perceived pace.
    let frame_ms = 8u64;
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\x1b[?25l");
    // Paint the static banner. Cursor lands on the line just below the bottom rail.
    let _ = write!(stdout, "{}", render_frame(None, true));
    // Save cursor position so each frame can restore back to this anchor before navigating.
    let _ = write!(stdout, "\x1b7");
    let _ = stdout.flush();

    let mut last_pos: Option<(usize, usize)> = None;
    for f in 0..TRACER_FRAMES {
        let Some((row, col)) = tracer_position(f) else {
            break;
        };
        // Skip duplicate-column frames — keeps motion paced even though we still sleep.
        if last_pos == Some((row, col)) {
            std::thread::sleep(Duration::from_millis(frame_ms));
            continue;
        }
        // Erase the previous dot (write a space at the old position).
        if let Some((pr, pc)) = last_pos {
            paint_cell(&mut stdout, pr, pc, ' ', None);
        }
        // Draw the current dot.
        paint_cell(&mut stdout, row, col, '●', Some(DOT_HEAD));
        let _ = stdout.flush();
        last_pos = Some((row, col));
        std::thread::sleep(Duration::from_millis(frame_ms));
    }

    // Settle: erase the last dot and stamp the version tag at the dock spot.
    if let Some((pr, pc)) = last_pos {
        paint_cell(&mut stdout, pr, pc, ' ', None);
    }
    let dock_tag = format!(" v{}", env!("CARGO_PKG_VERSION"));
    // Move to (BOTTOM_RAIL, COL_END) inside the border and write the dim-green tag. Anchor sits
    // below the bottom border line; +1 vertical for the border, +1 horizontal for the left
    // border.
    let _ = write!(stdout, "\x1b8"); // restore to anchor below banner
    let _ = write!(stdout, "\x1b[{}A", TOTAL_ROWS - BOTTOM_RAIL + 1);
    let _ = write!(stdout, "\x1b[{}G", COL_END + 2);
    let _ = write!(stdout, "{DOCK_TAG}{dock_tag}{RESET}");
    let _ = write!(stdout, "\x1b8");
    let _ = write!(stdout, "\x1b[?25h");
    let _ = stdout.flush();
}

/// Paint a single character at grid (row, col) relative to the anchor saved by `\x1b7` after
/// the static banner was printed. Accounts for the surrounding border: +1 row offset for the
/// bottom border line and +1 column for the left border. `color` is an optional SGR prefix
/// (RESET is always emitted after the char). Cursor is left at the anchor.
fn paint_cell(out: &mut std::io::Stdout, row: usize, col: usize, ch: char, color: Option<&str>) {
    let _ = write!(out, "\x1b8");
    let _ = write!(out, "\x1b[{}A", TOTAL_ROWS - row + 1);
    let _ = write!(out, "\x1b[{}G", col + 2);
    if let Some(c) = color {
        let _ = write!(out, "{c}{ch}{RESET}");
    } else {
        let _ = write!(out, "{ch}");
    }
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
