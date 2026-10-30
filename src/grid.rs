//! Finds the item cell under the mouse from the stash grid lines.
//!
//! RatEye's located box can be off by a large fraction of a slot (seen: 20 px off and 25 % too
//! large at 1440p). A bad box drags in neighbouring names and EFT's own hover tooltip, which the
//! OCR stage then reads with full confidence. The grid is exact: the item's border lines are the
//! nearest grid-coloured lines around the mouse, because the item covers the inner lines.

use crate::imgops::{GRID_RGB, PITCH};
use image::RgbImage;

/// Max summed |ΔR|+|ΔG|+|ΔB| from `GRID_RGB` for a pixel to count as grid line.
const TOL: i32 = 24;
/// Share of pixels in the probe band that must be grid-coloured for a line.
const LINE_FRAC: f32 = 0.6;
/// Widest item searched for, in slots.
const MAX_SLOTS: f32 = 12.0;
/// A measured side must be this close to a whole number of slots.
const SLOT_SNAP: f32 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub slots: (u32, u32),
}

fn is_grid(img: &RgbImage, x: i32, y: i32) -> bool {
    let p = img.get_pixel(x as u32, y as u32).0;
    let g = GRID_RGB.0;
    (0..3).map(|i| (p[i] as i32 - g[i] as i32).abs()).sum::<i32>() <= TOL
}

/// Fraction of grid pixels on the segment `fixed` × `[from, to)` (vertical when `vertical`).
fn line_at(img: &RgbImage, vertical: bool, fixed: i32, from: i32, to: i32) -> bool {
    let n = (to - from).max(1);
    let hits = (from..to).filter(|&t| if vertical { is_grid(img, fixed, t) } else { is_grid(img, t, fixed) }).count();
    hits as f32 / n as f32 >= LINE_FRAC
}

/// Nearest line on each side of `m` along one axis; `cross` is the other coordinate of the mouse.
fn edges(img: &RgbImage, vertical: bool, m: i32, cross: i32, pitch: f32) -> (Option<i32>, Option<i32>) {
    let (len, cross_len) = if vertical { (img.width() as i32, img.height() as i32) } else { (img.height() as i32, img.width() as i32) };
    let band = (pitch / 3.0) as i32;
    let (from, to) = ((cross - band).max(0), (cross + band).min(cross_len));
    let reach = (pitch * MAX_SLOTS) as i32;
    let low = (((m - reach).max(0))..=m.min(len - 1)).rev().find(|&v| line_at(img, vertical, v, from, to));
    let high = ((m + 1).max(0)..(m + reach).min(len)).find(|&v| line_at(img, vertical, v, from, to));
    (low, high)
}

/// Start, length and slot count along one axis. One missing edge (e.g. the stash viewport cuts
/// the last row) is filled in from `hint` slots.
fn side(low: Option<i32>, high: Option<i32>, pitch: f32, hint: u32) -> Option<(i32, i32, u32)> {
    let hint = hint.max(1);
    match (low, high) {
        (Some(a), Some(b)) => {
            let slots = (b - a) as f32 / pitch;
            let n = slots.round();
            (n >= 1.0 && (slots - n).abs() <= SLOT_SNAP).then(|| (a, b - a, n as u32))
        }
        (Some(a), None) => Some((a, (hint as f32 * pitch).round() as i32, hint)),
        (None, Some(b)) => {
            let len = (hint as f32 * pitch).round() as i32;
            Some((b - len, len, hint))
        }
        (None, None) => None,
    }
}

/// The item cell containing `mouse`, or `None` when the grid is not visible there.
/// `hint_slots` is only used when one edge of an axis is not visible.
pub fn locate_cell(img: &RgbImage, mouse: (i32, i32), scale: f32, hint_slots: (u32, u32)) -> Option<Cell> {
    let (mx, my) = mouse;
    if mx < 0 || my < 0 || mx >= img.width() as i32 || my >= img.height() as i32 {
        return None;
    }
    let pitch = PITCH * scale;
    let (l, r) = edges(img, true, mx, my, pitch);
    let (t, b) = edges(img, false, my, mx, pitch);
    let (x, w, sw) = side(l, r, pitch, hint_slots.0)?;
    let (y, h, sh) = side(t, b, pitch, hint_slots.1)?;
    Some(Cell { x, y, w, h, slots: (sw, sh) })
}
