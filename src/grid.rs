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
/// Tolerance for a second look when one edge of an axis is missing. Rig grids lighten towards
/// the top: a 1×2 slot's top border measured 36 off (78,95,101). Too loose for the first look:
/// in the stash it finds lines inside item art.
const LOOSE_TOL: i32 = 40;
/// Share of pixels in the probe band that must be grid-coloured for a line.
const LINE_FRAC: f32 = 0.6;
/// Widest item searched for, in slots.
const MAX_SLOTS: f32 = 12.0;
/// A measured side must be this close to a whole number of slots. Measured cells land within
/// 0.06; 0.2 accepted a grid-coloured run inside icon art as an edge (0.81 of a slot).
const SLOT_SNAP: f32 = 0.1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub slots: (u32, u32),
}

/// Grid lines lighten towards the top of the stash and of rigs: (73,81,84) at the bottom,
/// about (78,93,100) in the top stash rows at 1440p. First looks accept colours close to that
/// ramp (`RAMP_TOL`); the loose second look accepts either the ramp or `GRID_RGB` within its tolerance.
const GRID_LIGHT: [i32; 3] = [80, 98, 106];

fn is_grid(img: &RgbImage, x: i32, y: i32, tol: i32) -> bool {
    let p = img.get_pixel(x as u32, y as u32).0.map(i32::from);
    let g = GRID_RGB.0.map(i32::from);
    let d: [i32; 3] = std::array::from_fn(|i| GRID_LIGHT[i] - g[i]);
    let t = (((p[1] - g[1]) as f32) / d[1] as f32).clamp(0.0, 1.0);
    let ramp = (0..3).map(|i| (p[i] - g[i] - (t * d[i] as f32).round() as i32).abs()).sum::<i32>();
    let base = (0..3).map(|i| (p[i] - g[i]).abs()).sum::<i32>();
    // `tol` is `TOL` on the first look and `LOOSE_TOL` on the second.
    if tol == TOL { ramp <= RAMP_TOL } else { base.min(ramp) <= tol }
}
/// Distance from the lightening ramp for the first look; tighter than `TOL` because the ramp
/// covers more colours (wider tolerance found lines inside icon art).
const RAMP_TOL: i32 = 12;

/// Fraction of grid pixels on the segment `fixed` × `[from, to)` (vertical when `vertical`).
fn line_at(img: &RgbImage, vertical: bool, fixed: i32, from: i32, to: i32, tol: i32) -> bool {
    let n = (to - from).max(1);
    let hits = (from..to).filter(|&t| if vertical { is_grid(img, fixed, t, tol) } else { is_grid(img, t, fixed, tol) }).count();
    hits as f32 / n as f32 >= LINE_FRAC
}

/// Lines on each side of `m` along one axis, nearest first; `cross` is the other coordinate of
/// the mouse. Icon art can contain grid-coloured runs, so more than the nearest line is kept.
fn edges(img: &RgbImage, vertical: bool, m: i32, cross: i32, pitch: f32, tol: i32) -> (Vec<i32>, Vec<i32>) {
    let (len, cross_len) = if vertical { (img.width() as i32, img.height() as i32) } else { (img.height() as i32, img.width() as i32) };
    let band = (pitch / 3.0) as i32;
    let (from, to) = ((cross - band).max(0), (cross + band).min(cross_len));
    let reach = (pitch * MAX_SLOTS) as i32;
    let low = (((m - reach).max(0))..=m.min(len - 1)).rev().filter(|&v| line_at(img, vertical, v, from, to, tol)).collect();
    let high = ((m + 1).max(0)..(m + reach).min(len)).filter(|&v| line_at(img, vertical, v, from, to, tol)).collect();
    (low, high)
}

/// Start, length and slot count along one axis: the nearest pair of lines that spans a whole
/// number of slots. One missing edge (e.g. the stash viewport cuts the last row) is filled in
/// from `hint` slots.
fn side(low: &[i32], high: &[i32], pitch: f32, hint: u32) -> Option<(i32, i32, u32)> {
    let hint = hint.max(1);
    let len = (hint as f32 * pitch).round() as i32;
    match (low.first(), high.first()) {
        (Some(_), Some(_)) => {
            let mut pairs: Vec<(i32, i32)> = low.iter().flat_map(|&a| high.iter().map(move |&b| (a, b))).collect();
            // Nearest to the mouse first: the inner pair has the smallest span.
            pairs.sort_by_key(|&(a, b)| b - a);
            pairs.into_iter().find_map(|(a, b)| {
                let slots = (b - a) as f32 / pitch;
                let n = slots.round();
                (n >= 1.0 && (slots - n).abs() <= SLOT_SNAP).then(|| (a, b - a, n as u32))
            })
        }
        (Some(&a), None) => Some((a, len, hint)),
        (None, Some(&b)) => Some((b - len, len, hint)),
        (None, None) => None,
    }
}

/// One axis of the cell. When only one edge is found, a second look with `LOOSE_TOL` is tried
/// before the hint: it must find both edges a whole number of slots apart, else the hint stands.
fn axis(img: &RgbImage, vertical: bool, m: i32, cross: i32, pitch: f32, hint: u32) -> Option<(i32, i32, u32)> {
    let (low, high) = edges(img, vertical, m, cross, pitch, TOL);
    if low.is_empty() != high.is_empty() {
        let (loose_low, loose_high) = edges(img, vertical, m, cross, pitch, LOOSE_TOL);
        if !loose_low.is_empty() && !loose_high.is_empty() {
            if let Some(found) = side(&loose_low, &loose_high, pitch, hint) {
                return Some(found);
            }
        }
    }
    side(&low, &high, pitch, hint)
}

/// The item cell containing `mouse`, or `None` when the grid is not visible there.
/// `hint_slots` is only used when one edge of an axis is not visible.
pub fn locate_cell(img: &RgbImage, mouse: (i32, i32), scale: f32, hint_slots: (u32, u32)) -> Option<Cell> {
    let (mx, my) = mouse;
    if mx < 0 || my < 0 || mx >= img.width() as i32 || my >= img.height() as i32 {
        return None;
    }
    let pitch = PITCH * scale;
    let (y, h, sh) = axis(img, false, my, mx, pitch, hint_slots.1)?;
    // Vertical lines are probed around the mouse row and around the cell's middle row: near the
    // top border of a top-row cell the mouse band reaches into the stash header (no lines), while
    // icon art can hide a border at the middle row. The narrower whole-slot answer wins.
    let at_mouse = axis(img, true, mx, my, pitch, hint_slots.0);
    let at_middle = axis(img, true, mx, y + h / 2, pitch, hint_slots.0);
    let (x, w, sw) = match (at_mouse, at_middle) {
        (Some(a), Some(b)) => if b.1 < a.1 { b } else { a },
        (a, b) => a.or(b)?,
    };
    Some(Cell { x, y, w, h, slots: (sw, sh) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stash grid at 1440p (84 px slots) with the 1×1 item at slot (1, 1) and a grid-coloured
    /// stripe inside its icon art, as on the condensed milk can.
    #[test]
    fn ignores_grid_coloured_icon_art() {
        let scale = 4.0 / 3.0;
        let p = 84;
        let mut img = RgbImage::from_pixel(5 * p as u32, 5 * p as u32, image::Rgb([30, 32, 33]));
        for v in (0..5 * p).step_by(p as usize) {
            for t in 0..5 * p {
                img.put_pixel(v as u32, t as u32, GRID_RGB);
                img.put_pixel(t as u32, v as u32, GRID_RGB);
            }
        }
        for y in p..2 * p {
            img.put_pixel((p + 16) as u32, y as u32, GRID_RGB);
        }
        let cell = locate_cell(&img, (p + 30, p + 30), scale, (1, 1)).expect("cell");
        assert_eq!((cell.x, cell.y, cell.w, cell.h, cell.slots), (p, p, p, p, (1, 1)));
    }
}
