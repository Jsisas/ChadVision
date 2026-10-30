//! Hover tooltip: locate the item-name line next to the cursor, OCR it, match full item names.
use super::{ocr::Ocr, Candidate, Identifier, Input, Tooltip};
use crate::catalog::Catalog;
use crate::lexicon;
use image::{imageops, RgbImage};
use std::sync::Arc;

/// Minimum full-name score to trust a tooltip reading.
pub const NAME_ACCEPT: f32 = 0.8;
/// Text pixels: luma ≥ min(TEXT_LUMA, TEXT_REL × brightest pixel in the search region), never below TEXT_FLOOR.
/// Relative part keeps thin, blurred glyphs at small scales (peak luma drops to ~130).
const TEXT_LUMA: u32 = 150;
const TEXT_REL: f32 = 0.6;
const TEXT_FLOOR: u32 = 100;

fn luma(p: &image::Rgb<u8>) -> u32 {
    (299 * p[0] as u32 + 587 * p[1] as u32 + 114 * p[2] as u32) / 1000
}

/// Interior of EFT's tooltip box is pure black; nothing else in the inventory UI is.
fn is_box(p: &image::Rgb<u8>) -> bool {
    p.0.iter().all(|&c| c <= 4)
}

/// Interior (x0, y0, x1, y1 exclusive) of the black tooltip box nearest the cursor within
/// `[y0, y1)`, or `None` when no box is visible. Text search is confined to it: across the
/// whole window other labels and icon art merge every row into one band.
fn locate_box(img: &RgbImage, mouse: (u32, u32), y0: u32, y1: u32, s: f32) -> Option<(u32, u32, u32, u32)> {
    let min_run = (30.0 * s).round() as u32;
    let (mx, my) = mouse;
    let mut best: Option<(u32, u32, u32, u64)> = None; // (y, l, r, distance²)
    for y in y0..y1 {
        let mut x = 0;
        while x < img.width() {
            if !is_box(img.get_pixel(x, y)) {
                x += 1;
                continue;
            }
            let l = x;
            while x < img.width() && is_box(img.get_pixel(x, y)) {
                x += 1;
            }
            if x - l >= min_run {
                let dx = if mx < l { l - mx } else { mx.saturating_sub(x - 1) } as u64;
                let d = dx * dx + (y.abs_diff(my) as u64).pow(2);
                if best.is_none_or(|b| d < b.3) {
                    best = Some((y, l, x, d));
                }
            }
        }
    }
    let (y, l, r, _) = best?;
    // Just inside the left edge: the text starts further in, so this column is all box.
    let edge = l + 1;
    let col = |y: u32| is_box(img.get_pixel(edge, y));
    let top = (0..=y).rev().take_while(|&v| col(v)).last()?;
    let bottom = (y..img.height()).take_while(|&v| col(v)).last()? + 1;
    Some((l, top, r, bottom))
}

/// Bounding box (x, y, w, h) of the bright text line nearest the cursor, if any.
pub fn locate_name(tip: &Tooltip) -> Option<(u32, u32, u32, u32)> {
    let img = &tip.image;
    let s = tip.scale;
    let (mx, my) = tip.mouse;
    let reach = (100.0 * s).round() as u32;
    let (bx0, y0, bx1, y1) = locate_box(img, tip.mouse, my.saturating_sub(reach), (my + reach).min(img.height()), s)
        .unwrap_or((mx.min(img.width()), my.saturating_sub(reach), img.width(), (my + reach).min(img.height())));
    let (x0, xe) = (bx0, bx1);
    let peak = (y0..y1).flat_map(|y| (x0..xe).map(move |x| (x, y))).map(|(x, y)| luma(img.get_pixel(x, y))).max()?;
    let thr = TEXT_LUMA.min((peak as f32 * TEXT_REL) as u32).max(TEXT_FLOOR);
    let text = |x: u32, y: u32| luma(img.get_pixel(x, y)) >= thr;

    // Row bands of text.
    let rows: Vec<bool> = (y0..y1).map(|y| (x0..xe).filter(|&x| text(x, y)).count() >= 2).collect();
    let (min_h, max_h) = ((8.0 * s).round() as u32, (30.0 * s).round() as u32);
    let mut bands: Vec<(u32, u32)> = Vec::new(); // (top, bottom exclusive)
    let mut cur: Option<(u32, u32)> = None;
    for (k, &on) in rows.iter().enumerate() {
        let y = y0 + k as u32;
        if on {
            cur = Some(match cur {
                Some((t, b)) if y <= b + 2 => (t, y + 1),
                Some(band) => {
                    bands.push(band);
                    (y, y + 1)
                }
                None => (y, y + 1),
            });
        }
    }
    bands.extend(cur);
    let (top, bottom) = bands
        .into_iter()
        .filter(|&(t, b)| (min_h..=max_h).contains(&(b - t)))
        .min_by_key(|&(t, _)| t.abs_diff(my))?;

    // Column segments inside the band; keep the one nearest the cursor.
    // Blurred small text binarises spottily, so tolerate gaps up to one line height.
    let gap = ((12.0 * s).round() as u32).max(bottom - top);
    let mut segs: Vec<(u32, u32)> = Vec::new();
    for x in x0..xe {
        if (top..bottom).any(|y| text(x, y)) {
            match segs.last_mut() {
                Some((_, e)) if x <= *e + gap => *e = x + 1,
                _ => segs.push((x, x + 1)),
            }
        }
    }
    let (l, r) = segs.into_iter().min_by_key(|&(l, _)| l.abs_diff(mx))?;
    let pad = (3.0 * s).round() as u32;
    let (bx, by) = (l.saturating_sub(pad), top.saturating_sub(pad));
    let bw = (r + pad).min(img.width()) - bx;
    let bh = (bottom + pad).min(img.height()) - by;
    Some((bx, by, bw, bh))
}

/// OCR the item name from a tooltip; empty when no text line is found.
pub fn read_name(ocr: &Ocr, tip: &Tooltip) -> String {
    match locate_name(tip) {
        Some((x, y, w, h)) => ocr.read_line(&imageops::crop_imm(&tip.image, x, y, w, h).to_image()),
        None => String::new(),
    }
}

/// Rank same-size catalog items by full-name score.
pub fn rank_names(catalog: &Catalog, text: &str, slots: (u32, u32), k: usize) -> Vec<(usize, f32)> {
    let scored = catalog.candidates(slots).into_iter().map(|(i, _)| (i, lexicon::score(text, &catalog.items[i].name))).collect();
    super::top_k(scored, k)
}

/// Name-only identification from the tooltip.
pub struct TooltipName {
    catalog: Arc<Catalog>,
    ocr: Arc<Ocr>,
}

impl TooltipName {
    pub fn new(catalog: Arc<Catalog>, ocr: Arc<Ocr>) -> Self {
        Self { catalog, ocr }
    }
}

impl Identifier for TooltipName {
    fn name(&self) -> &'static str {
        "tooltip"
    }

    fn identify(&self, input: &Input) -> Vec<Candidate> {
        let Some(tip) = input.tooltip else { return Vec::new() };
        let text = read_name(&self.ocr, tip);
        let ranked = rank_names(&self.catalog, &text, input.slots, 5);
        if ranked.first().is_none_or(|c| c.1 < NAME_ACCEPT) {
            return Vec::new();
        }
        ranked.into_iter().map(|(i, s)| Candidate { item_id: self.catalog.items[i].id.clone(), score: s }).collect()
    }
}
