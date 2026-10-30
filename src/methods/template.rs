//! RatEye baseline: SqDiffNormed template matching against every same-size reference.
use super::{top_k, Candidate, Identifier};
use crate::catalog::Catalog;
use crate::imgops::crop_to_base_padded;
use image::{imageops, RgbImage};
use rayon::prelude::*;
use std::sync::Arc;

const SEARCH: i64 = 4;

struct Tpl {
    w: usize,
    h: usize,
    px: Vec<f32>,
    sumsq: f32,
}

impl Tpl {
    fn new(img: &RgbImage) -> Self {
        let px: Vec<f32> = img.as_raw().iter().map(|&v| v as f32).collect();
        let sumsq = px.iter().map(|v| v * v).sum();
        Self { w: img.width() as usize, h: img.height() as usize, px, sumsq }
    }
}

pub struct Template {
    catalog: Arc<Catalog>,
    /// Per catalog item: (as stored, rotated 90° clockwise).
    tpls: Vec<Option<(Tpl, Tpl)>>,
}

impl Template {
    pub fn new(catalog: Arc<Catalog>, bases: &[Option<RgbImage>]) -> Self {
        let tpls = bases
            .par_iter()
            .map(|b| b.as_ref().map(|b| (Tpl::new(b), Tpl::new(&imageops::rotate90(b)))))
            .collect();
        Self { catalog, tpls }
    }
}

/// 1 − min SqDiffNormed over a ±SEARCH px window around the centred position.
fn match_score(src: &[f32], sw: usize, sh: usize, t: &Tpl) -> f32 {
    if t.w > sw || t.h > sh {
        return 0.0;
    }
    let (cx, cy) = (((sw - t.w) / 2) as i64, ((sh - t.h) / 2) as i64);
    let mut best = f32::MAX;
    for oy in (cy - SEARCH).max(0)..=(cy + SEARCH).min((sh - t.h) as i64) {
        for ox in (cx - SEARCH).max(0)..=(cx + SEARCH).min((sw - t.w) as i64) {
            let (mut diff, mut isq) = (0f32, 0f32);
            for y in 0..t.h {
                let srow = &src[((oy as usize + y) * sw + ox as usize) * 3..][..t.w * 3];
                let trow = &t.px[y * t.w * 3..][..t.w * 3];
                for (a, b) in srow.iter().zip(trow) {
                    let d = a - b;
                    diff += d * d;
                    isq += a * a;
                }
            }
            let v = diff / (t.sumsq * isq).sqrt().max(1e-6);
            best = best.min(v);
        }
    }
    1.0 - best
}

impl Identifier for Template {
    fn name(&self) -> &'static str {
        "template"
    }

    fn identify(&self, crop: &RgbImage, slots: (u32, u32)) -> Vec<Candidate> {
        let src = crop_to_base_padded(crop, slots);
        let (sw, sh) = (src.width() as usize, src.height() as usize);
        let px: Vec<f32> = src.as_raw().iter().map(|&v| v as f32).collect();
        let scored: Vec<(usize, f32)> = self
            .catalog
            .candidates(slots)
            .par_iter()
            .filter_map(|&(i, rot)| {
                let (n, r) = self.tpls[i].as_ref()?;
                Some((i, match_score(&px, sw, sh, if rot { r } else { n })))
            })
            .collect();
        top_k(scored, 5)
            .into_iter()
            .map(|(i, s)| Candidate { item_id: self.catalog.items[i].id.clone(), score: s })
            .collect()
    }
}
