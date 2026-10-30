//! Constrained short-name OCR: PP-OCRv5 recognition on the icon's top strip, matched only
//! against short names of items with the same slot size.
use super::{Candidate, Identifier};
use crate::catalog::Catalog;
use crate::imgops::{crop_scale, resize, PITCH};
use crate::lexicon;
use anyhow::{Context, Result};
use image::{imageops, RgbImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Strip geometry at base (1080p) scale.
const STRIP_H: f32 = 14.0;
const LEFT_CUT: f32 = 4.0;

pub struct Ocr {
    catalog: Arc<Catalog>,
    session: Mutex<Session>,
    charset: Vec<String>,
    input_h: u32,
}

impl Ocr {
    pub fn new(catalog: Arc<Catalog>, model: &Path, dict: &Path) -> Result<Self> {
        let session = super::embed::session(model)?;
        let input_h = match &session.inputs[0].input_type {
            ValueType::Tensor { shape, .. } if shape.len() == 4 && shape[2] > 0 => shape[2] as u32,
            _ => 48,
        };
        let dict = std::fs::read_to_string(dict).with_context(|| format!("reading {}", dict.display()))?;
        // CTC classes: blank, dictionary lines, space.
        let mut charset = vec![String::new()];
        charset.extend(dict.lines().map(str::to_string));
        charset.push(" ".into());
        Ok(Self { catalog, session: Mutex::new(session), charset, input_h })
    }

    /// Top text strip in native pixels: skip the padding, cut 4 px on the left, keep 14 px of height.
    fn strip(crop: &RgbImage, slots: (u32, u32)) -> RgbImage {
        let s = crop_scale(crop, slots);
        let (cw, ch) = (PITCH * s * slots.0 as f32, PITCH * s * slots.1 as f32);
        let ox = ((crop.width() as f32 - cw) / 2.0).max(0.0);
        let oy = ((crop.height() as f32 - ch) / 2.0).max(0.0);
        let x = (ox + LEFT_CUT * s).round() as u32;
        let y = oy.round() as u32;
        let w = ((cw - LEFT_CUT * s).round() as u32).min(crop.width().saturating_sub(x)).max(1);
        let h = ((STRIP_H * s).round() as u32).min(crop.height().saturating_sub(y)).max(1);
        imageops::crop_imm(crop, x, y, w, h).to_image()
    }

    pub fn read(&self, crop: &RgbImage, slots: (u32, u32)) -> String {
        let strip = Self::strip(crop, slots);
        let h = self.input_h;
        let w = ((strip.width() as f32 * h as f32 / strip.height() as f32).round() as u32).max(8);
        let img = resize(&strip, w, h);
        let pw = w.div_ceil(8) * 8;
        let mut x = Array4::<f32>::zeros((1, 3, h as usize, pw as usize));
        for (px, py, p) in img.enumerate_pixels() {
            for c in 0..3 {
                x[[0, c, py as usize, px as usize]] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![Tensor::from_array(x).expect("tensor")]).expect("ocr inference");
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().expect("ocr output");
        let (steps, classes) = (shape[1] as usize, shape[2] as usize);
        let mut text = String::new();
        let mut prev = 0usize;
        for t in 0..steps {
            let row = &data[t * classes..][..classes];
            let k = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(k, _)| k).unwrap_or(0);
            if k != 0 && k != prev {
                text.push_str(self.charset.get(k).map(String::as_str).unwrap_or(""));
            }
            prev = k;
        }
        text.trim().to_string()
    }

    /// Fuzzy score of OCR text against an item's short name.
    pub fn score_item(&self, text: &str, idx: usize) -> f32 {
        lexicon::score(text, &self.catalog.items[idx].short_name)
    }
}

impl Identifier for Ocr {
    fn name(&self) -> &'static str {
        "ocr"
    }

    fn identify(&self, crop: &RgbImage, slots: (u32, u32)) -> Vec<Candidate> {
        let text = self.read(crop, slots);
        let cands = self.catalog.candidates(slots);
        lexicon::rank(&text, cands.iter().map(|&(i, _)| (i, self.catalog.items[i].short_name.as_str())), 5)
            .into_iter()
            .map(|(i, s)| Candidate { item_id: self.catalog.items[i].id.clone(), score: s })
            .collect()
    }
}
