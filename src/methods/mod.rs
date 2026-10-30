pub mod embed;
pub mod fusion;
pub mod ocr;
pub mod template;

use image::RgbImage;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub item_id: String,
    pub score: f32,
}

pub trait Identifier: Sync {
    fn name(&self) -> &'static str;
    /// `crop` is the padded cell crop; `slots` is the on-screen (width, height). Sorted desc, len <= 5.
    fn identify(&self, crop: &RgbImage, slots: (u32, u32)) -> Vec<Candidate>;
    /// Number of `identify` calls that ran OCR, for methods where OCR is conditional.
    fn ocr_runs(&self) -> Option<usize> {
        None
    }
}

/// Sort descending by score and keep the best `k`.
pub fn top_k(mut v: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}
