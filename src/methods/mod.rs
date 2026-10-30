pub mod corrected;
pub mod embed;
pub mod fusion;
pub mod ocr;
pub mod template;
pub mod tooltip;

use image::RgbImage;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub item_id: String,
    pub score: f32,
}

/// Hover tooltip search window; `mouse` is the cursor position inside `image`.
pub struct Tooltip {
    pub image: RgbImage,
    pub mouse: (u32, u32),
    pub scale: f32,
}

/// One identification request: `crop` is the padded cell crop, `slots` the on-screen (width, height).
pub struct Input<'a> {
    pub crop: &'a RgbImage,
    pub slots: (u32, u32),
    pub tooltip: Option<&'a Tooltip>,
}

/// First stage of a two-stage method.
#[derive(Debug, Clone)]
pub struct Stages {
    /// Wall time until the first (icon) answer.
    pub first_ms: f64,
    /// Whether a tooltip was read.
    pub used_tooltip: bool,
    /// Top-1 of the first answer; `None` when it rejected.
    pub first_id: Option<String>,
}

pub trait Identifier: Sync {
    fn name(&self) -> &'static str;
    /// Sorted desc, len <= 5.
    fn identify(&self, input: &Input) -> Vec<Candidate>;
    /// Number of `identify` calls that ran OCR, for methods where OCR is conditional.
    fn ocr_runs(&self) -> Option<usize> {
        None
    }
    /// For staged methods: timing and first answer of the last `identify` call.
    fn last_stages(&self) -> Option<Stages> {
        None
    }
}

/// Sort descending by score and keep the best `k`.
pub fn top_k(mut v: Vec<(usize, f32)>, k: usize) -> Vec<(usize, f32)> {
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(k);
    v
}
