//! Embedding top-10, rescored with short-name OCR only when retrieval is ambiguous.
use super::{embed::Embed, ocr::Ocr, Candidate, Identifier, Input};
use crate::catalog::Catalog;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const POOL: usize = 10;
const MARGIN: f32 = 0.05;
const CONFIDENT: f32 = 0.85;
const W_EMBED: f32 = 0.6;
const W_OCR: f32 = 0.4;

pub struct Fusion {
    catalog: Arc<Catalog>,
    embed: Arc<Embed>,
    ocr: Arc<Ocr>,
    ocr_runs: AtomicUsize,
}

impl Fusion {
    pub fn new(catalog: Arc<Catalog>, embed: Arc<Embed>, ocr: Arc<Ocr>) -> Self {
        Self { catalog, embed, ocr, ocr_runs: AtomicUsize::new(0) }
    }
}

impl Identifier for Fusion {
    fn name(&self) -> &'static str {
        "fusion"
    }

    fn identify(&self, input: &Input) -> Vec<Candidate> {
        let (crop, slots) = (input.crop, input.slots);
        let mut pool = self.embed.scores(crop, slots, POOL);
        let confident = match pool.as_slice() {
            [a, b, ..] => a.1 - b.1 >= MARGIN && a.1 >= CONFIDENT,
            [a] => a.1 >= CONFIDENT,
            [] => true,
        };
        if !confident {
            self.ocr_runs.fetch_add(1, Ordering::Relaxed);
            let text = self.ocr.read(crop, slots);
            for c in pool.iter_mut() {
                c.1 = W_EMBED * c.1 + W_OCR * self.ocr.score_item(&text, c.0);
            }
            pool.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
        pool.into_iter()
            .take(5)
            .map(|(i, s)| Candidate { item_id: self.catalog.items[i].id.clone(), score: s })
            .collect()
    }

    fn ocr_runs(&self) -> Option<usize> {
        Some(self.ocr_runs.load(Ordering::Relaxed))
    }
}
