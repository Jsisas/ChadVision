//! Two-stage: fusion answer first, then corrected by the tooltip's full item name.
use super::tooltip::{rank_names, read_name, NAME_ACCEPT};
use super::{fusion::Fusion, ocr::Ocr, Candidate, Identifier, Input, Stages};
use crate::catalog::Catalog;
use crate::lexicon;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub struct Corrected {
    catalog: Arc<Catalog>,
    fusion: Arc<Fusion>,
    ocr: Arc<Ocr>,
    last: Mutex<Option<Stages>>,
}

impl Corrected {
    pub fn new(catalog: Arc<Catalog>, fusion: Arc<Fusion>, ocr: Arc<Ocr>) -> Self {
        Self { catalog, fusion, ocr, last: Mutex::new(None) }
    }
}

impl Identifier for Corrected {
    fn name(&self) -> &'static str {
        "icon+tooltip"
    }

    fn identify(&self, input: &Input) -> Vec<Candidate> {
        let t = Instant::now();
        let first = self.fusion.identify(input);
        let t_first = t.elapsed().as_secs_f64() * 1e3;
        let mut stages = Stages { first_ms: t_first, used_tooltip: false, first_id: first.first().map(|c| c.item_id.clone()) };
        let Some(tip) = input.tooltip else {
            *self.last.lock().unwrap() = Some(stages);
            return first;
        };
        stages.used_tooltip = true;
        let text = read_name(&self.ocr, tip);
        let from_first = first
            .iter()
            .filter_map(|c| Some((c.item_id.as_str(), lexicon::score(&text, &self.catalog.get(&c.item_id)?.name))))
            // On equal name scores (duplicate full names) keep the icon's ranking.
            .reduce(|best, c| if c.1 > best.1 { c } else { best })
            .filter(|c| c.1 >= NAME_ACCEPT)
            .map(|(id, s)| (id.to_string(), s));
        // The catalog-wide best name wins when it reads strictly better: near-identical names
        // (PS gs / PRS gs) differ by one letter, so the icon's top-5 must not shadow an exact read.
        let global = rank_names(&self.catalog, &text, input.slots, 1)
            .into_iter()
            .find(|c| c.1 >= NAME_ACCEPT)
            .map(|(i, s)| (self.catalog.items[i].id.clone(), s));
        let chosen = match (from_first, global) {
            (Some(f), Some(g)) => Some(if g.1 > f.1 { g } else { f }),
            (f, g) => f.or(g),
        };
        let mut out = first;
        if let Some((id, s)) = chosen {
            if out.first().is_none_or(|c| c.item_id != id) {
                out.retain(|c| c.item_id != id);
                out.insert(0, Candidate { item_id: id, score: s });
                out.truncate(5);
            }
        }
        *self.last.lock().unwrap() = Some(stages);
        out
    }

    fn ocr_runs(&self) -> Option<usize> {
        self.fusion.ocr_runs()
    }

    fn last_stages(&self) -> Option<Stages> {
        self.last.lock().unwrap().clone()
    }
}
