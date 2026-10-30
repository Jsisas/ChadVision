//! Rust API: load the identifier once, then identify the item under the cursor in a screenshot.
use crate::catalog::Catalog;
use crate::imgops::PITCH;
use crate::methods::{corrected::Corrected, embed::Embed, fusion::Fusion, ocr::Ocr, Identifier, Input, Tooltip};
use crate::{dataset, grid, refs};
use image::RgbImage;
use std::path::Path;
use std::sync::Arc;

/// Without a grid cell the crop is a guessed one-slot box; only a strong tooltip name is trusted then.
pub const NAME_ONLY_ACCEPT: f32 = 0.8;

pub struct Engine {
    pub catalog: Arc<Catalog>,
    corrected: Corrected,
}

impl Engine {
    /// `data_dir` holds `refs/`, `models/` and `index/`.
    pub fn open(data_dir: &Path) -> anyhow::Result<Engine> {
        // Explicit path: a bare "onnxruntime.dll" could resolve to an older copy in System32.
        #[cfg(windows)]
        {
            let exe_dir = std::env::current_exe()?.parent().map(Path::to_path_buf).unwrap_or_default();
            ort::init_from(exe_dir.join("onnxruntime.dll").to_string_lossy()).commit()?;
        }
        let refs_dir = data_dir.join("refs");
        let models = data_dir.join("models");
        let catalog = Arc::new(Catalog::load(&refs_dir)?);
        let bases = refs::load_bases(&refs_dir, &catalog);
        let embed =
            Arc::new(Embed::new(catalog.clone(), &bases, &models.join("dinov2-small.onnx"), &data_dir.join("index"))?);
        let ocr = Arc::new(Ocr::new(
            catalog.clone(),
            &models.join("ppocrv5_latin_rec.onnx"),
            &models.join("ppocrv5_latin_dict.txt"),
        )?);
        drop(bases);
        let fusion = Arc::new(Fusion::new(catalog.clone(), embed, ocr.clone()));
        let corrected = Corrected::new(catalog.clone(), fusion, ocr);
        Ok(Engine { catalog, corrected })
    }

    /// Candidates for a padded cell crop plus the optional hover tooltip, best first.
    pub fn identify(&self, input: &Input) -> Vec<(String, f32)> {
        self.corrected.identify(input).into_iter().map(|c| (c.item_id, c.score)).collect()
    }

    /// Item under `mouse` in `screen` (RGB, EFT pixels at `scale` = height/1080), best first, at most `k`.
    /// Empty when no grid cell is found and the tooltip name does not reach `NAME_ONLY_ACCEPT`.
    pub fn identify_at(&self, screen: &RgbImage, mouse: (i32, i32), scale: f32, k: usize) -> Vec<(String, f32)> {
        let cell = grid::locate_cell(screen, mouse, scale, (1, 1));
        let slot = (PITCH * scale).round() as i32;
        let (x, y, w, h, slots) = match cell {
            Some(c) => (c.x, c.y, c.w, c.h, c.slots),
            None => (mouse.0 - slot / 2, mouse.1 - slot / 2, slot, slot, (1, 1)),
        };
        let crop = dataset::padded_crop(screen, x, y, w, h, scale);
        if crop.width() == 0 || crop.height() == 0 {
            return Vec::new();
        }
        let (image, tip_mouse) = dataset::tip_window(screen, (mouse.0 as i64, mouse.1 as i64), scale);
        let tooltip = (image.width() > 0 && image.height() > 0).then_some(Tooltip { image, mouse: tip_mouse, scale });
        let mut cands = self.identify(&Input { crop: &crop, slots, tooltip: tooltip.as_ref() });
        if cell.is_none() && cands.first().is_none_or(|c| c.1 < NAME_ONLY_ACCEPT) {
            return Vec::new();
        }
        cands.truncate(k);
        cands
    }
}
