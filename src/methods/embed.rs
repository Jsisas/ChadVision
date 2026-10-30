//! Embedding retrieval: DINOv2-small CLS token, cosine similarity against reference embeddings.
use super::{top_k, Candidate, Identifier};
use crate::catalog::Catalog;
use crate::imgops::{crop_to_base, letterbox, resize};
use anyhow::{bail, Context, Result};
use image::{imageops, RgbImage};
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use std::sync::{Arc, Mutex};

pub const MODEL_NAME: &str = "dinov2-small";
const SIDE: usize = 224;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const BUILD_BATCH: usize = 32;

pub fn session(model: &Path) -> Result<Session> {
    Ok(Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(num_cpus::get_physical())?
        .commit_from_file(model)
        .with_context(|| format!("loading {}", model.display()))?)
}

pub struct Embed {
    catalog: Arc<Catalog>,
    session: Mutex<Session>,
    dim: usize,
    /// Row-major unit vectors; `rows[i]` is the row of catalog item `i`.
    matrix: Vec<f32>,
    rows: Vec<Option<usize>>,
}

/// Base-scale cell image → letterboxed 224² square.
fn square(base: &RgbImage) -> RgbImage {
    resize(&letterbox(base), SIDE as u32, SIDE as u32)
}

impl Embed {
    pub fn new(catalog: Arc<Catalog>, bases: &[Option<RgbImage>], model: &Path, index_dir: &Path) -> Result<Self> {
        let mut session = session(model)?;
        let bin = index_dir.join(format!("embed_{MODEL_NAME}.bin"));
        let ids_path = index_dir.join("embed_ids.txt");
        let want: Vec<&str> = catalog.items.iter().zip(bases).filter(|(_, b)| b.is_some()).map(|(i, _)| i.id.as_str()).collect();

        let cached = std::fs::read_to_string(&ids_path).ok().filter(|t| t.lines().eq(want.iter().copied()));
        let matrix: Vec<f32> = match (cached, std::fs::read(&bin)) {
            (Some(_), Ok(bytes)) => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            _ => {
                let imgs: Vec<&RgbImage> = bases.iter().flatten().collect();
                let mut m = Vec::new();
                let t = std::time::Instant::now();
                for (n, chunk) in imgs.chunks(BUILD_BATCH).enumerate() {
                    let sq: Vec<RgbImage> = chunk.iter().map(|b| square(b)).collect();
                    m.extend(run(&mut session, &sq)?);
                    if n % 20 == 0 {
                        eprintln!("embed index {}/{} ({:.0}s)", n * BUILD_BATCH, imgs.len(), t.elapsed().as_secs_f32());
                    }
                }
                std::fs::create_dir_all(index_dir)?;
                std::fs::write(&bin, m.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
                std::fs::write(&ids_path, want.join("\n"))?;
                m
            }
        };
        if matrix.is_empty() || matrix.len() % want.len() != 0 {
            bail!("embedding index size mismatch");
        }
        let dim = matrix.len() / want.len();
        let mut next = 0;
        let rows = bases
            .iter()
            .map(|b| {
                b.as_ref().map(|_| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        Ok(Self { catalog, session: Mutex::new(session), dim, matrix, rows })
    }

    fn embed(&self, base: &RgbImage) -> Vec<f32> {
        run(&mut self.session.lock().unwrap(), &[square(base)]).expect("embed inference")
    }

    /// Cosine similarity of the crop against every same-size (or transposed) reference, best `k`.
    pub fn scores(&self, crop: &RgbImage, slots: (u32, u32), k: usize) -> Vec<(usize, f32)> {
        let base = crop_to_base(crop, slots);
        let q = self.embed(&base);
        // A transposed reference is compared against the crop rotated back (synth rotates refs clockwise).
        let q_rot = (slots.0 != slots.1).then(|| self.embed(&imageops::rotate270(&base)));
        let scored = self
            .catalog
            .candidates(slots)
            .into_iter()
            .filter_map(|(i, rot)| {
                let r = self.rows[i]?;
                let v = if rot { q_rot.as_ref()? } else { &q };
                let row = &self.matrix[r * self.dim..][..self.dim];
                Some((i, row.iter().zip(v).map(|(a, b)| a * b).sum::<f32>()))
            })
            .collect();
        top_k(scored, k)
    }
}

/// Batch inference; returns L2-normalised CLS rows, concatenated.
fn run(session: &mut Session, imgs: &[RgbImage]) -> Result<Vec<f32>> {
    let mut x = Array4::<f32>::zeros((imgs.len(), 3, SIDE, SIDE));
    for (n, img) in imgs.iter().enumerate() {
        for (px, py, p) in img.enumerate_pixels() {
            for c in 0..3 {
                x[[n, c, py as usize, px as usize]] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    let outputs = session.run(ort::inputs![Tensor::from_array(x)?])?;
    let (shape, data) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    // last_hidden_state: [batch, tokens, dim]; CLS is token 0.
    let (tokens, dim) = (shape[1] as usize, shape[2] as usize);
    let mut out = Vec::with_capacity(imgs.len() * dim);
    for n in 0..imgs.len() {
        let cls = &data[n * tokens * dim..][..dim];
        let norm = cls.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        out.extend(cls.iter().map(|v| v / norm));
    }
    Ok(out)
}

impl Identifier for Embed {
    fn name(&self) -> &'static str {
        "embed"
    }

    fn identify(&self, crop: &RgbImage, slots: (u32, u32)) -> Vec<Candidate> {
        self.scores(crop, slots, 5)
            .into_iter()
            .map(|(i, s)| Candidate { item_id: self.catalog.items[i].id.clone(), score: s })
            .collect()
    }
}
