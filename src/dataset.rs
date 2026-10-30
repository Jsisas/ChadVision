use crate::catalog::Catalog;
use crate::imgops::{resize, GRID_RGB, PITCH};
use anyhow::{bail, Context, Result};
use image::{imageops, RgbImage};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::Value;
use ab_glyph::{FontVec, PxScale};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Sample {
    pub file: String,
    pub item_id: String,
    /// On-screen size in slots (width, height).
    pub slots: (u32, u32),
    pub rotated: bool,
    pub scale: f32,
}

const HEADER: &str = "file,item_id,slot_w,slot_h,rotated,scale";

pub fn read_labels(dir: &Path) -> Result<Vec<Sample>> {
    let path = dir.join("labels.csv");
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 6 {
            bail!("{}:{}: expected 6 fields", path.display(), n + 1);
        }
        out.push(Sample {
            file: f[0].into(),
            item_id: f[1].into(),
            slots: (f[2].parse()?, f[3].parse()?),
            rotated: f[4].parse()?,
            scale: f[5].parse()?,
        });
    }
    Ok(out)
}

pub fn write_labels(dir: &Path, samples: &[Sample]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join("labels.csv"))?);
    writeln!(f, "{HEADER}")?;
    for s in samples {
        writeln!(f, "{},{},{},{},{},{:.4}", s.file, s.item_id, s.slots.0, s.slots.1, s.rotated, s.scale)?;
    }
    Ok(())
}

const TIP_HEADER: &str = "file,tip_file,mouse_x,mouse_y,scale";
const FONT: &str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";

/// Tooltip row: sample file → (tooltip image file, cursor inside it, scale).
pub type TipRow = (String, (String, (u32, u32), f32));

/// `tooltips.csv` next to `labels.csv`; empty map when absent.
pub fn read_tooltips(dir: &Path) -> Result<HashMap<String, (String, (u32, u32), f32)>> {
    let path = dir.join("tooltips.csv");
    let Ok(text) = std::fs::read_to_string(&path) else { return Ok(HashMap::new()) };
    let mut out = HashMap::new();
    for (n, line) in text.lines().enumerate().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        if f.len() != 5 {
            bail!("{}:{}: expected 5 fields", path.display(), n + 1);
        }
        out.insert(f[0].to_string(), (f[1].to_string(), (f[2].parse()?, f[3].parse()?), f[4].parse()?));
    }
    Ok(out)
}

fn write_tooltips(dir: &Path, rows: &[TipRow]) -> Result<()> {
    let path = dir.join("tooltips.csv");
    if rows.is_empty() {
        let _ = std::fs::remove_file(&path);
        return Ok(());
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(f, "{TIP_HEADER}")?;
    for (file, (tip, (x, y), scale)) in rows {
        writeln!(f, "{file},{tip},{x},{y},{scale:.4}")?;
    }
    Ok(())
}

/// Tooltip search window around the cursor: x ∈ [mx−40s, mx+700s], y ∈ [my−120s, my+120s], clamped.
/// Returns the crop and the cursor position inside it.
pub(crate) fn tip_window(img: &RgbImage, mouse: (i64, i64), scale: f32) -> (RgbImage, (u32, u32)) {
    let s = scale as f64;
    let x0 = (mouse.0 - (40.0 * s).round() as i64).clamp(0, img.width() as i64);
    let x1 = (mouse.0 + (700.0 * s).round() as i64).clamp(0, img.width() as i64);
    let y0 = (mouse.1 - (120.0 * s).round() as i64).clamp(0, img.height() as i64);
    let y1 = (mouse.1 + (120.0 * s).round() as i64).clamp(0, img.height() as i64);
    let crop = imageops::crop_imm(img, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32).to_image();
    let m = ((mouse.0 - x0).max(0) as u32, (mouse.1 - y0).max(0) as u32);
    (crop, m)
}

/// Cell rect padded like RatEye's `TryAddIcon` (round(63·scale/8) px per side), clamped to the image.
fn padded_crop(img: &RgbImage, x: i32, y: i32, w: i32, h: i32, scale: f32) -> RgbImage {
    let p = (PITCH * scale / 8.0).round() as i32;
    let x0 = (x - p).max(0);
    let y0 = (y - p).max(0);
    let x1 = (x + w + p).min(img.width() as i32);
    let y1 = (y + h + p).min(img.height() as i32);
    imageops::crop_imm(img, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32).to_image()
}

/// Hand-labelled RatEyeTest fixtures (`InventoryTest.cs`). Rects are the grid cells under the test
/// points, measured from the 63 px grid lines at scale 1.
const FIXTURES: &[(&str, &str, &str, (i32, i32, i32, i32), (u32, u32))] = &[
    ("Grid1.png", "fixture-propital", "Propital", (220, 109, 63, 63), (1, 1)),
    ("Grid1.png", "fixture-morphine", "Morphine", (94, 109, 63, 63), (1, 1)),
    ("Grid3.png", "fixture-water", "Water", (28, 39, 63, 126), (1, 2)),
];

pub fn import_gt(src: Option<&Path>, out: &Path, fixtures: Option<&Path>, catalog: &Catalog) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut samples = Vec::new();
    let mut tips: Vec<TipRow> = Vec::new();
    let mut skipped = 0;
    let mut entries: Vec<PathBuf> = match src {
        Some(src) => std::fs::read_dir(src)
            .with_context(|| format!("reading {}", src.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect(),
        None => Vec::new(),
    };
    entries.sort();
    for json in entries {
        let src = json.parent().unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&json)?)?;
        let truth = &v["truth"];
        let item_id = truth["itemId"].as_str().unwrap_or_default();
        if truth.is_null() || item_id.is_empty() {
            skipped += 1;
            continue;
        }
        let id = v["id"].as_str().map(str::to_string).unwrap_or_else(|| {
            json.file_stem().unwrap().to_string_lossy().into_owned()
        });
        let shot = v["screenshot"].as_str().map(|s| src.join(s)).filter(|p| p.exists());
        let shot = shot.unwrap_or_else(|| src.join(format!("{id}.png")));
        let img = image::open(&shot).with_context(|| format!("opening {}", shot.display()))?.to_rgb8();
        let scale = v["scale"].as_f64().unwrap_or(1.0) as f32;
        let r = &truth["rect"];
        let g = |k: &str| r[k].as_i64().unwrap_or(0) as i32;
        let crop = padded_crop(&img, g("x"), g("y"), g("width"), g("height"), scale);
        let file = format!("{id}.png");
        crop.save(out.join(&file))?;
        if let (Some(mx), Some(my)) = (v["mouse"]["x"].as_i64(), v["mouse"]["y"].as_i64()) {
            let (win, m) = tip_window(&img, (mx, my), scale);
            if win.width() > 0 && win.height() > 0 {
                let tip_file = format!("{id}.tip.png");
                win.save(out.join(&tip_file))?;
                tips.push((file.clone(), (tip_file, m, scale)));
            }
        }
        samples.push(Sample {
            file,
            item_id: item_id.into(),
            slots: (truth["gridWidth"].as_u64().unwrap_or(1) as u32, truth["gridHeight"].as_u64().unwrap_or(1) as u32),
            rotated: truth["rotated"].as_bool().unwrap_or(false),
            scale,
        });
    }
    if let Some(dir) = fixtures {
        for &(png, file_id, short, (x, y, w, h), slots) in FIXTURES {
            let item = catalog
                .items
                .iter()
                .find(|i| i.short_name == short && (i.width, i.height) == slots)
                .with_context(|| format!("fixture item {short} not in catalog"))?;
            let img = image::open(dir.join(png))?.to_rgb8();
            let file = format!("{file_id}.png");
            padded_crop(&img, x, y, w, h, 1.0).save(out.join(&file))?;
            samples.push(Sample { file, item_id: item.id.clone(), slots, rotated: false, scale: 1.0 });
        }
    }
    write_labels(out, &samples)?;
    write_tooltips(out, &tips)?;
    println!("imported {} samples ({} skipped without truth)", samples.len(), skipped);
    Ok(())
}

pub fn synth(refs: &Path, out: &Path, per_item: u32, seed: u64, tooltips: bool, catalog: &Catalog) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let font = if tooltips {
        let bytes = std::fs::read(FONT).with_context(|| format!("tooltip font missing: {FONT}"))?;
        Some(FontVec::try_from_vec(bytes).with_context(|| format!("parsing {FONT}"))?)
    } else {
        None
    };
    let rows: Vec<(Sample, Option<TipRow>)> = catalog
        .items
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, it)| {
            let base = refs.join("base").join(format!("{}.png", it.id));
            let img = image::open(&base).ok().map(|d| d.to_rgb8());
            let font = font.as_ref();
            (0..per_item).filter_map(move |k| {
                let img = img.as_ref()?;
                let mut rng = ChaCha8Rng::seed_from_u64(seed ^ ((i as u64) << 16) ^ k as u64);
                let (canvas, slots, rotated, scale) = augment(img, (it.width, it.height), &mut rng);
                let look = Degrade::sample(&mut rng);
                let file = format!("{}_{k}.jpg", it.id);
                std::fs::write(out.join(&file), look.apply(canvas)).ok()?;
                let tip = font.and_then(|font| {
                    let (canvas, mouse) = render_tooltip(font, &it.name, scale);
                    let tip_file = format!("{}_{k}.tip.jpg", it.id);
                    std::fs::write(out.join(&tip_file), look.apply(canvas)).ok()?;
                    Some((file.clone(), (tip_file, mouse, scale)))
                });
                Some((Sample { file, item_id: it.id.clone(), slots, rotated, scale }, tip))
            })
        })
        .collect();
    let (samples, tips): (Vec<Sample>, Vec<Option<TipRow>>) = rows.into_iter().unzip();
    write_labels(out, &samples)?;
    write_tooltips(out, &tips.into_iter().flatten().collect::<Vec<_>>())?;
    println!("synthesised {} samples{}", samples.len(), if tooltips { " with synthetic tooltips" } else { "" });
    Ok(())
}

/// Synthetic tooltip: name in a dark box right-below the cursor on a grid-coloured 740s × 240s canvas,
/// cursor at (40s, 120s). Development only: font and colours are not Tarkov's.
fn render_tooltip(font: &FontVec, name: &str, scale: f32) -> (RgbImage, (u32, u32)) {
    let px = |v: f32| (v * scale).round() as i32;
    let mut img = RgbImage::from_pixel(px(740.0) as u32, px(240.0) as u32, GRID_RGB);
    let mouse = (px(40.0), px(120.0));
    let size = PxScale::from(14.0 * scale);
    let (tw, th) = imageproc::drawing::text_size(size, font, name);
    let (pad, off) = (px(4.0), px(12.0));
    let (bx, by) = (mouse.0 + off, mouse.1 + off);
    let rect = imageproc::rect::Rect::at(bx, by).of_size(tw + 2 * pad as u32, th + 2 * pad as u32);
    imageproc::drawing::draw_filled_rect_mut(&mut img, rect, image::Rgb([25, 27, 27]));
    imageproc::drawing::draw_text_mut(&mut img, image::Rgb([210, 210, 200]), bx + pad, by + pad, size, font, name);
    (img, (mouse.0 as u32, mouse.1 as u32))
}

/// Capture degradation shared by a sample's crop and tooltip.
struct Degrade {
    sigma: f32,
    gain: f32,
    quality: u8,
}

impl Degrade {
    fn sample(rng: &mut ChaCha8Rng) -> Self {
        Self { sigma: rng.random_range(0.0..=0.8), gain: rng.random_range(0.92..=1.08), quality: rng.random_range(70..=95) }
    }

    /// Blur, gain, then JPEG-encode.
    fn apply(&self, mut canvas: RgbImage) -> Vec<u8> {
        if self.sigma > 0.05 {
            canvas = imageops::blur(&canvas, self.sigma);
        }
        for p in canvas.pixels_mut() {
            for c in p.0.iter_mut() {
                *c = (*c as f32 * self.gain).round().clamp(0.0, 255.0) as u8;
            }
        }
        let mut bytes = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, self.quality)
            .encode_image(&canvas)
            .expect("jpeg encode");
        bytes
    }
}

/// Returns the padded crop (before capture degradation), on-screen slots, rotation flag and scale.
fn augment(base: &RgbImage, size: (u32, u32), rng: &mut ChaCha8Rng) -> (RgbImage, (u32, u32), bool, f32) {
    let scale: f32 = rng.random_range(0.75..=2.0);
    let mut img = resize(base, (base.width() as f32 * scale).round() as u32, (base.height() as f32 * scale).round() as u32);
    let mut slots = size;
    let rotated = size.0 != size.1 && rng.random_bool(0.15);
    if rotated {
        img = imageops::rotate90(&img);
        slots = (size.1, size.0);
    }
    let pad = (PITCH * scale / 8.0).round() as i64;
    let mut canvas = RgbImage::from_pixel(img.width() + 2 * pad as u32, img.height() + 2 * pad as u32, GRID_RGB);
    let (dx, dy) = (rng.random_range(-2..=2i64), rng.random_range(-2..=2i64));
    imageops::overlay(&mut canvas, &img, pad + dx, pad + dy);
    (canvas, slots, rotated, scale)
}
