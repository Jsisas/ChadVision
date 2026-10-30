use crate::catalog::Item;
use crate::imgops::{resize, GRID_RGB};
use anyhow::{bail, Context, Result};
use image::{imageops, DynamicImage, RgbImage};
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;


/// Base references aligned with `catalog.items`; `None` where the file is missing.
pub fn load_bases(refs: &Path, catalog: &crate::catalog::Catalog) -> Vec<Option<RgbImage>> {
    catalog
        .items
        .par_iter()
        .map(|it| image::open(refs.join("base").join(format!("{}.png", it.id))).ok().map(|d| d.to_rgb8()))
        .collect()
}
const ITEMS_URL: &str = "https://json.tarkov.dev/regular/items";
const ITEMS_EN_URL: &str = "https://json.tarkov.dev/regular/items_en";

pub fn http() -> Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder()
        .user_agent("tarkov-vision/0.1")
        .timeout(std::time::Duration::from_secs(120))
        .build()?)
}

pub fn fetch_refs(out: &Path) -> Result<()> {
    let client = http()?;
    let items: Value = client.get(ITEMS_URL).send()?.error_for_status()?.json()?;
    let en: HashMap<String, String> = {
        let v: Value = client.get(ITEMS_EN_URL).send()?.error_for_status()?.json()?;
        serde_json::from_value(v["data"].clone()).context("items_en data")?
    };
    let raw = items["data"]["items"].as_object().context("data.items is not an object")?;

    let mut catalog = Vec::with_capacity(raw.len());
    let mut dropped = 0;
    for (id, it) in raw {
        let s = |k: &str| it[k].as_str().unwrap_or_default().to_string();
        let (Some(name), Some(short)) = (en.get(&s("name")), en.get(&s("shortName"))) else {
            eprintln!("drop {id}: missing translation");
            dropped += 1;
            continue;
        };
        let link = s("gridImageLink");
        if link.is_empty() {
            eprintln!("drop {id}: no gridImageLink");
            dropped += 1;
            continue;
        }
        catalog.push(Item {
            id: id.clone(),
            name: name.clone(),
            short_name: short.clone(),
            width: it["width"].as_u64().unwrap_or(1) as u32,
            height: it["height"].as_u64().unwrap_or(1) as u32,
            background_color: s("backgroundColor"),
            grid_image_link: link,
        });
    }
    catalog.sort_by(|a, b| a.id.cmp(&b.id));
    std::fs::create_dir_all(out.join("grid"))?;
    std::fs::create_dir_all(out.join("base"))?;
    std::fs::write(out.join("catalog.json"), serde_json::to_string_pretty(&catalog)?)?;
    println!("catalog: {} items ({} dropped)", catalog.len(), dropped);

    let failed_path = out.join("failed.txt");
    let failed = Mutex::new(std::fs::File::create(&failed_path)?);
    let n_failed = std::sync::atomic::AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    pool.install(|| {
        catalog.par_iter().for_each(|it| {
            if let Err(e) = fetch_one(&client, out, it) {
                n_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                writeln!(failed.lock().unwrap(), "{}\t{:#}", it.id, e).ok();
            }
        })
    });
    let n_failed = n_failed.into_inner();
    println!("images: {} ok, {} failed (see {})", catalog.len() - n_failed, n_failed, failed_path.display());
    if n_failed * 20 > catalog.len() {
        bail!("more than 5% of downloads failed");
    }
    Ok(())
}

fn fetch_one(client: &reqwest::blocking::Client, out: &Path, it: &Item) -> Result<()> {
    let grid = out.join("grid").join(format!("{}.webp", it.id));
    if !grid.exists() {
        let bytes = client.get(&it.grid_image_link).send()?.error_for_status()?.bytes()?;
        let tmp = grid.with_extension("part");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &grid)?;
    }
    let base = out.join("base").join(format!("{}.png", it.id));
    if !base.exists() {
        // Some tarkov.dev "webp" grid images (presets) are actually JPEG; sniff the content.
        let img = image::ImageReader::open(&grid)?
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("decoding {}", grid.display()))?;
        to_base(img, it.width, it.height).save(&base)?;
    }
    Ok(())
}

/// tarkov.dev grid images are `(63·w+1) × (63·h+1)`: 63 px pitch with a trailing 1 px grid line.
/// Cutting that line gives the exact 1080p cell; any other size is Lanczos-resized.
/// Transparent pixels (if any) are composited over the grid colour.
pub fn to_base(img: DynamicImage, w: u32, h: u32) -> RgbImage {
    let rgba = img.to_rgba8();
    let mut rgb = RgbImage::from_pixel(rgba.width(), rgba.height(), GRID_RGB);
    for (x, y, p) in rgba.enumerate_pixels() {
        let a = p[3] as u32;
        let bg = rgb.get_pixel(x, y).0;
        let c = |i: usize| ((p[i] as u32 * a + bg[i] as u32 * (255 - a) + 127) / 255) as u8;
        rgb.put_pixel(x, y, image::Rgb([c(0), c(1), c(2)]));
    }
    let (bw, bh) = (63 * w, 63 * h);
    if rgb.dimensions() == (bw + 1, bh + 1) {
        imageops::crop_imm(&rgb, 0, 0, bw, bh).to_image()
    } else {
        resize(&rgb, bw, bh)
    }
}
