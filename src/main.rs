mod bench;
mod catalog;
mod dataset;
mod imgops;
mod lexicon;
mod methods;
mod refs;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use methods::{embed::Embed, fusion::Fusion, ocr::Ocr, template::Template, Identifier};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MODELS: &[(&str, &str)] = &[
    ("dinov2-small.onnx", "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx"),
    ("ppocrv5_latin_rec.onnx", "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/rec.onnx"),
    ("ppocrv5_latin_dict.txt", "https://huggingface.co/monkt/paddleocr-onnx/resolve/main/languages/latin/dict.txt"),
];

#[derive(Parser)]
#[command(about = "Tarkov item identification benchmark")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the tarkov.dev catalog and grid images, normalise to 1080p base size.
    FetchRefs {
        #[arg(long, default_value = "data/refs")]
        out: PathBuf,
    },
    /// Download ONNX models into data/models.
    FetchModels {
        #[arg(long, default_value = "data/models")]
        out: PathBuf,
    },
    /// Import RatEyeGroundTruth captures (+ RatEyeTest fixtures) as a labelled set.
    ImportGt {
        /// RatEyeGroundTruth folder (Documents\RatEyeGroundTruth); optional when only importing fixtures.
        #[arg(long)]
        src: Option<PathBuf>,
        #[arg(long, default_value = "data/real")]
        out: PathBuf,
        /// RatEyeTest FHD fixture directory containing Grid1.png and Grid3.png.
        #[arg(long)]
        fixtures: Option<PathBuf>,
        #[arg(long, default_value = "data/refs")]
        refs: PathBuf,
    },
    /// Generate augmented synthetic samples from the references.
    Synth {
        #[arg(long, default_value = "data/refs")]
        refs: PathBuf,
        #[arg(long, default_value = "data/synth")]
        out: PathBuf,
        #[arg(long, default_value_t = 2)]
        per_item: u32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Run all methods over a labelled set and write a Markdown report.
    Bench {
        #[arg(long)]
        set: PathBuf,
        /// Defaults to reports/<set name>.md
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = "data/refs")]
        refs: PathBuf,
        #[arg(long, default_value = "data/models")]
        models: PathBuf,
        #[arg(long, default_value = "data/index")]
        index: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "template,embed,ocr,fusion")]
        methods: Vec<String>,
        /// Evenly subsample the set to at most this many samples.
        #[arg(long)]
        limit: Option<usize>,
    },
}

fn fetch_models(out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let client = refs::http()?;
    for (file, url) in MODELS {
        let path = out.join(file);
        if path.exists() {
            println!("{file}: present");
            continue;
        }
        let bytes = client.get(*url).send()?.error_for_status()?.bytes().with_context(|| url.to_string())?;
        std::fs::write(&path, &bytes)?;
        println!("{file}: {} bytes", bytes.len());
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::FetchRefs { out } => refs::fetch_refs(&out),
        Cmd::FetchModels { out } => fetch_models(&out),
        Cmd::ImportGt { src, out, fixtures, refs } => {
            dataset::import_gt(src.as_deref(), &out, fixtures.as_deref(), &catalog::Catalog::load(&refs)?)
        }
        Cmd::Synth { refs, out, per_item, seed } => {
            dataset::synth(&refs, &out, per_item, seed, &catalog::Catalog::load(&refs)?)
        }
        Cmd::Bench { set, out, refs, models, index, methods, limit } => {
            let catalog = Arc::new(catalog::Catalog::load(&refs)?);
            let bases = refs::load_bases(&refs, &catalog);
            let groups = bench::icon_groups(&bases);
            let want = |m: &str| methods.iter().any(|x| x == m);
            let template = want("template").then(|| Template::new(catalog.clone(), &bases));
            let embed = (want("embed") || want("fusion"))
                .then(|| Embed::new(catalog.clone(), &bases, &models.join("dinov2-small.onnx"), &index).map(Arc::new))
                .transpose()?;
            let ocr = (want("ocr") || want("fusion"))
                .then(|| {
                    Ocr::new(catalog.clone(), &models.join("ppocrv5_latin_rec.onnx"), &models.join("ppocrv5_latin_dict.txt"))
                        .map(Arc::new)
                })
                .transpose()?;
            let fusion = match (&embed, &ocr) {
                (Some(e), Some(o)) if want("fusion") => Some(Fusion::new(catalog.clone(), e.clone(), o.clone())),
                _ => None,
            };
            let mut list: Vec<&dyn Identifier> = Vec::new();
            for m in &methods {
                match m.as_str() {
                    "template" => list.push(template.as_ref().unwrap()),
                    "embed" => list.push(embed.as_deref().unwrap()),
                    "ocr" => list.push(ocr.as_deref().unwrap()),
                    "fusion" => list.push(fusion.as_ref().unwrap()),
                    other => bail!("unknown method {other}"),
                }
            }
            drop(bases);
            let name = set.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or("set".into());
            let out = out.unwrap_or_else(|| PathBuf::from(format!("reports/{name}.md")));
            bench::run(&set, &out, &catalog, &groups, &list, limit)
        }
    }
}
