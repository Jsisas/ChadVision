use crate::catalog::Catalog;
use crate::dataset::{read_labels, Sample};
use crate::methods::Identifier;
use anyhow::{Context, Result};
use image::RgbImage;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

const TARGET_TOP1: f64 = 99.0;
const TARGET_P95_MS: f64 = 50.0;
const TIE_PP: f64 = 1.0;

const CONTEXT: &str = r#"## 1. How existing tools identify items

|Tool|Method|Weak points|
|---|---|---|
|RatScanner / RatEye (C#)|Icon template matching: `SqDiffNormed` against every static icon of the same slot size, in parallel. Optional Tesseract short-name OCR. The fork `Jsisas/RatEye` branch `recognition-fix` adds a short-name OCR tie-break for look-alike icons.|Brute force over every same-size icon. Scale-sensitive. Confuses look-alikes (keys, ammo).|
|TarkovPriceViewer (C#)|Whole-window `PrintWindow` capture. `InRange(90,89,82)` contour pass finds the tooltip. PaddleOCR (Sdcb.PaddleOCR, EnglishV4) reads the name. Matched by plain Levenshtein to the nearest item, with no threshold.|Hard-coded colour. No reject path. One global OCR lock. DPI bugs (issue #26).|
|pado8/tarkov-price-overlay (Tauri + Python)|Fixed boxes at offsets from the cursor, read with EasyOCR (CPU). Matching order: exact → canonical fold (`0→o, 1/i/\|→l, 5→s, 8→b, 6→g, 2→z`) → `difflib` cutoff 0.6 → prefix.|~1 s per lookup (README). Neighbouring text pollutes results. Prefix matching mis-picks (e.g. SALEWA→SAW).|
|TarkovMonitor, Questy|**No vision at all.** They tail EFT logs: `notifications.log` gives quest started/failed/finished (`ChatMessageReceived` type 10/11/12) and flea sales. `application.log` gives the raid lifecycle.|n/a. This is the right source for the later quest tracker, not OCR.|

Methods beyond these tools, measured here:
- **embed**: DINOv2-small CLS embedding (384-d, L2-normalised), cosine nearest neighbour over same-size tarkov.dev grid images.
- **ocr**: PP-OCRv5 mobile Latin recognition on the icon's top strip, greedy CTC, folded Levenshtein against same-size short names, reject below 0.5.
- **fusion**: embed top-10; if top1−top2 < 0.05 or top1 < 0.85, rescore with 0.6·embed + 0.4·ocr.
- **template**: RatEye baseline reimplemented (1 − min SqDiffNormed over ±4 px, RGB, base scale).
"#;

#[derive(Default)]
struct Stats {
    top1: usize,
    top1_equiv: usize,
    top5: usize,
    rejects: usize,
    ms: Vec<f64>,
    confusions: HashMap<(String, String), usize>,
}

fn pct(n: usize, d: usize) -> f64 {
    if d == 0 { 0.0 } else { 100.0 * n as f64 / d as f64 }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| t.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split(':').nth(1)).map(|s| s.trim().to_string()))
        .unwrap_or_else(|| "unknown".into())
}

/// `equiv[i]` = group id of catalog items whose base images are byte-identical.
pub fn icon_groups(bases: &[Option<RgbImage>]) -> Vec<usize> {
    let mut seen: HashMap<(u32, u32, &[u8]), usize> = HashMap::new();
    bases
        .iter()
        .enumerate()
        .map(|(i, b)| match b {
            Some(b) => *seen.entry((b.width(), b.height(), b.as_raw().as_slice())).or_insert(i),
            None => i,
        })
        .collect()
}

pub fn run(
    set: &Path,
    out: &Path,
    catalog: &Catalog,
    groups: &[usize],
    methods: &[&dyn Identifier],
    limit: Option<usize>,
) -> Result<()> {
    let mut samples: Vec<Sample> = read_labels(set)?;
    if let Some(n) = limit.filter(|&n| n < samples.len()) {
        // Even stride keeps the subsample deterministic and spread over the catalog.
        let step = samples.len() as f64 / n as f64;
        samples = (0..n).map(|k| samples[(k as f64 * step) as usize].clone()).collect();
    }
    let crops: Vec<RgbImage> = samples
        .iter()
        .map(|s| image::open(set.join(&s.file)).map(|d| d.to_rgb8()).with_context(|| s.file.clone()))
        .collect::<Result<_>>()?;
    if let (Some(s), Some(c)) = (samples.first(), crops.first()) {
        for m in methods {
            m.identify(c, s.slots);
        }
    }
    let warm_ocr: Vec<usize> = methods.iter().map(|m| m.ocr_runs().unwrap_or(0)).collect();

    let set_name = set.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut log = String::from("file\ttruth\tslots\tmethod\tpred\tscore\tok\tms\n");
    let mut stats: Vec<Stats> = methods.iter().map(|_| Stats::default()).collect();
    let progress = Instant::now();
    for (n, (s, crop)) in samples.iter().zip(&crops).enumerate() {
        let truth_short = catalog.short_name(&s.item_id).to_string();
        let truth_group = catalog.index(&s.item_id).map(|i| groups[i]);
        for (m, st) in methods.iter().zip(stats.iter_mut()) {
            let t = Instant::now();
            let cands = m.identify(crop, s.slots);
            st.ms.push(t.elapsed().as_secs_f64() * 1e3);
            let pred = cands.first();
            let ok = pred.is_some_and(|c| c.item_id == s.item_id);
            match pred {
                None => st.rejects += 1,
                Some(c) if !ok => {
                    *st.confusions.entry((truth_short.clone(), catalog.short_name(&c.item_id).to_string())).or_default() += 1
                }
                _ => {}
            }
            st.top1 += ok as usize;
            st.top1_equiv += pred.is_some_and(|c| ok || (truth_group.is_some() && catalog.index(&c.item_id).map(|i| groups[i]) == truth_group)) as usize;
            st.top5 += cands.iter().any(|c| c.item_id == s.item_id) as usize;
            let _ = writeln!(
                log,
                "{}\t{}\t{}x{}\t{}\t{}\t{:.3}\t{}\t{:.2}",
                s.file,
                truth_short,
                s.slots.0,
                s.slots.1,
                m.name(),
                pred.map(|c| catalog.short_name(&c.item_id)).unwrap_or("<reject>"),
                pred.map(|c| c.score).unwrap_or(0.0),
                ok,
                st.ms.last().unwrap()
            );
        }
        if (n + 1) % 200 == 0 {
            eprintln!("{}/{} samples ({:.0}s)", n + 1, samples.len(), progress.elapsed().as_secs_f32());
        }
    }

    let total = samples.len();
    let mut rows = Vec::new();
    for ((m, st), warm) in methods.iter().zip(stats.iter_mut()).zip(&warm_ocr) {
        st.ms.sort_by(f64::total_cmp);
        rows.push((
            m.name(),
            pct(st.top1, total),
            pct(st.top1_equiv, total),
            pct(st.top5, total),
            pct(st.rejects, total),
            quantile(&st.ms, 0.5),
            quantile(&st.ms, 0.95),
            st.ms.last().copied().unwrap_or(0.0),
            m.ocr_runs().map(|r| pct(r - warm, total)),
        ));
    }

    let mut r = String::new();
    let _ = writeln!(r, "# Tarkov item identification benchmark: `{set_name}`\n");
    let _ = writeln!(r, "- Samples: {total} (`{}`)", set.display());
    let _ = writeln!(r, "- CPU: {} ({} physical cores), ONNX Runtime CPU EP", cpu_name(), num_cpus::get_physical());
    let _ = writeln!(r, "- Catalog: {} items from json.tarkov.dev; references are tarkov.dev grid images", catalog.items.len());
    if set_name != "real" || total < 300 {
        let _ = writeln!(
            r,
            "- **Provisional**: the decision requires ≥300 real RatEyeGroundTruth captures; this run has {total} samples from `{set_name}`."
        );
    }
    let _ = writeln!(r, "\n{CONTEXT}");
    let _ = writeln!(r, "## 2. Results\n");
    let _ = writeln!(r, "|Method|Top-1 %|Top-1 % (identical icons merged)|Top-5 %|Reject %|p50 ms|p95 ms|max ms|OCR trigger %|");
    let _ = writeln!(r, "|---|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (name, t1, t1e, t5, rej, p50, p95, max, ocr) in &rows {
        let ocr = ocr.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into());
        let _ = writeln!(r, "|{name}|{t1:.2}|{t1e:.2}|{t5:.2}|{rej:.2}|{p50:.2}|{p95:.2}|{max:.2}|{ocr}|");
    }
    let _ = writeln!(
        r,
        "\nLatency is wall time of one `identify` call, samples run sequentially (methods parallelise internally). \
        \"Identical icons merged\" counts a prediction as correct when its reference image is byte-identical to the truth's; it is informational only."
    );

    let _ = writeln!(r, "\n## 3. Decision\n");
    let _ = writeln!(
        r,
        "Rule: (1) highest top-1 wins; (2) methods within {TIE_PP:.1} pp of the best top-1 are tied and the lowest p95 among them wins; \
        (3) target is top-1 ≥ {TARGET_TOP1}% and p95 ≤ {TARGET_P95_MS} ms on CPU.\n"
    );
    let best_t1 = rows.iter().map(|r| r.1).fold(f64::MIN, f64::max);
    let winner = rows
        .iter()
        .filter(|r| best_t1 - r.1 <= TIE_PP)
        .min_by(|a, b| a.6.total_cmp(&b.6))
        .context("no methods")?;
    let _ = writeln!(r, "**Winner: `{}`** — top-1 {:.2}%, p95 {:.2} ms.", winner.0, winner.1, winner.6);
    let meets: Vec<&str> = rows.iter().filter(|r| r.1 >= TARGET_TOP1 && r.6 <= TARGET_P95_MS).map(|r| r.0).collect();
    if meets.is_empty() {
        let _ = writeln!(r, "\n**No method meets the target.** Confusions to address next are listed below.");
    } else {
        let _ = writeln!(r, "\nMethods meeting the target: {}.", meets.join(", "));
    }

    let _ = writeln!(r, "\n## 4. Worst confusions (truth → predicted ×count)\n");
    for (m, st) in methods.iter().zip(&stats) {
        let mut c: Vec<_> = st.confusions.iter().collect();
        c.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        let list: Vec<String> = c.iter().take(20).map(|((t, p), n)| format!("`{t}` → `{p}` ×{n}")).collect();
        let _ = writeln!(r, "- **{}** ({} misses, {} rejects): {}", m.name(), total - st.top1 - st.rejects, st.rejects,
            if list.is_empty() { "none".into() } else { list.join(", ") });
    }

    let fixtures: Vec<&str> = log.lines().filter(|l| l.starts_with("fixture-")).collect();
    if !fixtures.is_empty() || total <= 50 {
        let _ = writeln!(r, "\n## 5. Per-sample predictions\n");
        let _ = writeln!(r, "|File|Truth|Slots|Method|Predicted|Score|OK|ms|\n|---|---|---|---|---|---:|---|---:|");
        for l in log.lines().skip(1).filter(|l| total <= 50 || l.starts_with("fixture-")) {
            let _ = writeln!(r, "|{}|", l.replace('\t', "|"));
        }
    }
    let log_path = out.with_extension("samples.tsv");
    let _ = writeln!(r, "\nFull per-sample log: `{}`.", log_path.display());

    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, &r)?;
    std::fs::write(&log_path, log)?;
    println!("{}", &r[r.find("## 2.").unwrap_or(0)..r.find("## 4.").unwrap_or(r.len())]);
    println!("report: {}", out.display());
    Ok(())
}
