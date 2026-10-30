use crate::catalog::Catalog;
use crate::dataset::{read_labels, read_tooltips, Sample};
use crate::methods::{Identifier, Input, Tooltip};
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
- **tooltip**: name-only. The bright text line nearest the cursor in the hover tooltip is OCR'd (PP-OCRv5) and matched by folded Levenshtein against full names of same-size items; reject below 0.8.
- **icon+tooltip**: two-stage. `fusion` answers first; when a tooltip is present its name (score ≥ 0.8, first among fusion's top-5, then all same-size items) replaces the icon answer.
"#;

#[derive(Default)]
struct Stats {
    top1: usize,
    top1_equiv: usize,
    top5: usize,
    rejects: usize,
    ms: Vec<f64>,
    first_ms: Vec<f64>,
    /// Two-stage outcomes: icon wrong→final right, right→wrong, right→right, wrong→wrong, tooltip read.
    fixed: usize,
    broken: usize,
    kept_right: usize,
    kept_wrong: usize,
    attempted: usize,
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
    let tip_map = read_tooltips(set)?;
    let tips: Vec<Option<Tooltip>> = samples
        .iter()
        .map(|s| {
            tip_map
                .get(&s.file)
                .map(|(f, mouse, scale)| {
                    let image = image::open(set.join(f)).with_context(|| f.clone())?.to_rgb8();
                    Ok(Tooltip { image, mouse: *mouse, scale: *scale })
                })
                .transpose()
        })
        .collect::<Result<_>>()?;
    let inputs: Vec<Input> =
        samples.iter().zip(&crops).zip(&tips).map(|((s, c), t)| Input { crop: c, slots: s.slots, tooltip: t.as_ref() }).collect();
    if let Some(i) = inputs.first() {
        for m in methods {
            m.identify(i);
        }
    }
    let warm_ocr: Vec<usize> = methods.iter().map(|m| m.ocr_runs().unwrap_or(0)).collect();

    let set_name = set.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut log = String::from("file\ttruth\tslots\tmethod\tpred\tscore\tok\tms\n");
    let mut stats: Vec<Stats> = methods.iter().map(|_| Stats::default()).collect();
    let progress = Instant::now();
    for (n, (s, input)) in samples.iter().zip(&inputs).enumerate() {
        let truth_short = catalog.short_name(&s.item_id).to_string();
        let truth_group = catalog.index(&s.item_id).map(|i| groups[i]);
        for (m, st) in methods.iter().zip(stats.iter_mut()) {
            let t = Instant::now();
            let cands = m.identify(input);
            let ms = t.elapsed().as_secs_f64() * 1e3;
            st.ms.push(ms);
            let pred = cands.first();
            let ok = pred.is_some_and(|c| c.item_id == s.item_id);
            match m.last_stages() {
                Some(stg) => {
                    st.first_ms.push(stg.first_ms);
                    st.attempted += stg.used_tooltip as usize;
                    let first_ok = stg.first_id.as_deref() == Some(s.item_id.as_str());
                    match (first_ok, ok) {
                        (false, true) => st.fixed += 1,
                        (true, false) => st.broken += 1,
                        (true, true) => st.kept_right += 1,
                        (false, false) => st.kept_wrong += 1,
                    }
                }
                None => st.first_ms.push(ms),
            }
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
        st.first_ms.sort_by(f64::total_cmp);
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
            quantile(&st.first_ms, 0.95),
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
    let tip_cov = pct(tips.iter().filter(|t| t.is_some()).count(), total);
    let _ = writeln!(r, "- Tooltip coverage: {tip_cov:.1}% of samples");
    if set_name != "real" && tip_cov > 0.0 {
        let _ = writeln!(
            r,
            "- **Synthetic tooltips**: rendered with DejaVu Sans on a flat box, not Tarkov's font or styling; `tooltip`/`icon+tooltip` numbers are for development only."
        );
    }
    let _ = writeln!(r, "\n{CONTEXT}");
    let _ = writeln!(r, "## 2. Results\n");
    let _ = writeln!(r, "|Method|Top-1 %|Top-1 % (identical icons merged)|Top-5 %|Reject %|p50 ms|p95 ms|first p95 ms|max ms|OCR trigger %|Tooltip coverage %|");
    let _ = writeln!(r, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (name, t1, t1e, t5, rej, p50, p95, max, ocr, first) in &rows {
        let ocr = ocr.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into());
        let _ = writeln!(r, "|{name}|{t1:.2}|{t1e:.2}|{t5:.2}|{rej:.2}|{p50:.2}|{p95:.2}|{first:.2}|{max:.2}|{ocr}|{tip_cov:.1}|");
    }
    let _ = writeln!(
        r,
        "\nLatency is wall time of one `identify` call, samples run sequentially (methods parallelise internally). \
        \"first p95\" is time to the first displayed answer (the icon stage for `icon+tooltip`, otherwise = p95). \
        \"Identical icons merged\" counts a prediction as correct when its reference image is byte-identical to the truth's; it is informational only."
    );

    for (m, st) in methods.iter().zip(&stats).filter(|(m, _)| m.last_stages().is_some()) {
        let _ = writeln!(r, "\n### Correction stage: `{}`\n", m.name());
        let _ = writeln!(r, "|Fixed (icon wrong → right)|Broken (icon right → wrong)|Unchanged right|Unchanged wrong|Corrections attempted (tooltip read)|");
        let _ = writeln!(r, "|---:|---:|---:|---:|---:|");
        let _ = writeln!(r, "|{}|{}|{}|{}|{}|", st.fixed, st.broken, st.kept_right, st.kept_wrong, st.attempted);
    }

    let _ = writeln!(r, "\n## 3. Decision\n");
    let _ = writeln!(
        r,
        "Rule: (1) highest top-1 wins; (2) methods within {TIE_PP:.1} pp of the best top-1 are tied and the lowest p95 among them wins; \
        (3) target is top-1 ≥ {TARGET_TOP1}% and p95 ≤ {TARGET_P95_MS} ms on CPU. \
        For `icon+tooltip` latency is judged on first p95 (time to the icon answer); the tooltip correction arrives later.\n"
    );
    let latency = |r: &(&str, f64, f64, f64, f64, f64, f64, f64, Option<f64>, f64)| if r.0 == "icon+tooltip" { r.9 } else { r.6 };
    let best_t1 = rows.iter().map(|r| r.1).fold(f64::MIN, f64::max);
    let winner = rows
        .iter()
        .filter(|r| best_t1 - r.1 <= TIE_PP)
        .min_by(|a, b| latency(a).total_cmp(&latency(b)))
        .context("no methods")?;
    let _ = writeln!(r, "**Winner: `{}`** — top-1 {:.2}%, p95 {:.2} ms, first p95 {:.2} ms.", winner.0, winner.1, winner.6, winner.9);
    let meets: Vec<&str> = rows.iter().filter(|r| r.1 >= TARGET_TOP1 && latency(r) <= TARGET_P95_MS).map(|r| r.0).collect();
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
