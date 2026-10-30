//! Short-name normalisation and fuzzy matching for OCR output.

/// Lowercase, fold OCR-confusable digits/symbols and Cyrillic look-alikes onto Latin,
/// then drop whitespace and punctuation except `.` and `-`.
pub fn fold(s: &str) -> String {
    s.chars()
        .flat_map(char::to_lowercase)
        .filter_map(|c| {
            let c = match c {
                '0' => 'o',
                '1' | 'i' | '|' => 'l',
                '5' => 's',
                '8' => 'b',
                '6' => 'g',
                '2' => 'z',
                'т' => 't',
                'а' => 'a',
                'е' => 'e',
                'о' => 'o',
                'р' => 'p',
                'с' => 'c',
                'х' => 'x',
                'к' => 'k',
                'м' => 'm',
                'н' => 'h',
                'в' => 'b',
                c => c,
            };
            (c.is_alphanumeric() || c == '.' || c == '-').then_some(c)
        })
        .collect()
}

pub fn score(ocr: &str, short_name: &str) -> f32 {
    strsim::normalized_levenshtein(&fold(ocr), &fold(short_name)) as f32
}

pub const REJECT_BELOW: f32 = 0.5;

/// Rank `(index, short_name)` candidates against OCR text; top `k`, empty if the best is below the reject threshold.
pub fn rank<'a>(ocr: &str, cands: impl Iterator<Item = (usize, &'a str)>, k: usize) -> Vec<(usize, f32)> {
    let folded = fold(ocr);
    let mut scored: Vec<(usize, f32)> =
        cands.map(|(i, s)| (i, strsim::normalized_levenshtein(&folded, &fold(s)) as f32)).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    if scored.first().is_none_or(|c| c.1 < REJECT_BELOW) {
        return Vec::new();
    }
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_confusables() {
        assert_eq!(fold("Pr0pital"), fold("Propital"));
        assert_eq!(fold("МОРФИН"), fold("морфин"));
        assert_eq!(fold("М 855А1"), fold("m855a1"));
    }

    #[test]
    fn near_miss_scores_high() {
        assert!(score("Morphlne", "Morphine") > 0.85);
    }

    #[test]
    fn low_score_rejects() {
        let names = ["Propital", "Morphine", "Water"];
        assert!(rank("xqzvw", names.iter().copied().enumerate(), 5).is_empty());
        let r = rank("Pr0pital", names.iter().copied().enumerate(), 5);
        assert_eq!(r[0].0, 0);
    }
}
