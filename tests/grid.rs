use chadvision::grid::{locate_cell, Cell};

const SCALE_1440P: f32 = 4.0 / 3.0;

fn load(name: &str) -> image::RgbImage {
    image::open(format!("tests/fixtures/{name}")).unwrap().to_rgb8()
}

/// RatEye boxed this hovered Screws 20 px too high and 25 % too large; the crop then included EFT's
/// "Pack of screws" tooltip, OCR read "P" and fusion nearly picked CALOK-B. The stash viewport cuts
/// the bottom row, so the bottom edge comes from the 1-slot hint.
#[test]
fn hovered_bottom_row_cell_with_eft_tooltip() {
    let img = load("grid-hover-bottom-row.png");
    let cell = locate_cell(&img, (197, 197), SCALE_1440P, (1, 1));
    assert_eq!(cell, Some(Cell { x: 157, y: 173, w: 84, h: 84, slots: (1, 1) }));
}

#[test]
fn item_sizes_come_from_their_border_lines() {
    let img = load("grid-stash-1440p.png");
    // Hints are deliberately wrong: with both edges visible they must be ignored.
    let at = |m| locate_cell(&img, m, SCALE_1440P, (3, 3)).map(|c| (c.x, c.y, c.slots));
    assert_eq!(at((525, 455)), Some((483, 414, (1, 1))), "Screws 1x1");
    assert_eq!(at((400, 455)), Some((315, 414, (2, 1))), "SP-81 2x1");
    assert_eq!(at((160, 150)), Some((63, 78, (2, 2))), "Grizzly 2x2");
    assert_eq!(at((189, 370)), Some((147, 330, (1, 1))), "IFAK 1x1");
}
