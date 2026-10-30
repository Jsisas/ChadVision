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

/// AK-12 magazine in a 1×2 tactical rig slot, scanned without a RatEye icon (hint 1×1). The rig
/// grid lightens towards the top, so the top border (78,95,101) was not grid-coloured; the 1-slot
/// hint then cut the cell to the magazine's lower half, which matched an armor.
#[test]
fn rig_slot_top_border_is_lighter_than_the_grid() {
    let img = load("grid-rig-1x2-mag.png");
    let cell = locate_cell(&img, (597, 597), SCALE_1440P, (1, 1)).map(|c| (c.x, c.y, c.slots));
    assert_eq!(cell, Some((564, 510, (1, 2))));
}

/// Top stash rows: grid lines are lighter there ((78,93,100) vs (73,81,84)), so the hovered
/// Bandage's top border was missed and the cell merged with the Vaseline above it (1×2).
#[test]
fn top_row_lines_are_lighter() {
    let img = load("grid-top-row-hover-bandage.png");
    let cell = locate_cell(&img, (597, 227), SCALE_1440P, (1, 1)).unwrap();
    // One slot at 1440p is 84 px; lines are 2 px wide, so a few px either way.
    assert_eq!(cell.slots, (1, 1));
    assert!((cell.w - 84).abs() <= 8 && (cell.h - 84).abs() <= 8, "{cell:?}");
}

/// Mouse just under the stash's top border: a probe band centred on the mouse reaches into the
/// header, where no vertical lines exist, and the 7mm shell was read as 7×1.
#[test]
fn mouse_on_top_border_of_top_row() {
    let img = load("grid-top-row-mouse-on-border.png");
    let cell = locate_cell(&img, (597, 118), SCALE_1440P, (1, 1)).unwrap();
    // One slot at 1440p is 84 px; lines are 2 px wide, so a few px either way.
    assert_eq!(cell.slots, (1, 1));
    assert!((cell.w - 84).abs() <= 8 && (cell.h - 84).abs() <= 8, "{cell:?}");
}
