use chadvision::catalog::Catalog;
use chadvision::ffi::*;
use std::ffi::CString;

fn open() -> Option<*mut CvEngine> {
    if !std::path::Path::new("data/models").exists() {
        eprintln!("data/models missing; skipping");
        return None;
    }
    let dir = CString::new("data").unwrap();
    let e = cv_open(dir.as_ptr());
    if e.is_null() {
        let mut buf = [0 as std::ffi::c_char; 512];
        cv_last_error(buf.as_mut_ptr(), 512);
        panic!("cv_open failed: {}", unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy());
    }
    Some(e)
}

fn bgr(path: &str) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    (img.pixels().flat_map(|p| [p[2], p[1], p[0]]).collect(), w, h)
}

fn top_id(out: &[CvCandidate]) -> &str {
    std::str::from_utf8(&out[0].id).unwrap().trim_end_matches('\0')
}

#[test]
fn identifies_propital_via_c_abi() {
    let Some(e) = open() else { return };
    let (crop, w, h) = bgr("data/real/fixture-propital.png");
    let mut out: Vec<CvCandidate> = (0..5).map(|_| CvCandidate { id: [0; 32], score: 0.0 }).collect();
    let n = cv_identify(e, crop.as_ptr(), w, h, w * 3, 1, 1, std::ptr::null(), 0, 0, 0, 0, 0, 1.0, out.as_mut_ptr(), 5);
    assert!(n > 0);
    assert_eq!(top_id(&out), "5c0e530286f7747fa1419862");
    cv_close(e);
}

/// Real 3440x1440 scan: the icon stage alone picks Slug by 0.002, EFT's tooltip reads "Aluminum splint".
#[test]
fn tooltip_name_overrides_icon() {
    let Some(e) = open() else { return };
    let catalog = Catalog::load(std::path::Path::new("data/refs")).unwrap();
    let truth = catalog.items.iter().find(|i| i.name == "Aluminum splint").expect("catalog entry").id.clone();
    let (crop, w, h) = bgr("tests/fixtures/scan-alu-splint-crop.png");
    let (screen, sw, sh) = bgr("tests/fixtures/scan-alu-splint-screen.png");
    let mut out: Vec<CvCandidate> = (0..5).map(|_| CvCandidate { id: [0; 32], score: 0.0 }).collect();
    let n = cv_identify(
        e, crop.as_ptr(), w, h, w * 3, 1, 1,
        screen.as_ptr(), sw, sh, sw * 3, (sw / 2) as i32, (sh / 2) as i32, 4.0 / 3.0,
        out.as_mut_ptr(), 5,
    );
    assert!(n > 0);
    assert_eq!(top_id(&out), truth);
    cv_close(e);
}

/// Real scan of 5.45x39mm PS gs (1×1) where the cell came from a wrong RatEye box (1×2):
/// the tooltip name must still win, so names are matched across all sizes.
#[test]
fn tooltip_name_ignores_wrong_slot_size() {
    let Some(e) = open() else { return };
    let catalog = Catalog::load(std::path::Path::new("data/refs")).unwrap();
    let truth = catalog.items.iter().find(|i| i.name == "5.45x39mm PS gs").expect("catalog entry").id.clone();
    let (crop, w, h) = bgr("tests/fixtures/scan-ps-wrong-slots-crop.png");
    let (screen, sw, sh) = bgr("tests/fixtures/scan-ps-wrong-slots-screen.png");
    let mut out: Vec<CvCandidate> = (0..5).map(|_| CvCandidate { id: [0; 32], score: 0.0 }).collect();
    let n = cv_identify(
        e, crop.as_ptr(), w, h, w * 3, 1, 2,
        screen.as_ptr(), sw, sh, sw * 3, (sw / 2) as i32, (sh / 2) as i32, 4.0 / 3.0,
        out.as_mut_ptr(), 5,
    );
    assert!(n > 0);
    assert_eq!(top_id(&out), truth);
    cv_close(e);
}

/// Real scan: tooltip reads "5.45x39mm PS gs", the icon's top-5 holds only "5.45x39mm PRS gs"
/// (one letter off, 0.93). The exact catalog-wide name must win. `tooltip.png` is ChadScanner's
/// separate tooltip capture, cursor at (60·s, 140·s).
#[test]
fn exact_tooltip_name_beats_near_name_in_icon_pool() {
    let Some(e) = open() else { return };
    let catalog = Catalog::load(std::path::Path::new("data/refs")).unwrap();
    let truth = catalog.items.iter().find(|i| i.name == "5.45x39mm PS gs").expect("catalog entry").id.clone();
    let (crop, w, h) = bgr("tests/fixtures/scan-ps-vs-prs-crop.png");
    let (screen, sw, sh) = bgr("tests/fixtures/scan-ps-vs-prs-tooltip.png");
    let s = 4.0f32 / 3.0;
    let mut out: Vec<CvCandidate> = (0..5).map(|_| CvCandidate { id: [0; 32], score: 0.0 }).collect();
    let n = cv_identify(
        e, crop.as_ptr(), w, h, w * 3, 1, 1,
        screen.as_ptr(), sw, sh, sw * 3, (60.0 * s) as i32, (140.0 * s) as i32, s,
        out.as_mut_ptr(), 5,
    );
    assert!(n > 0);
    assert_eq!(top_id(&out), truth);
    cv_close(e);
}
