use std::ffi::CString;
use tv::ffi::*;

#[test]
fn identifies_propital_via_c_abi() {
    if !std::path::Path::new("data/models").exists() {
        eprintln!("data/models missing; skipping");
        return;
    }
    let dir = CString::new("data").unwrap();
    let e = tv_open(dir.as_ptr());
    if e.is_null() {
        let mut buf = [0 as std::ffi::c_char; 512];
        tv_last_error(buf.as_mut_ptr(), 512);
        panic!("tv_open failed: {}", unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }.to_string_lossy());
    }
    let img = image::open("data/real/fixture-propital.png").unwrap().to_rgb8();
    let (w, h) = img.dimensions();
    let bgr: Vec<u8> = img.pixels().flat_map(|p| [p[2], p[1], p[0]]).collect();
    let mut out: Vec<TvCandidate> = (0..5).map(|_| TvCandidate { id: [0; 32], score: 0.0 }).collect();
    let n = tv_identify(e, bgr.as_ptr(), w, h, w * 3, 1, 1, out.as_mut_ptr(), 5);
    assert!(n > 0);
    let id = std::str::from_utf8(&out[0].id).unwrap().trim_end_matches('\0');
    assert_eq!(id, "5c0e530286f7747fa1419862");
    tv_close(e);
}
