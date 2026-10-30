//! C ABI for embedding the Fusion icon identifier (see `TvCandidate`).
use crate::catalog::Catalog;
use crate::methods::{embed::Embed, fusion::Fusion, ocr::Ocr, Identifier, Input};
use crate::refs;
use image::RgbImage;
use std::cell::RefCell;
use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Arc;

/// `id` is a NUL-padded ASCII tarkov.dev item id.
#[repr(C)]
pub struct TvCandidate {
    pub id: [u8; 32],
    pub score: f32,
}

pub struct TvEngine {
    #[allow(dead_code)]
    catalog: Arc<Catalog>,
    fusion: Fusion,
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic".into())
}

fn open(dir: &Path) -> anyhow::Result<TvEngine> {
    let refs_dir = dir.join("refs");
    let models = dir.join("models");
    let catalog = Arc::new(Catalog::load(&refs_dir)?);
    let bases = refs::load_bases(&refs_dir, &catalog);
    let embed = Arc::new(Embed::new(catalog.clone(), &bases, &models.join("dinov2-small.onnx"), &dir.join("index"))?);
    let ocr = Arc::new(Ocr::new(
        catalog.clone(),
        &models.join("ppocrv5_latin_rec.onnx"),
        &models.join("ppocrv5_latin_dict.txt"),
    )?);
    drop(bases);
    let fusion = Fusion::new(catalog.clone(), embed, ocr);
    Ok(TvEngine { catalog, fusion })
}

/// Returns null on error; see `tv_last_error`.
#[no_mangle]
pub extern "C" fn tv_open(data_dir: *const c_char) -> *mut TvEngine {
    if data_dir.is_null() {
        set_error("data_dir is null".into());
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let dir = unsafe { CStr::from_ptr(data_dir) }.to_str()?;
        open(Path::new(dir))
    }));
    match result {
        Ok(Ok(e)) => Box::into_raw(Box::new(e)),
        Ok(Err(e)) => {
            set_error(format!("{e:#}"));
            std::ptr::null_mut()
        }
        Err(p) => {
            set_error(panic_message(p));
            std::ptr::null_mut()
        }
    }
}

/// `bgr`: rows of 3-byte B,G,R pixels, `stride` bytes per row. Returns candidate count or -1.
#[no_mangle]
pub extern "C" fn tv_identify(
    e: *const TvEngine,
    bgr: *const u8,
    width: u32,
    height: u32,
    stride: u32,
    slot_w: u32,
    slot_h: u32,
    out: *mut TvCandidate,
    cap: u32,
) -> i32 {
    if e.is_null() || bgr.is_null() || out.is_null() || width == 0 || height == 0 || stride < width.saturating_mul(3) {
        set_error("invalid argument".into());
        return -1;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let engine = unsafe { &*e };
        let (w, h, stride) = (width as usize, height as usize, stride as usize);
        let src = unsafe { std::slice::from_raw_parts(bgr, stride * (h - 1) + w * 3) };
        let mut buf = Vec::with_capacity(w * h * 3);
        for row in 0..h {
            for px in src[row * stride..row * stride + w * 3].chunks_exact(3) {
                buf.extend_from_slice(&[px[2], px[1], px[0]]);
            }
        }
        let crop = RgbImage::from_raw(width, height, buf).expect("buffer size");
        let cands = engine.fusion.identify(&Input { crop: &crop, slots: (slot_w, slot_h), tooltip: None });
        let out = unsafe { std::slice::from_raw_parts_mut(out, cap as usize) };
        let mut n = 0;
        for (slot, c) in out.iter_mut().zip(cands) {
            let mut id = [0u8; 32];
            let bytes = c.item_id.as_bytes();
            let len = bytes.len().min(31);
            id[..len].copy_from_slice(&bytes[..len]);
            *slot = TvCandidate { id, score: c.score };
            n += 1;
        }
        n as i32
    }));
    match result {
        Ok(n) => n,
        Err(p) => {
            set_error(panic_message(p));
            -1
        }
    }
}

#[no_mangle]
pub extern "C" fn tv_close(e: *mut TvEngine) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}

/// Copies the calling thread's last error into `buf` (NUL-terminated); returns bytes written excl. NUL.
#[no_mangle]
pub extern "C" fn tv_last_error(buf: *mut c_char, cap: u32) -> u32 {
    if buf.is_null() || cap == 0 {
        return 0;
    }
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        let n = e.len().min(cap as usize - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(e.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
        n as u32
    })
}
